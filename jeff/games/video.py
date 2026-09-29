"""Record a game episode as an MP4: each frame is the game view with a caption underneath showing the decision the player
made that turn (the options, the probability the model gave each, the chosen one highlighted, and the time it took)."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 640
BACKGROUND, TEXT, MUTED, CHOSEN = (18, 18, 24), (235, 235, 240), (150, 150, 165), (120, 200, 255)


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def wrap(text: str, width: int, face: ImageFont.ImageFont, draw: ImageDraw.ImageDraw) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=face) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line] if line else lines


def frame(view: Image.Image, title: str, options: dict[str, str], chosen: str,
          probabilities: dict[str, float] | None, milliseconds: float | None) -> Image.Image:
    """The game view scaled to the frame width, with the decision caption below it (as tall as the options need)."""
    scale = WIDTH / view.width
    view = view.convert("RGB").resize((WIDTH, round(view.height * scale)), Image.NEAREST)
    small, normal = font(15), font(17)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[tuple[int, str, tuple[int, int, int]]] = []
    for key, text in options.items():
        share = f"{probabilities[key]:.0%} " if probabilities and key in probabilities else ""
        colour = CHOSEN if key == chosen else TEXT
        marker = "> " if key == chosen else "  "
        first, *rest = wrap(f"{marker}{share}{key}: {text}".rstrip(": "), WIDTH - 24, normal, measure)
        lines.append((12, first, colour))
        if rest:
            lines.extend((40, line, colour) for line in wrap(" ".join(rest), WIDTH - 52, normal, measure))
    height = view.height + 40 + 21 * len(lines) + 10
    canvas = Image.new("RGB", (WIDTH, height), BACKGROUND)
    canvas.paste(view, (0, 0))
    draw = ImageDraw.Draw(canvas)
    timing = f" · {milliseconds:.0f} ms" if milliseconds is not None else ""
    draw.text((12, view.height + 8), f"{title}{timing}", fill=MUTED, font=small)
    y = view.height + 32
    for x, line, colour in lines:
        draw.text((x, y), line, fill=colour, font=normal)
        y += 21
    return canvas


def write(frames: list[Image.Image], path: Path, fps: float) -> None:
    import imageio.v2 as imageio
    import numpy as np

    if not frames:
        raise ValueError("No frames to write")
    height = max(f.height for f in frames)
    padded = []
    for f in frames:  # every frame must be the same size
        canvas = Image.new("RGB", (WIDTH, height + height % 2), BACKGROUND)
        canvas.paste(f, (0, 0))
        padded.append(np.asarray(canvas))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, padded, fps=fps, codec="libx264", quality=7, macro_block_size=1)
