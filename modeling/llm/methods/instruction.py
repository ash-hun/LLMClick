"""Instruction tuning: SFT on instruction/input/output records instead of ready-made conversations."""

from pydantic import BaseModel, Field

from modeling.tuning.method import Row
from modeling.llm.models.base import Message
from modeling.llm.methods.sft import SFT


class InstructionTuning(SFT):
    """Rows: {"instruction": str, "input"?: str, "output": str}. The loss is on the output only."""

    class Config(BaseModel):
        system: str | None = Field(default=None, description="System prompt put in front of every instruction")

    def messages(self, row: Row) -> list[Message]:
        request = f"{row['instruction']}\n\n{row['input']}" if row.get("input") else str(row["instruction"])
        system = [{"role": "system", "content": self.config.system}] if self.config.system else []
        return [*system, {"role": "user", "content": request}, {"role": "assistant", "content": str(row["output"])}]

    def check(self, row: Row) -> None:
        if not isinstance(row["instruction"], str) or not isinstance(row["output"], str):
            raise ValueError("instruction and output must be strings")
