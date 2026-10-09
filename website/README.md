# LLMClick docs site

A Docusaurus site whose pages are generated from the repository. Nothing under `docs/` is written by hand:

- `intros/<page>.md` holds the handwritten opening of each page (frontmatter and a few paragraphs for a reader who
  has not seen the code);
- `scripts/sync.py` appends the matching sections of `README.md`, `evaluation/README.md` and `data/README.md`, and
  generates the Reference pages from the code (`llmclick recipes`, the Pydantic config classes, the OpenAPI
  description, the CLI help, the settings class);
- `docs/` is the result and is ignored by git.

```bash
npm install --prefix website
npm run sync --prefix website      # uv run python website/scripts/sync.py
npm run start --prefix website     # live preview at http://localhost:3000/LLMClick/
npm run build --prefix website
```

To change a page: edit the README section or the code it comes from, or its intro under `intros/`, then run `sync`.
To add a page: add its intro, list it in `PAGES` of `scripts/sync.py` and in `sidebars.js`.

`.github/workflows/docs.yml` builds the site on every pull request that touches the sources and deploys it to GitHub
Pages on every push to `main`.
