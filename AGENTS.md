# Repository Guidelines

## Project Structure & Module Organization
- `lol_collector/`: main Python project.
  - `collect.py`: CLI entrypoint.
  - `pipeline.py`: 4-stage orchestration (players → matches → metadata → timelines).
  - `riot_api.py`: async Riot API client.
  - `rate_limiter.py`: per-key rate limits + 429 cooldown.
  - `state.py`: SQLite checkpointing/resume (`collection_state.db`).
  - `export.py`: Parquet writers.
- `lol_collector/draft_encoder/`: planned/experimental draft-embedding work (training/eval scripts).
- `lol_collector/data/`: local artifacts (Parquet outputs, logs, SQLite state). Don’t treat as source.
- Root `data/`: additional artifact storage.

## Build, Test, and Development Commands
Run from `lol_collector/`:
- `pip install -r requirements.txt`: install collector deps.
- `python collect.py`: run the full pipeline.
- `python collect.py --stage players|matches|metadata|timelines`: run a single stage.
- `python collect.py --stats`: show progress from the SQLite state DB.
- `python collect.py --config <file>.yaml`: use an alternate config.

## Coding Style & Naming Conventions
- Python: 4-space indentation; keep modules single-responsibility.
- Naming: `snake_case` (functions/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- Prefer explicit boundaries: API calls in `riot_api.py`, persistence in `state.py`, file outputs in `export.py`.

## Testing Guidelines
No dedicated test suite is present. If adding tests:
- Use `pytest` and place tests under `lol_collector/tests/` as `test_*.py`.
- Don’t hit live Riot endpoints; mock HTTP and use small JSON fixtures.

## Commit & Pull Request Guidelines
- Commit messages: imperative + scoped when helpful (e.g., `collector: handle Retry-After`).
- PRs should include: summary/rationale, how to validate (exact commands), and any data/schema changes (Parquet columns, SQLite migrations).

## Security & Configuration Tips
- Never commit real Riot API keys. Keep `lol_collector/config.yaml` local; use `config.example.yaml` for templates.
- Avoid running multiple collector instances against the same `collection_state.db` (SQLite contention). Use separate output/state paths.
