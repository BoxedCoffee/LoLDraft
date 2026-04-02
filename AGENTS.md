# Repository Guidelines

## Project Structure & Module Organization
- `lol_collector/`: Python data-collection pipeline.
  - `collect.py`: CLI entrypoint (`--stage`, `--stats`, `--config`).
  - `pipeline.py`: orchestrates the 4-stage flow.
  - `riot_api.py`: async Riot API client (aiohttp).
  - `rate_limiter.py`: per-key rate limits + 429 cooldown.
  - `state.py`: SQLite checkpointing/resume.
  - `export.py`: Parquet output writers.
- `lol_collector/data/`: local output (Parquet + `collection_state.db`). Treat as artifacts.
- `lol_collector/config.example.yaml`: template config; copy to `config.yaml` locally.

## Build, Test, and Development Commands
Run from `lol_collector/`:
- `pip install -r requirements.txt`: install dependencies.
- `python collect.py`: run the full pipeline (players → matches → metadata → timelines).
- `python collect.py --stage players|matches|metadata|timelines`: run a single stage.
- `python collect.py --stats`: show progress from the SQLite state DB.
- `python collect.py --config <file>.yaml`: run with an alternate config.

## Coding Style & Naming Conventions
- Python: 4-space indentation; prefer small, single-purpose functions.
- Naming: `snake_case` (functions/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- Keep I/O boundaries clear: API code in `riot_api.py`, persistence in `state.py`, transforms/writes in `export.py`.

## Testing Guidelines
No dedicated test suite is present yet. If you add tests:
- Use `pytest` under `lol_collector/tests/` with `test_*.py` naming.
- Avoid live Riot calls; mock HTTP (e.g., `aioresponses`) and use small JSON fixtures.

## Commit & Pull Request Guidelines
- Commits: use imperative, scoped messages when possible (e.g., `collector: handle Retry-After`).
- PRs should include: summary + rationale, how to validate (exact commands), and data/schema impact notes (Parquet columns, state DB migrations).

## Security & Configuration Tips
- Never commit real Riot API keys. Keep `lol_collector/config.yaml` local and redact logs/examples.
- Do not delete `lol_collector/data/collection_state.db` unless you intend to reset progress.
