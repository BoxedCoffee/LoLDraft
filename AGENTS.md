# Repository Guidelines

## Project Structure & Module Organization
- `lol_collector/`: Python data-collection pipeline (CLI + orchestration).
  - `collect.py`: main entrypoint (`--stage`, `--stats`, `--config`).
  - `pipeline.py`: stage orchestration.
  - `riot_api.py`: async HTTP client for Riot endpoints.
  - `rate_limiter.py`: per-key dual-window limiter + 429 cooldown.
  - `state.py`: SQLite checkpointing/resume state.
  - `export.py`: Parquet writers/output layout.
- `lol_collector/config.yaml`: runtime configuration (API keys, regions, limits).
- `5mLoLGames/` and `*.csv`: sample/processed datasets (treat as data artifacts).
- `backups/`: archived snapshots.

## Build, Test, and Development Commands
Run from `lol_collector/`.
- `pip install -r requirements.txt`: install runtime dependencies.
- `python collect.py`: run the full 4-stage pipeline.
- `python collect.py --stage players|matches|metadata|timelines`: run one stage.
- `python collect.py --stats`: print progress from the SQLite state DB.
- `python collect.py --config production.yaml`: use an alternate config file.

## Coding Style & Naming Conventions
- Language: Python 3 (asyncio/aiohttp).
- Indentation: 4 spaces; keep functions small and single-purpose.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, constants in `UPPER_SNAKE_CASE`.
- Prefer explicit types where practical (e.g., type hints on public functions) and clear docstrings for pipeline stages.

## Testing Guidelines
No dedicated test suite is currently present. When adding tests:
- Use `pytest` and place tests under `lol_collector/tests/`.
- Name files `test_*.py` and keep fixtures local and deterministic.
- Avoid live Riot API calls in CI; mock HTTP (e.g., `aioresponses`) and use small JSON fixtures.

## Commit & Pull Request Guidelines
- Commits: use imperative, scoped messages when possible (e.g., `collector: handle 429 Retry-After`), and keep commits focused.
- PRs should include:
  - A clear description of the change and rationale.
  - Config/data impact notes (e.g., schema changes to Parquet output).
  - Steps to validate locally (exact commands run).

## Security & Configuration Tips
- Never commit real Riot API keys. Keep keys in `config.yaml` locally and redact in examples.
- Treat `data/collection_state.db` as stateful; deleting it resets progress.
