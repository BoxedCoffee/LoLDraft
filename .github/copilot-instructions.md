# Copilot instructions for LoLDraft

## Project overview

LoLDraft is a Python League of Legends draft-analysis project with two connected workflows:

- `lol_collector/` is an asynchronous, resumable Riot API collection pipeline. It discovers high-elo players, discovers match IDs, fetches filtered match metadata, fetches timelines, checkpoints progress in SQLite, and exports Parquet data.
- `lol_collector/draft_encoder/` is the draft-embedding and prediction workflow. It prepares match data, trains a multi-task PyTorch model, evaluates it, extracts embeddings, and exposes Streamlit inference applications. Its requirements also include the scikit-learn and plotting dependencies used by the clustering/evaluation workflow.
- Root-level scripts such as `prepare_kaggle_data.py`, `extract_embeddings.py`, `clustering.py`, and `recommendation_engine.py` form the offline analysis path from processed drafts to embeddings, archetypes, and recommendations.

The repository also contains Data Dragon assets and large local datasets/model checkpoints. Treat generated data, logs, databases, checkpoints, and downloaded assets as runtime artifacts rather than source code.

## Build, test, and lint commands

There is no build system, dedicated test suite, or configured linter in the repository.

Run commands from the directory that contains the script, unless a command below uses an explicit path.

### Collector

From `lol_collector/`:

```bash
pip install -r requirements.txt
python collect.py                         # Run all collection stages
python collect.py --stage players
python collect.py --stage matches
python collect.py --stage metadata
python collect.py --stage timelines
python collect.py --stage reconcile
python collect.py --stats                  # Read progress from SQLite
python collect.py --config production.yaml
```

The collector configuration defaults to `lol_collector/config.yaml`. Use `config.example.yaml` as the template for a new local configuration.

### Draft encoder

From `lol_collector/draft_encoder/`:

```bash
pip install -r requirements.txt
python prepare_dataset.py --kaggle-dir <kaggle-data-dir> --output-dir ./data/processed
python train.py --config config/default.yaml
python train.py --config config/default.yaml --data-dir ./data/processed
python evaluate.py --checkpoint checkpoints/best.pt --data-dir ./data/processed
python predict.py --checkpoint checkpoints/best.pt
streamlit run app.py
```

Dataset preparation now validates objective labels and fails when objective
team ownership cannot be resolved. Use `--allow-missing-objectives` only when
intentionally training without those auxiliary targets.

The root-level Kaggle preparation command documented in `README.md` is also supported:

```bash
python prepare_kaggle_data.py --kaggle-dir ./5mLoLGames --output-dir ./data/processed
```

The Streamlit draft predictor expects its processed data and checkpoint paths relative to `lol_collector/draft_encoder/`.

### Tests and linting

No tests currently exist. If tests are added, follow the repository guidance and place them under `lol_collector/tests/` with `test_*.py` names. Use pytest, for example:

```bash
pytest lol_collector/tests/
pytest lol_collector/tests/test_state.py::test_name
```

No lint command is configured; do not introduce a formatter or linter-only change unless the task explicitly calls for it.

## Architecture and data flow

### Collection pipeline

`lol_collector/collect.py` is the CLI boundary. It loads and validates YAML configuration, creates storage directories, configures logging, builds the rate limiter and Riot client, opens the SQLite state database, and runs the pipeline.

`lol_collector/pipeline.py` owns the four-stage flow:

1. Discover Challenger/Grandmaster/Master players per configured region.
2. Pull match-v5 histories and deduplicate match IDs.
3. Fetch match metadata, applying queue, patch, and minimum-duration filters.
4. Fetch timelines and events for metadata-complete matches.

`riot_api.py` owns async HTTP calls, retries, exponential backoff, and Riot `Retry-After` handling. `rate_limiter.py` owns per-key one-second/two-minute limits and global concurrency. `state.py` is the resumability source of truth, including status transitions and run summaries. `export.py` buffers and writes match/timeline/event Parquet output. Keep these responsibilities separate when changing the collector.

The pipeline is intentionally independently resumable. A graceful Ctrl+C should leave SQLite checkpoints and Parquet output usable for a later run.

### Draft encoder and analysis

Dataset preparation produces `drafts.parquet`, vocabulary JSON, split JSON, and optional gold/objective tables. `dataset.py` turns those artifacts into training samples. `model.py` encodes five role-ordered champions per team with shared champion and role embeddings, passes each team through residual MLP blocks, combines blue and red team vectors with an optional patch embedding, and feeds the resulting 256-dimensional draft embedding to multi-task heads for win probability, gold curves, and first objectives. Training uses uncertainty-weighted losses and saves checkpoints used by evaluation, prediction, and Streamlit.

The offline analysis flow is:

```text
processed drafts -> model training/evaluation -> embedding extraction
                 -> clustering -> dashboard/recommendation inputs
```

The root `dashboard.py` consumes clustering and embedding artifacts. The draft-encoder `app.py` loads the model, champion vocabulary, patch vocabulary derived from `drafts.parquet`, and champion metadata for interactive prediction.

## Repository-specific conventions

- Run Python scripts from the directory their relative paths expect. In particular, collector imports and paths assume `lol_collector/`, while the draft encoder app expects `data/processed/` and `checkpoints/` below `lol_collector/draft_encoder/`.
- Preserve the collector boundaries: API access in `riot_api.py`, rate control in `rate_limiter.py`, persistence/checkpointing in `state.py`, and Parquet output in `export.py`.
- The collector uses async I/O and SQLite checkpoints. Avoid synchronous network work in pipeline stages and do not bypass state transitions when adding a fetch operation.
- Do not run multiple collector processes against the same `collection_state.db`; use separate state database and output paths for parallel experiments.
- Do not commit Riot API keys. Keep `lol_collector/config.yaml` local, redact keys from logs and snippets, and update `config.example.yaml` when configuration shape changes.
- Treat champion IDs as vocabulary indices, with `<UNK>` at index `0`; draft tensors are five champions ordered `[top, jng, mid, bot, sup]` for each team.
- Keep training and inference model dimensions, champion vocabulary, patch vocabulary, and checkpoint configuration aligned. The app should derive patch information from the actual processed draft data rather than from split metadata.
- Preserve side-swap augmentation and train/validation/test split grouping: augmented copies must remain in the same split as their original match.
- Use four-space Python indentation and the existing `snake_case`, `PascalCase`, and `UPPER_SNAKE_CASE` naming conventions.
- Keep generated Parquet files, SQLite databases, logs, embeddings, clustering results, and model checkpoints out of source-oriented changes unless the task explicitly concerns an artifact.

## Validation expectations

For collector changes, prefer `python collect.py --stats` or a single explicitly selected stage with a test/local configuration; do not accidentally start a large live collection. For model changes, validate with the smallest available processed dataset or a CPU configuration before attempting a full GPU training run. Record any schema changes to Parquet outputs or SQLite state when documenting a change.
