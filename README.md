# LoLDraft

An end-to-end League of Legends data project for studying how champion drafts
relate to early gold, objective control, comebacks, and victory.

## Project applications

- **Draft Analyzer** (`lol_collector/draft_analyzer/`): the current research
  application. It provides patch-aware, evidence-thresholded analysis of
  objectives, gold trajectories, comebacks, lane matchups, and composition
  indicators.
- **Draft Encoder** (`lol_collector/draft_encoder/`): the original neural draft
  predictor and training/evaluation workflow. It remains available as a safe
  baseline and fallback.
- **Collector** (`lol_collector/`): an asynchronous, resumable Riot API pipeline
  for match metadata and timeline/event Parquet exports.

## Quick start: Draft Analyzer

Install dependencies from the relevant project directories:

```powershell
cd lol_collector
python -m pip install -r requirements.txt
cd draft_encoder
python -m pip install -r requirements.txt
cd ..\draft_analyzer
```

Build canonical match-level artifacts from a local processed Riot dataset:

```powershell
python analysis.py `
  --data-dir ..\draft_encoder\data\processed_riot `
  --output-dir analysis_output
```

Launch the research application:

```powershell
streamlit run app.py
```

The analyzer reports historical associations rather than guaranteed
predictions. It deduplicates side-swapped rows and exposes evidence size,
comparison breadth, confidence intervals, patch scope, objective state, and
partial-draft fallbacks.

See [lol_collector/draft_analyzer/README.md](lol_collector/draft_analyzer/README.md)
for the research question and analysis details.

## Collector

Create a local configuration from the ignored template:

```powershell
cd lol_collector
Copy-Item config.example.yaml config.yaml
# Edit config.yaml locally with your Riot API key.
```

Run collector commands from `lol_collector/`:

```powershell
python collect.py --stats
python collect.py --stage players
python collect.py --stage matches
python collect.py --stage metadata
python collect.py --stage timelines
```

The collector stores resumable SQLite state and downloaded Parquet data under
`lol_collector/data/`. Do not run multiple collector processes against the same
state database.

## Draft Encoder fallback

Run from `lol_collector/draft_encoder/`:

```powershell
python prepare_riot_dataset.py --help
python train.py --config config\default.yaml --data-dir data\processed_riot
python train.py --config config\default.yaml --data-dir data\processed_riot --resume auto
python evaluate.py --checkpoint checkpoints\best.pt --data-dir data\processed_riot
streamlit run app.py
```

Training supports atomic recovery checkpoints, compatibility validation,
timestamped backups, and explicit resume. Checkpoints and evaluation outputs
are local artifacts and are excluded from Git.

## Original offline pipeline

The root scripts (`prepare_kaggle_data.py`, `extract_embeddings.py`,
`clustering.py`, `recommendation_engine.py`, and `dashboard.py`) support the
earlier Kaggle-based embedding, clustering, recommendation, and visualization
workflow. The current analyzer is the recommended entry point for the senior
project.

## Data and repository hygiene

Downloaded datasets, logs, SQLite state, model checkpoints, Parquet files,
evaluation outputs, and local credentials are excluded by `.gitignore`.
Never commit a Riot API key. Use `config.example.yaml` as the shareable
configuration template.

The shared `5mLoLGames/` CSV source data and the canonical processed Riot
Parquet artifacts are intentionally tracked with Git LFS so teammates can
clone the project with its reproducible data without putting large binaries in
normal Git history. Install Git LFS before cloning or pulling:

```powershell
git lfs install
git lfs pull
```

There is no dedicated test suite or configured linter. Validate Python changes
with:

```powershell
python -m py_compile path\to\changed_file.py
```
