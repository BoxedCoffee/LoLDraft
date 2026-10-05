# League of Legends Draft Predictor

This is a Streamlit application that predicts outcomes of League of Legends drafts using a neural network model trained on professional match data.

## Features
- Predict win probability for a given draft composition
- Show predicted gold curve at 5-minute intervals
- Predict first objectives (baron, dragon, tower, inhibitors)

## Running the Application

1. Navigate to the project directory:
```
cd C:/Users/Hamza/Senior Project/LoLDraft/lol_collector/draft_encoder
```

Training validates the dataset before starting, masks missing auxiliary labels,
uses a reproducible seed, and defaults to zero data-loader workers for reliable
Windows execution. Dataset joins are materialized once at load time so CPU
training does not perform a pandas lookup for every sample. Objective labels
must contain resolvable Blue/Red team IDs;
the preparation command stops when they are missing instead of silently training
an always-zero objective head.

For a dataset where objective events are intentionally unavailable:

```bash
python prepare_dataset.py --kaggle-dir ./kaggle_data --output-dir ./data/processed --allow-missing-objectives
```

This keeps the win/gold tasks trainable but objective predictions remain
unavailable until objective data is regenerated.

Train and evaluate the Riot dataset:

```powershell
python train.py --config config/default.yaml --data-dir .\data\processed_riot
python evaluate.py --checkpoint checkpoints\best.pt --data-dir .\data\processed_riot --output-dir eval_output_riot
```

New training runs select `best.pt` by validation win log loss
(`training.checkpoint_metric: win_logloss`) rather than the combined loss,
because raw late-game gold MSE can otherwise hide weak win predictions.
Evaluation reports accuracy confidence intervals, log loss, Brier score,
ROC-AUC, constant-probability log-loss/Brier baselines, gold baselines, and
objective class counts. The baseline comparison is important: a lower metric
than the baseline is required before treating a model as useful.

Training writes an atomic `latest.pt` recovery checkpoint every epoch and keeps
timestamped backups at the configured interval under `checkpoints/backups`. To resume
the latest compatible checkpoint, use:

```powershell
python train.py --config config/default.yaml --data-dir .\data\processed_riot --resume auto
```

To resume a specific checkpoint:

```powershell
python train.py --config config/default.yaml --data-dir .\data\processed_riot --resume .\checkpoints\latest.pt
```

Resume validation checks the champion vocabulary, patch vocabulary, split
sizes, checkpoint metric, model state, optimizer state, scheduler state, and
random-number-generator state before continuing. Starting without `--resume`
creates backups of existing checkpoints instead of silently discarding them.

The model is a draft-only estimate. It does not include player skill,
champion mastery, lane matchup execution, or in-game decisions; close
probabilities should be treated as low-confidence results.
2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The application will open in your browser at http://localhost:8501

## Model Information

The model was trained on professional League of Legends match data and uses:
- 173 champions (from champion_vocab.json)
- 50 patches (from drafts.parquet data)
- Multi-task learning for win probability, gold curves, and objectives

## File Structure
- `app.py` - Main Streamlit application
- `model.py` - Neural network model definition
- `data/processed/` - Processed data files (champion_vocab.json, drafts.parquet, splits.json)
- `checkpoints/best.pt` - Trained model checkpoint

## Troubleshooting
If you see a Streamlit onboarding prompt asking for email, simply leave the field blank and press Enter.

## Patch Vocabulary Fix

**Problem**: The application was incorrectly reading patch information from `splits.json` (3 splits) instead of actual patch data from `drafts.parquet` (~50 patches)

**Solution**: Updated `app.py` to load patches directly from `drafts.parquet` file to match the model training configuration

The application now correctly uses 50 patches (49 actual patches + <UNK>=0) matching what was used during training.