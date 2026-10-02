# League of Legends Draft Synergy Engine

A complete system for analyzing draft synergy in League of Legends with recommendation capabilities.

## Features

- **Multi-task Draft Encoder**: Predicts win probability, gold curves, and objectives
- **Champion Embeddings**: Role-aware champion representations
- **Archetype Clustering**: Group similar drafts together
- **Recommendation Engine**: Suggest picks and bans based on synergy
- **Interactive Dashboard**: Streamlit visualization of results

## Data Processing

### Kaggle Dataset Preparation
The system processes Kaggle LoL datasets using `prepare_kaggle_data.py`:

```bash
python prepare_kaggle_data.py --kaggle-dir ./5mLoLGames --output-dir ./data/processed
```

This creates:
- `drafts.parquet`: Draft data with 79,908 augmented drafts
- `champion_vocab.json`: Champion vocabulary (174 champions)
- `splits.json`: Train/validation/test splits

## Model Architecture

The draft encoder uses:
- Champion ID → 64-dim embeddings with role-aware positional encoding
- Team encoding with 3-layer MLP (512-dim residual blocks)
- Patch-aware embeddings for different game versions
- Multi-task loss with uncertainty weighting (Kendall et al. 2018)

## Usage

### Demo Script
```bash
python demo.py
```

This demonstrates the complete workflow including:
1. Data preparation (using dummy data if needed)
2. Model training
3. Embedding extraction
4. Archetype clustering
5. Recommendation engine
6. Dashboard visualization

## Requirements

- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Streamlit
- WandB (optional for logging)

## Training

The system is ready for training on the Kaggle dataset:
- Model can be trained with `lol_collector/draft_encoder/train_gpu.py`
- Supports both CPU and GPU training
- Includes checkpointing and early stopping