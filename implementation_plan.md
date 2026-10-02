# Draft Synergy and Archetype Engine - Implementation Plan

## Project Overview
This project implements a Draft Synergy and Archetype Engine for League of Legends that:
- Builds a multi-task draft encoder mapping 10 champion IDs to a 256-dimensional embedding
- Clusters drafts into strategic archetypes (Engage, Poke, Scaling)
- Implements a K-Nearest Neighbors recommendation engine for optimal picks and bans
- Evaluates success via win-rate prediction accuracy (>63%), clustering purity, and recommendation alignment

## Current State Analysis
The repository already contains:
- Complete draft encoder model with multi-task learning architecture (win prediction, gold curve, objectives)
- Data preparation pipeline 
- Training script with uncertainty-weighted loss functions
- Configuration files for training

What's missing for the complete implementation:
1. Clustering of drafts into strategic archetypes
2. KNN-based recommendation engine
3. Visualization/dashboard component

## Implementation Phases

### Phase 1: Data Preparation and Model Training (Completed)
**Goal**: Prepare data pipeline and train the core draft encoder model

#### Tasks Completed:
1. **Data Pipeline Setup**
   - Dataset loading and preprocessing
   - Champion ID standardization
   - Train/validation/test splits creation

2. **Draft Encoder Architecture Design**
   - Multi-task model architecture with uncertainty-weighted loss
   - Win prediction (BCE), Gold curve regression (MSE), Objective prediction (CE)

3. **Model Implementation**
   - Implemented using PyTorch
   - Configured with uncertainty weighting approach
   - Set up training pipeline

4. **Initial Training & Evaluation**
   - Model training pipeline established
   - Evaluation metrics monitoring implemented

### Phase 2: Embedding Extraction and Archetype Clustering (Current Focus)
**Goal**: Extract embeddings from trained model and create archetype clustering

#### Tasks:
1. **Embedding Extraction**
   - Use trained draft encoder to generate 256-dimensional embeddings for all drafts
   - Store embeddings in efficient data structure for quick access

2. **Archetype Clustering**
   - Implement KMeans clustering to group similar draft compositions
   - Identify three main archetypes: Engage, Poke, Scaling
   - Validate clustering quality using purity metrics

3. **Clustering Analysis**
   - Analyze characteristics of each archetype
   - Create visualizations showing archetype distributions
   - Document archetype definitions and examples

### Phase 3: Recommendation Engine Implementation (Next)
**Goal**: Build KNN-based recommendation system for picks and bans

#### Tasks:
1. **KNN Recommendation System**
   - Implement KNN algorithm using draft embeddings
   - Create pick recommendations (top 5 champions) based on current draft composition
   - Implement ban recommendations (top 3 champions to ban)

2. **Recommendation Evaluation**
   - Create evaluation metrics for recommendation quality
   - Test with sample drafts and compare to expert picks
   - Validate that recommendations align with strategic archetypes

### Phase 4: Dashboard and Visualization (Final)
**Goal**: Build interactive dashboard for exploring the engine

#### Tasks:
1. **Streamlit Dashboard**
   - Create interactive interface for inputting draft compositions
   - Display archetype classification for input drafts
   - Show recommendations for picks and bans

2. **Visualization Components**
   - Embedding space visualization (PCA/t-SNE)
   - Archetype distribution charts
   - Performance metrics dashboard

## Technical Implementation Details

### Data Flow:
1. Kaggle LoL dataset processed through existing pipeline
2. Trained model with multi-task loss functions
3. Embeddings extracted using `get_embedding()` method from trained model
4. Clustering performed on embeddings using scikit-learn KMeans
5. KNN recommendations built on top of embedding space

### Key Challenges:
1. Ensuring sufficient training data for good embedding quality
2. Accurately defining archetype boundaries in 256-dimensional space
3. Efficiently implementing recommendation system with large embedding database
4. Balancing multiple task losses during training

### Success Metrics:
- Win-rate prediction accuracy > 63%
- Clustering purity metrics > 0.7
- Recommendation alignment quality > 0.8 (based on expert validation)