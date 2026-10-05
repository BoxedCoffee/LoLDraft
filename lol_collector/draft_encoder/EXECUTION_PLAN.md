# Complete Execution Plan for Draft Encoder System

## Phase 1: System Preparation (Already Done)
- Champion vocabulary format verified (name → ID mapping)
- Current system structure confirmed working
- Checkpoint file identified and analyzed

## Phase 2: Data Acquisition
**You must obtain the complete 173-champion dataset from the original training data**

This dataset should contain:
- 173 champion entries
- Keys as numeric strings ("1", "2", "3", ...)
- Values as champion IDs (matching the checkpoint expectations)

## Phase 3: Data Preparation
Once you have the 173-champion dataset:

1. Replace your current `champion_data.json`:
   ```bash
   # Backup your current file
   cp draft_encoder/data/champion_data.json draft_encoder/data/champion_data.json.backup

   # Replace with complete dataset (this step requires your data)
   ```

2. Verify the new data format:
   ```bash
   python -c "
   import json
   with open('draft_encoder/data/champion_data.json', 'r') as f:
       data = json.load(f)
   print(f'Champions: {len(data)}')
   print('Sample entries:')
   for i, (key, value) in enumerate(list(data.items())[:5]):
       print(f'  {key} -> {value}')
   "
   ```

## Phase 4: Retraining
Run the training process to create a model compatible with your 173 champions:

```bash
# Navigate to project directory
cd C:\Users\Hamza\Senior Project\LoLDraft\lol_collector

# Start training (this will take time)
python draft_encoder/train.py --config config/default.yaml
```

## Phase 5: Verification
After training completes:

1. Verify checkpoint was created:
   ```bash
   ls -la draft_encoder/checkpoints/
   ```

2. Test evaluation:
   ```bash
   python draft_encoder/evaluate_fixed.py --checkpoint draft_encoder/checkpoints/best.pt
   ```

## Phase 6: Full System Validation
Once retrained, your system will be ready to work with all 173 champions:
- Win probability predictions
- Gold curve forecasts at 5, 10, 15, 20, 25, 30 minutes
- Objective predictions (dragon, herald, baron, tower)

## Expected Outcome
After completing this process:
- Your model will be trained on all 173 champions
- The checkpoint will match your champion data exactly
- You'll be able to make accurate predictions using the full champion set
- All functionality described in the original system will work correctly

## Important Notes
- The training process will take substantial time and computational resources
- Make sure you have sufficient GPU/CPU power for training
- The checkpoint file (`best.pt`) will be created automatically after training