# Champion Data Guide for Draft Encoder

## Understanding Your Current Data Format

Your current `champion_data.json` has champions with name keys:
```json
{
  "Annie": 1,
  "Olaf": 2,
  "Galio": 3,
  ...
}
```

But the checkpoint expects champion IDs as keys:
```json
{
  "1": 1,
  "2": 2,
  "3": 3,
  ...
}
```

## What You Need to Get

To make your system work with the original 173-champion model, you need to:

1. **Find a dataset** that contains all 173 champions as they existed during the original training
2. **Format it correctly** with numeric string keys ("1", "2", "3", ...)

## How to Find This Data

### Option A: Check Your Repository History
Look in your repository for any files that might contain the complete champion data:
```bash
find . -name "*champion*" -type f
git log --oneline --all | grep -i champion
```

### Option B: Use Riot API (if available)
You could potentially scrape or get champion data from the official Riot API, but this would require:
- API access
- Processing to match the exact format used during training

### Option C: Check Model Documentation
Look for any documentation that mentions:
- Where the original champion data came from
- Champion ID mappings used during training

## Required Format

The data should be in the exact format that the checkpoint expects:

```json
{
  "1": 1,
  "2": 2,
  "3": 3,
  ...
  "173": 173
}
```

## Verification Steps After Getting Data

1. Check that you have exactly 173 entries:
   ```bash
   python -c "
   import json
   with open('draft_encoder/data/champion_data.json', 'r') as f:
       data = json.load(f)
   print(f'Found {len(data)} champions')
   ```

2. Verify keys are numeric strings:
   ```bash
   python -c "
   import json
   with open('draft_encoder/data/champion_data.json', 'r') as f:
       data = json.load(f)
   sample_keys = list(data.keys())[:5]
   print('Sample keys:', sample_keys)
   print('Key types:', [type(k) for k in sample_keys])
   "
   ```

## Important Notes

- The champion IDs should match exactly what was used during training
- Different versions of League of Legends may have different champion ID mappings
- You might need to map from current champion names to the IDs used during training

This guide helps you identify what data you need to find, but I cannot search the internet or provide actual champion datasets.