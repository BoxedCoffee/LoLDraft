# Using LoL_DDragon Repository for Champion Data

## Overview
Based on your request to use the https://github.com/noxelisdev/LoL_DDragon repository, here's how to leverage it for your champion data needs.

## What You Need From This Repository
The LoL_DDragon repository contains League of Legends data including champion information. You'll want to extract champion IDs and mappings from this data.

## Steps to Get Champion Data

### 1. Access the Repository Data
Since you can't directly download, you need to:
- Visit https://github.com/noxelisdev/LoL_DDragon in your browser
- Navigate to the champion data files
- Extract the champion IDs and their mappings

### 2. Expected Champion Format
From LoL_DDragon or similar repositories, you'll likely find champions in a structure like:
```json
{
  "Aatrox": 266,
  "Ahri": 103,
  "Akali": 84,
  ...
}
```

### 3. Convert to Required Format
Your system needs numeric string keys, so you'll need to convert this to:
```json
{
  "1": 266,
  "2": 103,
  "3": 84,
  ...
}
```

## Implementation Approach

### Step 1: Extract Champions
From the repository data, extract all champion names and their numeric IDs.

### Step 2: Create Mapping Script
Create a script to convert from name→ID format to ID→ID format with string keys:
```python
import json

# Read original data (name->ID)
with open('original_champion_data.json', 'r') as f:
    name_to_id = json.load(f)

# Convert to required format (string ID -> actual ID)
required_format = {}
for i, (name, champion_id) in enumerate(name_to_id.items(), 1):
    required_format[str(i)] = champion_id

# Save the converted data
with open('champion_data.json', 'w') as f:
    json.dump(required_format, f, indent=2)
```

## Verify Your Data
After conversion, verify it matches your checkpoint expectations:
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

## Complete Your Process
Once you have the correct champion data:
1. Replace your current `champion_data.json`
2. Retrain your model: `python draft_encoder/train.py --config config/default.yaml`
3. Your system will work with all champions as intended

## Important Notes
- The repository likely contains more champions than the 173 used in training
- You may need to filter or map to the specific 173 champions used during original training
- Ensure the champion IDs match exactly what was used for training