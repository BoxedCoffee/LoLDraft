#!/usr/bin/env python3
"""
Vercel-ready Flask app for draft encoder demo
"""

import os
import torch
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Global variable to store the model and vocab
model = None
vocab = {}
champion_name_mapping = {}

def load_model_and_vocab():
    """Load the trained model and vocabulary"""
    global model, vocab, champion_name_mapping

    try:
        # Load vocabulary
        vocab_path = 'data/processed/champion_vocab.json'
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)

        # Load champion name mapping
        mapping_path = 'data/champion_name_mapping.json'
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                champion_name_mapping = json.load(f)

        # Load model checkpoint (if it exists)
        model_path = 'model_checkpoint.pth'
        if os.path.exists(model_path):
            # Create a mock model to test the loading
            from model import DraftModel

            # Initialize model with correct vocab size
            num_champions = len(vocab) if vocab else 1000  # Default fallback
            model = DraftModel(
                num_champions=num_champions,
                champion_dim=64,
                draft_dim=256,
                team_hidden=512,
                num_team_layers=3,
                num_patches=0,
                patch_dim=16
            )

            # Load checkpoint if available
            try:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                print("Model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load model checkpoint: {e}")
        else:
            print("No model checkpoint found - using mock predictions")

    except Exception as e:
        print(f"Error loading model: {e}")

# Load the model when the app starts
load_model_and_vocab()

# Simple mock prediction function - this would use the real model in production
def make_prediction(blue_team, red_team):
    """
    Mock prediction function. In a real deployment, this would load
    and use the trained model to make predictions.
    """
    import random

    # Mock win probability (between 30-70%)
    win_prob = round(random.uniform(0.3, 0.7), 3)

    # Mock gold curve (between -100 and 100 for each time point)
    gold_curve = [round(random.uniform(-100, 100), 2) for _ in range(6)]

    # Mock objective predictions (0=none, 1=blue, 2=red)
    obj_predictions = [random.randint(0, 2) for _ in range(4)]

    return {
        'win_probability': win_prob,
        'gold_curve': gold_curve,
        'objective_predictions': obj_predictions
    }

def convert_champion_names_to_ids(champions):
    """Convert champion names to IDs based on our vocabulary"""
    # This is a simplified version - in reality we'd want more robust mapping
    champion_ids = []
    for champ_name in champions:
        # Try to find matching champion ID in our vocab
        found = False
        for champ_id, name in champion_name_mapping.items():
            if name.lower() == champ_name.lower():
                champion_ids.append(int(champ_id))
                found = True
                break
        if not found:
            print(f"Warning: Champion '{champ_name}' not found in mapping")
            # Add a placeholder (0) for unknown champions
            champion_ids.append(0)
    return champion_ids

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/champions')
def get_champions():
    """Return list of champions with names and images from DataDragon"""
    try:
        import json
        import os
        import requests

        # Load the champion vocabulary
        vocab_path = 'data/processed/champion_vocab.json'
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)

            # Extract champion IDs (keys that are not <UNK>)
            champion_ids = [key for key in vocab.keys() if key != "<UNK>"]
            # Sort champions for consistent display
            champion_ids.sort()

            # Create list of champions with names and images
            champions_data = []
            for champ_id in champion_ids:
                # Use DataDragon to get champion name and image
                try:
                    # Fetch from DataDragon API
                    url = f"https://ddragon.leagueoflegends.com/cdn/14.20.1/data/en_US/champion/{champ_id}.json"
                    response = requests.get(url, timeout=5)

                    if response.status_code == 200:
                        data = response.json()
                        champ_data = data['data'][champ_id]

                        champion_info = {
                            'id': champ_id,
                            'name': champ_data['name'],
                            'title': champ_data['title'],
                            'image': f"https://ddragon.leagueoflegends.com/cdn/14.20.1/img/champion/{champ_id}.png"
                        }
                        champions_data.append(champion_info)
                    else:
                        # Fallback to basic info if API fails
                        champion_info = {
                            'id': champ_id,
                            'name': f"Champion {champ_id}",
                            'title': "Unknown",
                            'image': f"https://ddragon.leagueoflegends.com/cdn/14.20.1/img/champion/{champ_id}.png"
                        }
                        champions_data.append(champion_info)

                except Exception as e:
                    # Fallback to basic info if API fails
                    champion_info = {
                        'id': champ_id,
                        'name': f"Champion {champ_id}",
                        'title': "Unknown",
                        'image': f"https://ddragon.leagueoflegends.com/cdn/14.20.1/img/champion/{champ_id}.png"
                    }
                    champions_data.append(champion_info)

            return jsonify(champions_data)
        else:
            # Fallback to mock data if file doesn't exist
            champions = [
                {"id": "1", "name": "Annie", "title": "the Dark Child", "image": "https://ddragon.leagueoflegends.com/cdn/14.20.1/img/champion/Annie.png"},
                {"id": "2", "name": "Olaf", "title": "the Berserker", "image": "https://ddragon.leagueoflegends.com/cdn/14.20.1/img/champion/Olaf.png"}
            ]
            return jsonify(champions)
    except Exception as e:
        # Fallback to mock data if there's an error
        print(f"Error loading champions: {e}")
        champions = [
            {"id": "1", "name": "Annie", "title": "the Dark Child", "image": "https://ddragon.leagueoflegends.com/cdn/14.20.1/img/champion/Annie.png"},
            {"id": "2", "name": "Olaf", "title": "the Berserker", "image": "https://ddragon.leagueoflegends.com/cdn/14.20.1/img/champion/Olaf.png"}
        ]
        return jsonify(champions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        blue_team = data.get('blue_team', [])
        red_team = data.get('red_team', [])

        print(f"Received prediction request - Blue team: {blue_team}")
        print(f"Received prediction request - Red team: {red_team}")

        # Check if we have a real model
        if model is not None and len(model.state_dict()) > 0:
            # This is where we would use the actual model
            # For now, just use mock predictions for demonstration

            # Convert champion names to IDs (this is what would be done in real implementation)
            blue_ids = convert_champion_names_to_ids(blue_team)
            red_ids = convert_champion_names_to_ids(red_team)

            print(f"Converted blue team IDs: {blue_ids}")
            print(f"Converted red team IDs: {red_ids}")

            # Make prediction with real model (mocked here for now)
            result = make_prediction(blue_team, red_team)
        else:
            # Fallback to mock predictions
            result = make_prediction(blue_team, red_team)

        response = {
            'win_probability': result['win_probability'],
            'gold_curve': result['gold_curve'],
            'objective_predictions': result['objective_predictions'],
            'blue_team': blue_team,
            'red_team': red_team
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))