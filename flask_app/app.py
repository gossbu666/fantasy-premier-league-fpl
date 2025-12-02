"""
FPL Points Predictor - Flask Web Application
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import requests
from pathlib import Path

app = Flask(__name__)

# Simple wrapper for loaded models
class SimplePredictor:
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        return self.model.predict(X)

def load_models():
    """Load trained models"""
    models = {}
    
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        try:
            # Try tuned first, then regular
            for suffix in ['_tuned', '']:
                model_path = f'../models/{position}_ensemble{suffix}.pkl'
                if Path(model_path).exists():
                    raw_model = joblib.load(model_path)
                    models[position] = SimplePredictor(raw_model)
                    print(f"✓ Loaded {position} model{suffix}")
                    break
        except Exception as e:
            print(f"✗ Error loading {position}: {e}")
    
    return models

def get_fpl_data():
    """Fetch current FPL data"""
    try:
        response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/', timeout=10)
        data = response.json()
        
        current_gw = next((gw['id'] for gw in data['events'] if gw['is_current']), 13)
        
        players_df = pd.DataFrame(data['elements'])
        teams_df = pd.DataFrame(data['teams'])
        
        team_map = dict(zip(teams_df['id'], teams_df['name']))
        players_df['team_name'] = players_df['team'].map(team_map)
        
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        players_df['position_name'] = players_df['element_type'].map(position_map)
        
        return current_gw, players_df
    except Exception as e:
        print(f"Error fetching FPL data: {e}")
        return None, None

def prepare_features(player_row):
    """Prepare features - simplified version"""
    # Create 83 features (matching training)
    feature_array = np.zeros(83)
    
    # Fill with available data
    features = [
        player_row.get('minutes', 0),
        player_row.get('goals_scored', 0),
        player_row.get('assists', 0),
        player_row.get('clean_sheets', 0),
        player_row.get('goals_conceded', 0),
        player_row.get('bonus', 0),
        player_row.get('bps', 0),
        float(player_row.get('influence', 0)),
        float(player_row.get('creativity', 0)),
        float(player_row.get('threat', 0)),
        float(player_row.get('ict_index', 0)),
        float(player_row.get('form', 0)),
        float(player_row.get('selected_by_percent', 0)),
        player_row.get('now_cost', 0) / 10,
    ]
    
    feature_array[:len(features)] = features
    return feature_array

def predict_all_players(models, players_df):
    """Predict points for all players"""
    predictions = []
    
    for idx, player in players_df.iterrows():
        try:
            position = player['position_name']
            if position in models:
                X = prepare_features(player).reshape(1, -1)
                predicted_points = models[position].predict(X)[0]
                
                predictions.append({
                    'id': player['id'],
                    'name': player['web_name'],
                    'team': player['team_name'],
                    'position': position,
                    'cost': player['now_cost'] / 10,
                    'predicted_points': round(float(predicted_points), 2),
                    'form': float(player['form']),
                    'selected_by': float(player['selected_by_percent'])
                })
        except Exception as e:
            print(f"Error predicting {player['web_name']}: {e}")
    
    return pd.DataFrame(predictions)

# Load models on startup
print("Loading models...")
MODELS = load_models()
print(f"Loaded {len(MODELS)} models")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global MODELS
    
    try:
        if not MODELS:
            return render_template('error.html', 
                                 error="No models loaded. Please check model files.")
        
        position_filter = request.form.get('position', 'ALL')
        max_cost = float(request.form.get('max_cost', 15.0))
        min_points = float(request.form.get('min_points', 0))
        
        print("Fetching FPL data...")
        current_gw, all_players = get_fpl_data()
        
        if all_players is None:
            return render_template('error.html', 
                                 error="Failed to fetch FPL data. Please try again.")
        
        print("Making predictions...")
        predictions = predict_all_players(MODELS, all_players)
        
        if len(predictions) == 0:
            return render_template('error.html',
                                 error="No predictions generated. Check model compatibility.")
        
        filtered = predictions.copy()
        
        if position_filter != 'ALL':
            filtered = filtered[filtered['position'] == position_filter]
        
        filtered = filtered[filtered['cost'] <= max_cost]
        filtered = filtered[filtered['predicted_points'] >= min_points]
        filtered = filtered.sort_values('predicted_points', ascending=False)
        
        top_players = filtered.head(20)
        
        return render_template('results.html',
                             gameweek=current_gw,
                             players=top_players.to_dict('records'),
                             total_players=len(filtered),
                             position=position_filter,
                             max_cost=max_cost)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=f"Prediction failed: {str(e)}")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
