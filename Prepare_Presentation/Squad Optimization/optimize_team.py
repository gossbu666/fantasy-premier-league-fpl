#!/usr/bin/env python3
"""
Team optimization using Linear Programming
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import joblib
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD
from utils import logger

def load_models():
    """Load trained ensembles"""
    models = {}
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        try:
            models[pos] = joblib.load(f'models/{pos}_ensemble.pkl')
            logger.info(f"✓ Loaded {pos} model")
        except:
            logger.warning(f"✗ Could not load {pos} model")
    return models

def predict_all_players(models, players_df):
    """Make predictions for all players"""
    predictions = []
    
    for pos_code, pos_name in {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.items():
        if pos_name not in models:
            continue
        
        pos_players = players_df[players_df['element_type'] == pos_code].copy()
        
        if len(pos_players) == 0:
            continue
        
        # Create dummy features (simplified - ในจริงต้องใช้ features จาก preprocessing)
        n_features = 38 if pos_name == 'GK' else 44
        X_dummy = np.random.rand(len(pos_players), n_features) * 0.5  # Placeholder
        
        # Predict
        ensemble = models[pos_name]
        pred_points = ensemble.predict(X_dummy)
        
        # Denormalize (สมมติ max = 20 points)
        pred_points = pred_points * 20
        
        for idx, (_, player) in enumerate(pos_players.iterrows()):
            predictions.append({
                'player_id': player['id'],
                'name': player['web_name'],
                'position': pos_name,
                'team': player['team'],
                'price': player['now_cost'] / 10,  # Convert to millions
                'predicted_points': max(0, pred_points[idx])  # No negative
            })
    
    return pd.DataFrame(predictions)

def optimize_squad(predictions_df, budget=100.0):
    """
    Optimize squad using Linear Programming
    
    Args:
        predictions_df: DataFrame with columns [player_id, name, position, team, price, predicted_points]
        budget: Total budget in millions (default: 100.0)
    
    Returns:
        Optimized squad DataFrame
    """
    logger.info("\n" + "="*70)
    logger.info("TEAM OPTIMIZATION")
    logger.info("="*70)
    
    # Create LP problem
    prob = LpProblem("FPL_Squad_Optimization", LpMaximize)
    
    # Decision variables (binary)
    player_vars = {}
    for idx, row in predictions_df.iterrows():
        var_name = f"p_{row['player_id']}"
        player_vars[var_name] = LpVariable(var_name, cat='Binary')
    
    # Objective: Maximize predicted points
    prob += lpSum([
        predictions_df.iloc[idx]['predicted_points'] * player_vars[f"p_{predictions_df.iloc[idx]['player_id']}"]
        for idx in range(len(predictions_df))
    ]), "Total_Points"
    
    # Constraint 1: Squad size = 15
    prob += lpSum(player_vars.values()) == 15, "Squad_Size"
    
    # Constraint 2: Budget
    prob += lpSum([
        predictions_df.iloc[idx]['price'] * player_vars[f"p_{predictions_df.iloc[idx]['player_id']}"]
        for idx in range(len(predictions_df))
    ]) <= budget, "Budget"
    
    # Constraint 3: Position requirements
    positions_count = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    
    for idx, row in predictions_df.iterrows():
        var = player_vars[f"p_{row['player_id']}"]
        positions_count[row['position']] += var
    
    prob += positions_count['GK'] == 2, "GK_Count"
    prob += positions_count['DEF'] >= 3, "DEF_Min"
    prob += positions_count['DEF'] <= 5, "DEF_Max"
    prob += positions_count['MID'] >= 2, "MID_Min"
    prob += positions_count['MID'] <= 5, "MID_Max"
    prob += positions_count['FWD'] >= 1, "FWD_Min"
    prob += positions_count['FWD'] <= 3, "FWD_Max"
    
    # Constraint 4: Max 3 per team
    teams = predictions_df['team'].unique()
    for team in teams:
        team_players = predictions_df[predictions_df['team'] == team]
        team_vars = [player_vars[f"p_{pid}"] for pid in team_players['player_id']]
        prob += lpSum(team_vars) <= 3, f"Max_Team_{team}"
    
    # Solve
    logger.info("Solving optimization problem...")
    prob.solve(PULP_CBC_CMD(msg=0))
    
    # Extract solution
    selected_players = []
    total_points = 0
    total_cost = 0
    
    for idx, row in predictions_df.iterrows():
        var = player_vars[f"p_{row['player_id']}"]
        if var.varValue == 1:
            selected_players.append(row)
            total_points += row['predicted_points']
            total_cost += row['price']
    
    squad_df = pd.DataFrame(selected_players)
    
    # Sort by position
    position_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    squad_df['pos_order'] = squad_df['position'].map(position_order)
    squad_df = squad_df.sort_values('pos_order').drop('pos_order', axis=1)
    
    logger.info(f"\n✅ Optimization Complete!")
    logger.info(f"   Expected Points: {total_points:.1f}")
    logger.info(f"   Budget Used: £{total_cost:.1f}M / £{budget}M")
    logger.info(f"   Remaining: £{budget - total_cost:.1f}M")
    
    logger.info(f"\n📋 Optimized Squad:")
    for _, player in squad_df.iterrows():
        logger.info(f"   {player['position']:4} | {player['name']:20} | "
                   f"£{player['price']:4.1f}M | {player['predicted_points']:5.1f} pts")
    
    return squad_df

def main():
    logger.info("="*70)
    logger.info("FPL TEAM OPTIMIZATION")
    logger.info("="*70)
    
    # Load models
    models = load_models()
    
    if len(models) == 0:
        logger.error("No models found! Train models first.")
        return
    
    # Load player data
    players_df = pd.read_csv('data/raw/fpl_players.csv')
    logger.info(f"\n✓ Loaded {len(players_df)} players")
    
    # Make predictions
    logger.info("\nMaking predictions...")
    predictions_df = predict_all_players(models, players_df)
    logger.info(f"✓ Generated {len(predictions_df)} predictions")
    
    # Optimize squad
    squad_df = optimize_squad(predictions_df, budget=100.0)
    
    # Save
    squad_df.to_csv('results/optimized_squad.csv', index=False)
    logger.info(f"\n✓ Saved to results/optimized_squad.csv")

if __name__ == '__main__':
    main()

