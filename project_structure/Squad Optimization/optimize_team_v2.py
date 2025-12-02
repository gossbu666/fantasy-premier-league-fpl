#!/usr/bin/env python3
"""
Team optimization v2 - Using actual historical predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import joblib
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
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

def get_average_predictions():
    """
    Get average predictions from historical data
    (Simplified: use mean points from training data)
    """
    players_df = pd.read_csv('data/raw/fpl_players.csv')
    history_df = pd.read_csv('data/raw/fpl_history.csv')
    
    # Calculate average points per player
    avg_points = history_df.groupby('player_id')['total_points'].mean().reset_index()
    avg_points.columns = ['id', 'predicted_points']
    
    # Merge with player info
    players_pred = players_df.merge(avg_points, on='id', how='left')
    players_pred['predicted_points'] = players_pred['predicted_points'].fillna(2.0)  # Default 2 pts
    
    # Add position name
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    players_pred['position'] = players_pred['element_type'].map(pos_map)
    
    # Clean data
    players_pred = players_pred[[
        'id', 'web_name', 'position', 'team', 'now_cost', 'predicted_points'
    ]].copy()
    players_pred.columns = ['player_id', 'name', 'position', 'team', 'price', 'predicted_points']
    players_pred['price'] = players_pred['price'] / 10  # Convert to millions
    
    return players_pred

def optimize_squad(predictions_df, budget=100.0):
    """Optimize squad using Linear Programming"""
    logger.info("\n" + "="*70)
    logger.info("TEAM OPTIMIZATION")
    logger.info("="*70)
    
    # Filter only players with reasonable prices
    predictions_df = predictions_df[predictions_df['price'] >= 3.5].copy()
    
    # Create LP problem
    prob = LpProblem("FPL_Squad_Optimization", LpMaximize)
    
    # Decision variables
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
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = predictions_df[predictions_df['position'] == pos]
        pos_vars = [player_vars[f"p_{pid}"] for pid in pos_players['player_id']]
        
        if pos == 'GK':
            prob += lpSum(pos_vars) == 2, f"{pos}_Count"
        elif pos == 'DEF':
            prob += lpSum(pos_vars) >= 3, f"{pos}_Min"
            prob += lpSum(pos_vars) <= 5, f"{pos}_Max"
        elif pos == 'MID':
            prob += lpSum(pos_vars) >= 2, f"{pos}_Min"
            prob += lpSum(pos_vars) <= 5, f"{pos}_Max"
        elif pos == 'FWD':
            prob += lpSum(pos_vars) >= 1, f"{pos}_Min"
            prob += lpSum(pos_vars) <= 3, f"{pos}_Max"
    
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
    squad_df = squad_df.sort_values(['pos_order', 'predicted_points'], ascending=[True, False])
    squad_df = squad_df.drop('pos_order', axis=1)
    
    logger.info(f"\n✅ Optimization Complete!")
    logger.info(f"   Expected Points: {total_points:.1f} (average per gameweek)")
    logger.info(f"   Budget Used: £{total_cost:.1f}M / £{budget}M")
    logger.info(f"   Remaining: £{budget - total_cost:.1f}M")
    
    logger.info(f"\n📋 Optimized Squad:")
    logger.info(f"   {'Pos':4} | {'Player':20} | {'Price':8} | {'Avg Pts':8}")
    logger.info(f"   {'-'*4}-+-{'-'*20}-+-{'-'*8}-+-{'-'*8}")
    for _, player in squad_df.iterrows():
        logger.info(f"   {player['position']:4} | {player['name']:20} | "
                   f"£{player['price']:5.1f}M | {player['predicted_points']:7.1f}")
    
    return squad_df

def main():
    logger.info("="*70)
    logger.info("FPL TEAM OPTIMIZATION v2")
    logger.info("="*70)
    
    # Get predictions (using historical averages)
    logger.info("\nCalculating predictions from historical data...")
    predictions_df = get_average_predictions()
    logger.info(f"✓ Generated predictions for {len(predictions_df)} players")
    
    # Show top predictions
    logger.info("\n📊 Top 10 Predicted Players:")
    top_10 = predictions_df.nlargest(10, 'predicted_points')[
        ['name', 'position', 'price', 'predicted_points']
    ]
    logger.info("\n" + top_10.to_string(index=False))
    
    # Optimize squad
    squad_df = optimize_squad(predictions_df, budget=100.0)
    
    # Save
    Path('results').mkdir(exist_ok=True)
    squad_df.to_csv('results/optimized_squad_v2.csv', index=False)
    logger.info(f"\n✓ Saved to results/optimized_squad_v2.csv")

if __name__ == '__main__':
    main()

