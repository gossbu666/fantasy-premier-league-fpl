#!/usr/bin/env python3
"""
Predict next gameweek points for all players
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import joblib
from utils import logger

def predict_next_gameweek(target_gw=13):
    """
    Predict points for next gameweek
    
    Args:
        target_gw: Gameweek to predict (default: 13)
    """
    logger.info("="*70)
    logger.info(f"PREDICT GAMEWEEK {target_gw}")
    logger.info("="*70)
    
    # Load models
    models = {}
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        models[pos] = joblib.load(f'models/{pos}_ensemble.pkl')
        logger.info(f"✓ Loaded {pos} model")
    
    # Load latest player data
    players_df = pd.read_csv('data/raw/fpl_players.csv')
    history_df = pd.read_csv('data/raw/fpl_history.csv')
    
    # Calculate recent form (last 5 GW)
    recent_form = history_df.groupby('player_id')['total_points'].apply(
        lambda x: x.tail(5).mean()
    ).reset_index()
    recent_form.columns = ['id', 'form_5gw']
    
    # Merge
    players_pred = players_df.merge(recent_form, on='id', how='left')
    players_pred['form_5gw'] = players_pred['form_5gw'].fillna(2.0)
    
    # Map position
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    players_pred['position'] = players_pred['element_type'].map(pos_map)
    
    # Use form as prediction (simplified)
    players_pred['gw13_prediction'] = players_pred['form_5gw']
    
    # Clean and sort
    results = players_pred[[
        'web_name', 'position', 'team', 'now_cost', 'gw13_prediction'
    ]].copy()
    results.columns = ['Player', 'Position', 'Team', 'Price', f'GW{target_gw}_Pred']
    results['Price'] = results['Price'] / 10
    results = results.sort_values(f'GW{target_gw}_Pred', ascending=False)
    
    # Display top picks
    logger.info(f"\n🔥 TOP 20 PREDICTED PERFORMERS - GW{target_gw}")
    logger.info(f"\n{'Rank':4} | {'Player':20} | {'Pos':4} | {'Price':7} | {'Pred Pts':9}")
    logger.info(f"{'-'*4}-+-{'-'*20}-+-{'-'*4}-+-{'-'*7}-+-{'-'*9}")
    
    for idx, (_, row) in enumerate(results.head(20).iterrows(), 1):
        logger.info(f"{idx:4} | {row['Player']:20} | {row['Position']:4} | "
                   f"£{row['Price']:5.1f}M | {row[f'GW{target_gw}_Pred']:8.1f}")
    
    # By position
    logger.info(f"\n📊 TOP 5 BY POSITION - GW{target_gw}")
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        logger.info(f"\n{pos}:")
        pos_players = results[results['Position'] == pos].head(5)
        for _, row in pos_players.iterrows():
            logger.info(f"  {row['Player']:20} £{row['Price']:5.1f}M → {row[f'GW{target_gw}_Pred']:5.1f} pts")
    
    # Captain suggestions
    logger.info(f"\n⭐ CAPTAIN RECOMMENDATIONS - GW{target_gw}")
    logger.info(f"   (2× points multiplier)")
    top_3 = results.head(3)
    for idx, (_, row) in enumerate(top_3.iterrows(), 1):
        expected_2x = row[f'GW{target_gw}_Pred'] * 2
        logger.info(f"   {idx}. {row['Player']:20} → {expected_2x:5.1f} pts (2×)")
    
    # Save
    Path('results').mkdir(exist_ok=True)
    results.to_csv(f'results/gw{target_gw}_predictions.csv', index=False)
    logger.info(f"\n✓ Saved to results/gw{target_gw}_predictions.csv")

if __name__ == '__main__':
    predict_next_gameweek(target_gw=13)

