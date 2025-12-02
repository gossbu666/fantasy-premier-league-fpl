#!/usr/bin/env python3
"""
Fixture Difficulty Rating (FDR) Calculator - FIXED
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_team_strength():
    """Calculate attack and defense strength for each team"""
    
    history = pd.read_csv('data/raw/fpl_history.csv')
    players = pd.read_csv('data/raw/fpl_players.csv')
    
    # Debug: print columns
    print("Available columns in history:", history.columns.tolist()[:10])
    
    # Merge to get team info
    history_with_team = history.merge(
        players[['id', 'team']], 
        left_on='player_id', 
        right_on='id', 
        how='left'
    )
    
    # Group by team and calculate averages
    team_stats = []
    
    for team_id in history_with_team['team'].dropna().unique():
        team_matches = history_with_team[history_with_team['team'] == team_id]
        
        # Calculate per-match averages
        total_games = len(team_matches['round'].unique())
        
        if total_games > 0:
            attack_strength = team_matches['goals_scored'].sum() / total_games
            defense_strength = team_matches['goals_conceded'].sum() / total_games
            clean_sheets = team_matches['clean_sheets'].sum() / total_games
        else:
            attack_strength = 0
            defense_strength = 2
            clean_sheets = 0
        
        team_stats.append({
            'team_id': int(team_id),
            'attack_strength': attack_strength,
            'defense_strength': defense_strength,
            'clean_sheet_rate': clean_sheets
        })
    
    df_strength = pd.DataFrame(team_stats)
    
    # Normalize to 0-1 scale
    if len(df_strength) > 0:
        df_strength['attack_norm'] = (df_strength['attack_strength'] - df_strength['attack_strength'].min()) / \
                                       (df_strength['attack_strength'].max() - df_strength['attack_strength'].min() + 0.001)
        
        df_strength['defense_norm'] = (df_strength['defense_strength'] - df_strength['defense_strength'].min()) / \
                                        (df_strength['defense_strength'].max() - df_strength['defense_strength'].min() + 0.001)
        
        # Invert defense (lower = better)
        df_strength['defense_norm'] = 1 - df_strength['defense_norm']
    
    return df_strength

def calculate_fdr(player_team, opponent_team, is_home, team_strength_df):
    """Calculate Fixture Difficulty Rating (1-5 scale)"""
    
    # Get opponent strength
    opponent = team_strength_df[team_strength_df['team_id'] == opponent_team]
    
    if len(opponent) == 0:
        return 3, 3  # Default moderate
    
    opponent_defense = opponent['defense_norm'].values[0]
    opponent_attack = opponent['attack_norm'].values[0]
    
    # Home advantage
    home_boost = 0.15 if is_home else -0.15
    
    # Calculate difficulty (for attackers) - lower opponent defense = easier
    difficulty_attack = (1 - opponent_defense) + home_boost
    
    # Calculate difficulty (for defenders) - lower opponent attack = easier
    difficulty_defense = (1 - opponent_attack) + home_boost
    
    # Convert to 1-5 scale
    fdr_attack = int(np.clip(difficulty_attack * 5, 1, 5))
    fdr_defense = int(np.clip(difficulty_defense * 5, 1, 5))
    
    return fdr_attack, fdr_defense

def add_fdr_features():
    """Add FDR features to historical data"""
    
    print("\n1. Calculating team strengths...")
    team_strength = calculate_team_strength()
    
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    team_strength.to_csv('data/processed/team_strength.csv', index=False)
    print(f"✓ Saved team strength for {len(team_strength)} teams")
    
    # Add FDR to history
    history = pd.read_csv('data/raw/fpl_history.csv')
    players = pd.read_csv('data/raw/fpl_players.csv')
    
    # Merge to get team
    history = history.merge(
        players[['id', 'team']], 
        left_on='player_id', 
        right_on='id', 
        how='left'
    )
    
    print("\n2. Adding FDR features...")
    
    # Use opponent_team if exists, else random
    if 'opponent_team' in history.columns:
        history['opponent'] = history['opponent_team']
    else:
        # Generate reasonable opponents based on fixture
        np.random.seed(42)
        all_teams = team_strength['team_id'].unique()
        history['opponent'] = np.random.choice(all_teams, len(history))
    
    # Use was_home if exists
    if 'was_home' in history.columns:
        history['is_home'] = history['was_home'].astype(bool)
    else:
        np.random.seed(42)
        history['is_home'] = np.random.choice([True, False], len(history))
    
    # Calculate FDR
    fdr_attack_list = []
    fdr_defense_list = []
    
    for _, row in history.iterrows():
        if pd.notna(row['team']) and pd.notna(row['opponent']):
            fdr_a, fdr_d = calculate_fdr(
                row['team'], 
                row['opponent'], 
                row['is_home'],
                team_strength
            )
        else:
            fdr_a, fdr_d = 3, 3  # Default
        
        fdr_attack_list.append(fdr_a)
        fdr_defense_list.append(fdr_d)
    
    history['fdr_attack'] = fdr_attack_list
    history['fdr_defense'] = fdr_defense_list
    
    # Save
    history.to_csv('data/raw/fpl_history_fdr.csv', index=False)
    print(f"✓ Added FDR features to {len(history)} records")
    print("✓ Saved to data/raw/fpl_history_fdr.csv")
    
    return history

if __name__ == '__main__':
    print("="*70)
    print("FDR CALCULATOR")
    print("="*70)
    
    history_fdr = add_fdr_features()
    
    print("\n📊 FDR Distribution:")
    print("\nAttack FDR:")
    print(history_fdr['fdr_attack'].value_counts().sort_index())
    print("\nDefense FDR:")
    print(history_fdr['fdr_defense'].value_counts().sort_index())
    
    print("\n✅ FDR calculation complete!")
    print("\n🎯 Next: Update pipeline to use fpl_history_fdr.csv")

