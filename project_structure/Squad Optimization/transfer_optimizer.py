#!/usr/bin/env python3
"""
Transfer Optimizer - Suggest optimal 1-2 player transfers
Inspired by OpenFPL methodology
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import joblib
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
from utils import logger

def get_player_predictions():
    """Get predictions for all players using historical averages"""
    players_df = pd.read_csv('data/raw/fpl_players.csv')
    history_df = pd.read_csv('data/raw/fpl_history.csv')
    
    # Calculate 5-gameweek form
    avg_points = history_df.groupby('player_id').agg({
        'total_points': lambda x: x.tail(5).mean()
    }).reset_index()
    avg_points.columns = ['id', 'predicted_points']
    
    # Merge
    players_pred = players_df.merge(avg_points, on='id', how='left')
    players_pred['predicted_points'] = players_pred['predicted_points'].fillna(2.0)
    
    # Position mapping
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    players_pred['position'] = players_pred['element_type'].map(pos_map)
    
    # Clean
    players_pred = players_pred[[
        'id', 'web_name', 'position', 'team', 'now_cost', 'predicted_points'
    ]].copy()
    players_pred.columns = ['player_id', 'name', 'position', 'team', 'price', 'predicted_points']
    players_pred['price'] = players_pred['price'] / 10
    
    return players_pred

def load_current_squad(squad_file='results/optimized_squad_v2.csv'):
    """Load current squad from file"""
    try:
        squad = pd.read_csv(squad_file)
        logger.info(f"✓ Loaded current squad from {squad_file}")
        return squad
    except:
        logger.warning(f"Could not load squad from {squad_file}")
        return None

def optimize_transfers(current_squad, all_players, max_transfers=2, transfer_cost=4):
    """Optimize transfers to maximize expected points"""
    logger.info("\n" + "="*70)
    logger.info("TRANSFER OPTIMIZATION")
    logger.info("="*70)
    
    # Current squad stats
    current_value = current_squad['price'].sum()
    current_points = current_squad['predicted_points'].sum()
    
    logger.info(f"\n📊 Current Squad:")
    logger.info(f"   Total Value: £{current_value:.1f}M")
    logger.info(f"   Expected Points (GW): {current_points:.1f}")
    logger.info(f"   Squad Size: {len(current_squad)}")
    
    # Filter available players
    current_ids = set(current_squad['player_id'].values)
    available = all_players[~all_players['player_id'].isin(current_ids)].copy()
    available = available[available['price'] >= 3.5]
    
    logger.info(f"\n📋 Available Players: {len(available)}")
    
    # Create LP problem
    prob = LpProblem("Transfer_Optimization", LpMaximize)
    
    # Decision variables
    transfer_out = {pid: LpVariable(f"out_{pid}", cat='Binary') 
                    for pid in current_squad['player_id']}
    transfer_in = {pid: LpVariable(f"in_{pid}", cat='Binary') 
                   for pid in available['player_id']}
    
    # Objective
    points_gained = lpSum([
        available[available['player_id'] == pid]['predicted_points'].values[0] * var
        for pid, var in transfer_in.items()
    ])
    
    points_lost = lpSum([
        current_squad[current_squad['player_id'] == pid]['predicted_points'].values[0] * var
        for pid, var in transfer_out.items()
    ])
    
    num_transfers = lpSum(transfer_out.values())
    
    prob += (points_gained - points_lost - transfer_cost * num_transfers), "Net_Points"
    
    # Constraints
    prob += num_transfers <= max_transfers, "Max_Transfers"
    prob += num_transfers == lpSum(transfer_in.values()), "Balanced_Transfers"
    
    # Budget
    money_in = lpSum([
        current_squad[current_squad['player_id'] == pid]['price'].values[0] * var
        for pid, var in transfer_out.items()
    ])
    
    money_out = lpSum([
        available[available['player_id'] == pid]['price'].values[0] * var
        for pid, var in transfer_in.items()
    ])
    
    prob += money_out <= money_in, "Budget_Balance"
    
    # Position balance
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_current = len(current_squad[current_squad['position'] == pos])
        
        pos_out = lpSum([
            transfer_out[pid] for pid in current_squad[current_squad['position'] == pos]['player_id']
        ])
        
        pos_in = lpSum([
            transfer_in[pid] for pid in available[available['position'] == pos]['player_id']
        ])
        
        final_count = pos_current - pos_out + pos_in
        
        if pos == 'GK':
            prob += final_count == 2, f"{pos}_Count"
        elif pos == 'DEF':
            prob += final_count >= 3, f"{pos}_Min"
            prob += final_count <= 5, f"{pos}_Max"
        elif pos == 'MID':
            prob += final_count >= 2, f"{pos}_Min"
            prob += final_count <= 5, f"{pos}_Max"
        elif pos == 'FWD':
            prob += final_count >= 1, f"{pos}_Min"
            prob += final_count <= 3, f"{pos}_Max"
    
    # Team constraints
    for team in all_players['team'].unique():
        team_current = len(current_squad[current_squad['team'] == team])
        
        team_out = lpSum([
            transfer_out[pid] for pid in current_squad[current_squad['team'] == team]['player_id']
            if pid in transfer_out
        ])
        
        team_in = lpSum([
            transfer_in[pid] for pid in available[available['team'] == team]['player_id']
            if pid in transfer_in
        ])
        
        final_team = team_current - team_out + team_in
        prob += final_team <= 3, f"Max_Team_{team}"
    
    # Solve
    logger.info("\n🔄 Solving optimization problem...")
    prob.solve(PULP_CBC_CMD(msg=0))
    
    # Extract solution
    transfers_out = []
    transfers_in = []
    
    for pid, var in transfer_out.items():
        if var.varValue == 1:
            player_info = current_squad[current_squad['player_id'] == pid].iloc[0]
            transfers_out.append(player_info)
    
    for pid, var in transfer_in.items():
        if var.varValue == 1:
            player_info = available[available['player_id'] == pid].iloc[0]
            transfers_in.append(player_info)
    
    if len(transfers_out) > 0:
        out_points = sum([p['predicted_points'] for p in transfers_out])
        in_points = sum([p['predicted_points'] for p in transfers_in])
        net_gain = in_points - out_points - (transfer_cost * len(transfers_out))
        
        logger.info(f"\n✅ Optimization Complete!")
        logger.info(f"   Transfers: {len(transfers_out)}")
        logger.info(f"   Points Lost: {out_points:.1f}")
        logger.info(f"   Points Gained: {in_points:.1f}")
        logger.info(f"   Transfer Cost: -{transfer_cost * len(transfers_out)}")
        logger.info(f"   Net Gain: {net_gain:+.1f} pts")
        
        logger.info(f"\n🔴 TRANSFERS OUT:")
        for player in transfers_out:
            logger.info(f"   ❌ {player['position']:4} | {player['name']:20} | "
                       f"£{player['price']:5.1f}M | {player['predicted_points']:5.1f} pts")
        
        logger.info(f"\n🟢 TRANSFERS IN:")
        for player in transfers_in:
            logger.info(f"   ✅ {player['position']:4} | {player['name']:20} | "
                       f"£{player['price']:5.1f}M | {player['predicted_points']:5.1f} pts")
        
        summary = {
            'transfers_out': pd.DataFrame(transfers_out),
            'transfers_in': pd.DataFrame(transfers_in),
            'net_gain': net_gain,
            'num_transfers': len(transfers_out)
        }
        
        return summary
    else:
        logger.info("\n✅ No beneficial transfers found!")
        logger.info("   Your current squad is already optimal.")
        return None

def main():
    logger.info("="*70)
    logger.info("FPL TRANSFER OPTIMIZER")
    logger.info("="*70)
    
    current_squad = load_current_squad()
    
    if current_squad is None:
        logger.error("❌ No current squad found. Run optimize_team_v2.py first!")
        return
    
    logger.info("\nLoading player predictions...")
    all_players = get_player_predictions()
    logger.info(f"✓ Generated predictions for {len(all_players)} players")
    
    result = optimize_transfers(current_squad, all_players, max_transfers=2, transfer_cost=4)
    
    if result is not None:
        Path('results').mkdir(exist_ok=True)
        result['transfers_out'].to_csv('results/transfers_out.csv', index=False)
        result['transfers_in'].to_csv('results/transfers_in.csv', index=False)
        logger.info(f"\n✓ Saved transfer suggestions to results/")
        
        logger.info(f"\n📊 Updated Squad Preview:")
        logger.info(f"   New Expected Points: {current_squad['predicted_points'].sum() + result['net_gain']:.1f}")
        logger.info(f"   Improvement: {result['net_gain']:+.1f} pts")

if __name__ == '__main__':
    main()

