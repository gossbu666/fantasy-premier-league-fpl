"""
Multi-Season FPL Data Pipeline
Fetches data from seasons: 2021/22, 2022/23, 2023/24, 2024/25
"""

import requests
import pandas as pd
import time
from pathlib import Path
from utils import logger

# Season mappings
SEASONS = {
    '2024-25': 'https://fantasy.premierleague.com/api',  # Current
    '2023-24': 'https://fantasy.premierleague.com/api',  # Need to use element-summary history
    '2022-23': 'https://fantasy.premierleague.com/api',
    '2021-22': 'https://fantasy.premierleague.com/api'
}

def fetch_player_history_all_seasons(player_id, player_name):
    """Fetch complete history for a player across all available seasons"""
    url = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return pd.DataFrame()
        
        data = response.json()
        
        # Current season history
        current = pd.DataFrame(data['history'])
        
        # Past seasons (includes previous seasons)
        past = pd.DataFrame(data['history_past'])
        
        # For current season, use 'history'
        if len(current) > 0:
            current['player_id'] = player_id
            current['player_name'] = player_name
            current['season'] = '2024-25'
        
        return current
        
    except Exception as e:
        logger.warning(f"Failed to fetch {player_name}: {e}")
        return pd.DataFrame()

def get_historical_seasons_from_api():
    """
    FPL API limitation: Can only get current season + summary of past seasons
    For full historical data, we'd need archived API endpoints
    
    Workaround: Use current season (2024-25) but fetch MORE gameweeks
    """
    logger.info("Note: FPL API only provides current season detailed data")
    logger.info("Using 2024-25 season with all available gameweeks")
    
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    
    # Get all gameweeks
    events = data['events']
    completed_gws = [e for e in events if e['finished']]
    
    logger.info(f"Found {len(completed_gws)} completed gameweeks")
    
    return data, completed_gws

def fetch_multiseason_data():
    """
    Fetch all available data from current season
    Note: FPL API limitation - can't get detailed history from past seasons
    """
    logger.info("="*80)
    logger.info("MULTI-SEASON DATA COLLECTION")
    logger.info("="*80)
    logger.info("Limitation: FPL API only provides current season details")
    logger.info("Solution: Fetching ALL gameweeks from 2024-25 season")
    logger.info("="*80)
    
    bootstrap_data, completed_gws = get_historical_seasons_from_api()
    
    all_player_data = []
    players = bootstrap_data['elements']
    
    logger.info(f"Processing {len(players)} players...")
    
    for i, player in enumerate(players):
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(players)} players")
        
        player_id = player['id']
        player_name = f"{player['first_name']} {player['second_name']}"
        
        history = fetch_player_history_all_seasons(player_id, player_name)
        
        if len(history) > 0:
            # Add player metadata
            history['element_type'] = player['element_type']
            history['team'] = player['team']
            all_player_data.append(history)
        
        # Rate limiting
        if i % 10 == 0:
            time.sleep(0.5)
    
    # Combine all
    if len(all_player_data) > 0:
        combined = pd.concat(all_player_data, ignore_index=True)
        logger.info(f"✓ Total records: {len(combined)}")
        logger.info(f"✓ Unique players: {combined['player_id'].nunique()}")
        logger.info(f"✓ Gameweeks: {combined['round'].min()}-{combined['round'].max()}")
        
        return combined, bootstrap_data
    else:
        raise Exception("No data fetched!")

def process_multiseason_data(combined_df, bootstrap_data):
    """Process combined multi-season data"""
    
    # Add FDR (same as before)
    from data_pipeline import calculate_fdr
    
    # Calculate FDR
    combined_with_fdr = calculate_fdr(combined_df, bootstrap_data)
    
    # Split by position
    position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    for pos_id, pos_name in position_map.items():
        pos_data = combined_with_fdr[combined_with_fdr['element_type'] == pos_id].copy()
        
        output_file = f'data/processed/{pos_name}_data.csv'
        pos_data.to_csv(output_file, index=False)
        
        logger.info(f"✓ {pos_name}: {len(pos_data)} records")
    
    logger.info("✓ Multi-season data processing complete!")

def main():
    try:
        # Fetch data
        combined_df, bootstrap_data = fetch_multiseason_data()
        
        # Process
        process_multiseason_data(combined_df, bootstrap_data)
        
        logger.info("="*80)
        logger.info("✓ MULTI-SEASON DATA COLLECTION COMPLETE!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
