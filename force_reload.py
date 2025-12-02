#!/usr/bin/env python3
"""Force reload all data (ignore cache)"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import FPLDataLoader
from utils import logger
import pandas as pd

def main():
    logger.info("="*70)
    logger.info("FORCE RELOAD - ALL 755 PLAYERS")
    logger.info("="*70)
    
    # Create directories
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # Initialize loader
    loader = FPLDataLoader()
    
    # Fetch bootstrap
    logger.info("\n1. Fetching bootstrap data...")
    bootstrap = loader.fetch_bootstrap_data()
    
    # Parse
    logger.info("\n2. Parsing players and teams...")
    players = loader.parse_players(bootstrap)
    teams = loader.parse_teams(bootstrap)
    
    # Save
    logger.info("\n3. Saving players and teams...")
    players.to_csv('data/raw/fpl_players.csv', index=False)
    teams.to_csv('data/raw/fpl_teams.csv', index=False)
    logger.info(f"✓ Saved {len(players)} players")
    logger.info(f"✓ Saved {len(teams)} teams")
    
    # Load ALL player history
    logger.info("\n4. Loading ALL player histories (755 players)...")
    logger.info("   ⏱️  This will take 1-1.5 hours")
    logger.info("   ☕ Go take a break!\n")
    
    history_df = loader.load_all_player_history(players, max_players=None)
    
    # Save
    logger.info("\n5. Saving history...")
    history_df.to_csv('data/raw/fpl_history.csv', index=False)
    logger.info(f"✓ Saved {len(history_df)} gameweek records")
    
    logger.info("\n" + "="*70)
    logger.info("✅ DATA COLLECTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Players: {len(players)}")
    logger.info(f"   Teams: {len(teams)}")
    logger.info(f"   Gameweek records: {len(history_df)}")
    logger.info(f"\n📁 Files saved to data/raw/")

if __name__ == '__main__':
    main()

