#!/usr/bin/env python3
"""
Main data pipeline - Process raw data into position-specific datasets
Uses FDR-enhanced data if available
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from utils import logger

def create_position_datasets():
    """Create position-specific datasets from raw data"""
    
    logger.info("="*70)
    logger.info("CREATING POSITION-SPECIFIC DATASETS")
    logger.info("="*70)
    
    # Load raw data - prefer FDR-enhanced version
    logger.info("\n1. Loading raw data...")
    
    if Path('data/raw/fpl_history_fdr.csv').exists():
        history = pd.read_csv('data/raw/fpl_history_fdr.csv')
        logger.info("✓ Using FDR-enhanced data (fpl_history_fdr.csv)")
    else:
        history = pd.read_csv('data/raw/fpl_history.csv')
        logger.info("✓ Using standard data (fpl_history.csv)")
    
    players = pd.read_csv('data/raw/fpl_players.csv')
    
    logger.info(f"✓ Loaded {len(history)} gameweek records")
    logger.info(f"✓ Loaded {len(players)} players")
    
    # Position mapping
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    # Create processed directory
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Split by position
    logger.info("\n2. Splitting by position...")
    for pos_code, pos_name in pos_map.items():
        pos_history = history[history['element_type'] == pos_code].copy()
        
        # Ensure essential columns exist
        if 'round' not in pos_history.columns:
            logger.warning(f"⚠️  No 'round' column in {pos_name} data!")
            # Try to find round/gameweek column
            for col in pos_history.columns:
                if col.lower() in ['gameweek', 'gw', 'event']:
                    pos_history['round'] = pos_history[col]
                    logger.info(f"   Using '{col}' as round for {pos_name}")
                    break
        
        if 'player_id' not in pos_history.columns:
            if 'element' in pos_history.columns:
                pos_history['player_id'] = pos_history['element']
                logger.info(f"   Using 'element' as player_id for {pos_name}")
        
        # Save
        output_file = f'data/processed/{pos_name}_data.csv'
        pos_history.to_csv(output_file, index=False)
        
        logger.info(f"✓ {pos_name}: {len(pos_history)} records → {output_file}")
        if 'round' in pos_history.columns:
            logger.info(f"   Gameweeks: {pos_history['round'].min():.0f} to {pos_history['round'].max():.0f}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ POSITION DATASETS CREATED!")
    logger.info("="*70)

def main():
    """Main pipeline"""
    
    logger.info("="*70)
    logger.info("EPL FANTASY DATA PIPELINE - PHASE 1")
    logger.info("="*70)
    
    # Check if raw data exists
    if not Path('data/raw/fpl_history.csv').exists() and not Path('data/raw/fpl_history_fdr.csv').exists():
        logger.error("❌ Raw data not found!")
        logger.error("   Run: python3 force_reload.py first")
        return
    
    logger.info("✓ Raw data exists")
    
    # Create position datasets
    create_position_datasets()
    
    logger.info("\n📁 Next step:")
    logger.info("   python3 feature_pipeline.py")

if __name__ == '__main__':
    main()
