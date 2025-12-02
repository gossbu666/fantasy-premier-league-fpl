import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from utils import logger

for pos in ['GK', 'DEF', 'MID', 'FWD']:
    df = pd.read_csv(f'data/processed/{pos}_data.csv')

    logger.info(f"\n{'='*60}")
    logger.info(f"{pos} Position")
    logger.info(f"{'='*60}")

    xg_cols = ['expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded']

    for col in xg_cols:
        if col in df.columns:
            non_zero = (df[col] > 0).sum()
            pct = non_zero / len(df) * 100
            mean_val = df[col].mean()
            max_val = df[col].max()

            logger.info(f"{col:30s}: Non-zero: {non_zero:4d}/{len(df)} ({pct:5.1f}%), Mean: {mean_val:.3f}, Max: {max_val:.3f}")

