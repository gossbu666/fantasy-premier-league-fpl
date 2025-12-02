"""
Model Training - Season-based Split
Train: 2021-22, 2022-23, 2023-24
Test:  2024-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import time
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import logger

class EnsemblePredictor:
    def __init__(self, position):
        self.position = position
        self.models = self._create_models()
        
    def _create_models(self):
        return [
            ('XGB_1', XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1
            )),
            ('XGB_2', XGBRegressor(
                n_estimators=200, max_depth=7, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                random_state=43, n_jobs=-1
            )),
            ('RF_1', RandomForestRegressor(
                n_estimators=200, max_depth=15,
                min_samples_split=5, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            )),
            ('RF_2', RandomForestRegressor(
                n_estimators=300, max_depth=20,
                min_samples_split=10, min_samples_leaf=4,
                random_state=43, n_jobs=-1
            ))
        ]
    
    def fit(self, X_train, y_train, sample_weight=None):
        logger.info(f"\nTraining 4 models for {self.position}...")
        
        for i, (name, model) in enumerate(self.models, 1):
            logger.info(f"  [{i}/4] Training {name}...")
            try:
                if sample_weight is not None and 'XGB' in name:
                    model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
                elif sample_weight is not None:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                else:
                    model.fit(X_train, y_train)
                logger.info(f"      ✓ {name} trained successfully")
            except Exception as e:
                logger.error(f"      ✗ {name} failed: {e}")
    
    def predict(self, X):
        predictions = []
        for name, model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.mean(predictions, axis=0)

def load_data(position):
    """Load feature data"""
    file_path = f'data/processed/{position}_features.csv'
    df = pd.read_csv(file_path)
    logger.info(f"✓ Loaded {len(df)} samples")
    logger.info(f"✓ Features: {len([c for c in df.columns if c not in ['target', 'sample_weight', 'round', 'player_id']])} columns")
    logger.info(f"✓ Target range: {df['target'].min()} to {df['target'].max()}")
    logger.info(f"✓ Gameweeks: {df['round'].min():.0f} to {df['round'].max():.0f}")
    return df

def season_based_split(df):
    """
    Split by season:
    Train: 2021-22, 2022-23, 2023-24
    Test:  2024-25
    """
    # Need to add season info
    # Infer season from gameweek patterns
    df_sorted = df.sort_values('round').copy()
    
    # Assume GW 1-38 repeats for each season
    # Calculate season based on round cycling
    df_sorted['season_group'] = (df_sorted['round'] - 1) // 38
    
    # Last season_group = 2024-25
    max_group = df_sorted['season_group'].max()
    
    train_mask = df_sorted['season_group'] < max_group
    test_mask = df_sorted['season_group'] == max_group
    
    train_df = df_sorted[train_mask].copy()
    test_df = df_sorted[test_mask].copy()
    
    logger.info(f"\n🎯 Season-based split:")
    logger.info(f"  Train: Seasons 2021-2024 ({len(train_df)} samples)")
    logger.info(f"  Test:  Season 2024-25 ({len(test_df)} samples)")
    
    return train_df, test_df

def evaluate_model(ensemble, X, y, set_name='Test'):
    """Evaluate model performance"""
    y_pred = ensemble.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # MAPE (excluding zeros)
    non_zero_mask = y != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / y[non_zero_mask])) * 100
    else:
        mape = 0
    
    # Correlation
    corr = np.corrcoef(y, y_pred)[0, 1] if len(y) > 1 else 0
    
    logger.info(f"\n📊 {set_name} Set Performance:")
    logger.info(f"  Overall RMSE:        {rmse:.3f}")
    logger.info(f"  Overall MAE:         {mae:.3f}")
    logger.info(f"  Overall R²:          {r2:.3f}")
    logger.info(f"  Overall MAPE:        {mape:.2f}%")
    logger.info(f"  Overall Correlation: {corr:.3f}")
    
    # By category
    evaluate_by_category(y, y_pred, set_name)
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'corr': corr}

def evaluate_by_category(y_true, y_pred, set_name):
    """Evaluate by return category"""
    def categorize(points):
        if points == 0: return 'Zeros'
        elif points <= 2: return 'Blanks'
        elif points <= 4: return 'Tickers'
        else: return 'Haulers'
    
    categories = pd.Series([categorize(p) for p in y_true])
    
    logger.info(f"\n📊 By Return Category ({set_name} Set):")
    logger.info(f"  {'Category':12s} | {'RMSE':6s} | {'MAE':6s} | {'R²':6s} | {'MAPE':7s} | {'N':5s}")
    logger.info(f"  {'-'*11}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*9}+{'-'*5}")
    
    for cat in ['Zeros', 'Blanks', 'Tickers', 'Haulers']:
        mask = categories == cat
        if mask.sum() == 0:
            continue
        
        y_cat = y_true[mask]
        pred_cat = y_pred[mask]
        
        rmse_cat = np.sqrt(mean_squared_error(y_cat, pred_cat))
        mae_cat = mean_absolute_error(y_cat, pred_cat)
        
        # R² for category
        ss_res = np.sum((y_cat - pred_cat) ** 2)
        ss_tot = np.sum((y_cat - np.mean(y_cat)) ** 2)
        r2_cat = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # MAPE
        non_zero = y_cat != 0
        if non_zero.sum() > 0:
            mape_cat = np.mean(np.abs((y_cat[non_zero] - pred_cat[non_zero]) / y_cat[non_zero])) * 100
            mape_str = f"{mape_cat:6.1f}%"
        else:
            mape_str = "    N/A"
        
        logger.info(f"  {cat:12s} | {rmse_cat:6.3f} | {mae_cat:6.3f} | {r2_cat:6.3f} | {mape_str} | {mask.sum():5d}")

def train_position_model(position):
    """Train model for a position using season-based split"""
    logger.info("\n" + "="*70)
    logger.info(f"TRAINING {position} MODEL - SEASON-BASED SPLIT")
    logger.info("="*70)
    
    # Load data
    df = load_data(position)
    
    # Season-based split
    train_df, test_df = season_based_split(df)
    
    # Prepare data
    feature_cols = [c for c in df.columns if c not in ['target', 'sample_weight', 'round', 'player_id', 'season_group']]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    weights_train = train_df['sample_weight'].values if 'sample_weight' in train_df.columns else None
    
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # Train ensemble
    logger.info(f"\n🎯 Training final model...")
    logger.info(f"  Train: 2021-2024 ({len(X_train)} samples)")
    logger.info(f"  Test:  2024-25 ({len(X_test)} samples)")
    
    ensemble = EnsemblePredictor(position)
    logger.info(f"Created ensemble with {len(ensemble.models)} models")
    
    ensemble.fit(X_train, y_train, weights_train)
    
    # Evaluate
    logger.info("\n" + "="*70)
    logger.info(f"EVALUATION - {position}")
    logger.info("="*70)
    
    train_metrics = evaluate_model(ensemble, X_train, y_train, 'Training')
    test_metrics = evaluate_model(ensemble, X_test, y_test, 'Test')
    
    # Save
    Path('models').mkdir(exist_ok=True)
    
    model_file = f'models/{position}_ensemble.pkl'
    joblib.dump(ensemble, model_file)
    logger.info(f"\n✓ Saved ensemble to {model_file}")
    
    metrics_file = f'models/{position}_metrics.pkl'
    joblib.dump({
        'train': train_metrics,
        'test': test_metrics,
        'position': position
    }, metrics_file)
    logger.info(f"✓ Saved metrics to {metrics_file}")
    
    logger.info(f"\n✅ {position} model training complete!\n")
    
    return test_metrics

def main():
    logger.info("="*80)
    logger.info("SEASON-BASED MODEL TRAINING")
    logger.info("Train: 2021-22, 2022-23, 2023-24")
    logger.info("Test:  2024-25 (Current Season)")
    logger.info("="*80)
    
    start_time = time.time()
    positions = ['GK', 'DEF', 'MID', 'FWD']
    
    all_metrics = {}
    for position in positions:
        try:
            metrics = train_position_model(position)
            all_metrics[position] = metrics
        except Exception as e:
            logger.error(f"✗ Error training {position}: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = (time.time() - start_time) / 60
    
    logger.info("\n" + "="*80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nTime elapsed: {elapsed:.1f} minutes")
    logger.info(f"Models trained: {len(all_metrics)}/{len(positions)}")
    
    logger.info("\n📊 Final Test Performance Summary:")
    logger.info(f"  {'Position':<8} | {'R²':>6s} | {'RMSE':>6s}")
    logger.info(f"  {'-'*8}+{'-'*8}+{'-'*8}")
    for pos, metrics in all_metrics.items():
        logger.info(f"  {pos:<8} | {metrics['r2']:>6.3f} | {metrics['rmse']:>6.3f}")
    
    logger.info(f"\n📁 Generated files:")
    for pos in positions:
        logger.info(f"  ✓ models/{pos}_ensemble.pkl")
        logger.info(f"  ✓ models/{pos}_metrics.pkl")
    
    logger.info(f"\n🎯 Next steps:")
    logger.info(f"  1. python3 predict_next_gw.py")
    logger.info(f"  2. streamlit run app.py")

if __name__ == '__main__':
    main()
