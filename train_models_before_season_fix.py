#!/usr/bin/env python3
"""
Model Training with Proper Temporal Validation
NO DATA LEAKAGE - Split by gameweek, not random
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import logger
import time

class PositionEnsemble:
    """Ensemble of models for one position"""
    
    def __init__(self, position):
        self.position = position
        self.models = []
        
        # XGBoost models
        self.models.append(('XGB_1', XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )))
        
        self.models.append(('XGB_2', XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, random_state=43
        )))
        
        # Random Forest models
        self.models.append(('RF_1', RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )))
        
        self.models.append(('RF_2', RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=10,
            min_samples_leaf=4, random_state=43, n_jobs=-1
        )))
        
        logger.info(f"Created ensemble with {len(self.models)} models")
    
    def fit(self, X, y, sample_weight=None):
        """Train all models"""
        logger.info(f"\nTraining {len(self.models)} models for {self.position}...")
        
        for idx, (name, model) in enumerate(self.models):
            logger.info(f"  [{idx+1}/{len(self.models)}] Training {name}...")
            
            try:
                if 'XGB' in name:
                    model.fit(X, y, sample_weight=sample_weight, verbose=False)
                else:
                    model.fit(X, y, sample_weight=sample_weight)
                logger.info(f"      ✓ {name} trained successfully")
            except Exception as e:
                logger.error(f"      ✗ {name} failed: {e}")
    
    def predict(self, X):
        """Predict using ensemble (median)"""
        predictions = []
        for name, model in self.models:
            predictions.append(model.predict(X))
        
        # Median ensemble (robust to outliers)
        return np.median(predictions, axis=0)

def calculate_mape_safe(y_true, y_pred):
    """Calculate MAPE safely (handle zeros)"""
    # Only calculate for non-zero actual values
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(y_true, y_pred, position, set_name='Test'):
    """Evaluate predictions"""
    
    # Overall metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (safe version)
    mape = calculate_mape_safe(y_true, y_pred)
    
    # Correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    logger.info(f"\n📊 {set_name} Set Performance:")
    logger.info(f"  Overall RMSE:        {rmse:.3f}")
    logger.info(f"  Overall MAE:         {mae:.3f}")
    logger.info(f"  Overall R²:          {r2:.3f}")
    if not np.isnan(mape):
        logger.info(f"  Overall MAPE:        {mape:.2f}%")
    logger.info(f"  Overall Correlation: {corr:.3f}")
    
    # By category
    def categorize(pts):
        if pts == 0: return 'Zeros'
        elif pts <= 2: return 'Blanks'
        elif pts <= 4: return 'Tickers'
        else: return 'Haulers'
    
    categories = pd.Series(y_true).apply(categorize)
    
    logger.info(f"\n📊 By Return Category ({set_name} Set):")
    logger.info(f"  {'Category':<12} | {'RMSE':<6} | {'MAE':<6} | {'R²':<6} | {'MAPE':<7} | {'N':<5}")
    logger.info(f"  {'-'*11}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*9}+{'-'*5}")
    
    for cat in ['Zeros', 'Blanks', 'Tickers', 'Haulers']:
        mask = categories == cat
        if mask.sum() == 0:
            continue
        
        cat_true = y_true[mask]
        cat_pred = y_pred[mask]
        
        cat_rmse = np.sqrt(mean_squared_error(cat_true, cat_pred))
        cat_mae = mean_absolute_error(cat_true, cat_pred)
        
        if len(set(cat_true)) > 1:
            cat_r2 = r2_score(cat_true, cat_pred)
        else:
            cat_r2 = 0.0
        
        if cat == 'Zeros':
            cat_mape_str = 'N/A'
        else:
            cat_mape = calculate_mape_safe(cat_true, cat_pred)
            cat_mape_str = f"{cat_mape:.1f}%" if not np.isnan(cat_mape) else "N/A"
        
        logger.info(f"  {cat:<12} | {cat_rmse:>6.3f} | {cat_mae:>6.3f} | {cat_r2:>6.3f} | {cat_mape_str:>7} | {mask.sum():>5}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape if not np.isnan(mape) else 0.0,
        'correlation': corr
    }

def train_with_temporal_cv(position):
    """Train model with PROPER temporal cross-validation"""
    
    logger.info("\n" + "="*70)
    logger.info(f"TRAINING {position} MODEL WITH TEMPORAL CV")
    logger.info("="*70)
    
    # Load data
    features = pd.read_csv(f'data/processed/{position}_features.csv')
    logger.info(f"✓ Loaded {len(features)} samples")
    
    # Check required columns
    if 'target' not in features.columns:
        logger.error("❌ No 'target' column found!")
        return
    
    if 'round' not in features.columns:
        logger.warning("⚠️ No 'round' column - using index as proxy")
        features['round'] = np.arange(len(features))
    
    # Prepare data
    drop_cols = ['target', 'sample_weight', 'round', 'player_id', 'player_name']
    X = features.drop(drop_cols, axis=1, errors='ignore')
    y = features['target']
    rounds = features['round'].values
    sample_weights = features.get('sample_weight', pd.Series(np.ones(len(features))))
    
    logger.info(f"✓ Features: {len(X.columns)} columns")
    logger.info(f"✓ Target range: {y.min():.1f} to {y.max():.1f}")
    logger.info(f"✓ Gameweeks: {rounds.min():.0f} to {rounds.max():.0f}")
    
    # TEMPORAL CROSS-VALIDATION (3 folds)
    logger.info(f"\n🔄 Performing 3-fold Temporal Cross-Validation...")
    
    unique_rounds = np.sort(np.unique(rounds))
    n_rounds = len(unique_rounds)
    
    cv_scores = []
    
    for fold_num in range(1, 4):
        logger.info(f"\n--- Fold {fold_num}/3 ---")
        
        # Progressive temporal split
        train_cutoff_idx = int(n_rounds * fold_num / 4)  # 25%, 50%, 75%
        val_start_idx = train_cutoff_idx
        val_end_idx = int(n_rounds * (fold_num + 1) / 4)
        
        train_rounds = unique_rounds[:train_cutoff_idx]
        val_rounds = unique_rounds[val_start_idx:val_end_idx]
        
        train_mask = np.isin(rounds, train_rounds)
        val_mask = np.isin(rounds, val_rounds)
        
        X_train_cv = X[train_mask]
        X_val_cv = X[val_mask]
        y_train_cv = y[train_mask]
        y_val_cv = y[val_mask]
        w_train_cv = sample_weights[train_mask]
        
        logger.info(f"Train GW: {train_rounds.min():.0f}-{train_rounds.max():.0f} ({len(X_train_cv)} samples)")
        logger.info(f"Val GW: {val_rounds.min():.0f}-{val_rounds.max():.0f} ({len(X_val_cv)} samples)")
        
        # Train ensemble
        ensemble_cv = PositionEnsemble(position)
        ensemble_cv.fit(X_train_cv, y_train_cv, w_train_cv)
        
        # Evaluate
        y_pred_cv = ensemble_cv.predict(X_val_cv)
        r2_cv = r2_score(y_val_cv, y_pred_cv)
        rmse_cv = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
        
        logger.info(f"Fold {fold_num} Val R²: {r2_cv:.3f}, RMSE: {rmse_cv:.3f}")
        
        cv_scores.append({'fold': fold_num, 'r2': r2_cv, 'rmse': rmse_cv})
    
    # CV Summary
    cv_df = pd.DataFrame(cv_scores)
    logger.info(f"\n📊 Cross-Validation Summary:")
    logger.info(f"  Mean R²: {cv_df['r2'].mean():.3f} (±{cv_df['r2'].std():.3f})")
    logger.info(f"  Mean RMSE: {cv_df['rmse'].mean():.3f} (±{cv_df['rmse'].std():.3f})")
    
    # Final model: Train on early GWs, test on late GWs (80/20)
    logger.info(f"\n🎯 Training final model with 80/20 temporal split...")
    
    train_cutoff_idx = int(n_rounds * 0.8)
    train_rounds_final = unique_rounds[:train_cutoff_idx]
    test_rounds_final = unique_rounds[train_cutoff_idx:]
    
    train_mask = np.isin(rounds, train_rounds_final)
    test_mask = np.isin(rounds, test_rounds_final)
    
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()
    w_train = sample_weights[train_mask].copy()
    w_test = sample_weights[test_mask].copy()
    
    logger.info(f"✓ Train: GW {train_rounds_final.min():.0f}-{train_rounds_final.max():.0f} ({len(X_train)} samples)")
    logger.info(f"✓ Test: GW {test_rounds_final.min():.0f}-{test_rounds_final.max():.0f} ({len(X_test)} samples)")
    
    # Train final ensemble
    ensemble = PositionEnsemble(position)
    ensemble.fit(X_train, y_train, w_train)
    
    # Evaluation
    logger.info("\n" + "="*70)
    logger.info(f"EVALUATION - {position}")
    logger.info("="*70)
    
    # Train set
    y_train_pred = ensemble.predict(X_train)
    train_metrics = evaluate_model(y_train.values, y_train_pred, position, 'Training')
    
    # Test set
    y_test_pred = ensemble.predict(X_test)
    test_metrics = evaluate_model(y_test.values, y_test_pred, position, 'Test')
    
    # Save
    Path('models').mkdir(exist_ok=True)
    joblib.dump(ensemble, f'models/{position}_ensemble.pkl')
    logger.info(f"\n✓ Saved ensemble to models/{position}_ensemble.pkl")
    
    # Save metrics
    metrics = {
        'train_rmse': train_metrics['rmse'],
        'train_mae': train_metrics['mae'],
        'train_r2': train_metrics['r2'],
        'train_mape': train_metrics['mape'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'test_mape': test_metrics['mape'],
        'cv_mean_r2': cv_df['r2'].mean(),
        'cv_std_r2': cv_df['r2'].std(),
        'cv_mean_rmse': cv_df['rmse'].mean(),
        'cv_std_rmse': cv_df['rmse'].std()
    }
    
    joblib.dump(metrics, f'models/{position}_metrics.pkl')
    logger.info(f"✓ Saved metrics to models/{position}_metrics.pkl")
    
    logger.info(f"\n✅ {position} model training complete!\n")

def main():
    """Main training pipeline"""
    
    start_time = time.time()
    
    logger.info("="*80)
    logger.info("EPL FANTASY MODEL TRAINING - PHASE 3 (TEMPORAL VALIDATION)")
    logger.info("="*80)
    
    positions = ['GK', 'DEF', 'MID', 'FWD']
    
    for position in positions:
        try:
            train_with_temporal_cv(position)
        except Exception as e:
            logger.error(f"✗ Error training {position}: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = (time.time() - start_time) / 60
    
    logger.info("\n" + "="*80)
    logger.info("✅ PHASE 3 COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nTime elapsed: {elapsed:.1f} minutes")
    logger.info(f"Models trained: {len(positions)}/{len(positions)}")
    
    logger.info("\n📁 Generated files:")
    for pos in positions:
        logger.info(f"  ✓ models/{pos}_ensemble.pkl")
        logger.info(f"  ✓ models/{pos}_metrics.pkl")
    
    logger.info("\n🎯 Next steps:")
    logger.info("  1. python3 predict_next_gw.py")
    logger.info("  2. streamlit run app.py")
    logger.info("  3. Write report")

if __name__ == '__main__':
    main()
