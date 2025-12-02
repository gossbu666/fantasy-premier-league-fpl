"""
Optuna Hyperparameter Tuning for FPL Models
Quick tune: 50 trials per position
"""

import pandas as pd
import numpy as np
import optuna
from pathlib import Path
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')
from utils import logger

def load_data(position):
    """Load and split data"""
    df = pd.read_csv(f'data/processed/{position}_features.csv')
    
    # Proper temporal split
    train_seasons = ['2021-22', '2022-23']
    test_season = '2023-24'
    
    train_mask = df['season'].isin(train_seasons)
    test_mask = df['season'] == test_season
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    feature_cols = [c for c in df.columns if c not in ['target', 'sample_weight', 'round', 'player_id', 'season']]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    weights_train = train_df['sample_weight'].values if 'sample_weight' in train_df.columns else None
    
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    logger.info(f"{position}: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, y_train, X_test, y_test, weights_train

def objective_xgb(trial, X_train, y_train, weights):
    """Optuna objective for XGBoost"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = XGBRegressor(**params)
    
    # 3-fold CV with sample weights
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3,
        scoring='neg_mean_squared_error',
        fit_params={'sample_weight': weights} if weights is not None else None,
        n_jobs=1  # Optuna already parallelizes
    )
    
    rmse = np.sqrt(-scores.mean())
    return rmse

def objective_rf(trial, X_train, y_train, weights):
    """Optuna objective for Random Forest"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestRegressor(**params)
    
    # 3-fold CV
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3,
        scoring='neg_mean_squared_error',
        fit_params={'sample_weight': weights} if weights is not None else None,
        n_jobs=1
    )
    
    rmse = np.sqrt(-scores.mean())
    return rmse

def tune_position(position, n_trials=50):
    """Tune both XGBoost and Random Forest for a position"""
    
    logger.info("="*70)
    logger.info(f"TUNING {position} - {n_trials} trials per algorithm")
    logger.info("="*70)
    
    # Load data
    X_train, y_train, X_test, y_test, weights = load_data(position)
    
    # Tune XGBoost
    logger.info(f"\n🔍 Tuning XGBoost...")
    study_xgb = optuna.create_study(
        direction='minimize',
        study_name=f'{position}_XGB',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study_xgb.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, weights),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Single job for stability
    )
    
    best_xgb_params = study_xgb.best_params
    best_xgb_rmse = study_xgb.best_value
    
    logger.info(f"✓ Best XGB CV RMSE: {best_xgb_rmse:.3f}")
    logger.info(f"✓ Best XGB params: {best_xgb_params}")
    
    # Tune Random Forest
    logger.info(f"\n�� Tuning Random Forest...")
    study_rf = optuna.create_study(
        direction='minimize',
        study_name=f'{position}_RF',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study_rf.optimize(
        lambda trial: objective_rf(trial, X_train, y_train, weights),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )
    
    best_rf_params = study_rf.best_params
    best_rf_rmse = study_rf.best_value
    
    logger.info(f"✓ Best RF CV RMSE: {best_rf_rmse:.3f}")
    logger.info(f"✓ Best RF params: {best_rf_params}")
    
    # Evaluate on test set with best params
    logger.info(f"\n📊 Evaluating on test set...")
    
    # Train XGBoost with best params
    xgb_model = XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train, sample_weight=weights)
    pred_xgb = xgb_model.predict(X_test)
    rmse_xgb_test = np.sqrt(mean_squared_error(y_test, pred_xgb))
    r2_xgb_test = r2_score(y_test, pred_xgb)
    
    # Train RF with best params
    rf_model = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train, sample_weight=weights)
    pred_rf = rf_model.predict(X_test)
    rmse_rf_test = np.sqrt(mean_squared_error(y_test, pred_rf))
    r2_rf_test = r2_score(y_test, pred_rf)
    
    # Ensemble
    pred_ensemble = (pred_xgb + pred_rf) / 2
    rmse_ensemble = np.sqrt(mean_squared_error(y_test, pred_ensemble))
    r2_ensemble = r2_score(y_test, pred_ensemble)
    
    logger.info(f"  XGBoost:  Test RMSE={rmse_xgb_test:.3f}, R²={r2_xgb_test:.3f}")
    logger.info(f"  RF:       Test RMSE={rmse_rf_test:.3f}, R²={r2_rf_test:.3f}")
    logger.info(f"  Ensemble: Test RMSE={rmse_ensemble:.3f}, R²={r2_ensemble:.3f}")
    
    # Save results
    results = {
        'position': position,
        'xgb_params': best_xgb_params,
        'rf_params': best_rf_params,
        'xgb_cv_rmse': best_xgb_rmse,
        'rf_cv_rmse': best_rf_rmse,
        'test_rmse_xgb': rmse_xgb_test,
        'test_r2_xgb': r2_xgb_test,
        'test_rmse_rf': rmse_rf_test,
        'test_r2_rf': r2_rf_test,
        'test_rmse_ensemble': rmse_ensemble,
        'test_r2_ensemble': r2_ensemble
    }
    
    Path('tuning_results').mkdir(exist_ok=True)
    joblib.dump(results, f'tuning_results/{position}_optuna_results.pkl')
    
    logger.info(f"\n✓ Saved to tuning_results/{position}_optuna_results.pkl")
    
    return results

def main():
    logger.info("="*80)
    logger.info("OPTUNA HYPERPARAMETER TUNING")
    logger.info("Trials: 50 per algorithm per position")
    logger.info("Total: 400 trials (50×2×4 positions)")
    logger.info("="*80)
    
    import time
    start_time = time.time()
    
    positions = ['GK', 'DEF', 'MID', 'FWD']
    n_trials = 50  # Adjust if needed (30=faster, 100=better)
    
    all_results = {}
    
    for position in positions:
        try:
            results = tune_position(position, n_trials)
            all_results[position] = results
        except Exception as e:
            logger.error(f"✗ Error tuning {position}: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = (time.time() - start_time) / 60
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ TUNING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nTime elapsed: {elapsed:.1f} minutes")
    
    logger.info(f"\n{'Position':<8} | {'Ensemble R²':>12} | {'Ensemble RMSE':>14} | {'Improvement':>12}")
    logger.info("-"*60)
    
    # Load original metrics for comparison
    original_metrics = {}
    for pos in positions:
        try:
            orig = joblib.load(f'models/{pos}_metrics.pkl')
            original_metrics[pos] = orig['test']
        except:
            original_metrics[pos] = {'r2': 0, 'rmse': 999}
    
    for pos, results in all_results.items():
        orig_r2 = original_metrics[pos]['r2']
        new_r2 = results['test_r2_ensemble']
        improvement = new_r2 - orig_r2
        
        logger.info(
            f"{pos:<8} | "
            f"{new_r2:>12.3f} | "
            f"{results['test_rmse_ensemble']:>14.3f} | "
            f"{improvement:>+12.3f}"
        )
    
    logger.info("\n✓ Best parameters saved in tuning_results/")
    logger.info("✓ Now retrain models with tuned params: python3 retrain_tuned.py")

if __name__ == '__main__':
    main()
