#!/usr/bin/env python3
"""
FWD HYPERPARAMETER TUNING (Optuna + Cross-validation)
"""
import optuna
import joblib
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import numpy as np

# Load FWD optimized features
with open('analysis/FWD_clean_ensemble_features.txt', 'r') as f:
    features = [line.strip() for line in f if line.strip()]

df = pd.read_csv("data/processed/FWD_features_enhanced.csv")
X = df[features].select_dtypes(include=['number']).fillna(0)
y = df['target']

def objective(trial):
    params = {
        'xgb': {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3),
            'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0),
        },
        'lgb': {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.3),
            'subsample': trial.suggest_float('lgb_subsample', 0.7, 1.0),
        }
    }
    
    xgb_model = xgb.XGBRegressor(**params['xgb'], random_state=42, n_jobs=-1)
    lgb_model = lgb.LGBMRegressor(**params['lgb'], random_state=42, n_jobs=-1, verbose=-1)
    
    ensemble = StackingRegressor(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
        final_estimator=xgb.XGBRegressor(n_estimators=100, random_state=42),
        cv=3
    )
    
    scores = GroupKFold(n_splits=5).split(X, y, groups=df['round'])
    r2_scores = []
    
    for train_idx, val_idx in scores:
        ensemble.fit(X.iloc[train_idx], y.iloc[train_idx])
        r2_scores.append(r2_score(y.iloc[val_idx], ensemble.predict(X.iloc[val_idx])))
    
    return np.mean(r2_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"✅ BEST FWD R²: {study.best_value:.4f}")
print(f"PARAMS: {study.best_params}")

# Train final model
best_params = study.best_params
joblib.dump(study.best_params, "models/FWD_hyperopt_params.pkl")
print("✅ SAVED HYPEROPT PARAMS!")
