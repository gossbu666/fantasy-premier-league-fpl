#!/usr/bin/env python3
"""
FWD TURBO ENSEMBLE: XGBoost + LightGBM + CatBoost + Neural Net
"""
import joblib
import pandas as pd
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load super features
with open('analysis/FWD_super_features.txt', 'r') as f:
    features = [line.strip() for line in f]

df = pd.read_csv("data/processed/FWD_features_enhanced.csv")
available_features = [f for f in features if f in df.columns]
X = df[available_features].fillna(0)
y = df['target']

# TURBO ENSEMBLE
models = [
    ('xgb', xgb.XGBRegressor(n_estimators=500, max_depth=6, random_state=42)),
    ('lgb', lgb.LGBMRegressor(n_estimators=500, max_depth=6, random_state=42, verbose=-1)),
    ('cat', CatBoostRegressor(iterations=500, depth=6, random_state=42, verbose=0)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42))
]

turbo_ensemble = VotingRegressor(models)
turbo_ensemble.fit(X, y)

joblib.dump(turbo_ensemble, "models/FWD_turbo_ensemble.pkl")
print("🚀 FWD TURBO ENSEMBLE SAVED! (4 models)")
