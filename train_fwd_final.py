#!/usr/bin/env python3
"""
Train FINAL FWD stacking model using Optuna best hyperparameters
and a FIXED temporal train/test split.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import StackingRegressor

import xgboost as xgb
import lightgbm as lgb

# 1. Load data
df = pd.read_csv("data/processed/FWD_features_enhanced.csv")

feature_cols = df.select_dtypes(include=["number"]).columns.tolist()
feature_cols = [c for c in feature_cols if c != "target"]

X = df[feature_cols].fillna(0)
y = df["target"].fillna(0)

# 2. Load FIXED split (same for baseline + tuned model)
train_idx = np.load("analysis/FWD_train_idx.npy")
test_idx  = np.load("analysis/FWD_test_idx.npy")

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"FWD train size = {len(X_train)}, test size = {len(X_test)}")

# 3. Best hyperparameters from Optuna (CV R^2 ≈ 0.152)
best_params = {
    "xgb_n_estimators": 837,
    "xgb_max_depth": 6,
    "xgb_lr": 0.010782958714031302,
    "xgb_subsample": 0.7002760369964917,
    "lgb_n_estimators": 836,
    "lgb_max_depth": 4,
    "lgb_lr": 0.09184717195169345,
    "lgb_subsample": 0.9797480770012145,
}

xgb_model = xgb.XGBRegressor(
    n_estimators=best_params["xgb_n_estimators"],
    max_depth=best_params["xgb_max_depth"],
    learning_rate=best_params["xgb_lr"],
    subsample=best_params["xgb_subsample"],
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=best_params["lgb_n_estimators"],
    max_depth=best_params["lgb_max_depth"],
    learning_rate=best_params["lgb_lr"],
    subsample=best_params["lgb_subsample"],
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

ensemble = StackingRegressor(
    estimators=[("xgb", xgb_model), ("lgb", lgb_model)],
    final_estimator=xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    ),
    cv=3,
)

# 4. Train & evaluate
print("Training FINAL FWD ensemble with best hyperparameters...")
ensemble.fit(X_train, y_train)

y_pred_train = ensemble.predict(X_train)
y_pred_test  = ensemble.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2  = r2_score(y_test,  y_pred_test)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

print("\n📈 FINAL FWD MODEL PERFORMANCE (FIXED SPLIT)")
print(f"Train R^2: {train_r2:.3f}")
print(f"Test  R^2: {test_r2:.3f}   (baseline ≈ 0.065 on same split)")
print(f"Test  RMSE: {test_rmse:.3f}")

# 5. Save model + features
Path("models").mkdir(exist_ok=True)
joblib.dump(ensemble, "models/FWD_final_ensemble.pkl")
with open("models/FWD_final_features.txt", "w") as f:
    f.write("\n".join(feature_cols))

print("\n✅ Saved FINAL FWD model to models/FWD_final_ensemble.pkl")
