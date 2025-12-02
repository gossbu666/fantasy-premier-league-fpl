#!/usr/bin/env python3
"""
FINAL VALIDATION: Train 2021-24 + 2024-25 GW1-10 → Test 2024-25 GW11+
ใช้ safe features (no leakage) - FIXED
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb

TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24"]
TEST_SEASON = "2024-25"
TEST_START_GW = 11
POSITIONS = ["GK", "DEF", "MID", "FWD"]

def load_safe_df(pos: str) -> pd.DataFrame:
    path = f"data/processed/{pos}_features_enhanced_safe.csv"
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    df["round"] = df["round"].astype(int)
    # target มีอยู่แล้ว
    return df

def make_final_split(df):
    train_mask = (
        df["season"].isin(TRAIN_SEASONS) | 
        ((df["season"] == TEST_SEASON) & (df["round"] <= TEST_START_GW))
    )
    test_mask = (df["season"] == TEST_SEASON) & (df["round"] > TEST_START_GW)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"  Train: {len(train_df)} samples | Test: {len(test_df)} samples")
    print(f"  Split: {TRAIN_SEASONS} + {TEST_SEASON} GW1-{TEST_START_GW} → GW{TEST_START_GW+1}+")
    return train_df, test_df

def build_ensemble():
    xgb_model = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, 
                                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    lgb_model = lgb.LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, 
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, 
                                 n_jobs=-1, verbose=-1)
    meta = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    
    return StackingRegressor(estimators=[("xgb", xgb_model), ("lgb", lgb_model)], 
                            final_estimator=meta, cv=3)

def train_position(pos: str):
    print(f"\n================ {pos.upper()} ================")
    df = load_safe_df(pos)
    train_df, test_df = make_final_split(df)
    
    # Features (exclude target/season/round/player_id/team)
    exclude_cols = ["target", "season", "round", "player_id", "team", "player_name", "element_type", "opponent_team"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target"]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target"]
    
    print(f"  Features: {len(feature_cols)}")
    
    model = build_ensemble()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    print(f"  Test R²: {r2:.3f} | RMSE: {rmse:.3f}")
    
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, f"models/{pos}_202425_safe_final.pkl")
    with open(f"models/{pos}_202425_features.txt", "w") as f:
        f.write("\n".join(feature_cols))
    
    print(f"  ✅ Saved {pos} model")
    return r2, rmse

def main():
    results = {}
    for pos in POSITIONS:
        r2, rmse = train_position(pos)
        results[pos] = (r2, rmse)
    
    print("\n===== FINAL 2024-25 RESULTS (Safe, No Leakage) =====")
    print("Pos  | R²    | RMSE  | Test Samples")
    print("-" * 35)
    for pos in POSITIONS:
        r2, rmse = results[pos]
        df = load_safe_df(pos)
        test_samples = len(df[(df["season"] == "2024-25") & (df["round"] > 10)])
        print(f"{pos:<3} | {r2:5.3f} | {rmse:5.3f} | {test_samples:3}")

if __name__ == "__main__":
    main()
