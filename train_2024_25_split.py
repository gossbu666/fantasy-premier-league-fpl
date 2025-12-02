#!/usr/bin/env python3
"""
NEW SPLIT: Train 2021-24 full + 2024-25 GW1-10 → Test 2024-25 GW11-38
Validate model with MOST RECENT season data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb

TRAIN_SEASONS_FULL = ["2021-22", "2022-23", "2023-24"]
TEST_SEASON = "2024-25"
TEST_START_GW = 11  # Test GW11-38 2024-25
POSITIONS = ["GK", "DEF", "MID", "FWD"]

def load_position_df(pos: str) -> pd.DataFrame:
    path = f"data/processed/{pos}_data.csv"  # ใช้ raw data + compute rolling on-the-fly
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    df["round"] = df["round"].astype(int)
    return df

def make_new_split(df: pd.DataFrame):
    # Train: 2021-24 full + 2024-25 GW1-10
    train_mask = (
        df["season"].isin(TRAIN_SEASONS_FULL) | 
        ((df["season"] == TEST_SEASON) & (df["round"] <= TEST_START_GW))
    )
    # Test: 2024-25 GW11-38
    test_mask = (df["season"] == TEST_SEASON) & (df["round"] > TEST_START_GW)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"  Train: {TRAIN_SEASONS_FULL} + {TEST_SEASON} GW1-{TEST_START_GW}")
    print(f"  Test:  {TEST_SEASON} GW{TEST_START_GW+1}-38 ({len(test_df)} samples)")
    return train_df, test_df

def compute_features(df):
    """Compute rolling features on-the-fly"""
    from utils_rolling import add_rolling_features_season_player
    
    player_roll_cols = ["total_points", "goals_scored", "assists", "minutes", "bps", "influence"]
    player_roll_cols = [c for c in player_roll_cols if c in df.columns]
    
    df = add_rolling_features_season_player(df, player_roll_cols, windows=(1,3,5,10))
    
    # Team rolling
    df = df.sort_values(["season", "team", "round"])
    team_group = df.groupby(["season", "team"], group_keys=False)
    team_cols = ["goals_scored", "goals_conceded"]
    team_cols = [c for c in team_cols if c in df.columns]
    
    for col in team_cols:
        for w in (3, 5, 10):
            new_col = f"{col}_roll_team{w}"
            df[new_col] = team_group[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    
    return df

def build_models(best_params=None):
    if best_params is None:
        best_params = {"xgb_n_estimators": 600, "xgb_max_depth": 6, "xgb_lr": 0.05, "xgb_subsample": 0.8,
                      "lgb_n_estimators": 600, "lgb_max_depth": 6, "lgb_lr": 0.05, "lgb_subsample": 0.8}
    
    xgb_model = xgb.XGBRegressor(**{k.replace("xgb_", ""): v for k, v in best_params.items() if "xgb_" in k},
                                colsample_bytree=0.8, random_state=42, n_jobs=-1)
    lgb_model = lgb.LGBMRegressor(**{k.replace("lgb_", ""): v for k, v in best_params.items() if "lgb_" in k},
                                 colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    meta = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    
    return StackingRegressor(estimators=[("xgb", xgb_model), ("lgb", lgb_model)], final_estimator=meta, cv=3)

def train_eval_pos(pos: str):
    print(f"\n================ {pos.upper()} ================")
    df = load_position_df(pos)
    train_df, test_df = make_new_split(df)
    
    # Compute features separately
    train_features = compute_features(train_df)
    test_features = compute_features(test_df)
    
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != "target"]
    
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features["total_points"]  # Use raw points as target
    X_test = test_features[feature_cols].fillna(0)
    y_test = test_features["total_points"]
    
    print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_cols)}")
    
    ensemble = build_models()
    ensemble.fit(X_train, y_train)
    
    y_pred_test = ensemble.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    
    print(f"  Test R²: {test_r2:.3f} | RMSE: {test_rmse:.3f}")
    
    # Save
    Path("models").mkdir(exist_ok=True)
    joblib.dump(ensemble, f"models/{pos}_202425_final.pkl")
    print(f"  ✅ Saved {pos} model")
    
    return test_r2, test_rmse

def main():
    best_params_fwd = {"xgb_n_estimators": 837, "xgb_max_depth": 6, "xgb_lr": 0.0108, "xgb_subsample": 0.70,
                      "lgb_n_estimators": 836, "lgb_max_depth": 4, "lgb_lr": 0.0918, "lgb_subsample": 0.98}
    
    results = {}
    for pos in POSITIONS:
        r2, rmse = train_eval_pos(pos)
        results[pos] = (r2, rmse)
    
    print("\n===== 2024-25 VALIDATION (GW11-38) =====")
    print("Pos  | R²    | RMSE")
    for pos in POSITIONS:
        r2, rmse = results[pos]
        print(f"{pos:<3} | {r2:5.3f} | {rmse:5.3f}")

if __name__ == "__main__":
    main()
