#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import StackingRegressor

import xgboost as xgb
import lightgbm as lgb

TRAIN_SEASONS = ["2021-22", "2022-23"] 
TEST_SEASON   = "2023-24"
TEST_START_GW = 14  # test = GW14-38 ของ 2023-24 (สมจริง)
POSITIONS     = ["GK", "DEF", "MID", "FWD"]

def load_position_df(pos: str) -> pd.DataFrame:
    path = f"data/processed/{pos}_features_enhanced_safe.csv"
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    df["round"] = df["round"].astype(int)
    return df

def make_season_split(df: pd.DataFrame):
    # Train: seasons 2021-22 + 2022-23 เต็ม + 2023-24 GW1-13
    train_mask = (
        df["season"].isin(TRAIN_SEASONS) | 
        ((df["season"] == TEST_SEASON) & (df["round"] < TEST_START_GW))
    )
    
    # Test: 2023-24 GW14-38 เท่านั้น (สมจริง)
    test_mask = (df["season"] == TEST_SEASON) & (df["round"] >= TEST_START_GW)
    
    train_df = df[train_mask].copy()
    test_df  = df[test_mask].copy()
    
    print(f"  Train: {sorted(train_df['season'].unique())} + {TEST_SEASON} GW1-{TEST_START_GW-1}")
    print(f"  Test:  {TEST_SEASON} GW{TEST_START_GW}-38")
    return train_df, test_df

def build_models(best_params=None):
    if best_params is None:
        best_params = {
            "xgb_n_estimators": 600, "xgb_max_depth": 6, "xgb_lr": 0.05, "xgb_subsample": 0.8,
            "lgb_n_estimators": 600, "lgb_max_depth": 6, "lgb_lr": 0.05, "lgb_subsample": 0.8,
        }

    xgb_model = xgb.XGBRegressor(n_estimators=best_params["xgb_n_estimators"],
                                max_depth=best_params["xgb_max_depth"],
                                learning_rate=best_params["xgb_lr"],
                                subsample=best_params["xgb_subsample"],
                                colsample_bytree=0.8, random_state=42, n_jobs=-1)

    lgb_model = lgb.LGBMRegressor(n_estimators=best_params["lgb_n_estimators"],
                                 max_depth=best_params["lgb_max_depth"],
                                 learning_rate=best_params["lgb_lr"],
                                 subsample=best_params["lgb_subsample"],
                                 colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)

    meta = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

    return StackingRegressor(estimators=[("xgb", xgb_model), ("lgb", lgb_model)],
                            final_estimator=meta, cv=3)

def train_and_eval_position(pos: str, best_params_fwd=None):
    print(f"\n================ {pos} ================")
    df = load_position_df(pos)
    train_df, test_df = make_season_split(df)

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "target"]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target"].fillna(0)
    X_test  = test_df[feature_cols].fillna(0)
    y_test  = test_df["target"].fillna(0)

    print(f"  Train samples={len(X_train)}, Test samples={len(X_test)}, Features={len(feature_cols)}")

    if pos == "FWD" and best_params_fwd:
        ensemble = build_models(best_params_fwd)
        print("  Using Optuna-tuned params")
    else:
        ensemble = build_models()

    ensemble.fit(X_train, y_train)
    y_pred_train = ensemble.predict(X_train)
    y_pred_test  = ensemble.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2  = r2_score(y_test, y_pred_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

    print(f"  Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f} | RMSE: {test_rmse:.3f}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(ensemble, f"models/{pos}_seasonstack_final.pkl")
    with open(f"models/{pos}_seasonstack_final_features.txt", "w") as f:
        f.write("\n".join(feature_cols))
    print(f"  ✅ Saved {pos} model")

    return test_r2, test_rmse

def main():
    best_params_fwd = {
        "xgb_n_estimators": 837, "xgb_max_depth": 6, "xgb_lr": 0.010782958714031302, "xgb_subsample": 0.7002760369964917,
        "lgb_n_estimators": 836, "lgb_max_depth": 4, "lgb_lr": 0.09184717195169345, "lgb_subsample": 0.9797480770012145,
    }

    results = {}
    for pos in POSITIONS:
        r2, rmse = train_and_eval_position(pos, best_params_fwd if pos == "FWD" else None)
        results[pos] = (r2, rmse)

    print("\n===== FINAL SUMMARY (Realistic split) =====")
    print("Pos  | R²    | RMSE")
    for pos in POSITIONS:
        r2, rmse = results[pos]
        print(f"{pos:<3} | {r2:5.3f} | {rmse:5.3f}")

if __name__ == "__main__":
    main()
