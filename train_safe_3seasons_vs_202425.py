#!/usr/bin/env python3
"""
Train safe-final models แบบใหม่:
- Train: seasons 2021-22, 2022-23, 2023-24
- Test:  season 2024-25 (ทั้งฤดูกาล GW1-38)

ผลลัพธ์:
- models/{POS}_safe_3seasons_train_202425_test.pkl
- พิมพ์ R2 / RMSE ของแต่ละตำแหน่งบน 2024-25
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
POSITIONS = ["GK", "DEF", "MID", "FWD"]


def load_safe_features(pos: str) -> pd.DataFrame:
    path = f"data/processed/{pos}_features_enhanced_safe.csv"
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    df["round"] = df["round"].astype(int)
    return df


def split_train_test(df: pd.DataFrame):
    train_mask = df["season"].isin(TRAIN_SEASONS)
    test_mask = df["season"] == TEST_SEASON

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(
        f"  Train seasons: {sorted(train_df['season'].unique())} "
        f"| Test season: {sorted(test_df['season'].unique())}"
    )
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    return train_df, test_df


def build_ensemble():
    xgb_model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgb_model = lgb.LGBMRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    meta = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )

    return StackingRegressor(
        estimators=[("xgb", xgb_model), ("lgb", lgb_model)],
        final_estimator=meta,
        cv=3,
        n_jobs=-1,
    )


def train_position(pos: str):
    print(f"\n================ {pos} ================")
    df = load_safe_features(pos)
    train_df, test_df = split_train_test(df)

    # โหลด feature_cols เดิมจากไฟล์ safe-final ปัจจุบัน ถ้ามี
    feat_file = Path(f"models/{pos}_202425_features.txt")
    if feat_file.exists():
        with open(feat_file) as f:
            feature_cols = [c for c in f.read().strip().split("\n") if c != "target"]
    else:
        # fallback: ตัด column ที่ไม่ใช่ฟีเจอร์พื้นฐาน
        drop_cols = [
            "target",
            "season",
            "round",
            "player_id",
            "team",
            "player_name",
            "web_name",
        ]
        feature_cols = [c for c in df.columns if c not in drop_cols]

    print(f"  Features: {len(feature_cols)}")

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target"].astype(float)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target"].astype(float)

    model = build_ensemble()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"  Test (season {TEST_SEASON}) R²: {r2:.3f} | RMSE: {rmse:.3f}")

    Path("models").mkdir(exist_ok=True)
    out_model = f"models/{pos}_safe_3seasons_train_202425_test.pkl"
    joblib.dump(model, out_model)

    # เซฟ feature_cols สำหรับโมเดลนี้
    with open(f"models/{pos}_safe_3seasons_features.txt", "w") as f:
        f.write("\n".join(feature_cols))

    print(f"  ✅ Saved model: {out_model}")
    return r2, rmse, len(test_df)


def main():
    results = {}
    for pos in POSITIONS:
        r2, rmse, n_test = train_position(pos)
        results[pos] = (r2, rmse, n_test)

    print("\n===== 3-SEASON TRAIN → 2024-25 TEST (FULL SEASON) =====")
    print("Pos | R²    | RMSE  | Test rows")
    print("--------------------------------")
    for pos in POSITIONS:
        r2, rmse, n_test = results[pos]
        print(f"{pos:<3} | {r2:5.3f} | {rmse:5.3f} | {n_test:8d}")


if __name__ == "__main__":
    main()
