"""
Retrain ensembles using enhanced features and existing tuned hyperparameters.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import sys

sys.path.insert(0, 'src')
from utils import logger


class TunedEnsemble:
    def __init__(self, xgb_params, rf_params):
        self.models = [
            ("XGB_1", XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)),
            ("XGB_2", XGBRegressor(**xgb_params, random_state=43, n_jobs=-1)),
            ("RF_1", RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)),
            ("RF_2", RandomForestRegressor(**rf_params, random_state=43, n_jobs=-1)),
        ]

    def fit(self, X, y, sample_weight=None):
        for name, model in self.models:
            if sample_weight is not None:
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)

    def predict(self, X):
        preds = [m.predict(X) for _, m in self.models]
        return np.mean(preds, axis=0)


def load_data(position: str):
    path = Path(f"data/processed/{position}_features_enhanced.csv")
    df = pd.read_csv(path)

    train_seasons = ["2021-22", "2022-23"]
    test_season = "2023-24"

    train_mask = df["season"].isin(train_seasons)
    test_mask = df["season"] == test_season

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # ดรอปคอลัมน์ที่ไม่ใช่ feature
    drop_cols = [
        "target",
        "sample_weight",
        "round",
        "player_id",
        "element",
        "season",
        "team",
        "opponent_team",
        "name",
        "position",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    # เอาเฉพาะ numeric columns
    numeric_df = df.drop(columns=drop_cols, errors="ignore")

    # ★★ เติม NaN เป็น 0 (interpret ว่า "no history / no stat")
    numeric_df = numeric_df.fillna(0.0)

    feature_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()

    # apply column set + fillna ให้ train/test ด้วย
    train_num = train_df[feature_cols].fillna(0.0)
    test_num = test_df[feature_cols].fillna(0.0)

    X_train = train_num.values
    y_train = train_df["target"].values

    weights = None
    if "sample_weight" in train_df.columns:
        weights = train_df["sample_weight"].values

    X_test = test_num.values
    y_test = test_df["target"].values

    logger.info(
        f"{position}: {len(feature_cols)} numeric features, Train={len(X_train)}, Test={len(X_test)}"
    )

    return X_train, y_train, X_test, y_test, weights



def retrain_position(position: str):
    logger.info("\n" + "=" * 70)
    logger.info(f"RETRAINING {position} WITH ENHANCED FEATURES")
    logger.info("=" * 70)

    res_path = Path(f"tuning_results/{position}_optuna_results.pkl")
    tuned = joblib.load(res_path)
    xgb_params = tuned["xgb_params"]
    rf_params = tuned["rf_params"]

    X_train, y_train, X_test, y_test, weights = load_data(position)

    ens = TunedEnsemble(xgb_params, rf_params)
    logger.info("Training ensemble...")
    ens.fit(X_train, y_train, weights)

    y_pred_test = ens.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    logger.info(f"Enhanced tuned (test):  R²={test_r2:.3f}, RMSE={test_rmse:.3f}")

    joblib.dump(ens, f"models/{position}_ensemble_enhanced.pkl")
    joblib.dump(
        {"test": {"r2": test_r2, "rmse": test_rmse}, "position": position},
        f"models/{position}_metrics_enhanced.pkl",
    )

    return position, test_r2, test_rmse



def main():
    logger.info("=" * 80)
    logger.info("RETRAINING WITH ENHANCED EXPECTED-STATS FEATURES")
    logger.info("=" * 80)

    positions = ["GK", "DEF", "MID", "FWD"]
    summary = []

    for pos in positions:
        try:
            summary.append(retrain_position(pos))
        except Exception as e:
            logger.error(f"Error retraining {pos}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Enhanced tuned performance")
    logger.info("=" * 80)

    logger.info(f"\n{'Position':<8} | {'R²':>8} | {'RMSE':>8}")
    logger.info("-" * 30)
    for pos, r2, rmse in summary:
        logger.info(f"{pos:<8} | {r2:>8.3f} | {rmse:>8.3f}")



if __name__ == "__main__":
    main()
