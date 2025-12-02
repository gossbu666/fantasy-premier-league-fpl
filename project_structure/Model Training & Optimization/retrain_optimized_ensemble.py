"""
OPTIMIZED ENSEMBLE - EXACT SAME SPLIT AS ORIGINAL retrain_enhanced.py
"""

import pandas as pd
import numpy as np
import argparse
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb

def load_optimized_features(position):
    feature_file = f"analysis/GK_clean_ensemble_features.txt"
    with open(feature_file, 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    return features

def get_original_split_indices(position):
    """Get EXACT same train/test indices as original retrain_enhanced.py."""
    df = pd.read_csv(f"data/processed/{position}_features_enhanced.csv")
    
    # EXACT same split logic as original (by round + random_state)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.366, random_state=42)
    
    # Use 'round' as group (same as original time-series split)
    train_idx, test_idx = next(gss.split(df, df['target'], groups=df['round']))
    
    print(f"🔍 {position}: Train={len(train_idx)}, Test={len(test_idx)}")
    return train_idx, test_idx

def retrain_position_optimized(position):
    print(f"\n{'='*70}")
    print(f"🔥 RETRAINING {position} OPTIMIZED ENSEMBLE")
    print(f"{'='*70}")
    
    # Load optimized features
    feature_list = load_optimized_features(position)
    df = pd.read_csv(f"data/processed/{position}_features_enhanced.csv")
    
    # Select features
    available_features = [f for f in feature_list if f in df.columns]
    X = df[available_features].select_dtypes(include=['number']).fillna(0)
    y = df['target'].fillna(0)
    
    print(f"📊 {position}: {len(available_features)} optimized features")
    
    # EXACT SAME SPLIT AS ORIGINAL
    train_idx, test_idx = get_original_split_indices(position)
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    print(f"✅ MATCHED original split: Train={len(X_train)}, Test={len(X_test)}")
    
    # SAME ensemble models + params as original
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        verbose=-1
    )
    
    ensemble = StackingRegressor(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
        final_estimator=xgb.XGBRegressor(n_estimators=100, random_state=42),
        cv=5
    )
    
    print("🚀 Training optimized ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    train_pred = ensemble.predict(X_train)
    test_pred = ensemble.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = mean_squared_error(y_test, test_pred, squared=False)
    
    print(f"\n📈 {position} OPTIMIZED ENSEMBLE RESULTS:")
    print(f"   Train R²: {train_r2:.3f}")
    print(f"   Test  R²: {test_r2:.3f}")
    print(f"   Test  RMSE: {test_rmse:.3f}")
    print(f"   vs Original: 0.451 → {test_r2:.3f} {'✅' if test_r2 > 0.451 else '❌'}")
    
    # Save
    Path("models").mkdir(exist_ok=True)
    joblib.dump(ensemble, f"models/{position}_optimized_ensemble.pkl")
    with open(f"models/{position}_optimized_features.txt", 'w') as f:
        f.write('\n'.join(available_features))
    
    return test_r2, test_rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--position', required=True)
    args = parser.parse_args()
    
    r2, rmse = retrain_position_optimized(args.position)
    print(f"\n✅ {args.position} OPTIMIZED ENSEMBLE COMPLETE!")
    print(f"📊 Final: R²={r2:.3f}, RMSE={rmse:.3f}")

if __name__ == "__main__":
    main()
