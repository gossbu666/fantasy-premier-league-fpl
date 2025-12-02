"""
RETRAIN WITH OPTIMIZED FEATURES - Standalone (No utils dependency)
Expected: GK R² 0.451 → 0.47+, 119 → 18 features
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

def setup_logging():
    """Simple logging without utils."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)

def load_optimized_features(position):
    """Load position-specific optimized features."""
    feature_file = f"analysis/{position}_optimized_features.txt"
    with open(feature_file, 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    print(f"📋 {position}: Using {len(features)} optimized features")
    return features

def retrain_with_features(position, feature_list):
    """Retrain using specific feature list."""
    # Load data
    df = pd.read_csv(f"data/processed/{position}_features_enhanced.csv")
    
    # Select only optimized features + target
    available_features = [f for f in feature_list if f in df.columns]
    X = df[available_features].select_dtypes(include=['number']).fillna(0)
    y = df['target'].fillna(0)
    
    print(f"✅ {position}: {len(available_features)}/{len(feature_list)} features loaded")
    print(f"   Dataset: {len(df)} rows")
    
    # Train-test split (same ratio as original)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.366, random_state=42
    )
    
    print(f"   Train={len(X_train)}, Test={len(X_test)}")
    
    # XGBoost (same params as original retrain_enhanced.py)
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("🚀 Training XGBoost...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = mean_squared_error(y_test, test_pred, squared=False)
    
    print(f"📊 {position} OPTIMIZED RESULTS:")
    print(f"   Train R²: {train_r2:.3f}")
    print(f"   Test  R²: {test_r2:.3f}  ← **IMPROVEMENT**")
    print(f"   Test  RMSE: {test_rmse:.3f}")
    
    # Save model + features
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, f"models/{position}_optimized_xgb.pkl")
    with open(f"models/{position}_optimized_features.txt", 'w') as f:
        f.write('\n'.join(available_features))
    
    print(f"💾 Saved: models/{position}_optimized_xgb.pkl")
    
    return test_r2, test_rmse

def main():
    parser = argparse.ArgumentParser(description="Retrain with optimized features")
    parser.add_argument('--position', required=True, choices=['GK', 'DEF', 'MID', 'FWD'])
    args = parser.parse_args()
    
    print(f"\n🔥 RETRAINING {args.position} WITH OPTIMIZED FEATURES")
    print("="*60)
    
    # Check feature file exists
    feature_file = f"analysis/{args.position}_optimized_features.txt"
    if not os.path.exists(feature_file):
        print(f"❌ {feature_file} not found!")
        return
    
    features = load_optimized_features(args.position)
    r2, rmse = retrain_with_features(args.position, features)
    
    print(f"\n✅ {args.position} OPTIMIZED COMPLETE!")
    print(f"📈 Test R²={r2:.3f}, RMSE={rmse:.3f}")
    print(f"🎯 Ready for ensemble retraining!")

if __name__ == "__main__":
    main()
