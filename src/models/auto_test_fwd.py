#!/usr/bin/env python3
"""
AUTO TEST FWD Optimized Performance
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb

def test_fwd_optimized():
    print("\n🔥 TESTING FWD OPTIMIZED ENSEMBLE")
    
    # Load clean features
    with open('analysis/FWD_clean_ensemble_features.txt', 'r') as f:
        feature_list = [line.strip() for line in f if line.strip()]
    
    df = pd.read_csv("data/processed/FWD_features_enhanced.csv")
    available_features = [f for f in feature_list if f in df.columns]
    X = df[available_features].select_dtypes(include=['number']).fillna(0)
    y = df['target'].fillna(0)
    
    # Original split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.366, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['round']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"📊 FWD: {len(available_features)} features | Train={len(X_train)} Test={len(X_test)}")
    
    # Ensemble
    xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    lgb_model = lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    
    ensemble = StackingRegressor(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
        final_estimator=xgb.XGBRegressor(n_estimators=100, random_state=42),
        cv=5
    )
    
    ensemble.fit(X_train, y_train)
    test_pred = ensemble.predict(X_test)
    
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = mean_squared_error(y_test, test_pred, squared=False)
    
    print(f"\n📈 FWD OPTIMIZED RESULTS:")
    print(f"   Test R²: {test_r2:.3f} (vs Original 0.065)")
    print(f"   Test RMSE: {test_rmse:.3f}")
    
    # Save if improved
    if test_r2 > 0.065:
        Path("models").mkdir(exist_ok=True)
        joblib.dump(ensemble, "models/FWD_optimized_ensemble.pkl")
        print("✅ SAVED IMPROVED MODEL!")
    else:
        print("❌ No improvement over original")
    
    return test_r2

if __name__ == "__main__":
    test_fwd_optimized()
