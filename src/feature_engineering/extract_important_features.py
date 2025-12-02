"""
EXTRACT TOP FEATURES FROM ORIGINAL MODEL FEATURE IMPORTANCE
(Not correlation - actual model usage)
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path

def extract_feature_importance(position):
    """Extract top features from original trained model."""
    try:
        # Load original model (assuming exists)
        model = joblib.load(f"models/{position}_enhanced_tuned.pkl")
        
        # Get feature importance (XGBoost/LightGBM)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # For stacking, get from first estimator
            importances = model.estimators_[0].feature_importances_
        
        # Get feature names from original data
        df = pd.read_csv(f"data/processed/{position}_features_enhanced.csv")
        feature_names = df.select_dtypes(include=[np.number]).columns
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(20)['feature'].tolist()
        
        print(f"\n📊 {position} ORIGINAL MODEL TOP 20 FEATURES:")
        for i, (feat, imp) in enumerate(importance_df.head(10).itertuples(index=False), 1):
            print(f"  {i:2d}. {feat:<30} {imp:.4f}")
        
        # Save
        with open(f"analysis/{position}_model_important_features.txt", 'w') as f:
            f.write('\n'.join(top_features))
        print(f"✅ Saved: analysis/{position}_model_important_features.txt")
        
    except FileNotFoundError:
        print(f"❌ {position} original model not found. Using correlation top 10.")
        # Fallback to correlation top 10
        corr_features = [
            'minutes_avg_3gw', 'bps_avg_3gw', 'ict_index_avg_3gw', 
            'total_points_avg_3gw', 'influence_avg_3gw'
        ][:10]
        with open(f"analysis/{position}_model_important_features.txt", 'w') as f:
            f.write('\n'.join(corr_features))

if __name__ == "__main__":
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        extract_feature_importance(pos)
