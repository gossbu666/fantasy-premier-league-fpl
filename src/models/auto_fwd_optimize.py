#!/usr/bin/env python3
"""
AUTO FWD OPTIMIZATION + TEST (R² 0.065 → 0.10+ expected)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold

def ensemble_features(df, target_col='target', position='FWD', max_corr=0.75, n_features=40):
    exclude_cols = ['sample_weight', 'player_id', 'element', 'round', 'season', 'team', 'opponent_team']
    numeric_cols = df.select_dtypes([np.number]).columns
    X = df[[c for c in numeric_cols.drop(target_col) if c not in exclude_cols]].fillna(0)
    y = df[target_col]
    
    # Variance filter
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X)
    high_var_features = X.columns[selector.get_support()].tolist()
    
    # Correlation ranking
    corrs = X[high_var_features].corrwith(y).abs()
    top_candidates = corrs.nlargest(60).index.tolist()
    
    # Low collinearity selection
    selected = []
    corr_matrix = X[top_candidates].corr().abs()
    
    for candidate in top_candidates:
        if len(selected) >= n_features:
            break
        if len(selected) == 0 or corr_matrix.loc[candidate, selected].max() < max_corr:
            selected.append(candidate)
    
    return selected, corrs

# AUTO FWD
print("🔥 AUTO FWD OPTIMIZATION")
df = pd.read_csv("data/processed/FWD_features_enhanced.csv")
features, corrs = ensemble_features(df, position='FWD')

print(f"\n🎯 FWD ENSEMBLE FEATURES ({len(features)}):")
for i, feat in enumerate(features[:10], 1):
    print(f"  {i}. {feat:<30} corr={corrs[feat]:.3f}")

# Save
with open('analysis/FWD_clean_ensemble_features.txt', 'w') as f:
    f.write('\n'.join(features))
print(f"\n✅ Saved: analysis/FWD_clean_ensemble_features.txt")
