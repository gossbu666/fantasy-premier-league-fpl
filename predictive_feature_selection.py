"""
COMPLETE PREDICTIVE FEATURE SELECTION (Correlation + MI + Redundancy)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

def load_position_data(position):
    df = pd.read_csv(f"data/processed/{position}_features_enhanced.csv")
    
    # Exclude ID columns, target, weights, categorical
    exclude_cols = ['target', 'sample_weight', 'player_id', 'element', 'name', 'position', 'round', 'season', 'team', 'opponent_team']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    numeric_df = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    target = df['target'].fillna(0)
    
    # Remove constant features (std=0)
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0.001]
    
    return numeric_df, target

def correlation_with_target(df, target):
    corrs = df.corrwith(target).abs().sort_values(ascending=False)
    return corrs

def top_predictive_features(df, target, position, top_n=20):
    corrs = correlation_with_target(df, target)
    
    print(f"\n{'='*70}")
    print(f"🎯 TOP {top_n} PREDICTIVE FEATURES - {position}")
    print(f"{'='*70}")
    print(corrs.head(top_n).round(4))
    
    plt.figure(figsize=(10, 8))
    top_corr = corrs.head(25)
    colors = ['red' if x > 0.15 else 'orange' if x > 0.10 else 'blue' for x in top_corr.values]
    plt.barh(range(len(top_corr)), top_corr.values, color=colors)
    plt.yticks(range(len(top_corr)), top_corr.index, fontsize=9)
    plt.axvline(0.10, color='orange', linestyle='--', label='Moderate corr (>0.10)')
    plt.axvline(0.15, color='red', linestyle='--', label='Strong corr (>0.15)')
    plt.xlabel('Absolute Correlation with Target')
    plt.title(f'{position}: Top Predictive Features')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"analysis/{position}_target_corr.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return corrs.head(20).index.tolist()

def mutual_information_scores(df, target, position, top_n=15):
    """Non-linear relationships."""
    mi_scores = mutual_info_regression(df.fillna(0), target, random_state=42, n_jobs=-1)
    mi_df = pd.Series(mi_scores, index=df.columns).sort_values(ascending=False)
    
    print(f"\n🔗 {position} Mutual Information (top {top_n}):")
    print(mi_df.head(top_n).round(4))
    
    plt.figure(figsize=(10, 6))
    top_mi = mi_df.head(20)
    plt.barh(range(len(top_mi)), top_mi.values)
    plt.yticks(range(len(top_mi)), top_mi.index, fontsize=9)
    plt.xlabel('Mutual Information Score')
    plt.title(f'{position}: Mutual Information with Target')
    plt.tight_layout()
    plt.savefig(f"analysis/{position}_mutual_info.png", dpi=150)
    plt.close()
    
    return mi_df.head(top_n).index.tolist()

def feature_redundancy(df, position):
    """High correlation pairs to drop."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = []
    for col in upper.columns:
        high_corr = upper[col][upper[col] > 0.85]
        high_corr_pairs.extend([(col, high_corr.index[i], high_corr.values[i]) 
                               for i in range(len(high_corr))])
    
    print(f"\n🔄 {position} REDUNDANT PAIRS (>0.85 corr): {len(high_corr_pairs)}")
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:5]:
        print(f"   {feat1:<25} ↔ {feat2:<25} ({corr:.3f})")
    
    return [pair[0] for pair in high_corr_pairs[:3]]  # Suggest dropping one from each pair

def recommend_features(position, corr_features, mi_features, redundant):
    """Recommend final feature set."""
    keep_features = list(set(corr_features[:15] + mi_features[:10]))
    drop_features = [f for f in redundant if f in keep_features]
    
    print(f"\n✅ {position} RECOMMENDATION:")
    print(f"   KEEP ({len(keep_features)}): {keep_features[:8]}...")
    print(f"   CONSIDER DROP ({len(drop_features)}): {drop_features}")
    
    return keep_features

def main():
    Path("analysis").mkdir(exist_ok=True)
    
    positions = ['GK', 'DEF', 'MID', 'FWD']
    recommendations = {}
    
    for pos in positions:
        print(f"\n🔍 Analyzing {pos}...")
        try:
            df_features, target = load_position_data(pos)
            print(f"   Features: {df_features.shape[1]}")
            
            # Step 1: Correlation
            corr_features = top_predictive_features(df_features, target, pos)
            
            # Step 2: Mutual Information  
            mi_features = mutual_information_scores(df_features, target, pos)
            
            # Step 3: Redundancy
            redundant = feature_redundancy(df_features, pos)
            
            # Step 4: Recommendation
            keep_features = recommend_features(pos, corr_features, mi_features, redundant)
            recommendations[pos] = keep_features
            
        except Exception as e:
            print(f"❌ Error {pos}: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("🚀 FINAL FEATURE SELECTION RECOMMENDATIONS")
    print("="*80)
    for pos, features in recommendations.items():
        print(f"\n{pos:>3}: KEEP {len(features)} features")
        print(f"     {', '.join(features[:6])}...")

if __name__ == "__main__":
    main()
