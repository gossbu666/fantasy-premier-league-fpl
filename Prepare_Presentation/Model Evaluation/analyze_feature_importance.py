"""
Extract and analyze feature importance from trained enhanced ensemble models.
FIXED: Import TunedEnsemble class before loading pickled models
"""

import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import traceback

# ★★★ FIX: Import TunedEnsemble from retrain_enhanced.py ★★★
sys.path.insert(0, '.')
from retrain_enhanced import TunedEnsemble  # noqa: F401

def get_feature_names(position):
    """Get feature names for a position from enhanced features file."""
    df = pd.read_csv(f"data/processed/{position}_features_enhanced.csv")
    
    drop_cols = [
        "target", "sample_weight", "round", "player_id", "element",
        "season", "team", "opponent_team", "name", "position"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    numeric_df = df.drop(columns=drop_cols, errors="ignore")
    numeric_df = numeric_df.fillna(0.0)
    feature_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()
    
    return feature_cols

def extract_importance_from_ensemble(position):
    """Extract feature importance from all models in ensemble."""
    ensemble_path = f"models/{position}_ensemble_enhanced.pkl"
    ensemble = joblib.load(ensemble_path)
    
    feature_names = get_feature_names(position)
    
    importances = []
    model_names = []
    
    for name, model in ensemble.models:
        try:
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                importances.append(imp)
                model_names.append(name)
            else:
                print(f"  Warning: {name} has no feature_importances_")
        except Exception as e:
            print(f"  Error extracting importance from {name}: {e}")
    
    if not importances:
        raise ValueError(f"No models with feature_importances_ found in {ensemble_path}")
    
    # Average importance across all models in ensemble
    avg_importance = np.mean(importances, axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    return importance_df

def categorize_features(feature_name):
    """Categorize features into groups (Xp/Xt/Xo/Xs)."""
    feature_lower = feature_name.lower()
    
    if any(x in feature_lower for x in ['opp_', 'opponent']):
        return 'Xo (Opponent)'
    if any(x in feature_lower for x in ['team_', 'matchup']):
        return 'Xt (Team)'
    if any(x in feature_lower for x in ['was_home', 'start_rate', 'minutes_std', 'points_std']):
        return 'Xs (Status)'
    return 'Xp (Player)'

def analyze_position(position):
    """Analyze feature importance for one position."""
    print(f"\n{'='*70}")
    print(f"ANALYZING {position} FEATURE IMPORTANCE")
    print(f"{'='*70}")
    
    try:
        importance_df = extract_importance_from_ensemble(position)
        importance_df['category'] = importance_df['feature'].apply(categorize_features)
        
        # Save full importance
        Path("analysis").mkdir(exist_ok=True)
        importance_df.to_csv(f"analysis/{position}_feature_importance.csv", index=False)
        
        # Top 20 features
        print(f"\nTop 20 most important features for {position}:")
        print("-" * 70)
        for idx, row in importance_df.head(20).iterrows():
            print(f"{row['feature']:<50} {row['importance']:>8.4f}  [{row['category']}]")
        
        # Group by category
        print(f"\n\nImportance by feature group:")
        print("-" * 70)
        category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
        total_imp = importance_df['importance'].sum()
        for cat, imp in category_importance.items():
            pct = (imp / total_imp) * 100
            print(f"{cat:<20} {imp:>10.4f}  ({pct:>5.1f}%)")
        
        return importance_df, category_importance
        
    except Exception as e:
        print(f"❌ Failed to analyze {position}: {e}")
        traceback.print_exc()
        return None, None

def plot_importance(position, importance_df, top_n=25):
    """Plot top N feature importances."""
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = {
        'Xp (Player)': 'steelblue',
        'Xt (Team)': 'darkorange', 
        'Xo (Opponent)': 'forestgreen',
        'Xs (Status)': 'crimson'
    }
    top_features['color'] = top_features['category'].map(colors)
    
    plt.barh(range(len(top_features)), top_features['importance'], color=top_features['color'])
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=9)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Features - {position} (Enhanced Ensemble)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label) for label, color in colors.items()]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.savefig(f"analysis/{position}_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot to analysis/{position}_feature_importance.png")

def main():
    print("🔍 Starting feature importance analysis...")
    Path("analysis").mkdir(exist_ok=True)
    
    positions = ['GK', 'DEF', 'MID', 'FWD']
    all_results = {}
    
    for pos in positions:
        importance_df, category_importance = analyze_position(pos)
        if importance_df is not None:
            plot_importance(pos, importance_df)
            all_results[pos] = {
                'importance_df': importance_df,
                'category_importance': category_importance
            }
    
    # Summary table
    if all_results:
        print("\n" + "="*80)
        print("📊 SUMMARY: Feature Group Importance by Position (%)")
        print("="*80)
        print(f"\n{'Category':<15} | {'GK':>6} | {'DEF':>6} | {'MID':>6} | {'FWD':>6}")
        print("-" * 60)
        
        categories = ['Xp (Player)', 'Xt (Team)', 'Xo (Opponent)', 'Xs (Status)']
        for cat in categories:
            row = [cat[:14]]
            for pos in positions:
                if pos in all_results:
                    cat_imp = all_results[pos]['category_importance']
                    total_imp = cat_imp.sum()
                    pct = (cat_imp.get(cat, 0) / total_imp) * 100 if total_imp > 0 else 0
                    row.append(f"{pct:>5.1f}")
                else:
                    row.append("  N/A")
            print(" | ".join(row))
    
    print("\n✅ Feature importance analysis complete!")
    print("📁 Check analysis/ folder for:")
    print("   • *_feature_importance.csv (full rankings)")
    print("   • *_feature_importance.png (visualizations)")

if __name__ == "__main__":
    main()
