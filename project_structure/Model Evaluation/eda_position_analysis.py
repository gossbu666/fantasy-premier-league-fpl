"""
Position-specific EDA - MATCHES ACTUAL COLUMN NAMES
Columns: *_avg_1gw, *_avg_3gw, *_avg_5gw, *_avg_10gw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_enhanced_data(position):
    df = pd.read_csv(f"data/processed/{position}_features_enhanced.csv")
    print(f"\n📊 {position}: {len(df)} rows, {len(df.columns)} columns")
    print(f"   Key columns: {list(df.columns[:15])}...")
    return df

def eda_gk(df_gk):
    """GK EDA - uses *_avg_*gw columns."""
    print("\n" + "="*70)
    print("🧤 GOALKEEPER EDA INSIGHTS")
    print("="*70)
    
    # Recent form vs performance
    recent_cols = [c for c in df_gk.columns if 'avg_3gw' in c]
    print(f"🔥 Recent 3GW features ({len(recent_cols)}): {recent_cols[:8]}...")
    
    # Minutes correlation (using avg_3gw as proxy)
    if 'minutes_avg_3gw' in df_gk.columns:
        print(f"⏱️  Minutes 3GW vs other metrics:")
        min_cols = [c for c in df_gk.columns if 'minutes_avg_3gw' in df_gk.columns and any(x in c for x in ['bps', 'ict', 'influence'])]
        print(f"   Top correlated: {min_cols[:3]}")
    
    # Clean sheets rolling
    cs_cols = [c for c in df_gk.columns if 'clean_sheets_avg' in c]
    print(f"🏆 Clean sheet trends: {cs_cols}")

def eda_def(df_def):
    """DEF EDA."""
    print("\n" + "="*70)
    print("🛡️ DEFENDER EDA INSIGHTS") 
    print("="*70)
    
    # Team clean sheets (if available)
    team_cs_cols = [c for c in df_def.columns if 'team_clean_sheets' in c]
    print(f"🏆 Team CS features: {team_cs_cols}")
    
    # Attacking threat for defenders
    att_cols = [c for c in df_def.columns if any(x in c for x in ['goals_scored_avg', 'assists_avg', 'threat', 'creativity'])]
    print(f"⚽ DEF attacking ({len(att_cols)}): {att_cols[:5]}...")
    
    # Price vs performance (if now_cost exists)
    if 'now_cost' in df_def.columns:
        print(f"💰 DEF price range: {df_def['now_cost'].min():.1f}-{df_def['now_cost'].max():.1f}")

def eda_mid(df_mid):
    """MID EDA."""
    print("\n" + "="*70)
    print("⚽ MIDFIELDER EDA INSIGHTS")
    print("="*70)
    
    # xGI rolling averages
    xgi_cols = [c for c in df_mid.columns if 'goal_involvements' in c or 'expected_goal' in c]
    print(f"🎯 xGI features ({len(xgi_cols)}): {xgi_cols}")
    
    # Form stability
    form_cols = [c for c in df_mid.columns if 'total_points' in c and ('std' in c or 'roll' in c)]
    print(f"📈 Form stability: {form_cols}")

def eda_fwd(df_fwd):
    """FWD EDA."""
    print("\n" + "="*70)
    print("🔥 FORWARD EDA INSIGHTS")
    print("="*70)
    
    # Recent form (most important per feature importance)
    recent_form = [c for c in df_fwd.columns if 'avg_3gw' in c][:5]
    print(f"🚀 Top recent form features: {recent_form}")
    
    # Expected goals chain
    xg_cols = [c for c in df_fwd.columns if 'expected_goals' in c]
    print(f"📊 xG features ({len(xg_cols)}): {xg_cols}")

def feature_summary(df, position):
    """Summary of feature groups."""
    print(f"\n📋 {position} FEATURE GROUPS:")
    
    player_cols = [c for c in df.columns if any(x in c for x in ['avg_', 'bps', 'ict', 'influence', 'threat', 'creativity'])]
    team_cols = [c for c in df.columns if 'team_' in c]
    opp_cols = [c for c in df.columns if 'opp_' in c]
    
    print(f"   👤 Player features: {len(player_cols)}")
    print(f"   🏃‍♂️ Team features: {len(team_cols)}") 
    print(f"   👥 Opponent features: {len(opp_cols)}")
    
    # Top 10 most variable features (good for modeling)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        variability = df[num_cols].std().sort_values(ascending=False)
        print(f"   📊 Most variable (top 5): {variability.head().index.tolist()}")

def main():
    Path("analysis").mkdir(exist_ok=True)
    
    positions = {
        'GK': eda_gk,
        'DEF': eda_def,
        'MID': eda_mid,
        'FWD': eda_fwd
    }
    
    for pos, eda_func in positions.items():
        try:
            df = load_enhanced_data(pos)
            feature_summary(df, pos)
            eda_func(df)
            print(f"\n✅ {pos} EDA complete ✓")
        except Exception as e:
            print(f"❌ Error in {pos}: {e}")
    
    print("\n" + "="*70)
    print("🎉 EDA ANALYSIS COMPLETE!")
    print("💡 Key insights ready for paper Results/Discussion")

if __name__ == "__main__":
    main()
