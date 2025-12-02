#!/usr/bin/env python3
"""
FWD SUPER FEATURES (Form + Fixture + Momentum)
"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/FWD_features_enhanced.csv")

# SUPER FEATURES
df['form_momentum'] = df['total_points_avg_3gw'] - df['total_points_avg_10gw']
df['minutes_trend'] = df['minutes_avg_1gw'] - df['minutes_avg_10gw']
df['goal_scoring_rate'] = df['goals_scored_avg_3gw'] / (df['minutes_avg_3gw'] + 1)
df['fixture_difficulty_score'] = 10 - df['opponent_team_strength']  # Easier = higher score

new_features = ['form_momentum', 'minutes_trend', 'goal_scoring_rate', 'fixture_difficulty_score']
print("🆕 NEW FWD SUPER FEATURES:", new_features)

# Combine with best ensemble features
with open('analysis/FWD_clean_ensemble_features.txt', 'r') as f:
    base_features = [line.strip() for line in f if line.strip()]

all_features = base_features + new_features

# Save
with open('analysis/FWD_super_features.txt', 'w') as f:
    f.write('\n'.join(all_features))
print(f"✅ FWD SUPER FEATURES ({len(all_features)}): Ready!")
