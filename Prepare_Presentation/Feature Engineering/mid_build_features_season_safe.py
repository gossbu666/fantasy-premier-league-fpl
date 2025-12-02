#!/usr/bin/env python3
import pandas as pd
import numpy as np
from utils_rolling import add_rolling_features_season_player

# 1) โหลด base data
df = pd.read_csv("data/processed/MID_data.csv")
df["season"] = df["season"].astype(str)

print(f"Loaded MID_data.csv: {len(df)} rows")

# 2) Rolling per (season, player) 
player_roll_cols = [
    "total_points",
    "goals_scored", 
    "assists",
    "minutes",
    "bps",
    "influence",
    "creativity",
    "threat",
]

player_roll_cols = [c for c in player_roll_cols if c in df.columns]
print(f"Player roll cols: {player_roll_cols}")

df = add_rolling_features_season_player(
    df,
    value_cols=player_roll_cols,
    windows=(1, 3, 5, 10),
    prefix="",
)

# 3) Rolling per (season, team)
df = df.sort_values(["season", "team", "round"]).copy()
team_group = df.groupby(["season", "team"], group_keys=False)

team_roll_cols = []
if "goals_scored" in df.columns:
    team_roll_cols.append("goals_scored")
if "goals_conceded" in df.columns:
    team_roll_cols.append("goals_conceded")

print(f"Team roll cols: {team_roll_cols}")

for col in team_roll_cols:
    for w in (3, 5, 10):
        new_col = f"{col}_roll_team{w}"
        df[new_col] = team_group[col].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )

# 4) Target
df["target"] = df["total_points"].astype(float)

# 5) เลือกคอลัมน์
base_feature_cols = [
    "season", "round", "player_id", "team", "opponent_team", 
    "element_type", "now_cost", "minutes", "goals_scored", 
    "assists", "bps", "influence", "creativity", "threat",
    "ict_index", "fdr_attack", "fdr_defense", "is_home"
]

base_feature_cols = [c for c in base_feature_cols if c in df.columns]
roll_feature_cols = [c for c in df.columns if "roll" in c]

feature_cols = base_feature_cols + roll_feature_cols
print(f"Final features: {len(feature_cols)} total")

out = df[feature_cols + ["target"]].copy()
out.to_csv("data/processed/MID_features_enhanced_safe.csv", index=False)
print("✅ Saved season-safe MID features")
