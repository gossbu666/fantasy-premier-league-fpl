#!/usr/bin/env python3
"""
Predict Salah GW ล่าสุดใน 2025-26 season
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils_rolling import add_rolling_features_season_player

def load_model(pos):
    model = joblib.load(f"models/{pos}_seasonstack_final.pkl")
    with open(f"models/{pos}_seasonstack_final_features.txt") as f:
        feature_cols = f.read().strip().split("\n")
    return model, feature_cols

print("🔍 Checking Salah 2025-26 data...")
df = pd.read_csv("data/processed/MID_data.csv")

# หา Salah 2025-26
salah_2025_26 = df[
    (df["player_name"].str.contains("Salah", case=False)) & 
    (df["season"] == "2025-26")
]

if salah_2025_26.empty:
    print("❌ No Salah 2025-26 data found")
    print("Available seasons for Salah:")
    print(df[df["player_name"].str.contains("Salah", case=False)]["season"].unique())
    exit()

print(f"✅ Found {len(salah_2025_26)} Salah rows in 2025-26")
print("Latest GWs:")
print(salah_2025_26[["round", "total_points", "season"]].tail())

# GW ล่าสุด
latest = salah_2025_26.iloc[-1]
gw = int(latest["round"])
print(f"\n🔮 Predicting GW{gw} 2025-26...")

model, feature_cols = load_model("MID")

# คำนวณ rolling features
player_roll_cols = ["total_points", "goals_scored", "assists", "minutes", "bps", "influence", "creativity", "threat"]
player_roll_cols = [c for c in player_roll_cols if c in df.columns]

season_data = df[
    (df["player_name"].str.contains("Salah", case=False)) & 
    (df["season"] == "2025-26") & 
    (df["round"] <= gw)
]

if len(season_data) > 0:
    season_with_roll = add_rolling_features_season_player(season_data, player_roll_cols)
    gw_row = season_with_roll[season_with_roll["round"] == gw].iloc[0]
else:
    gw_row = latest

# Predict
X = gw_row[feature_cols].fillna(0).values.reshape(1, -1)
pred = model.predict(X)[0]

actual = latest["total_points"]
print(f"\n🎯 GW{gw} 2025-26:")
print(f"   Actual:   {actual:.1f} points")
print(f"   Predicted: {pred:.1f} points")
print(f"   Error:    {abs(actual-pred):.1f} points")

print(f"\n📈 Rolling form (roll3): {gw_row.get('total_points_roll3', 'N/A'):.1f}")
