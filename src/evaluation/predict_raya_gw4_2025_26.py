#!/usr/bin/env python3
"""
Predict Raya (Arsenal GK #1) GW4 2025-26
"""
import pandas as pd
import numpy as np
import requests
import joblib
from pathlib import Path

print("🔮 Predicting Raya GW4 2025-26...")

# 1. Load GK model
model = joblib.load("models/GK_202425_safe_final.pkl")
with open("models/GK_202425_features.txt") as f:
    feature_cols = f.read().strip().split("\n")

print(f"✅ GK model loaded ({len(feature_cols)} features)")

# 2. Fetch FPL API
print("🌐 Fetching FPL data...")
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
r = requests.get(url, timeout=10)
data = r.json()

# หา Raya (ID: 1)
raya_id = 1
raya = data["elements"][raya_id-1]
print(f"✅ Raya: {raya['web_name']} (ID: {raya_id}, Team: {data['teams'][raya['team']-1]['name']})")

# 3. Fetch history
history_url = f"https://fantasy.premierleague.com/api/element-summary/{raya_id}/"
h = requests.get(history_url, timeout=10)
history = h.json()["history"]

# หา GW4
gw4_data = None
for game in history:
    if game["round"] == 4:
        gw4_data = game
        break

if not gw4_data:
    print("❌ GW4 not found. Using latest GW...")
    gw4_data = history[-1]

print(f"📊 Raya GW{gw4_data['round']}: {gw4_data['total_points']} pts (actual)")

# 4. Feature row
feature_row = pd.Series({
    "season": "2025-26", "round": gw4_data["round"], "player_id": raya_id,
    "team": raya["team"], "element_type": 1,
    "now_cost": raya["now_cost"]/10,
    "minutes": gw4_data["minutes"],
    "total_points": gw4_data["total_points"],
    "goals_conceded": gw4_data["goals_conceded"],
    "saves": gw4_data["saves"],
    "bonus": gw4_data["bonus"],
    "bps": gw4_data["bps"],
    "influence": gw4_data["influence"],
    "ict_index": gw4_data["ict_index"],
    "clean_sheets": gw4_data["clean_sheets"],
})

# เติม features ที่ขาด
for col in feature_cols:
    if col not in feature_row:
        feature_row[col] = 0

# 5. Predict!
X = feature_row[feature_cols].fillna(0).values.reshape(1, -1)
prediction = model.predict(X)[0]

print(f"\n🎯 RAYA GW{gw4_data['round']} 2025-26")
print(f"   Team: {data['teams'][raya['team']-1]['name']}")
print(f"   Actual:   {gw4_data['total_points']:.1f} pts")
print(f"   Predicted: {prediction:.1f} pts")
print(f"   Error:    {abs(gw4_data['total_points']-prediction):.1f} pts")
print(f"   Saves: {gw4_data['saves']} | Goals conceded: {gw4_data['goals_conceded']}")
print(f"   Clean sheet: {'Yes' if gw4_data['clean_sheets'] else 'No'} | Bonus: {gw4_data['bonus']}")

print(f"\n✅ Model MAE: {abs(gw4_data['total_points']-prediction):.1f} pts")
print(f"📊 SUMMARY: Salah(1.5) | Haaland(0.7) | Raya(?)")
