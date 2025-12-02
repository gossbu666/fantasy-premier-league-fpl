#!/usr/bin/env python3
"""
Predict Haaland GW10 2025-26 จาก FPL API จริง
"""
import pandas as pd
import numpy as np
import requests
import joblib
from pathlib import Path

print("🔮 Predicting Haaland GW10 2025-26...")

# 1. Load FWD model (2024-25 final)
model = joblib.load("models/FWD_202425_safe_final.pkl")
with open("models/FWD_202425_features.txt") as f:
    feature_cols = f.read().strip().split("\n")

print(f"✅ FWD model loaded ({len(feature_cols)} features)")

# 2. Fetch FPL API
print("🌐 Fetching FPL data...")
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
r = requests.get(url, timeout=10)
data = r.json()

# หา Haaland (Manchester City FWD)
haaland_id = None
for player in data["elements"]:
    if "Haaland" in player["web_name"] or "Haaland" in player["second_name"]:
        haaland_id = player["id"]
        print(f"✅ Found Haaland: {player['web_name']} (ID: {haaland_id})")
        break

if not haaland_id:
    print("❌ Haaland not found!")
    exit()

# 3. Fetch player history
history_url = f"https://fantasy.premierleague.com/api/element-summary/{haaland_id}/"
h = requests.get(history_url, timeout=10)
history = h.json()["history"]

# หา GW10 2025-26
gw10_data = None
for game in history:
    if game["round"] == 10:
        gw10_data = game
        break

if not gw10_data:
    print("❌ GW10 2025-26 not found. Using latest GW...")
    gw10_data = history[-1]

print(f"📊 Haaland GW{gw10_data['round']}: {gw10_data['total_points']} pts (actual)")

# 4. สร้าง feature row
feature_row = pd.Series({
    "season": "2025-26", "round": gw10_data["round"], "player_id": haaland_id,
    "team": data["elements"][haaland_id-1]["team"], "element_type": 4,
    "now_cost": data["elements"][haaland_id-1]["now_cost"]/10,
    "minutes": gw10_data["minutes"],
    "total_points": gw10_data["total_points"],
    "goals_scored": gw10_data["goals_scored"],
    "assists": gw10_data["assists"],
    "bps": gw10_data["bps"],
    "influence": gw10_data["influence"],
    "creativity": gw10_data["creativity"],
    "threat": gw10_data["threat"],
    "ict_index": gw10_data["ict_index"],
})

# เติม features ที่ขาด
for col in feature_cols:
    if col not in feature_row:
        feature_row[col] = 0

# 5. Predict!
X = feature_row[feature_cols].fillna(0).values.reshape(1, -1)
prediction = model.predict(X)[0]

print(f"\n🎯 HAALAND GW{gw10_data['round']} 2025-26")
print(f"   Team: {data['teams'][data['elements'][haaland_id-1]['team']-1]['name']}")
print(f"   Actual:   {gw10_data['total_points']:.1f} pts")
print(f"   Predicted: {prediction:.1f} pts")
print(f"   Error:    {abs(gw10_data['total_points']-prediction):.1f} pts")
print(f"   Goals: {gw10_data['goals_scored']} | Assists: {gw10_data['assists']}")
print(f"   Minutes: {gw10_data['minutes']} | Bonus: {gw10_data['bonus']}")

print(f"\n✅ Model MAE: {abs(gw10_data['total_points']-prediction):.1f} pts")
