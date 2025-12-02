#!/usr/bin/env python3
"""
Predict Sanchez GW5 2025-26 (GK)
"""
import pandas as pd
import numpy as np
import requests
import joblib
from pathlib import Path

print("🔮 Predicting Sanchez GW5 2025-26...")

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

# หา Sanchez (GK, ค้นชื่อ)
sanchez_id = None
for player in data["elements"]:
    if "Sanchez" in player["web_name"] or "Sanchez" in player["second_name"]:
        if player["element_type"] == 1:  # GK only
            sanchez_id = player["id"]
            print(f"✅ Found Sanchez: {player['web_name']} (ID: {sanchez_id})")
            break

if not sanchez_id:
    print("❌ Sanchez not found. Listing top GKs...")
    gks = [p for p in data["elements"] if p["element_type"] == 1][:5]
    for p in gks:
        print(f"   {p['web_name']} (ID: {p['id']}, Team: {data['teams'][p['team']-1]['name']})")
    exit()

# 3. Fetch player history
history_url = f"https://fantasy.premierleague.com/api/element-summary/{sanchez_id}/"
h = requests.get(history_url, timeout=10)
history = h.json()["history"]

# หา GW5 2025-26
gw5_data = None
for game in history:
    if game["round"] == 5:
        gw5_data = game
        break

if not gw5_data:
    print("❌ GW5 2025-26 not found. Using latest GW...")
    gw5_data = history[-1]

print(f"📊 Sanchez GW{gw5_data['round']}: {gw5_data['total_points']} pts (actual)")

# 4. สร้าง feature row
feature_row = pd.Series({
    "season": "2025-26", "round": gw5_data["round"], "player_id": sanchez_id,
    "team": data["elements"][sanchez_id-1]["team"], "element_type": 1,
    "now_cost": data["elements"][sanchez_id-1]["now_cost"]/10,
    "minutes": gw5_data["minutes"],
    "total_points": gw5_data["total_points"],
    "goals_conceded": gw5_data["goals_conceded"],
    "saves": gw5_data["saves"],
    "bonus": gw5_data["bonus"],
    "bps": gw5_data["bps"],
    "influence": gw10_data["influence"],
    "ict_index": gw5_data["ict_index"],
})

# เติม features ที่ขาด
for col in feature_cols:
    if col not in feature_row:
        feature_row[col] = 0

# 5. Predict!
X = feature_row[feature_cols].fillna(0).values.reshape(1, -1)
prediction = model.predict(X)[0]

print(f"\n🎯 SANCHEZ GW{gw5_data['round']} 2025-26")
print(f"   Team: {data['teams'][data['elements'][sanchez_id-1]['team']-1]['name']}")
print(f"   Actual:   {gw5_data['total_points']:.1f} pts")
print(f"   Predicted: {prediction:.1f} pts")
print(f"   Error:    {abs(gw5_data['total_points']-prediction):.1f} pts")
print(f"   Saves: {gw5_data['saves']} | Goals conceded: {gw5_data['goals_conceded']}")
print(f"   Clean sheet: {'Yes' if gw5_data['clean_sheets'] else 'No'} | Bonus: {gw5_data['bonus']}")

print(f"\n✅ Model MAE: {abs(gw5_data['total_points']-prediction):.1f} pts")
print(f"📊 SUMMARY: Salah(1.5) | Haaland(0.7) | Sanchez(?)")
