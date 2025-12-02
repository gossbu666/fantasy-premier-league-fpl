#!/usr/bin/env python3
"""
Predict Palmer GW1 2025-26
"""
import pandas as pd
import numpy as np
import requests
import joblib
from pathlib import Path

# Load model
model = joblib.load("models/MID_seasonstack_final.pkl")
with open("models/MID_seasonstack_final_features.txt") as f:
    feature_cols = f.read().strip().split("\n")

print(f"✅ MID model loaded ({len(feature_cols)} features)")

# Fetch FPL API
print("🌐 Fetching Palmer GW1 2025-26...")
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
r = requests.get(url, timeout=10)
data = r.json()

# หา Palmer (Chelsea MID, ค้นชื่อ)
palmer_id = None
for player in data["elements"]:
    if "Palmer" in player["web_name"] or "Palmer" in player["first_name"] + player["second_name"]:
        palmer_id = player["id"]
        print(f"✅ Found Palmer: {player['web_name']} (ID: {palmer_id})")
        break

if not palmer_id:
    print("❌ Palmer not found, listing top Chelsea MIDs...")
    chelsea_players = [p for p in data["elements"] if p["team"] == 5]  # Chelsea team_id
    mids = [p for p in chelsea_players if p["element_type"] == 3][:5]
    for p in mids:
        print(f"   {p['web_name']} (ID: {p['id']})")
    exit()

# Fetch player history
history_url = f"https://fantasy.premierleague.com/api/element-summary/{palmer_id}/"
h = requests.get(history_url)
history = h.json()["history"]

# GW1 2025-26
gw1_data = None
for game in history:
    if game["round"] == 1 and "2025" in str(game.get("kickoff_time", "")):
        gw1_data = game
        break

if not gw1_data:
    print("❌ GW1 2025-26 not found. Using latest GW...")
    gw1_data = history[-1]

# สร้าง feature row
feature_row = pd.Series({
    "season": "2025-26", "round": gw1_data["round"], "player_id": palmer_id,
    "team": data["elements"][palmer_id-1]["team"], "element_type": 3,
    "now_cost": data["elements"][palmer_id-1]["now_cost"]/10,
    "total_points": gw1_data["total_points"],
    "minutes": gw1_data["minutes"],
    "goals_scored": gw1_data["goals_scored"],
    "assists": gw1_data["assists"],
    "bps": gw1_data["bps"],
    "influence": gw1_data["influence"],
    "creativity": gw1_data["creativity"],
    "threat": gw1_data["threat"],
    "ict_index": gw1_data["ict_index"],
})

# เติม features ที่ขาด
for col in feature_cols:
    if col not in feature_row:
        feature_row[col] = 0

# Predict!
X = feature_row[feature_cols].fillna(0).values.reshape(1, -1)
prediction = model.predict(X)[0]

print(f"\n🎯 PALMER GW1 2025-26")
print(f"   Actual:   {gw1_data['total_points']:.1f} pts")
print(f"   Predicted: {prediction:.1f} pts")
print(f"   Error:    {abs(gw1_data['total_points']-prediction):.1f} pts")

print(f"\n✅ Model accuracy: Salah 1.5pts error → Palmer ?")
