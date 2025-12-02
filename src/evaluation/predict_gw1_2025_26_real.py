#!/usr/bin/env python3
"""
Predict GW1 2025-26 จริงจาก FPL API (ไม่ mock)
"""
import pandas as pd
import numpy as np
import requests
import joblib
from utils_rolling import add_rolling_features_season_player

# 1. Load model
model = joblib.load("models/MID_seasonstack_final.pkl")
with open("models/MID_seasonstack_final_features.txt") as f:
    feature_cols = f.read().strip().split("\n")
print(f"✅ MID model loaded ({len(feature_cols)} features)")

# 2. Fetch FPL GW1 2025-26 (Salah)
print("🌐 Fetching FPL GW1 2025-26...")
url = "https://fantasy.premierleague.com/api/bootstrap-static/"

try:
    r = requests.get(url, timeout=10)
    data = r.json()
    
    # หา Salah (player_id = 381)
    salah_id = 381
    salah_data = None
    
    for player in data["elements"]:
        if player["id"] == salah_id:
            salah_data = player
            break
    
    if not salah_data:
        print("❌ Salah not found in current data")
        exit()
    
    # GW1 history (จาก player history API)
    history_url = f"https://fantasy.premierleague.com/api/element-summary/{salah_id}/"
    h = requests.get(history_url)
    history = h.json()["history"]
    
    # หา GW1 2025-26
    gw1_data = None
    for game in history:
        if game["round"] == 1 and "2025" in game.get("kickoff_time", ""):
            gw1_data = game
            break
    
    if not gw1_data:
        print("❌ GW1 2025-26 not found. Using latest GW...")
        gw1_data = history[-1]  # latest GW
    
    print(f"📊 Salah GW{gw1_data['round']}: {gw1_data['total_points']} pts actual")
    
    # 3. สร้าง feature row
    feature_row = pd.Series({
        "season": "2025-26", "round": gw1_data["round"], "player_id": salah_id,
        "team": salah_data["team"], "element_type": 3,
        "now_cost": salah_data["now_cost"]/10,
        "total_points": gw1_data["total_points"],
        "minutes": gw1_data["minutes"],
        "goals_scored": gw1_data["goals_scored"],
        "assists": gw1_data["assists"],
        "bps": gw1_data["bps"],
        "influence": gw1_data["influence"],
        "creativity": gw1_data["creativity"],
        "threat": gw1_data["threat"],
        "ict_index": gw1_data["ict_index"],
        # เติม rolling = 0 (GW1 ไม่มีข้อมูลก่อนหน้า)
    })
    
    # เติม feature ที่ขาด
    for col in feature_cols:
        if col not in feature_row:
            feature_row[col] = 0
    
    # 4. Predict
    X = feature_row[feature_cols].fillna(0).values.reshape(1, -1)
    prediction = model.predict(X)[0]
    
    print(f"\n🎯 SALAH GW1 2025-26")
    print(f"   Actual:   {gw1_data['total_points']:.1f} pts")
    print(f"   Predicted: {prediction:.1f} pts")
    print(f"   Error:    {abs(gw1_data['total_points']-prediction):.1f} pts")
    
except Exception as e:
    print(f"API error: {e}")
    print("Using 2024-25 GW1 instead...")
    
    # Fallback: ใช้ 2024-25 GW1
    df = pd.read_csv("data/processed/MID_data.csv")
    salah_gw1 = df[(df["player_name"].str.contains("Salah")) & 
                   (df["season"] == "2024-25") & 
                   (df["round"] == 1)].iloc[0]
    
    feature_row = salah_gw1.copy()
    for col in feature_cols:
        if col not in feature_row:
            feature_row[col] = 0
    
    X = feature_row[feature_cols].fillna(0).values.reshape(1, -1)
    prediction = model.predict(X)[0]
    
    print(f"\n🎯 SALAH GW1 2024-25 (fallback)")
    print(f"   Actual:   {salah_gw1['total_points']:.1f} pts")
    print(f"   Predicted: {prediction:.1f} pts")
    print(f"   Error:    {abs(salah_gw1['total_points']-prediction):.1f} pts")

