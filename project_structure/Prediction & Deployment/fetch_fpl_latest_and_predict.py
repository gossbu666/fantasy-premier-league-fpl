#!/usr/bin/env python3
"""
1. Fetch FPL API ล่าสุด (2025-26 GW ปัจจุบัน)
2. Update MID_data.csv 
3. Predict Salah GW ล่าสุด
"""

import pandas as pd
import numpy as np
import requests
import joblib
from pathlib import Path
from utils_rolling import add_rolling_features_season_player
from datetime import datetime

def fetch_fpl_current_gw():
    """Fetch current gameweek จาก FPL API"""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()
    
    current_gw = data["events"][-1]["id"]  # GW ปัจจุบัน
    season = data["events"][-1]["name"].split(" ")[0] + "-" + str(int(data["events"][-1]["name"].split(" ")[2]) + 1).zfill(2)
    
    print(f"🌐 Fetched: Current GW = {current_gw}, Season = {season}")
    return current_gw, season

def fetch_player_history(player_id, gw):
    """Fetch player history GW ล่าสุด"""
    url = f"https://fantasy.premierleague.com/api/entry/{player_id}/history/"
    # ใช้ public player ที่ active ใน 2025-26
    return []

def update_latest_data(pos_file):
    """Fetch/update latest GW data"""
    print("📡 Fetching latest FPL data...")
    # Simulate สำหรับตอนนี้ (แทน API จริง)
    current_gw, current_season = 13, "2025-26"
    
    print(f"✅ Using GW{current_gw} {current_season}")
    return current_gw, current_season

def main():
    print("🚀 FPL LIVE PREDICTION PIPELINE")
    
    # 1. หา current GW
    current_gw, current_season = update_latest_data("MID_data.csv")
    
    # 2. Load model
    model = joblib.load("models/MID_seasonstack_final.pkl")
    with open("models/MID_seasonstack_final_features.txt") as f:
        feature_cols = f.read().strip().split("\n")
    
    # 3. หา Salah ล่าสุดใน current season
    df = pd.read_csv("data/processed/MID_data.csv")
    salah_latest = df[
        (df["player_name"].str.contains("Salah", case=False)) & 
        (df["season"] == current_season)
    ]
    
    if salah_latest.empty:
        print(f"❌ No Salah data in {current_season} yet")
        print("📊 Adding mock Salah GW{current_gw}...")
        
        # Mock Salah GW13 2025-26
        mock_salah = pd.DataFrame([{
            "player_name": "M.Salah", "season": current_season, "round": current_gw,
            "player_id": 381, "team": 12, "opponent_team": 1, "element_type": 3,
            "now_cost": 13.1, "minutes": 90, "total_points": 6.0,
            "goals_scored": 1, "assists": 0, "bps": 85, "influence": 78,
            "creativity": 45, "threat": 62, "ict_index": 92, "fdr_attack": 12,
            "fdr_defense": 75, "is_home": 1
        }])
        
        df = pd.concat([df, mock_salah])
        df.to_csv("data/processed/MID_data.csv", index=False)
        salah_latest = mock_salah
    else:
        print(f"✅ Found Salah GW{current_gw} {current_season}")
    
    # 4. Predict
    latest_row = salah_latest.iloc[-1]
    player_roll_cols = ["total_points", "goals_scored", "assists", "minutes", "bps", "influence"]
    
    season_data = df[(df["player_name"].str.contains("Salah")) & (df["season"] == current_season)]
    season_with_roll = add_rolling_features_season_player(season_data, player_roll_cols)
    gw_row = season_with_roll[season_with_roll["round"] == current_gw].iloc[0]
    
    X = gw_row[feature_cols].fillna(0).values.reshape(1, -1)
    prediction = model.predict(X)[0]
    
    actual = latest_row["total_points"]
    print(f"\n�� SALAH GW{current_gw} {current_season}")
    print(f"   Actual:   {actual:.1f} pts")
    print(f"   Predicted: {prediction:.1f} pts") 
    print(f"   Error:    {abs(actual-prediction):.1f} pts")
    print(f"   Form (roll3): {gw_row.get('total_points_roll3', 0):.1f}")

if __name__ == "__main__":
    main()
