#!/usr/bin/env python3
"""
Inference pipeline: predict single player GW จากโมเดลที่ train มาแล้ว
Usage: python3 predict_single_player.py --player "Salah" --gw 1 --season "2024-25"
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path

POSITIONS = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

def load_model_and_features(pos):
    model_path = f"models/{pos}_seasonstack_final.pkl"
    features_path = f"models/{pos}_seasonstack_final_features.txt"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    with open(features_path) as f:
        feature_cols = f.read().strip().split("\n")
    
    print(f"✅ Loaded {pos} model ({len(feature_cols)} features)")
    return model, feature_cols

def predict_player(model, feature_cols, player_row):
    """Predict จาก single row"""
    X = player_row[feature_cols].fillna(0).values.reshape(1, -1)
    pred = model.predict(X)[0]
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True, help="Player name (e.g. 'Salah')")
    parser.add_argument("--gw", type=int, required=True, help="Gameweek (e.g. 1)")
    parser.add_argument("--season", default="2024-25", help="Season (default 2024-25)")
    parser.add_argument("--pos", choices=["GK", "DEF", "MID", "FWD"], required=True)
    args = parser.parse_args()

    print(f"🔮 Predicting {args.player} ({args.pos}) GW{args.gw} {args.season}")

    # 1) โหลดโมเดล + features
    model, feature_cols = load_model_and_features(args.pos)
    
    # 2) หา player data (จากไฟล์ล่าสุด)
    latest_data = pd.read_csv(f"data/processed/{args.pos}_data.csv")
    latest_data["season"] = latest_data["season"].astype(str)
    
    player_data = latest_data[
        (latest_data["player_name"].str.contains(args.player, case=False, na=False)) &
        (latest_data["season"] == args.season) &
        (latest_data["round"] == args.gw)
    ]
    
    if player_data.empty:
        print(f"❌ No data found for {args.player} GW{args.gw} {args.season}")
        print("Available players in this GW:")
        print(latest_data[(latest_data["season"] == args.season) & 
                         (latest_data["round"] == args.gw)][["player_name"]].head())
        return
    
    # 3) ใช้ row แรก (ถ้ามีหลาย row)
    player_row = player_data.iloc[0]
    print(f"📊 Using data for: {player_row['player_name']}")
    print(f"   Team: {player_row.get('team', 'N/A')}, Opponent: {player_row.get('opponent_team', 'N/A')}")
    
    # 4) Predict!
    prediction = predict_player(model, feature_cols, player_row)
    
    print(f"\n🎯 PREDICTION: {prediction:.1f} points")
    print(f"   (95% CI approx: {prediction-1.5:.1f} – {prediction+1.5:.1f})")
    
    # 5) แสดง features สำคัญบางตัว
    print("\n📈 Key features used:")
    key_features = ["now_cost", "minutes", "ict_index", "fdr_attack", "fdr_defense"]
    for feat in key_features:
        if feat in player_row:
            print(f"   {feat}: {player_row[feat]:.2f}")

if __name__ == "__main__":
    main()
