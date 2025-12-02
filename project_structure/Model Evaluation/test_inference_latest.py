#!/usr/bin/env python3
"""
test_inference_latest.py

เอาโมเดลล่าสุด (GK/DEF/MID/FWD *_202425_safe_final.pkl)
มาทดสอบ inference บน FPL API สำหรับผู้เล่นคนเดียว
โดยใช้ฟีเจอร์ชุดเดียวกับตอน train:

- ใช้ history ทั้งฤดูกาลของ player จาก element-summary
- ทำ rolling per (season, player) ด้วย add_rolling_features_season_player
- ทำ rolling per (season, team) ด้วย groupby + shift + rolling
- base features: season, round, player_id, team, opponent_team, element_type,
  now_cost, minutes, goals_scored, assists, bps, influence, creativity,
  threat, ict_index, fdr_attack, fdr_defense, is_home

เป้าคือทดสอบ pipeline ให้ตรงกับ *_build_features_season_safe.py มากที่สุด
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib

from utils_rolling import add_rolling_features_season_player

SEASON_LABEL = "2025-26"
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def fetch_bootstrap():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_element_summary(player_id: int):
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def find_player(bootstrap, name: str):
    name = name.lower().strip()
    for p in bootstrap["elements"]:
        full = (p["first_name"] + " " + p["second_name"]).lower()
        if name in p["web_name"].lower() or name in full:
            return p
    return None


def load_position_models():
    models = {}
    positions = ["GK", "DEF", "MID", "FWD"]
    for pos in positions:
        mp = Path(f"models/{pos}_202425_safe_final.pkl")
        fp = Path(f"models/{pos}_202425_features.txt")
        if mp.exists() and fp.exists():
            with open(fp) as f:
                cols = [c for c in f.read().strip().split("\n") if c != "target"]
            models[pos] = {
                "model": joblib.load(mp),
                "features": cols,
            }
    return models


def build_feature_series_exact(player, feature_cols):
    """
    ใช้ logic เหมือน *_build_features_season_safe.py
    เพื่อสร้าง feature vector สำหรับ GW ล่าสุดของ player คนนี้
    """
    player_id = player["id"]
    team_id = player["team"]
    element_type = player["element_type"]

    summ = fetch_element_summary(player_id)
    hist = summ["history"]
    df = pd.DataFrame(hist)
    if df.empty:
        raise ValueError("No history for this player")

    # ตรงกับ step 1 ของ mid_build_features_season_safe.py
    df["season"] = SEASON_LABEL
    df["player_id"] = player_id
    df["team"] = team_id
    df["element_type"] = element_type

    # approx now_cost จาก bootstrap
    df["now_cost"] = player["now_cost"] / 10.0

    # จริง ๆ fdr_attack / fdr_defense ถูกสร้างไว้ตอน build MID_data.csv จาก team_strength.csv
    # ในที่นี้ให้ใช้ค่าจากคอลัมน์เดิมถ้ามี ไม่งั้นใส่ 50 เป็น default
    if "fdr_attack" not in df.columns:
        df["fdr_attack"] = 50.0
    if "fdr_defense" not in df.columns:
        df["fdr_defense"] = 50.0

    df["is_home"] = df["was_home"].astype(int)

    df["season"] = df["season"].astype(str)
    df["round"] = df["round"].astype(int)

    last_gw = int(df["round"].max())
    target_gw = last_gw + 1

    # ----- Rolling per (season, player) -----
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

    df = add_rolling_features_season_player(
        df,
        value_cols=player_roll_cols,
        windows=(1, 3, 5, 10),
        prefix="",
    )

    # ----- Rolling per (season, team) -----
    df = df.sort_values(["season", "team", "round"]).copy()
    team_group = df.groupby(["season", "team"], group_keys=False)

    team_roll_cols = []
    if "goals_scored" in df.columns:
        team_roll_cols.append("goals_scored")
    if "goals_conceded" in df.columns:
        team_roll_cols.append("goals_conceded")

    for col in team_roll_cols:
        for w in (3, 5, 10):
            new_col = f"{col}_roll_team{w}"
            df[new_col] = team_group[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )

    # ----- เลือก row ล่าสุด -----
    row_last = df[df["round"] == last_gw].iloc[0]
    last_points = float(row_last.get("total_points", 0.0))

    # map ให้ตรงกับ feature_cols ของโมเดล
    feat = pd.Series(dtype=float)
    for col in feature_cols:
        if col in row_last.index:
            feat[col] = row_last[col]
        else:
            feat[col] = 0.0

    return feat, last_gw, target_gw, last_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", type=str, required=True,
                        help="Player name, e.g. Haaland, Salah, Pickford")
    args = parser.parse_args()

    models = load_position_models()
    if not models:
        print("❌ No models loaded.")
        return

    bootstrap = fetch_bootstrap()
    player = find_player(bootstrap, args.name)
    if player is None:
        print(f"❌ Player '{args.name}' not found.")
        return

    pos = POS_MAP[player["element_type"]]
    if pos not in models:
        print(f"❌ No model for position {pos}.")
        return

    model = models[pos]["model"]
    feature_cols = models[pos]["features"]

    print(f"\nPlayer: {player['web_name']} | Position: {pos}")

    feat, last_gw, target_gw, last_points = build_feature_series_exact(
        player, feature_cols
    )
    X = feat[feature_cols].fillna(0).to_numpy().reshape(1, -1)
    pred = float(model.predict(X)[0])

    print(f"\nUsing history up to GW{last_gw} (last points = {last_points:.1f})")
    print(f"Predicting next GW (GW{target_gw}) with latest model...")
    print(f"\n🔮 Predicted points: {pred:.2f} pts")
    print(f"   Last GW (GW{last_gw}) actual: {last_points:.2f} pts")
    print(f"   Rounded: {int(round(pred))} pts")


if __name__ == "__main__":
    main()
