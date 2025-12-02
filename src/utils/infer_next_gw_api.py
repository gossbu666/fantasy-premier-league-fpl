#!/usr/bin/env python3
"""
infer_next_gw_api.py

Test inference pipeline:
- โหลดโมเดลสุดท้าย (2021-24 + 2024-25 GW1-11 train, GW12+ valid)
- ดึง FPL API ของผู้เล่นที่ระบุ (2025-26)
- ทำ feature engineering ให้ schema ใกล้กับ *_features_enhanced_safe.csv:
    * rolling per (season, player) ด้วย add_rolling_features_season_player
    * rolling per (season, team) แบบ groupby+shift+rolling
    * base features: season, round, player_id, team, opponent_team, element_type,
      now_cost, minutes, goals_scored, assists, bps, influence, creativity,
      threat, ict_index, fdr_attack, fdr_defense, is_home
- ทำนาย next GW (last_gw + 1) แล้วพิมพ์ผลใน console
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


# ---------- FPL API helpers ----------

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


# ---------- Feature engineering (match *_build_features_season_safe.py) ----------

def build_features_for_player_from_api(player, feature_cols):
    """
    สร้าง feature vector สำหรับ player จาก FPL API
    โดยใช้ logic เดียวกับ *_build_features_season_safe.py เท่าที่ข้อมูล API อนุญาต
    """
    player_id = player["id"]
    team_id = player["team"]
    element_type = player["element_type"]

    # 1) history ของ player ~ MID_data.csv/FWD_data.csv แต่อยู่ใน API
    summ = fetch_element_summary(player_id)
    hist = summ["history"]
    df = pd.DataFrame(hist)
    if df.empty:
        raise ValueError("No history for this player")

    # เติมคอลัมน์ที่ pipeline เดิมคาดหวัง
    df["season"] = SEASON_LABEL
    df["player_id"] = player_id
    df["team"] = team_id
    df["element_type"] = element_type
    # approx now_cost / fdr / is_home จากข้อมูลที่มี
    df["now_cost"] = player["now_cost"] / 10.0

    # fdr_attack / fdr_defense: ยังไม่มี mapping ของโปรเจกต์ปีที่แล้ว → ให้ค่า default 50 ไว้ก่อน
    df["fdr_attack"] = 50.0
    df["fdr_defense"] = 50.0

    # is_home, opponent_team มีใน history อยู่แล้ว
    df["is_home"] = df["was_home"].astype(int)

    df["season"] = df["season"].astype(str)
    df["round"] = df["round"].astype(int)

    last_gw = int(df["round"].max())
    target_gw = last_gw + 1

    # 2) rolling per (season, player) — copy logic จาก *_build_features_season_safe.py
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
        prefix=""
    )

    # 3) rolling per (season, team)
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

    # 4) เลือก row ล่าสุดมาเป็น input
    row_last = df[df["round"] == last_gw].iloc[0]

    # map ตาม feature_cols ของโมเดล (จาก *_202425_features.txt)
    feat = pd.Series(dtype=float)
    for col in feature_cols:
        if col in row_last.index:
            feat[col] = row_last[col]
        else:
            feat[col] = 0.0

    # total_points ของ last_gw ไว้ใช้เปรียบเทียบ
    last_points = float(row_last.get("total_points", 0.0))

    return feat, last_gw, target_gw, last_points


# ---------- main inference ----------

def load_position_models():
    models = {}
    positions = ["GK", "DEF", "MID", "FWD"]
    for pos in positions:
        model_path = Path(f"models/{pos}_202425_safe_final.pkl")
        feat_path = Path(f"models/{pos}_202425_features.txt")
        if model_path.exists() and feat_path.exists():
            models[pos] = {}
            models[pos]["model"] = joblib.load(model_path)
            with open(feat_path) as f:
                cols = [c for c in f.read().strip().split("\n") if c != "target"]
            models[pos]["features"] = cols
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", type=str, required=True,
                        help="Player name, e.g. Haaland, Salah, Pickford")
    args = parser.parse_args()

    models = load_position_models()
    if not models:
        print("❌ No models loaded. Check models/*.pkl and features txt files.")
        return

    bootstrap = fetch_bootstrap()
    player = find_player(bootstrap, args.name)
    if player is None:
        print(f"❌ Player '{args.name}' not found in bootstrap.")
        return

    pos = POS_MAP[player["element_type"]]
    if pos not in models:
        print(f"❌ No model for position {pos}.")
        return

    model = models[pos]["model"]
    feature_cols = models[pos]["features"]

    print(f"\nPlayer: {player['web_name']} | Team id: {player['team']} | Position: {pos}")

    feat, last_gw, target_gw, last_points = build_features_for_player_from_api(
        player, feature_cols
    )
    X = feat[feature_cols].fillna(0).to_numpy().reshape(1, -1)
    pred = float(model.predict(X)[0])

    print(f"\nUsing history up to GW{last_gw} (last points = {last_points:.1f})")
    print(f"Predicting next GW (GW{target_gw})...")
    print(f"\n🔮 Predicted points: {pred:.2f} pts")
    print(f"   Last GW (GW{last_gw}) actual: {last_points:.2f} pts")
    print(f"   Rounded: {int(round(pred))} pts")


if __name__ == "__main__":
    main()
