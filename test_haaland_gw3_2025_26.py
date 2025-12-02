#!/usr/bin/env python3
"""
Test: Haaland GW3 2025-26
ใช้ฟีเจอร์จาก GW2 → ทำนายคะแนน GW3
"""

import pandas as pd
import numpy as np
import requests
import joblib

from utils_rolling import add_rolling_features_season_player

SEASON_LABEL = "2025-26"
HAALAND_NAME = "Haaland"  # search ด้วยชื่อ

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

def find_haaland(data):
    for p in data["elements"]:
        full = (p["first_name"] + " " + p["second_name"]).lower()
        if HAALAND_NAME.lower() in p["web_name"].lower() or HAALAND_NAME.lower() in full:
            return p
    return None

def main():
    # 1) โหลดโมเดล FWD ล่าสุด
    model = joblib.load("models/FWD_202425_safe_final.pkl")
    with open("models/FWD_202425_features.txt") as f:
        feature_cols = [c for c in f.read().strip().split("\n") if c != "target"]
    print(f"✅ FWD model loaded ({len(feature_cols)} features)")

    # 2) ดึงข้อมูล Haaland จาก FPL API
    data = fetch_bootstrap()
    player = find_haaland(data)
    if player is None:
        print("❌ Haaland not found in bootstrap")
        return
    pid = player["id"]
    team_id = player["team"]
    print(f"✅ Found Haaland: {player['web_name']} (ID {pid}, team {team_id})")

    summ = fetch_element_summary(pid)
    hist = summ["history"]
    df = pd.DataFrame(hist)
    if df.empty:
        print("❌ No history for Haaland")
        return

    # 3) เตรียม DataFrame ให้เหมือน MID/FWD_data.csv
    df["season"] = SEASON_LABEL
    df["player_id"] = pid
    df["team"] = team_id
    df["element_type"] = player["element_type"]
    df["now_cost"] = player["now_cost"] / 10.0
    df["is_home"] = df["was_home"].astype(int)

    df["season"] = df["season"].astype(str)
    df["round"] = df["round"].astype(int)

    # ต้องมีข้อมูลถึงอย่างน้อย GW3
    if df["round"].max() < 3:
        print("❌ Haaland does not have data up to GW3 yet.")
        return

    # 4) rolling per (season, player) เหมือน build_features_season_safe
    player_roll_cols = [
        "total_points", "goals_scored", "assists",
        "minutes", "bps", "influence", "creativity", "threat",
    ]
    player_roll_cols = [c for c in player_roll_cols if c in df.columns]

    df_roll = add_rolling_features_season_player(
        df,
        value_cols=player_roll_cols,
        windows=(1, 3, 5, 10),
        prefix=""
    )

    # 5) rolling per (season, team) แบบง่าย ๆ จากประวัติ Haaland (proxy)
    df_roll = df_roll.sort_values(["season", "team", "round"]).copy()
    team_group = df_roll.groupby(["season", "team"], group_keys=False)

    team_roll_cols = []
    if "goals_scored" in df_roll.columns:
        team_roll_cols.append("goals_scored")
    if "goals_conceded" in df_roll.columns:
        team_roll_cols.append("goals_conceded")

    for col in team_roll_cols:
        for w in (3, 5, 10):
            new_col = f"{col}_roll_team{w}"
            df_roll[new_col] = team_group[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )

    # 6) ใช้แถว GW2 เป็น input → ทำนาย GW3
    gw2_row = df_roll[df_roll["round"] == 2]
    gw3_row = df_roll[df_roll["round"] == 3]

    if gw2_row.empty or gw3_row.empty:
        print("❌ Missing GW2 or GW3 in history.")
        return

    gw2 = gw2_row.iloc[0]
    gw3 = gw3_row.iloc[0]

    feature_row = pd.Series(dtype=float)
    for col in feature_cols:
        if col in gw2.index:
            feature_row[col] = gw2[col]
        else:
            feature_row[col] = 0.0

    X = feature_row[feature_cols].fillna(0).to_numpy().reshape(1, -1)
    pred = float(model.predict(X)[0])

    print("\n🎯 HAALAND BACKTEST – GW2 → predict GW3 (2025-26)")
    print(f"Input GW: 2")
    print(f"Predicted GW3: {pred:.2f} pts")
    print(f"Actual   GW3: {float(gw3['total_points']):.2f} pts")
    print(f"Error:        {abs(pred - float(gw3['total_points'])):.2f} pts")

    # optional: print GW1–3 form
    print("\nRecent form (GW1–3):")
    print(df_roll[df_roll["round"].isin([1,2,3])][
        ["round","total_points","goals_scored","assists","minutes"]
    ])


if __name__ == "__main__":
    main()
