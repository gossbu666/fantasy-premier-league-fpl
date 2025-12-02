#!/usr/bin/env python3
# helper_api_features.py

import pandas as pd
import numpy as np
import requests

from utils_rolling import add_rolling_features_season_player

SEASON_LABEL = "2025-26"

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

def build_feature_series_from_api(player, feature_cols):
    """
    ทำ feature engineering จาก FPL API ให้ schema ตรงกับ *_features_enhanced_safe.csv
    โดย reuse logic จาก *_build_features_season_safe.py
    สำหรับผู้เล่นคนเดียว (MID/FWD/DEF/GK ใช้แพทเทิร์นเดียวกันได้)
    """
    player_id = player["id"]
    team_id = player["team"]
    element_type = player["element_type"]

    # 1) ดึง history ของ player (เหมือน MID_data.csv แต่เฉพาะคนนี้)
    summ = fetch_element_summary(player_id)
    hist = summ["history"]
    df = pd.DataFrame(hist)
    if df.empty:
        raise ValueError("No history for this player")

    df["season"] = SEASON_LABEL
    df["player_id"] = player_id
    df["team"] = team_id

    # now_cost / fdr / is_home / opponent_team จะใช้จากแต่ละแถวใน df อยู่แล้วตอนเลือก base_feature_cols
    # (ใน MID_data.csv เดิมนายควรจะมีสิ่งเหล่านี้อยู่แล้วเหมือนกัน)

    df["season"] = df["season"].astype(str)
    last_gw = int(df["round"].max())
    target_gw = last_gw + 1

    # 2) Rolling per (season, player) — logic เดิม
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

    # 3) Rolling per (season, team) — logic เดิม
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

    # 4) เลือกคอลัมน์แบบเดียวกับ build_features_season_safe.py
    base_feature_cols = [
        "season", "round", "player_id", "team", "opponent_team",
        "element_type", "now_cost", "minutes", "goals_scored",
        "assists", "bps", "influence", "creativity", "threat",
        "ict_index", "fdr_attack", "fdr_defense", "is_home",
    ]
    base_feature_cols = [c for c in base_feature_cols if c in df.columns]
    roll_feature_cols = [c for c in df.columns if "roll" in c]

    # ให้ order ของ feature_cols ตามที่โมเดลใช้
    used_feature_cols = feature_cols[:]  # จากไฟล์ {pos}_202425_features.txt

    # เลือก row ล่าสุด (round == last_gw) มาเป็น input
    row_last = df[df["round"] == last_gw].iloc[0]

    feat_series = pd.Series(dtype=float)
    for col in used_feature_cols:
        if col in row_last.index:
            feat_series[col] = row_last[col]
        else:
            # ถ้าใน safe features มี แต่จาก API ยังไม่มีคอลัมน์นี้ ให้เติม 0 ไว้ก่อน
            feat_series[col] = 0.0

    return feat_series, last_gw, target_gw
