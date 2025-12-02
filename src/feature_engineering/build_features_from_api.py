#!/usr/bin/env python3
"""
Build 2025-26 features from live FPL API
ให้ schema แมตช์ *_features_enhanced_safe.csv (ยกเว้นค่า fdr/team-roll เป็นเวอร์ชัน 2025-26)
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path

from utils_rolling import add_rolling_features_season_player

SEASON_LABEL = "2025-26"

# ---------- FPL API ----------

def fetch_bootstrap():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def fetch_element_summary(player_id):
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

# ---------- raw player history (เหมือน *_data.csv ดิบ) ----------

def build_player_history_df(bootstrap):
    elements = bootstrap["elements"]
    records = []

    for el in elements:
        pid = el["id"]
        team_id = el["team"]
        el_type = el["element_type"]   # 1 GK, 2 DEF, 3 MID, 4 FWD

        summ = fetch_element_summary(pid)
        hist = summ["history"]   # matches ใน season ปัจจุบัน

        for h in hist:
            gw = h["round"]
            rec = {
                "season": SEASON_LABEL,
                "round": gw,
                "player_id": pid,
                "team": team_id,
                "opponent_team": h["opponent_team"],
                "element_type": el_type,
                "minutes": h["minutes"],
                "goals_scored": h["goals_scored"],
                "assists": h["assists"],
                "bps": h["bps"],
                "influence": h["influence"],
                "creativity": h["creativity"],
                "threat": h["threat"],
                "ict_index": h["ict_index"],
                "total_points": h["total_points"],
                "goals_conceded": h.get("goals_conceded", 0),
                "is_home": 1 if h["was_home"] else 0,
            }
            records.append(rec)

    df = pd.DataFrame(records)
    df["season"] = df["season"].astype(str)
    df["round"] = df["round"].astype(int)
    return df

# ---------- team rolling (goals_for / goals_against per team) ----------

def add_team_rolling(df):
    df = df.sort_values(["season", "team", "round"])
    grp = df.groupby(["season", "team"], group_keys=False)

    for col in ["goals_scored", "goals_conceded"]:
        for w in (3, 5, 10):
            new_col = f"{col}_roll_team{w}"
            df[new_col] = grp[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return df

# ---------- FDR-like features จาก team strength 2025-26 ----------

def add_fdr_like_features(df, bootstrap):
    teams = pd.DataFrame(bootstrap["teams"])
    ta = teams.set_index("id")["strength_attack"].to_dict()
    td = teams.set_index("id")["strength_defence"].to_dict()

    def norm(d):
        vals = np.array(list(d.values()), dtype=float)
        mn, mx = vals.min(), vals.max()
        return {k: 100.0 * (v - mn) / (mx - mn + 1e-9) for k, v in d.items()}

    ta_n = norm(ta)
    td_n = norm(td)

    df["fdr_attack"] = df["opponent_team"].map(ta_n).astype(float)
    df["fdr_defense"] = df["opponent_team"].map(td_n).astype(float)
    return df

# ---------- สร้าง features หลักทั้ง season ----------

def build_features_for_season_2025_26():
    bootstrap = fetch_bootstrap()
    raw = build_player_history_df(bootstrap)

    # 1) player rolling (เหมือน safe pipeline เดิม)
    roll_cols = [
        "total_points", "goals_scored", "assists",
        "minutes", "bps", "influence", "creativity", "threat",
    ]
    roll_cols = [c for c in roll_cols if c in raw.columns]

    feats = add_rolling_features_season_player(
        raw,
        value_cols=roll_cols,
        windows=(1, 3, 5, 10),
        prefix=""
    )

    # 2) team rolling
    feats = add_team_rolling(feats)

    # 3) FDR-like features
    feats = add_fdr_like_features(feats, bootstrap)

    return feats

# ---------- split เป็นไฟล์ per position ให้ schema ตรง safe ----------

def split_by_position_and_save(feats: pd.DataFrame, out_dir="data/processed"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    base_cols_order = [
        "season","round","player_id","team","opponent_team","element_type",
        "minutes","goals_scored","assists","bps","influence","creativity",
        "threat","ict_index","fdr_attack","fdr_defense","is_home",
        "total_points_roll1","total_points_roll3","total_points_roll5","total_points_roll10",
        "goals_scored_roll1","goals_scored_roll3","goals_scored_roll5","goals_scored_roll10",
        "assists_roll1","assists_roll3","assists_roll5","assists_roll10",
        "minutes_roll1","minutes_roll3","minutes_roll5","minutes_roll10",
        "bps_roll1","bps_roll3","bps_roll5","bps_roll10",
        "influence_roll1","influence_roll3","influence_roll5","influence_roll10",
        "creativity_roll1","creativity_roll3","creativity_roll5","creativity_roll10",
        "threat_roll1","threat_roll3","threat_roll5","threat_roll10",
        "goals_scored_roll_team3","goals_scored_roll_team5","goals_scored_roll_team10",
        "goals_conceded_roll_team3","goals_conceded_roll_team5","goals_conceded_roll_team10",
    ]

    for etype, pos in pos_map.items():
        df_pos = feats[feats["element_type"] == etype].copy()

        # เผื่อบางคอลัมน์ยังไม่ถูกสร้าง เติม 0 ให้ครบ schema
        for c in base_cols_order:
            if c not in df_pos.columns:
                df_pos[c] = 0.0

        df_pos["target"] = df_pos["total_points"]
        df_pos = df_pos[base_cols_order + ["target"]]

        out_path = Path(out_dir) / f"{pos}_features_2025_26_from_api_safe.csv"
        df_pos.to_csv(out_path, index=False)
        print(f"Saved {pos}: {len(df_pos)} rows → {out_path}")

# ---------- main ----------

def main():
    feats = build_features_for_season_2025_26()
    split_by_position_and_save(feats)

if __name__ == "__main__":
    main()
