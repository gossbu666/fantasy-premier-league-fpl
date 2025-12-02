#!/usr/bin/env python3
"""
ดึงข้อมูลผู้เล่นจาก FPL API + ใช้โมเดล safe-3seasons (train 2021-24, test 2024-25)
ทำนายคะแนนคาดหวังแบบง่าย ๆ แล้วเซฟเป็น players_predictions_live.csv

NOTE: เวอร์ชันนี้ใช้สถิติ season-to-date (รวม) เป็นฟีเจอร์พื้นฐาน
ไม่ได้ทำ rolling หลาย GW เพื่อให้ pipeline ง่ายสำหรับ production แรก
"""

import requests
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

API_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"

POSITIONS = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

MODEL_INFO = {
    "GK": (
        "models/GK_safe_3seasons_train_202425_test.pkl",
        "models/GK_safe_3seasons_features.txt",
    ),
    "DEF": (
        "models/DEF_safe_3seasons_train_202425_test.pkl",
        "models/DEF_safe_3seasons_features.txt",
    ),
    "MID": (
        "models/MID_safe_3seasons_train_202425_test.pkl",
        "models/MID_safe_3seasons_features.txt",
    ),
    "FWD": (
        "models/FWD_safe_3seasons_train_202425_test.pkl",
        "models/FWD_safe_3seasons_features.txt",
    ),
}


def fetch_bootstrap():
    print("Fetching bootstrap-static from FPL API...")
    r = requests.get(API_BOOTSTRAP)
    r.raise_for_status()
    data = r.json()
    return data


def build_players_df(bootstrap_json: dict) -> pd.DataFrame:
    elements = pd.DataFrame(bootstrap_json["elements"])
    teams = pd.DataFrame(bootstrap_json["teams"])[["id", "name", "short_name"]]
    teams = teams.rename(columns={"id": "team_id"})

    # map team id -> short name
    elements = elements.merge(
        teams, left_on="team", right_on="team_id", how="left"
    )

    # basic columns we care about
    cols = [
        "id",           # player_id
        "web_name",     # short display name
        "first_name",
        "second_name",
        "team",         # team id
        "short_name",   # team short
        "element_type", # position code 1-4
        "now_cost",     # in tenths of million (e.g. 55 => 5.5m)
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "form",
        "points_per_game",
        "total_points",
    ]
    cols = [c for c in cols if c in elements.columns]
    df = elements[cols].copy()
    df = df.rename(
        columns={
            "id": "player_id",
            "short_name": "team_name",
        }
    )
    return df


def load_model_and_features(pos: str):
    model_path, feat_path = MODEL_INFO[pos]
    model = joblib.load(model_path)
    with open(feat_path) as f:
        feat_cols = [c for c in f.read().strip().split("\n") if c != "target"]
    return model, feat_cols


def map_features_for_position(df_pos: pd.DataFrame, pos: str, feat_cols):
    """
    สร้าง DataFrame X ที่มี column ตรงกับ feat_cols ของโมเดล safe-3seasons
    โดยใช้ข้อมูลจาก bootstrap (season-to-date) แม็ปเท่าที่มี, ที่เหลือใส่ 0
    """

    # base feature mapping ที่คาดว่าน่าจะตรง/คล้ายกับ safe features เดิม
    base_map = {
        "season": None,           # ใส่ dummy season string
        "round": None,            # ใส่ dummy GW, เราทำนาย "GW ถัดไป"
        "player_id": "player_id",
        "team": "team",           # team id
        "opponent_team": None,    # ไม่มีใน bootstrap -> 0
        "element_type": "element_type",
        "now_cost": "now_cost",
        "minutes": "minutes",
        "goals_scored": "goals_scored",
        "assists": "assists",
        "bps": "bps",
        "influence": "influence",
        "saves": "saves",
        "goals_conceded": "goals_conceded",
        "ict_index": "ict_index",
        "fdr_attack": None,
        "fdr_defense": None,
        "is_home": None,
        # season aggregate stats
        "total_points": "total_points",
        "points_per_game": "points_per_game",
        "form": "form",
    }

    # เริ่มจาก DataFrame ว่างที่มีทุก feat_cols
    X = pd.DataFrame(index=df_pos.index, columns=feat_cols, dtype=float)

    # ใส่ค่า 0 default
    X[:] = 0.0

    # เติมจาก mapping เท่าที่มี
    for feat in feat_cols:
        if feat in base_map and base_map[feat] is not None:
            src = base_map[feat]
            if src in df_pos.columns:
                X[feat] = df_pos[src].astype(float)

    # season/round ใส่ dummy ไว้ เฉย ๆ (โมเดลจะไม่สนถ้าไม่ได้ใช้เป็น feature จริง)
    if "season" in X.columns:
        X["season"] = 2025.0  # dummy numeric / ถ้าเป็น object โมเดลจะไม่ใช้
    if "round" in X.columns:
        X["round"] = 1.0

    # ที่เหลือ (เช่น *_roll*) ปล่อย 0 ไว้
    return X.fillna(0.0)


def main():
    bootstrap = fetch_bootstrap()
    df_all = build_players_df(bootstrap)
    print(f"Loaded {len(df_all)} players from bootstrap-static.")

    records = []

    for pos_code, pos_name in POSITIONS.items():
        if pos_name not in MODEL_INFO:
            continue

        model, feat_cols = load_model_and_features(pos_name)

        df_pos = df_all[df_all["element_type"] == pos_code].copy()
        if df_pos.empty:
            continue

        X = map_features_for_position(df_pos, pos_name, feat_cols)
        raw_preds = model.predict(X.to_numpy())

# heuristic scaling ให้สเกลเฉลี่ยใกล้ ๆ 4–5 แต้มต่อคน
        SCALE = 0.25
        preds = raw_preds * SCALE

        df_pos["position"] = pos_name
        df_pos["predicted_points"] = preds
        df_pos["raw_predicted_points"] = raw_preds  # เผื่ออยาก debug ทีหลัง


        records.append(
            df_pos[
                [
                    "player_id",
                    "web_name",
                    "first_name",
                    "second_name",
                    "team",
                    "team_name",
                    "position",
                    "now_cost",
                    "minutes",
                    "goals_scored",
                    "assists",
                    "clean_sheets",
                    "goals_conceded",
                    "saves",
                    "bps",
                    "influence",
                    "creativity",
                    "threat",
                    "ict_index",
                    "form",
                    "points_per_game",
                    "total_points",
                    "predicted_points",
                ]
            ]
        )

        print(
            f"{pos_name}: {len(df_pos)} players, "
            f"predicted_points mean={preds.mean():.2f}"
        )

    df_out = pd.concat(records, ignore_index=True)

    out_path = Path("data/processed/players_predictions_live.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Saved live predictions to {out_path} ({len(df_out)} rows).")


if __name__ == "__main__":
    main()
