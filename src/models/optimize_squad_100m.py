#!/usr/bin/env python3
"""
อ่าน players_predictions_live.csv แล้วหา team 15 ตัวภายใต้งบ 100m
- 2 GK, 5 DEF, 5 MID, 3 FWD
- ไม่เกิน 3 คนต่อทีม
- maximize sum(predicted_points)
"""

import pandas as pd
from pathlib import Path
import pulp


def load_players(path="data/processed/players_predictions_live.csv"):
    df = pd.read_csv(path)
    # now_cost ใน FPL เป็น x10 (เช่น 55 = 5.5m) แปลงกลับเป็นล้าน
    df["cost_m"] = df["now_cost"] / 10.0
    return df


def optimize_team(df, budget=100.0):
    # index ทุกแถว
    idx = list(df.index)

    # binary decision vars
    x = pulp.LpVariable.dicts("pick", idx, lowBound=0, upBound=1, cat="Binary")

    model = pulp.LpProblem("FPL_Optimize_100m", pulp.LpMaximize)

    # objective
    model += pulp.lpSum(x[i] * df.loc[i, "predicted_points"] for i in idx)

    # budget
    model += pulp.lpSum(x[i] * df.loc[i, "cost_m"] for i in idx) <= budget

    # position constraints
    for pos, count in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        model += (
            pulp.lpSum(x[i] for i in idx if df.loc[i, "position"] == pos) == count
        )

    # max 3 per team
    for team_id in df["team"].unique():
        model += (
            pulp.lpSum(x[i] for i in idx if df.loc[i, "team"] == team_id) <= 3
        )

    # solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    picked = [i for i in idx if x[i].value() == 1]

    team_df = df.loc[picked].copy()
    team_df["cost_m"] = team_df["cost_m"].round(1)
    total_cost = team_df["cost_m"].sum()
    total_points = team_df["predicted_points"].sum()

    return team_df, total_cost, total_points


def main():
    path = Path("data/processed/players_predictions_live.csv")
    if not path.exists():
        print(f"File not found: {path}. Run build_players_predictions_live.py first.")
        return

    df = load_players(path)
    team_df, total_cost, total_points = optimize_team(df, budget=100.0)

    print("Optimized squad (15 players):")
    print(
        team_df[
            [
                "position",
                "web_name",
                "team_name",
                "cost_m",
                "predicted_points",
            ]
        ]
        .sort_values(["position", "predicted_points"], ascending=[True, False])
        .to_string(index=False)
    )
    print(f"\nTotal cost: {total_cost:.1f} m")
    print(f"Total predicted points: {total_points:.2f}")


if __name__ == "__main__":
    main()
