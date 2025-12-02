#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from pathlib import Path
import subprocess
import sys
import pulp

# -------- CONFIG --------
PRED_PATH = Path("data/processed/players_predictions_live.csv")
BUDGET = 100.0
MIN_SPEND = 99.0   # force squad cost >= 99m to use budget efficiently


def load_players(path=PRED_PATH):
    df = pd.read_csv(path)
    df["cost_m"] = df["now_cost"] / 10.0
    return df


def optimize_team(df, budget=BUDGET, min_spend=MIN_SPEND):
    idx = list(df.index)
    x = pulp.LpVariable.dicts("pick", idx, lowBound=0, upBound=1, cat="Binary")

    model = pulp.LpProblem("FPL_Optimize_100m", pulp.LpMaximize)

    # objective: maximize predicted points
    model += pulp.lpSum(x[i] * df.loc[i, "predicted_points"] for i in idx)

    # budget constraints
    model += pulp.lpSum(x[i] * df.loc[i, "cost_m"] for i in idx) <= budget
    model += pulp.lpSum(x[i] * df.loc[i, "cost_m"] for i in idx) >= min_spend

    # squad size by position
    for pos, count in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        model += (
            pulp.lpSum(x[i] for i in idx if df.loc[i, "position"] == pos) == count
        )

    # max 3 per real team
    for team_id in df["team"].unique():
        model += (
            pulp.lpSum(x[i] for i in idx if df.loc[i, "team"] == team_id) <= 3
        )

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    picked = [i for i in idx if x[i].value() == 1]

    team_df = df.loc[picked].copy()
    team_df["cost_m"] = team_df["cost_m"].round(1)
    total_cost = team_df["cost_m"].sum()
    total_points = team_df["predicted_points"].sum()
    return team_df, total_cost, total_points


# -------- STREAMLIT APP --------
st.set_page_config(page_title="FPL Optimizer 100m", layout="wide")
st.title("🔮 FPL Optimizer – Budget 100m")

st.markdown(
    "This tool uses **safe-3seasons models** (trained on 2021–22 to 2023–24, "
    "evaluated on 2024–25) to predict expected points for current FPL players "
    "from the official API, then finds a 15-player squad that maximizes expected "
    "points for the next gameweek under a **100m** budget "
    "(2 GK, 5 DEF, 5 MID, 3 FWD, max 3 players per real team)."
)

col_refresh, _ = st.columns([1, 4])
with col_refresh:
    if st.button("🔄 Refresh live predictions"):
        with st.spinner("Fetching FPL data and running models..."):
            cmd = [sys.executable, "build_players_predictions_live.py"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                st.error("build_players_predictions_live.py failed:")
                st.code(proc.stderr)
            else:
                st.success("Updated players_predictions_live.csv ✅")
                st.expander("Logs (stdout/stderr)").code(
                    proc.stdout + "\n" + proc.stderr
                )

st.markdown("---")

if not PRED_PATH.exists():
    st.error(f"File not found: {PRED_PATH}. Click **Refresh live predictions** first.")
    st.stop()

df = load_players(PRED_PATH)

if st.button("🧮 Optimize 100m squad"):
    with st.spinner("Solving optimization problem..."):
        team_df, total_cost, total_points = optimize_team(
            df, budget=BUDGET, min_spend=MIN_SPEND
        )

    st.subheader("Optimized squad (15 players)")
    st.write(
        f"Total cost (15): **{total_cost:.1f} m**  |  "
        f"Total expected points (15): **{total_points:.2f}**"
    )

    # sort and create integer display points
    show_df = (
        team_df[
            ["position", "web_name", "team_name", "cost_m", "predicted_points"]
        ]
        .sort_values(["position", "predicted_points"], ascending=[True, False])
        .reset_index(drop=True)
    )
    show_df["expected_points"] = show_df["predicted_points"].round(0).astype(int)

    st.dataframe(
        show_df[["position", "web_name", "team_name", "cost_m", "expected_points"]],
        use_container_width=True,
    )

    # -------- STARTING XI 4-3-3 --------
    st.subheader("Suggested starting XI (4-3-3)")

    gk_start = show_df[show_df["position"] == "GK"].nlargest(1, "predicted_points")
    def_start = show_df[show_df["position"] == "DEF"].nlargest(4, "predicted_points")
    mid_start = show_df[show_df["position"] == "MID"].nlargest(3, "predicted_points")
    fwd_start = show_df[show_df["position"] == "FWD"].nlargest(3, "predicted_points")

    xi_df = pd.concat([gk_start, def_start, mid_start, fwd_start], ignore_index=True)
    xi_points = xi_df["predicted_points"].sum()
    xi_points_rounded = int(round(xi_points))
    xi_df["expected_points"] = xi_df["predicted_points"].round(0).astype(int)

    st.write(f"Total expected points (XI): **{xi_points_rounded}**")
    st.dataframe(
        xi_df[["position", "web_name", "team_name", "cost_m", "expected_points"]],
        use_container_width=True,
    )

    st.caption(
        "Expected points are per-gameweek values from the production-scaled model. "
        "Displayed values are rounded to integers for easier reading."
    )
else:
    st.info("Click **Optimize 100m squad** to generate a recommended team.")
