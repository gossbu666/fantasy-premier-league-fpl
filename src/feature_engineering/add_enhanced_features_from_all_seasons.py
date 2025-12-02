"""
Create enhanced features (Xp/Xt/Xo/Xs-style) using FPL expected stats
and merge into per-position feature files.

Input:
    data/historical/all_seasons.csv
    data/processed/{GK,DEF,MID,FWD}_features.csv

Output:
    data/processed/{GK,DEF,MID,FWD}_features_enhanced.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

HIST_PATH = Path("data/historical/all_seasons.csv")
PROC_DIR = Path("data/processed")


def load_history():
    df = pd.read_csv(HIST_PATH)

    # basic types
    df['season'] = df['season'].astype(str)
    df['GW'] = df['GW'].astype(int)
    df['round'] = df['round'].astype(int)

    # team / opponent_team เป็นชื่อทีม (string) อยู่แล้ว ใช้แบบนี้เลย

    # normalize cost name
    df = df.rename(columns={'value': 'now_cost'})

    # team goals for/against per fixture (from home/away)
    df['team_goals_for'] = np.where(df['was_home'],
                                    df['team_h_score'],
                                    df['team_a_score'])
    df['team_goals_against'] = np.where(df['was_home'],
                                        df['team_a_score'],
                                        df['team_h_score'])
    return df


def add_player_expected_roll(df_hist):
    """Xp: player-level rolling expected stats & stability."""
    df_hist = df_hist.sort_values(['season', 'element', 'round'])
    g = df_hist.groupby(['season', 'element'])

    base_cols = [
        'total_points',
        'expected_goals',
        'expected_assists',
        'expected_goal_involvements',
        'expected_goals_conceded',
    ]
    for col in base_cols:
        if col not in df_hist.columns:
            continue
        for window in [3, 5, 10]:
            new_col = f'{col}_roll{window}'
            df_hist[new_col] = g[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

    for base in ['minutes', 'total_points']:
        if base not in df_hist.columns:
            continue
        for window in [5, 10]:
            new_col = f'{base}_std{window}'
            df_hist[new_col] = g[base].transform(
                lambda x: x.rolling(window, min_periods=2).std().fillna(0)
            )

    if 'starts' in df_hist.columns:
        df_hist['starts_5'] = g['starts'].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )
        df_hist['start_rate_5'] = df_hist['starts_5'] / 5.0
        df_hist.drop(columns=['starts_5'], inplace=True)

    return df_hist


def add_team_expected_roll(df_hist):
    """Xt: team-level rolling goals and expected goals."""
    df_hist = df_hist.sort_values(['season', 'team', 'round'])
    g_team = df_hist.groupby(['season', 'team'])

    cols = ['team_goals_for', 'team_goals_against', 'expected_goals_conceded']
    for col in cols:
        if col not in df_hist.columns:
            continue
        for window in [3, 5, 10]:
            new_col = f'{col}_team_roll{window}'
            df_hist[new_col] = g_team[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

    df_hist['team_attack_strength_5'] = df_hist.get(
        'team_goals_for_team_roll5', 0.0
    )
    df_hist['team_defense_strength_5'] = -df_hist.get(
        'team_goals_against_team_roll5', 0.0
    )

    return df_hist


def add_opponent_expected_roll(df_hist):
    """Xo: opponent-level rolling goals / xGC + matchup features."""
    # ensure same dtype for merge keys
    df_hist['season'] = df_hist['season'].astype(str)
    df_hist['opponent_team'] = df_hist['opponent_team'].astype(str)

    team_cols = [
        'season', 'team', 'round',
        'team_goals_for_team_roll5',
        'team_goals_against_team_roll5',
        'expected_goals_conceded_team_roll5',
    ]
    team_cols = [c for c in team_cols if c in df_hist.columns]

    team_stats = df_hist[team_cols].drop_duplicates(
        subset=['season', 'team', 'round']
    ).copy()
    team_stats['season'] = team_stats['season'].astype(str)
    team_stats['team'] = team_stats['team'].astype(str)

    opp_stats = team_stats.rename(columns={
        'team': 'opponent_team',
        'team_goals_for_team_roll5': 'opp_goals_for_team_roll5',
        'team_goals_against_team_roll5': 'opp_goals_against_team_roll5',
        'expected_goals_conceded_team_roll5': 'opp_xgc_team_roll5',
    })
    opp_stats['opponent_team'] = opp_stats['opponent_team'].astype(str)

    df_hist = df_hist.merge(
        opp_stats,
        on=['season', 'opponent_team', 'round'],
        how='left'
    )

    df_hist['attack_matchup_5'] = (
        df_hist['team_attack_strength_5']
        - df_hist['opp_goals_against_team_roll5']
    )
    df_hist['defense_matchup_5'] = (
        df_hist['team_defense_strength_5']
        + df_hist['opp_goals_for_team_roll5']
    )

    return df_hist


def build_enhanced_features():
    print("Loading historical data...")
    hist = load_history()

    print("Adding player-level expected rolling features (Xp)...")
    hist = add_player_expected_roll(hist)

    print("Adding team-level features (Xt)...")
    hist = add_team_expected_roll(hist)

    print("Adding opponent-level features (Xo)...")
    hist = add_opponent_expected_roll(hist)

    core_cols = [
        'season', 'round', 'element', 'team', 'opponent_team', 'was_home'
    ]
    dyn_cols = [c for c in hist.columns if any(tag in c for tag in [
        '_roll3', '_roll5', '_roll10',
        '_std5', '_std10',
        'attack_matchup_5', 'defense_matchup_5',
        'team_attack_strength_5', 'team_defense_strength_5',
        'start_rate_5',
    ])]

    keep_cols = core_cols + dyn_cols
    hist_small = hist[keep_cols].copy()
    return hist_small


def merge_with_position_features(hist_small, position):
    in_path = PROC_DIR / f"{position}_features.csv"
    out_path = PROC_DIR / f"{position}_features_enhanced.csv"

    print(f"\n=== {position}: merging enhanced features ===")
    df = pd.read_csv(in_path)
    before_cols = df.shape[1]

    if 'element' in df.columns:
        key = 'element'
    elif 'player_id' in df.columns:
        key = 'player_id'
    else:
        raise ValueError(
            f"{position}_features.csv must contain 'element' or 'player_id'"
        )

    hist_pos = hist_small.rename(columns={'element': key})
    merge_keys = ['season', 'round', key]

    df_merged = df.merge(
        hist_pos,
        on=merge_keys,
        how='left',
        suffixes=('', '_hist'),
    )

    after_cols = df_merged.shape[1]
    added = after_cols - before_cols
    print(f"  Original features:  {before_cols}")
    print(f"  Enhanced features:  {after_cols}  (added {added})")

    df_merged.to_csv(out_path, index=False)
    print(f"  ✓ Saved to {out_path}")


def main():
    hist_small = build_enhanced_features()
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        merge_with_position_features(hist_small, pos)

    print("\n✅ Enhanced feature files created in data/processed/*_features_enhanced.csv")


if __name__ == "__main__":
    main()
