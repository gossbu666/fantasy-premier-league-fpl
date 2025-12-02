import pandas as pd

def add_rolling_features_season_player(df: pd.DataFrame,
                                       value_cols,
                                       windows=(1, 3, 5, 10),
                                       prefix=""):
    """
    Rolling mean per (season, player_id) โดยใช้เฉพาะข้อมูล GW ก่อนหน้า (shift(1)).
    df ต้องมี: ['season', 'player_id', 'round'] + value_cols
    """
    df = df.sort_values(["season", "player_id", "round"]).copy()
    group = df.groupby(["season", "player_id"], group_keys=False)

    for col in value_cols:
        for w in windows:
            new_col = f"{prefix}{col}_roll{w}"
            df[new_col] = group[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return df
