import requests
import pandas as pd
from pathlib import Path

SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25']
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

def download_season(season):
    print(f"Downloading {season}...")
    
    # Download merged_gw.csv (all gameweeks combined)
    url = f"{BASE_URL}/{season}/gws/merged_gw.csv"
    
    try:
        df = pd.read_csv(url)
        df['season'] = season
        print(f"  ✓ {season}: {len(df)} records")
        return df
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

# Download all seasons
all_data = []
for season in SEASONS:
    data = download_season(season)
    if data is not None:
        all_data.append(data)

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    
    # Save
    output_file = 'data/historical/all_seasons.csv'
    combined.to_csv(output_file, index=False)
    
    print(f"\n✓ Combined: {len(combined)} records")
    print(f"✓ Seasons: {combined['season'].unique()}")
    print(f"✓ Output: {output_file}")
else:
    print("✗ No data downloaded!")

