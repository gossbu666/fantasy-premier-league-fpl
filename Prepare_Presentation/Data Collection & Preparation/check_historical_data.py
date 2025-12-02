import pandas as pd

df = pd.read_csv('data/historical/all_seasons.csv')

print("="*60)
print("HISTORICAL DATA QUALITY CHECK")
print("="*60)

print(f"\nTotal records: {len(df)}")
print(f"Columns: {len(df.columns)}")

print(f"\nBy Season:")
print(df.groupby('season').size())

print(f"\nColumns available:")
print(df.columns.tolist()[:20])  # First 20

print(f"\nMissing critical columns check:")
critical = ['name', 'position', 'total_points', 'minutes', 
            'goals_scored', 'assists', 'clean_sheets', 'GW']

for col in critical:
    if col in df.columns:
        missing = df[col].isna().sum()
        print(f"  ✓ {col:20s}: {missing} missing ({missing/len(df)*100:.1f}%)")
    else:
        print(f"  ✗ {col:20s}: NOT FOUND!")

print(f"\nxG/xA columns:")
xg_cols = ['expected_goals', 'expected_assists', 'xG', 'xA']
for col in xg_cols:
    if col in df.columns:
        non_zero = (df[col] > 0).sum()
        print(f"  ✓ {col:20s}: {non_zero} non-zero ({non_zero/len(df)*100:.1f}%)")

