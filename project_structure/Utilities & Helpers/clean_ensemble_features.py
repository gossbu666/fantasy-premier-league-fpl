"""
CLEAN ENSEMBLE FEATURES (Remove ID/weight columns)
"""

exclude_cols = ['sample_weight', 'player_id', 'element', 'round', 'season']

with open('analysis/GK_ensemble_features.txt', 'r') as f:
    all_features = f.read().splitlines()

clean_features = [f for f in all_features if f not in exclude_cols]
print(f"🧹 GK Clean ensemble features: {len(clean_features)}")

print("\n📋 CLEAN GK ENSEMBLE FEATURES:")
for i, feat in enumerate(clean_features[:15], 1):
    print(f"  {i:2d}. {feat}")

# Save clean version
with open('analysis/GK_clean_ensemble_features.txt', 'w') as f:
    f.write('\n'.join(clean_features))

print(f"\n✅ Saved: analysis/GK_clean_ensemble_features.txt")
