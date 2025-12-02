"""
CREATE OPTIMIZED FEATURE SETS จาก Correlation Analysis
"""

top_features = {
    'GK': [
        'minutes_avg_3gw', 'minutes_avg_5gw', 'minutes_avg_10gw',
        'bps_avg_3gw', 'bps_avg_5gw', 'bps_avg_10gw',
        'ict_index_avg_3gw', 'ict_index_avg_5gw', 'ict_index_avg_10gw',
        'total_points_avg_3gw', 'total_points_avg_5gw', 'total_points_avg_10gw',
        'influence_avg_3gw', 'influence_avg_5gw', 'influence_avg_10gw',
        'consistency_10gw', 'consistency_5gw', 'goals_conceded_avg_5gw'
    ],
    
    'DEF': [
        'bps_avg_10gw', 'minutes_avg_10gw', 'minutes_avg_5gw', 'bps_avg_5gw',
        'ict_index_avg_10gw', 'influence_avg_10gw', 'minutes_avg_3gw',
        'total_points_avg_10gw', 'bps_avg_3gw', 'ict_index_avg_5gw',
        'influence_avg_5gw', 'ict_index_avg_3gw', 'total_points_avg_5gw',
        'consistency_10gw', 'clean_sheets_avg_10gw'
    ],
    
    'MID': [
        'ict_index_avg_10gw', 'minutes_avg_10gw', 'total_points_avg_10gw',
        'bps_avg_10gw', 'minutes_avg_5gw', 'influence_avg_10gw',
        'ict_index_avg_5gw', 'minutes_avg_3gw', 'creativity_avg_10gw',
        'threat_avg_10gw', 'bps_avg_5gw', 'total_points_avg_5gw'
    ],
    
    'FWD': [
        'minutes_avg_5gw', 'minutes_avg_10gw', 'ict_index_avg_10gw',
        'minutes_avg_3gw', 'total_points_avg_10gw', 'ict_index_avg_5gw',
        'total_points_avg_5gw', 'threat_avg_10gw', 'ict_index_avg_3gw',
        'influence_avg_10gw', 'bps_avg_10gw', 'threat_avg_5gw'
    ]
}

positions = ['GK', 'DEF', 'MID', 'FWD']

for pos in positions:
    features = top_features[pos]
    
    print(f"\n📋 {pos} OPTIMIZED FEATURES ({len(features)}):")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")
    
    # Save feature list
    with open(f"analysis/{pos}_optimized_features.txt", 'w') as f:
        f.write('\n'.join(features))
    print(f"   ✓ Saved: analysis/{pos}_optimized_features.txt")

print("\n🚀 READY FOR RETRAINING!")
print("python3 retrain_enhanced.py --features-file analysis/{POS}_optimized_features.txt")
