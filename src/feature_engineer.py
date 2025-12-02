"""
Feature Engineering for FPL - WITH xG/xA + Team Features
"""

import pandas as pd
import numpy as np
from utils import logger

def create_rolling_features(data, horizons=[1, 3, 5, 10]):
    """Create rolling averages for all metrics including xG/xA"""
    features = pd.DataFrame(index=data.index)
    data = data.sort_values(['player_id', 'round']).copy()
    
    # Base metrics
    metrics = ['total_points', 'minutes', 'goals_scored', 'assists', 
               'clean_sheets', 'goals_conceded', 'bonus', 'bps',
               'influence', 'creativity', 'threat', 'ict_index']
    
    # xG/xA metrics
    xg_metrics = ['expected_goals', 'expected_assists', 
                  'expected_goal_involvements', 'expected_goals_conceded']
    
    # Position-specific
    if 'saves' in data.columns:
        metrics.append('saves')
    if 'value' in data.columns:
        metrics.append('value')
    if 'selected' in data.columns:
        metrics.append('selected')
    if 'transfers_in' in data.columns:
        metrics.extend(['transfers_in', 'transfers_out'])
    
    # Add xG metrics if available
    for metric in xg_metrics:
        if metric in data.columns:
            metrics.append(metric)
    
    # Create rolling features
    feature_count = 0
    for metric in metrics:
        if metric not in data.columns:
            continue
        for horizon in horizons:
            col_name = f'{metric}_avg_{horizon}gw'
            features[col_name] = data.groupby('player_id')[metric].transform(
                lambda x: x.shift(1).rolling(window=horizon, min_periods=1).mean()
            )
            feature_count += 1
    
    # xG Differential features (overperformance/underperformance)
    if 'expected_goals' in data.columns and 'goals_scored' in data.columns:
        for horizon in [3, 5]:
            xg_col = f'expected_goals_avg_{horizon}gw'
            goals_col = f'goals_scored_avg_{horizon}gw'
            
            if xg_col in features.columns and goals_col in features.columns:
                features[f'xg_diff_{horizon}gw'] = features[goals_col] - features[xg_col]
                feature_count += 1
    
    if 'expected_assists' in data.columns and 'assists' in data.columns:
        for horizon in [3, 5]:
            xa_col = f'expected_assists_avg_{horizon}gw'
            assists_col = f'assists_avg_{horizon}gw'
            
            if xa_col in features.columns and assists_col in features.columns:
                features[f'xa_diff_{horizon}gw'] = features[assists_col] - features[xa_col]
                feature_count += 1
    
    logger.info(f"   ✓ Created {feature_count} rolling features")
    return features

def create_team_features(data, horizons=[3, 5]):
    """Create team-level aggregated features"""
    features = pd.DataFrame(index=data.index)
    data = data.sort_values(['team', 'round']).copy()
    
    feature_count = 0
    
    # Team-level metrics
    team_metrics = ['goals_scored', 'goals_conceded', 'clean_sheets']
    
    # Add xG if available
    if 'expected_goals' in data.columns:
        team_metrics.append('expected_goals')
    if 'expected_goals_conceded' in data.columns:
        team_metrics.append('expected_goals_conceded')
    
    for metric in team_metrics:
        if metric not in data.columns:
            continue
            
        for horizon in horizons:
            col_name = f'team_{metric}_avg_{horizon}gw'
            features[col_name] = data.groupby('team')[metric].transform(
                lambda x: x.shift(1).rolling(window=horizon, min_periods=1).mean()
            )
            feature_count += 1
    
    # Opponent team features
    if 'opponent' in data.columns:
        for metric in ['goals_scored', 'goals_conceded', 'clean_sheets']:
            if metric not in data.columns:
                continue
                
            for horizon in [3, 5]:
                col_name = f'opp_{metric}_avg_{horizon}gw'
                features[col_name] = data.groupby('opponent')[metric].transform(
                    lambda x: x.shift(1).rolling(window=horizon, min_periods=1).mean()
                )
                feature_count += 1
    
    if feature_count > 0:
        logger.info(f"   ✓ Created {feature_count} team-level features")
    
    return features

def create_form_features(data):
    """Create form and consistency metrics"""
    features = pd.DataFrame(index=data.index)
    data = data.sort_values(['player_id', 'round']).copy()
    
    # Points per 90 (recent form)
    if 'total_points' in data.columns and 'minutes' in data.columns:
        for horizon in [3, 5]:
            points_col = f'total_points_avg_{horizon}gw'
            minutes_col = f'minutes_avg_{horizon}gw'
            
            # These should exist from rolling features
            if points_col in data.columns and minutes_col in data.columns:
                features[f'points_per_90_{horizon}gw'] = (
                    data.groupby('player_id')[points_col].shift(1) / 
                    (data.groupby('player_id')[minutes_col].shift(1) / 90)
                ).fillna(0).replace([np.inf, -np.inf], 0)
    
    # Consistency score (std of recent points)
    if 'total_points' in data.columns:
        for horizon in [5, 10]:
            features[f'consistency_{horizon}gw'] = data.groupby('player_id')['total_points'].transform(
                lambda x: x.shift(1).rolling(window=horizon, min_periods=3).std()
            )
    
    feature_count = len([c for c in features.columns if features[c].notna().sum() > 0])
    if feature_count > 0:
        logger.info(f"   ✓ Created {feature_count} form features")
    
    return features

def create_interaction_features(data):
    """Create feature interactions"""
    features = pd.DataFrame(index=data.index)
    
    # xG * FDR interaction
    if 'expected_goals' in data.columns and 'fdr_attack' in data.columns:
        for horizon in [3, 5]:
            xg_col = f'expected_goals_avg_{horizon}gw'
            if xg_col in data.columns:
                features[f'xg_fdr_interact_{horizon}gw'] = data[xg_col] * data['fdr_attack']
    
    # Minutes * Form interaction
    if 'minutes' in data.columns and 'total_points' in data.columns:
        minutes_5 = f'minutes_avg_5gw'
        points_5 = f'total_points_avg_5gw'
        if minutes_5 in data.columns and points_5 in data.columns:
            features['minutes_form_interact'] = data[minutes_5] * data[points_5]
    
    feature_count = len([c for c in features.columns if features[c].notna().sum() > 0])
    if feature_count > 0:
        logger.info(f"   ✓ Created {feature_count} interaction features")
    
    return features

def create_features(data, position):
    logger.info(f"Creating advanced features for {position}...")
    data = data.sort_values(['player_id', 'round']).reset_index(drop=True)
    
    features = pd.DataFrame(index=data.index)
    features['player_id'] = data['player_id'].values
    features['round'] = data['round'].values
    
    if 'player_name' in data.columns:
        features['player_name'] = data['player_name'].values
    
    features['target'] = data['total_points'].values
    
    # 1. Rolling features (base + xG/xA)
    rolling = create_rolling_features(data)
    features = pd.concat([features, rolling], axis=1)
    
    # 2. Team-level features
    team_features = create_team_features(data)
    if len(team_features.columns) > 0:
        features = pd.concat([features, team_features], axis=1)
    
    # 3. Form features
    form_features = create_form_features(data)
    if len(form_features.columns) > 0:
        features = pd.concat([features, form_features], axis=1)
    
    # 4. Interaction features
    interact_features = create_interaction_features(data)
    if len(interact_features.columns) > 0:
        features = pd.concat([features, interact_features], axis=1)
    
    # 5. FDR features
    if 'fdr_attack' in data.columns:
        features['fdr_attack'] = data['fdr_attack'].values
        features['fdr_defense'] = data['fdr_defense'].values
        features['is_home'] = data['is_home'].astype(int).values
        logger.info("   ✓ Added FDR features")
    
    # Remove GW1
    initial_len = len(features)
    features = features[features['round'] > 1].copy()
    logger.info(f"   ✓ Removed GW1: {initial_len - len(features)} rows")
    
    # Clean up
    features = features.dropna(subset=['target'])
    features = features.fillna(0)
    
    # Summary
    logger.info(f"✓ Feature creation summary:")
    logger.info(f"  - Total columns: {len(features.columns)}")
    logger.info(f"  - Total rows: {len(features)}")
    logger.info(f"  - Gameweeks: {features['round'].min():.0f} to {features['round'].max():.0f}")
    
    xg_features = [c for c in features.columns if 'expected' in c or 'xg' in c.lower() or 'xa' in c.lower()]
    team_features = [c for c in features.columns if 'team_' in c or 'opp_' in c]
    form_features = [c for c in features.columns if 'form' in c or 'consistency' in c or 'per_90' in c]
    interact_features = [c for c in features.columns if 'interact' in c]
    
    logger.info(f"  - xG/xA features: {len(xg_features)}")
    logger.info(f"  - Team features: {len(team_features)}")
    logger.info(f"  - Form features: {len(form_features)}")
    logger.info(f"  - Interaction features: {len(interact_features)}")
    
    return features

def prepare_features(features_df, position):
    logger.info(f"Preparing features for training...")
    
    metadata_cols = ['target', 'player_id', 'player_name', 'round']
    X = features_df.drop(metadata_cols, axis=1, errors='ignore')
    y = features_df['target']
    rounds = features_df['round'].values
    player_ids = features_df['player_id'].values
    
    logger.info(f"✓ Final feature matrix: {X.shape}")
    
    return X, y, rounds, player_ids

def calculate_sample_weights(y):
    def categorize(points):
        if points == 0: return 'zero'
        elif points <= 2: return 'blank'
        elif points <= 4: return 'ticker'
        else: return 'hauler'
    
    categories = y.apply(categorize)
    weights = categories.map({'zero': 0.5, 'blank': 1.0, 'ticker': 1.5, 'hauler': 3.0})
    return weights

def process_position(position):
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    from pathlib import Path
    
    logger.info("="*60)
    logger.info(f"Processing {position} (ADVANCED FEATURES)")
    logger.info("="*60)
    
    data_file = f'data/processed/{position}_data.csv'
    data = pd.read_csv(data_file)
    logger.info(f"✓ Loaded {len(data)} records from {data_file}")
    
    features = create_features(data, position)
    X, y, rounds, player_ids = prepare_features(features, position)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    logger.info("✓ Normalized features")
    
    sample_weights = calculate_sample_weights(y)
    
    result = pd.DataFrame()
    for col in X_scaled_df.columns:
        result[col] = X_scaled_df[col].values
    
    result['target'] = y.values
    result['sample_weight'] = sample_weights.values
    result['round'] = rounds
    result['player_id'] = player_ids
    
    output_file = f'data/processed/{position}_features.csv'
    result.to_csv(output_file, index=False)
    logger.info(f"✓ Saved to {output_file}")
    
    Path('models').mkdir(exist_ok=True)
    joblib.dump(scaler, f'models/{position}_preprocessor.pkl')
    logger.info(f"✓ Saved preprocessor\n")

def main():
    logger.info("="*80)
    logger.info("ADVANCED FEATURE ENGINEERING - xG/xA + Team + Form + Interactions")
    logger.info("="*80)
    
    positions = ['GK', 'DEF', 'MID', 'FWD']
    
    for position in positions:
        try:
            process_position(position)
        except Exception as e:
            logger.error(f"✗ Error: {position}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("="*80)
    logger.info("✓ ADVANCED FEATURE ENGINEERING COMPLETE!")
    logger.info("="*80)

if __name__ == '__main__':
    main()
