import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from typing import List, Tuple
from utils import logger

class FeatureEngineer:
    """Create features for ML models"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns = []
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               value_col: str,
                               horizons: List[int] = [1, 3, 5, 10, 38]) -> pd.DataFrame:
        """
        Create rolling average features
        
        Args:
            df: DataFrame with player_id, fixture (gameweek), and value columns
            value_col: Column to calculate rolling average
            horizons: List of windows (gameweeks) to average over
        """
        df = df.sort_values(['player_id', 'fixture'])
        
        for horizon in horizons:
            col_name = f'{value_col}_avg_{horizon}gw'
            df[col_name] = df.groupby('player_id')[value_col].transform(
                lambda x: x.rolling(window=horizon, min_periods=1).mean()
            )
        
        return df
    
    def create_position_features(self, df: pd.DataFrame, position: str) -> pd.DataFrame:
        """
        Create position-specific features
        
        Position codes: 1=GK, 2=DEF, 3=MID, 4=FWD
        """
        logger.info(f"Creating features for {position}...")
        
        features_df = df.copy()
        
        # Basic features (all positions)
        basic_cols = ['total_points', 'minutes', 'value', 'selected', 
                     'transfers_in', 'transfers_out']
        
        for col in basic_cols:
            if col in features_df.columns:
                features_df = self.create_rolling_features(
                    features_df, col, horizons=[1, 3, 5, 10]
                )
        
        # Position-specific features
        if position == 'GK':
            gk_cols = ['saves', 'goals_conceded', 'clean_sheets']
            for col in gk_cols:
                if col in features_df.columns:
                    features_df = self.create_rolling_features(
                        features_df, col, horizons=[1, 3, 5]
                    )
        
        elif position in ['DEF', 'MID', 'FWD']:
            outfield_cols = ['goals_scored', 'assists', 'clean_sheets', 
                           'goals_conceded', 'bonus']
            for col in outfield_cols:
                if col in features_df.columns:
                    features_df = self.create_rolling_features(
                        features_df, col, horizons=[1, 3, 5]
                    )
        
        # Additional features
        if 'ict_index' in features_df.columns:
            features_df = self.create_rolling_features(
                features_df, 'ict_index', horizons=[1, 3, 5]
            )
        
        if 'influence' in features_df.columns:
            features_df = self.create_rolling_features(
                features_df, 'influence', horizons=[1, 3]
            )
        
        logger.info(f"✓ Created {len(features_df.columns)} total columns for {position}")
        
        return features_df
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix X and target y
        """
        # Target
        y = df['total_points'].copy()
        
        # Features: select only numeric rolling features
        feature_cols = [col for col in df.columns 
                       if ('_avg_' in col or '_roll_' in col) 
                       and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].copy()
        
        # Fill NaN with 0
        X = X.fillna(0)
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        self.feature_columns = feature_cols
        
        logger.info(f"✓ Prepared {len(feature_cols)} features")
        logger.info(f"✓ Target range: {y.min():.1f} to {y.max():.1f} points")
        
        return X, y
    
    def normalize_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("✓ Fitted and transformed features")
        else:
            X_scaled = self.scaler.transform(X)
            logger.info("✓ Transformed features")
        
        return X_scaled
    
    def compute_sample_weights(self, y: pd.Series, n_bins: int = 4) -> np.ndarray:
        """
        Compute sample weights based on target distribution
        Emphasizes high-return players
        
        Bins:
        - Zeros: 0 points
        - Blanks: 1-2 points
        - Tickers: 3-4 points
        - Haulers: 5+ points
        """
        # Discretize into bins
        bins = [y.min() - 1, 0.5, 2.5, 4.5, y.max() + 1]
        labels = [0, 1, 2, 3]  # Zeros, Blanks, Tickers, Haulers
        
        y_binned = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
        
        # Compute class weights
        weights = compute_sample_weight('balanced', y_binned)
        
        # Clip extreme weights
        weights = np.clip(weights, 0, np.percentile(weights, 95))
        
        # Normalize to unit mean
        weights = weights / weights.mean()
        
        logger.info(f"✓ Sample weights - mean: {weights.mean():.2f}, std: {weights.std():.2f}")
        logger.info(f"  Zeros weight: {weights[y == 0].mean():.2f}")
        logger.info(f"  High-return weight: {weights[y >= 5].mean():.2f}")
        
        return weights
    
    def save_preprocessor(self, path: str = 'models/preprocessor.pkl'):
        """Save scaler and feature info"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
        logger.info(f"✓ Saved preprocessor to {path}")


def process_position_data(position: str) -> pd.DataFrame:
    """
    Process data for a specific position
    
    Args:
        position: 'GK', 'DEF', 'MID', or 'FWD'
    
    Returns:
        DataFrame with features and target ready for training
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {position} data")
    logger.info(f"{'='*60}")
    
    # Load raw data
    input_file = f'data/processed/{position}_data.csv'
    df = pd.read_csv(input_file)
    
    logger.info(f"✓ Loaded {len(df)} records from {input_file}")
    
    # Feature engineering
    engineer = FeatureEngineer()
    df_features = engineer.create_position_features(df, position)
    
    # Prepare X and y
    X, y = engineer.prepare_features_and_target(df_features)
    
    # Normalize
    X_scaled = engineer.normalize_features(X, fit=True)
    
    # Compute sample weights
    sample_weights = engineer.compute_sample_weights(y)
    
    # Create output dataframe
    output_df = pd.DataFrame(X_scaled, columns=engineer.feature_columns)
    output_df['target'] = y.values
    output_df['sample_weight'] = sample_weights
    output_df['player_id'] = df['player_id'].values
    output_df['player_name'] = df['player_name'].values
    
    # Save
    output_file = f'data/processed/{position}_features.csv'
    output_df.to_csv(output_file, index=False)
    
    logger.info(f"✓ Saved {len(output_df)} samples with {len(engineer.feature_columns)} features")
    logger.info(f"✓ Output: {output_file}")
    
    # Save preprocessor for this position
    engineer.save_preprocessor(f'models/{position}_preprocessor.pkl')
    
    return output_df

