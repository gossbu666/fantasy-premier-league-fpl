import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from typing import Dict
from utils import logger
import warnings
warnings.filterwarnings('ignore')

class SimpleEnsemble:
    """Simple ensemble of XGBoost and Random Forest"""
    
    def __init__(self, position: str):
        self.position = position
        self.models = []
        self.model_names = []
        
    def add_model(self, model, name: str):
        """Add a model to ensemble"""
        self.models.append(model)
        self.model_names.append(name)
        
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Fit all models in ensemble"""
        logger.info(f"\nTraining {len(self.models)} models for {self.position}...")
        
        for idx, (model, name) in enumerate(zip(self.models, self.model_names)):
            logger.info(f"  [{idx+1}/{len(self.models)}] Training {name}...")
            
            try:
                if 'XGB' in name:
                    model.fit(X, y, sample_weight=sample_weight, verbose=False)
                else:
                    model.fit(X, y, sample_weight=sample_weight)
                logger.info(f"      ✓ {name} trained successfully")
            except Exception as e:
                logger.error(f"      ✗ Error training {name}: {e}")
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using median of all models"""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.median(predictions, axis=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate ensemble performance"""
        y_pred = self.predict(X)
        
        # Overall metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # MAPE (only for non-zero y to avoid division by zero)
        mask_nonzero = y > 0
        if mask_nonzero.sum() > 0:
            mape = np.mean(np.abs((y[mask_nonzero] - y_pred[mask_nonzero]) / y[mask_nonzero])) * 100
        else:
            mape = np.nan
        
        # Correlation
        correlation = np.corrcoef(y, y_pred)[0, 1]
        
        # By return category
        categories = {
            'Zeros': (y == 0),
            'Blanks': (y > 0) & (y <= 2),
            'Tickers': (y > 2) & (y <= 4),
            'Haulers': (y >= 5)
        }
        
        category_metrics = {}
        for cat_name, mask in categories.items():
            if mask.sum() > 0:
                cat_rmse = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
                cat_mae = mean_absolute_error(y[mask], y_pred[mask])
                cat_r2 = r2_score(y[mask], y_pred[mask]) if mask.sum() > 1 else np.nan
                
                # MAPE for category (only non-zero)
                cat_mask_nonzero = mask & (y > 0)
                if cat_mask_nonzero.sum() > 0:
                    cat_mape = np.mean(np.abs((y[cat_mask_nonzero] - y_pred[cat_mask_nonzero]) / y[cat_mask_nonzero])) * 100
                else:
                    cat_mape = np.nan
                
                category_metrics[cat_name] = {
                    'rmse': cat_rmse,
                    'mae': cat_mae,
                    'r2': cat_r2,
                    'mape': cat_mape,
                    'n': mask.sum()
                }
        
        return {
            'overall_rmse': rmse,
            'overall_mae': mae,
            'overall_r2': r2,
            'overall_mape': mape,
            'overall_correlation': correlation,
            'by_category': category_metrics
        }
    
    def save(self, path: str):
        """Save ensemble to file"""
        joblib.dump(self, path)
        logger.info(f"✓ Saved {self.position} ensemble to {path}")


def create_ensemble(position: str, n_models: int = 2):
    """Create ensemble for a position"""
    ensemble = SimpleEnsemble(position)
    
    # Add XGBoost models
    for i in range(n_models):
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + i,
            n_jobs=-1
        )
        ensemble.add_model(xgb, f'XGB_{i+1}')
    
    # Add Random Forest models
    for i in range(n_models):
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42 + i,
            n_jobs=-1
        )
        ensemble.add_model(rf, f'RF_{i+1}')
    
    logger.info(f"Created ensemble with {len(ensemble.models)} models")
    return ensemble


def train_position_model(position: str):
    """Train model for a specific position"""
    logger.info("\n" + "="*70)
    logger.info(f"TRAINING {position} MODEL")
    logger.info("="*70)
    
    # Load features
    feature_file = f'data/processed/{position}_features.csv'
    df = pd.read_csv(feature_file)
    
    logger.info(f"✓ Loaded {len(df)} samples from {feature_file}")
    
    # Prepare data
    feature_cols = [col for col in df.columns 
                   if col not in ['target', 'sample_weight', 'player_id', 'player_name']]
    
    X = df[feature_cols].values
    y = df['target'].values
    sample_weights = df['sample_weight'].values if 'sample_weight' in df.columns else None
    
    logger.info(f"✓ Features: {len(feature_cols)} columns")
    logger.info(f"✓ Target range: {y.min():.1f} to {y.max():.1f}")
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )
    
    logger.info(f"✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Create and train ensemble
    ensemble = create_ensemble(position, n_models=2)
    ensemble.fit(X_train, y_train, sample_weight=sw_train)
    
    # Evaluate
    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATION - {position}")
    logger.info(f"{'='*70}")
    
    train_metrics = ensemble.evaluate(X_train, y_train)
    test_metrics = ensemble.evaluate(X_test, y_test)
    
    logger.info("\n📊 Training Set Performance:")
    logger.info(f"  Overall RMSE:        {train_metrics['overall_rmse']:.3f}")
    logger.info(f"  Overall MAE:         {train_metrics['overall_mae']:.3f}")
    logger.info(f"  Overall R²:          {train_metrics['overall_r2']:.3f}")
    logger.info(f"  Overall MAPE:        {train_metrics['overall_mape']:.2f}%")
    logger.info(f"  Overall Correlation: {train_metrics['overall_correlation']:.3f}")
    
    logger.info("\n📊 Test Set Performance:")
    logger.info(f"  Overall RMSE:        {test_metrics['overall_rmse']:.3f}")
    logger.info(f"  Overall MAE:         {test_metrics['overall_mae']:.3f}")
    logger.info(f"  Overall R²:          {test_metrics['overall_r2']:.3f}")
    logger.info(f"  Overall MAPE:        {test_metrics['overall_mape']:.2f}%")
    logger.info(f"  Overall Correlation: {test_metrics['overall_correlation']:.3f}")
    
    logger.info("\n📊 By Return Category (Test Set):")
    logger.info(f"  {'Category':10} | {'RMSE':6} | {'MAE':6} | {'R²':6} | {'MAPE':7} | {'N':4}")
    logger.info(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*4}")
    
    for cat_name, metrics in test_metrics['by_category'].items():
        mape_str = f"{metrics['mape']:.1f}%" if not np.isnan(metrics['mape']) else "N/A"
        r2_str = f"{metrics['r2']:.3f}" if not np.isnan(metrics['r2']) else "N/A"
        logger.info(f"  {cat_name:10} | {metrics['rmse']:6.3f} | {metrics['mae']:6.3f} | "
                   f"{r2_str:6} | {mape_str:7} | {metrics['n']:4}")
    
    # Save model
    model_path = f'models/{position}_ensemble.pkl'
    ensemble.save(model_path)
    
    # Save metrics
    metrics_data = {
        'position': position,
        'train_rmse': train_metrics['overall_rmse'],
        'test_rmse': test_metrics['overall_rmse'],
        'train_mae': train_metrics['overall_mae'],
        'test_mae': test_metrics['overall_mae'],
        'train_r2': train_metrics['overall_r2'],
        'test_r2': test_metrics['overall_r2'],
        'train_mape': train_metrics['overall_mape'],
        'test_mape': test_metrics['overall_mape'],
        'test_correlation': test_metrics['overall_correlation'],
        'by_category': test_metrics['by_category']
    }
    
    metrics_path = f'models/{position}_metrics.pkl'
    joblib.dump(metrics_data, metrics_path)
    logger.info(f"✓ Saved metrics to {metrics_path}")
    
    return ensemble

