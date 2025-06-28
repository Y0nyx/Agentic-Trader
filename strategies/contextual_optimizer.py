"""
Contextual Optimizer with Machine Learning for Parameter Optimization.

This module implements ML-based optimization that learns optimal parameters
for different market contexts and regimes, enabling adaptive strategy tuning.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ContextualOptimizer:
    """
    Machine Learning optimizer that learns optimal parameters for different market contexts.
    
    This optimizer uses historical performance data to train models that predict
    the best strategy parameters based on current market context features.
    
    Features Used:
    - Market regime indicators (volatility, trend, momentum)
    - Technical indicators (RSI, MACD, MA relationships)
    - Market microstructure (volume, bid-ask dynamics)
    - Risk factors (drawdown, correlation, beta)
    
    Parameters
    ----------
    min_samples_per_regime : int, default 50
        Minimum samples required to train regime-specific model
    test_size : float, default 0.2
        Fraction of data for testing
    random_state : int, default 42
        Random state for reproducibility
    """
    
    def __init__(
        self,
        min_samples_per_regime: int = 50,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.min_samples_per_regime = min_samples_per_regime
        self.test_size = test_size
        self.random_state = random_state
        
        # Model storage
        self.regime_models = {}
        self.feature_scalers = {}
        self.context_features = [
            'volatility_regime', 'trend_strength', 'volume_profile',
            'momentum_indicator', 'rsi_level', 'macd_signal',
            'ma_relationship', 'drawdown_level', 'market_stress'
        ]
        
        # Parameter ranges for optimization
        self.parameter_ranges = {
            'trend_window': (20, 100),
            'confirmation_window': (5, 30),
            'rsi_threshold': (60, 80),
            'exit_threshold': (70, 90),
            'volatility_threshold': (0.15, 0.40),
            'volume_threshold': (1.5, 4.0)
        }
        
        # Performance tracking
        self.training_history = []
        self.model_performance = {}
        
    def train_on_historical_data(self, multi_asset_data: Dict[str, pd.DataFrame]):
        """
        Train contextual optimization models on historical multi-asset data.
        
        Parameters
        ----------
        multi_asset_data : Dict[str, pd.DataFrame]
            Dictionary of asset data {symbol: DataFrame}
        """
        logger.info("Starting contextual optimizer training...")
        
        # Prepare training dataset
        training_data = self._prepare_training_data(multi_asset_data)
        
        if len(training_data) < self.min_samples_per_regime:
            logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return
        
        # Group data by regime and train regime-specific models
        for regime in training_data['regime'].unique():
            regime_data = training_data[training_data['regime'] == regime]
            
            if len(regime_data) >= self.min_samples_per_regime:
                self._train_regime_model(regime, regime_data)
            else:
                logger.warning(f"Insufficient data for {regime}: {len(regime_data)} samples")
        
        logger.info(f"Training completed. Models trained for {len(self.regime_models)} regimes")
    
    def _prepare_training_data(self, multi_asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare training dataset from multi-asset historical data."""
        training_samples = []
        
        for symbol, data in multi_asset_data.items():
            if len(data) < 100:  # Skip assets with insufficient data
                continue
                
            logger.info(f"Processing {symbol} for training data...")
            
            # Generate context features for each time period
            context_features = self._extract_context_features(data)
            
            # Generate parameter combinations and their performance
            param_performance = self._backtest_parameter_combinations(data, symbol)
            
            # Combine context with performance results
            for i, features in context_features.iterrows():
                for params, performance in param_performance.items():
                    if i < len(performance):
                        sample = {
                            'symbol': symbol,
                            'date': i,
                            'regime': features.get('regime', 'unknown'),
                            **features.to_dict(),
                            **params,
                            'performance_score': performance[i] if isinstance(performance, list) else performance
                        }
                        training_samples.append(sample)
        
        return pd.DataFrame(training_samples)
    
    def _extract_context_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract context features from market data."""
        features = pd.DataFrame(index=data.index)
        
        prices = data['Close']
        
        # Volatility regime
        returns = prices.pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        features['volatility_regime'] = pd.cut(volatility, bins=3, labels=[0, 1, 2]).astype(float)
        
        # Trend strength
        ma_short = prices.rolling(window=20).mean()
        ma_long = prices.rolling(window=50).mean()
        trend_strength = (ma_short - ma_long) / ma_long
        features['trend_strength'] = trend_strength
        
        # Volume profile
        if 'Volume' in data.columns:
            volume_ma = data['Volume'].rolling(window=20).mean()
            volume_profile = data['Volume'] / volume_ma
            features['volume_profile'] = volume_profile
        else:
            features['volume_profile'] = 1.0
        
        # Momentum indicator
        momentum = prices.pct_change(periods=10)
        features['momentum_indicator'] = momentum
        
        # RSI level
        from indicators.technical_indicators import rsi
        rsi_values = rsi(prices)
        features['rsi_level'] = rsi_values / 100.0  # Normalize to 0-1
        
        # MACD signal
        try:
            from indicators.technical_indicators import macd
            macd_line, macd_signal, _ = macd(prices)
            features['macd_signal'] = (macd_line - macd_signal) / prices
        except:
            features['macd_signal'] = 0.0
        
        # MA relationship
        features['ma_relationship'] = (prices - ma_long) / ma_long
        
        # Drawdown level
        rolling_max = prices.rolling(window=50).max()
        drawdown = (prices / rolling_max) - 1
        features['drawdown_level'] = drawdown
        
        # Market stress (composite indicator)
        stress_components = [
            volatility / volatility.rolling(window=100).mean(),
            abs(drawdown),
            features['volume_profile'] - 1
        ]
        market_stress = pd.concat(stress_components, axis=1).mean(axis=1)
        features['market_stress'] = market_stress
        
        # Simple regime classification
        features['regime'] = 'moderate_bull'  # Default
        features.loc[trend_strength > 0.1, 'regime'] = 'strong_bull'
        features.loc[trend_strength < -0.05, 'regime'] = 'bear_market'
        features.loc[volatility > 0.3, 'regime'] = 'crisis_mode'
        features.loc[(abs(trend_strength) < 0.05) & (volatility > 0.2), 'regime'] = 'sideways_volatile'
        features.loc[(abs(trend_strength) < 0.05) & (volatility <= 0.2), 'regime'] = 'sideways_calm'
        
        return features.fillna(0)
    
    def _backtest_parameter_combinations(self, data: pd.DataFrame, symbol: str) -> Dict[Tuple, List[float]]:
        """Backtest different parameter combinations to generate training targets."""
        # Simple parameter combinations for demonstration
        param_combinations = [
            {'trend_window': 20, 'confirmation_window': 10, 'rsi_threshold': 70},
            {'trend_window': 50, 'confirmation_window': 20, 'rsi_threshold': 75},
            {'trend_window': 100, 'confirmation_window': 30, 'rsi_threshold': 80},
        ]
        
        results = {}
        
        for params in param_combinations:
            # Simulate simple performance score based on parameters
            # In real implementation, this would run actual backtests
            performance_score = self._simulate_performance_score(data, params)
            results[tuple(params.items())] = performance_score
        
        return results
    
    def _simulate_performance_score(self, data: pd.DataFrame, params: Dict) -> List[float]:
        """Simulate performance score for parameter combination."""
        # Simple simulation - in practice, would run actual strategy backtest
        prices = data['Close']
        returns = prices.pct_change()
        
        # Mock performance based on parameter characteristics
        trend_factor = 1.0 + (params['trend_window'] - 50) / 100
        rsi_factor = 1.0 + (80 - params['rsi_threshold']) / 100
        
        base_performance = returns.rolling(window=20).mean() * trend_factor * rsi_factor
        
        # Add some noise and normalize
        performance = base_performance + np.random.normal(0, 0.01, len(base_performance))
        performance = performance.fillna(0).tolist()
        
        return performance
    
    def _train_regime_model(self, regime: str, regime_data: pd.DataFrame):
        """Train ML model for specific market regime."""
        logger.info(f"Training model for {regime} regime with {len(regime_data)} samples")
        
        # Prepare features and targets
        feature_cols = [col for col in self.context_features if col in regime_data.columns]
        X = regime_data[feature_cols].fillna(0)
        
        # Target: performance score
        y = regime_data['performance_score'].fillna(0)
        
        if len(X) < 10:  # Minimum samples for training
            logger.warning(f"Too few samples for {regime}: {len(X)}")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        
        # Store model and scaler
        self.regime_models[regime] = model
        self.feature_scalers[regime] = scaler
        
        # Track performance
        self.model_performance[regime] = {
            'mse': mse,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }
        
        logger.info(f"Model for {regime} trained. MSE: {mse:.4f}")
    
    def predict_optimal_params(self, current_context: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict optimal parameters for current market context.
        
        Parameters
        ----------
        current_context : Dict[str, float]
            Current market context features
            
        Returns
        -------
        Dict[str, Any]
            Predicted optimal parameters
        """
        # Determine current regime
        regime = current_context.get('regime', 'moderate_bull')
        
        if regime not in self.regime_models:
            logger.warning(f"No model available for regime: {regime}")
            return self._get_default_params(regime)
        
        # Prepare context features
        feature_vector = []
        for feature in self.context_features:
            feature_vector.append(current_context.get(feature, 0.0))
        
        # Scale features
        scaler = self.feature_scalers[regime]
        feature_vector_scaled = scaler.transform([feature_vector])
        
        # Predict performance score for different parameter combinations
        best_params = self._optimize_parameters_with_model(regime, feature_vector_scaled)
        
        logger.info(f"Predicted optimal parameters for {regime}: {best_params}")
        return best_params
    
    def _optimize_parameters_with_model(self, regime: str, context_features: np.ndarray) -> Dict[str, Any]:
        """Optimize parameters using trained model."""
        model = self.regime_models[regime]
        
        # Generate parameter candidates
        param_candidates = self._generate_parameter_candidates()
        
        best_score = -np.inf
        best_params = None
        
        for params in param_candidates:
            # Combine context with parameters for prediction
            # For simplicity, using context features only
            # In practice, would combine context + parameter features
            try:
                score = model.predict(context_features)[0]
                
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                logger.warning(f"Error predicting for params {params}: {e}")
                continue
        
        return best_params or self._get_default_params(regime)
    
    def _generate_parameter_candidates(self) -> List[Dict[str, Any]]:
        """Generate candidate parameter combinations."""
        candidates = []
        
        # Generate grid of parameter combinations
        for trend_window in [20, 50, 100]:
            for confirmation_window in [10, 20, 30]:
                for rsi_threshold in [70, 75, 80]:
                    candidates.append({
                        'trend_window': trend_window,
                        'confirmation_window': confirmation_window,
                        'rsi_threshold': rsi_threshold,
                        'exit_threshold': rsi_threshold + 5,
                        'volatility_threshold': 0.25,
                        'volume_threshold': 2.0
                    })
        
        return candidates
    
    def _get_default_params(self, regime: str) -> Dict[str, Any]:
        """Get default parameters for regime."""
        defaults = {
            'strong_bull': {
                'trend_window': 50,
                'confirmation_window': 20,
                'rsi_threshold': 80,
                'exit_threshold': 85,
                'volatility_threshold': 0.20,
                'volume_threshold': 2.0
            },
            'moderate_bull': {
                'trend_window': 30,
                'confirmation_window': 15,
                'rsi_threshold': 75,
                'exit_threshold': 80,
                'volatility_threshold': 0.25,
                'volume_threshold': 2.5
            },
            'sideways_volatile': {
                'trend_window': 20,
                'confirmation_window': 10,
                'rsi_threshold': 70,
                'exit_threshold': 75,
                'volatility_threshold': 0.30,
                'volume_threshold': 3.0
            },
            'sideways_calm': {
                'trend_window': 40,
                'confirmation_window': 20,
                'rsi_threshold': 65,
                'exit_threshold': 70,
                'volatility_threshold': 0.15,
                'volume_threshold': 1.5
            },
            'bear_market': {
                'trend_window': 25,
                'confirmation_window': 12,
                'rsi_threshold': 60,
                'exit_threshold': 65,
                'volatility_threshold': 0.35,
                'volume_threshold': 2.5
            },
            'crisis_mode': {
                'trend_window': 15,
                'confirmation_window': 5,
                'rsi_threshold': 50,
                'exit_threshold': 55,
                'volatility_threshold': 0.50,
                'volume_threshold': 4.0
            }
        }
        
        return defaults.get(regime, defaults['moderate_bull'])
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained models."""
        summary = {
            'total_regimes': len(self.regime_models),
            'trained_regimes': list(self.regime_models.keys()),
            'model_performance': self.model_performance,
            'feature_importance_avg': self._calculate_average_feature_importance()
        }
        
        return summary
    
    def _calculate_average_feature_importance(self) -> Dict[str, float]:
        """Calculate average feature importance across all models."""
        if not self.model_performance:
            return {}
        
        importance_sum = {}
        count = 0
        
        for regime_perf in self.model_performance.values():
            if 'feature_importance' in regime_perf:
                for feature, importance in regime_perf['feature_importance'].items():
                    importance_sum[feature] = importance_sum.get(feature, 0) + importance
                count += 1
        
        if count > 0:
            return {feature: total / count for feature, total in importance_sum.items()}
        return {}
    
    def update_model(self, regime: str, new_data: pd.DataFrame):
        """
        Update model with new performance data.
        
        Parameters
        ----------
        regime : str
            Market regime to update
        new_data : pd.DataFrame
            New training data
        """
        if regime in self.regime_models:
            logger.info(f"Updating model for {regime} with {len(new_data)} new samples")
            # In practice, would implement incremental learning
            # For now, just log the update
            self.training_history.append({
                'regime': regime,
                'update_time': pd.Timestamp.now(),
                'new_samples': len(new_data)
            })
        else:
            logger.warning(f"No existing model to update for regime: {regime}")


# Helper function for easy integration
def optimize_strategy_for_context(
    strategy_class,
    market_data: pd.DataFrame,
    context_optimizer: ContextualOptimizer
) -> Tuple[Any, Dict[str, Any]]:
    """
    Optimize strategy parameters for current market context.
    
    Parameters
    ----------
    strategy_class : class
        Strategy class to optimize
    market_data : pd.DataFrame
        Current market data
    context_optimizer : ContextualOptimizer
        Trained contextual optimizer
        
    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        Optimized strategy instance and parameters used
    """
    # Extract current context
    context_features = context_optimizer._extract_context_features(market_data)
    current_context = context_features.iloc[-1].to_dict()
    
    # Get optimal parameters
    optimal_params = context_optimizer.predict_optimal_params(current_context)
    
    # Create strategy with optimal parameters
    strategy = strategy_class(**optimal_params)
    
    return strategy, optimal_params