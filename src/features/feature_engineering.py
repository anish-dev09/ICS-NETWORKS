"""
Feature Engineering for ICS Network Intrusion Detection

This module provides feature extraction and engineering for ICS sensor data.
It creates statistical, temporal, and correlation-based features to improve
detection performance.

Author: Anish
Date: November 6, 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for ICS sensor data.
    
    Creates:
    - Statistical features (mean, std, min, max, variance, skewness, kurtosis)
    - Temporal features (rolling windows, rate of change, differences)
    - Correlation-based features
    - Lag features
    """
    
    def __init__(self, window_sizes: List[int] = [10, 30, 60]):
        """
        Initialize feature engineer.
        
        Args:
            window_sizes: List of window sizes for rolling statistics
        """
        self.window_sizes = window_sizes
        self.feature_names = []
        self.sensor_cols = None
        
    def fit(self, X: pd.DataFrame, sensor_cols: Optional[List[str]] = None):
        """
        Fit the feature engineer (store sensor column names).
        
        Args:
            X: Input dataframe
            sensor_cols: List of sensor column names (optional)
        """
        if sensor_cols is not None:
            self.sensor_cols = sensor_cols
        else:
            # Auto-detect numeric columns (exclude timestamp and attack)
            exclude_cols = ['timestamp', 'attack', 'time']
            self.sensor_cols = [col for col in X.columns 
                               if col not in exclude_cols and 
                               pd.api.types.is_numeric_dtype(X[col])]
        
        print(f"✅ Feature engineer fitted on {len(self.sensor_cols)} sensors")
        return self
    
    def transform(self, X: pd.DataFrame, include_original: bool = True) -> pd.DataFrame:
        """
        Transform data by creating engineered features.
        
        Args:
            X: Input dataframe
            include_original: Whether to include original sensor values
            
        Returns:
            DataFrame with engineered features
        """
        if self.sensor_cols is None:
            raise ValueError("FeatureEngineer must be fitted first!")
        
        print(f"🔧 Engineering features from {len(self.sensor_cols)} sensors...")
        
        # Start with original features if requested
        if include_original:
            result = X[self.sensor_cols].copy()
        else:
            result = pd.DataFrame(index=X.index)
        
        # Extract numeric sensor data
        sensor_data = X[self.sensor_cols].apply(pd.to_numeric, errors='coerce')
        
        # 1. Statistical features
        print("  → Creating statistical features...")
        stat_features = self._create_statistical_features(sensor_data)
        result = pd.concat([result, stat_features], axis=1)
        
        # 2. Temporal features (rolling windows)
        print("  → Creating temporal features...")
        temp_features = self._create_temporal_features(sensor_data)
        result = pd.concat([result, temp_features], axis=1)
        
        # 3. Rate of change features
        print("  → Creating rate of change features...")
        roc_features = self._create_rate_of_change_features(sensor_data)
        result = pd.concat([result, roc_features], axis=1)
        
        # 4. Lag features
        print("  → Creating lag features...")
        lag_features = self._create_lag_features(sensor_data)
        result = pd.concat([result, lag_features], axis=1)
        
        # 5. Interaction features (selected pairs)
        print("  → Creating interaction features...")
        interaction_features = self._create_interaction_features(sensor_data)
        result = pd.concat([result, interaction_features], axis=1)
        
        # Fill NaN values (caused by rolling windows and lags)
        result = result.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        print(f"✅ Created {result.shape[1]} features from {len(self.sensor_cols)} sensors")
        
        return result
    
    def fit_transform(self, X: pd.DataFrame, sensor_cols: Optional[List[str]] = None,
                     include_original: bool = True) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Input dataframe
            sensor_cols: List of sensor column names (optional)
            include_original: Whether to include original sensor values
            
        Returns:
            DataFrame with engineered features
        """
        self.fit(X, sensor_cols)
        return self.transform(X, include_original)
    
    def _create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features across sensors."""
        features = pd.DataFrame(index=X.index)
        
        # Global statistics across all sensors
        features['mean_all_sensors'] = X.mean(axis=1)
        features['std_all_sensors'] = X.std(axis=1)
        features['min_all_sensors'] = X.min(axis=1)
        features['max_all_sensors'] = X.max(axis=1)
        features['range_all_sensors'] = features['max_all_sensors'] - features['min_all_sensors']
        features['median_all_sensors'] = X.median(axis=1)
        features['skew_all_sensors'] = X.skew(axis=1)
        features['kurtosis_all_sensors'] = X.kurtosis(axis=1)
        
        # Count of sensors at extreme values
        features['num_sensors_at_zero'] = (X == 0).sum(axis=1)
        features['num_sensors_at_max'] = (X == X.max()).sum(axis=1)
        
        return features
    
    def _create_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features using rolling windows."""
        features = pd.DataFrame(index=X.index)
        
        for window in self.window_sizes:
            # Rolling statistics
            rolling = X.rolling(window=window, min_periods=1)
            
            features[f'rolling_mean_{window}'] = rolling.mean().mean(axis=1)
            features[f'rolling_std_{window}'] = rolling.std().mean(axis=1)
            features[f'rolling_min_{window}'] = rolling.min().mean(axis=1)
            features[f'rolling_max_{window}'] = rolling.max().mean(axis=1)
            features[f'rolling_range_{window}'] = (features[f'rolling_max_{window}'] - 
                                                   features[f'rolling_min_{window}'])
        
        return features
    
    def _create_rate_of_change_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create rate of change features."""
        features = pd.DataFrame(index=X.index)
        
        # First-order differences (change from previous time step)
        diff_1 = X.diff(periods=1)
        features['mean_diff_1'] = diff_1.mean(axis=1)
        features['std_diff_1'] = diff_1.std(axis=1)
        features['max_abs_diff_1'] = diff_1.abs().max(axis=1)
        
        # Second-order differences (acceleration)
        diff_2 = X.diff(periods=2)
        features['mean_diff_2'] = diff_2.mean(axis=1)
        features['std_diff_2'] = diff_2.std(axis=1)
        
        # Rate of change percentage
        roc = X.pct_change(periods=1).replace([np.inf, -np.inf], 0)
        features['mean_roc'] = roc.mean(axis=1)
        features['std_roc'] = roc.std(axis=1)
        features['max_abs_roc'] = roc.abs().max(axis=1)
        
        return features
    
    def _create_lag_features(self, X: pd.DataFrame, lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Create lag features for temporal patterns."""
        features = pd.DataFrame(index=X.index)
        
        # Use only a subset of key sensors for lag features (to avoid explosion)
        # Select first 5 sensors as representatives
        key_sensors = self.sensor_cols[:5]
        
        for lag in lags:
            lagged = X[key_sensors].shift(periods=lag)
            features[f'lag_{lag}_mean'] = lagged.mean(axis=1)
            features[f'lag_{lag}_std'] = lagged.std(axis=1)
        
        return features
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between highly correlated sensors."""
        features = pd.DataFrame(index=X.index)
        
        # Select first 10 sensors for interactions (to avoid explosion)
        selected_sensors = self.sensor_cols[:10]
        
        # Pairwise differences
        for i in range(min(5, len(selected_sensors))):
            for j in range(i+1, min(5, len(selected_sensors))):
                col_i = selected_sensors[i]
                col_j = selected_sensors[j]
                
                features[f'diff_{col_i}_{col_j}'] = X[col_i] - X[col_j]
                features[f'ratio_{col_i}_{col_j}'] = X[col_i] / (X[col_j] + 1e-6)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.feature_names


class FeatureSelector:
    """
    Feature selection for ICS intrusion detection.
    
    Selects most important features using:
    - Variance threshold
    - Correlation threshold
    - Feature importance (from tree-based models)
    """
    
    def __init__(self, variance_threshold: float = 0.01, 
                 correlation_threshold: float = 0.95):
        """
        Initialize feature selector.
        
        Args:
            variance_threshold: Minimum variance for feature to be kept
            correlation_threshold: Maximum correlation for keeping both features
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit feature selector.
        
        Args:
            X: Feature dataframe
            y: Target labels (optional)
        """
        # First, ensure all columns are numeric
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]
        
        selected = list(X_numeric.columns)
        
        print(f"🔍 Starting with {len(selected)} features...")
        
        # 1. Remove low variance features
        print("  → Removing low variance features...")
        variances = X_numeric.var()
        high_var_features = variances[variances > self.variance_threshold].index.tolist()
        selected = [f for f in selected if f in high_var_features]
        print(f"     Kept {len(selected)} features (removed {len(X_numeric.columns) - len(selected)})")
        
        # 2. Remove highly correlated features
        print("  → Removing highly correlated features...")
        X_selected = X_numeric[selected]
        corr_matrix = X_selected.corr().abs()
        
        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > self.correlation_threshold)]
        
        selected = [f for f in selected if f not in to_drop]
        print(f"     Kept {len(selected)} features (removed {len(to_drop)} correlated)")
        
        self.selected_features = selected
        print(f"✅ Feature selection complete: {len(self.selected_features)} features selected")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with selected features only
        """
        if self.selected_features is None:
            raise ValueError("FeatureSelector must be fitted first!")
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature dataframe
            y: Target labels (optional)
            
        Returns:
            DataFrame with selected features only
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        return self.selected_features if self.selected_features is not None else []


def create_features_pipeline(X: pd.DataFrame, y: Optional[pd.Series] = None,
                            window_sizes: List[int] = [10, 30, 60],
                            include_original: bool = True,
                            apply_selection: bool = True) -> Tuple[pd.DataFrame, FeatureEngineer, FeatureSelector]:
    """
    Complete feature engineering pipeline.
    
    Args:
        X: Input dataframe
        y: Target labels (optional)
        window_sizes: List of window sizes for rolling statistics
        include_original: Whether to include original sensor values
        apply_selection: Whether to apply feature selection
        
    Returns:
        Tuple of (engineered_features, engineer, selector)
    """
    # Feature engineering
    engineer = FeatureEngineer(window_sizes=window_sizes)
    X_engineered = engineer.fit_transform(X, include_original=include_original)
    
    # Feature selection
    if apply_selection:
        selector = FeatureSelector()
        X_selected = selector.fit_transform(X_engineered, y)
        return X_selected, engineer, selector
    else:
        return X_engineered, engineer, None


if __name__ == "__main__":
    print("Feature Engineering Module for ICS Intrusion Detection")
    print("=" * 60)
    print("\nThis module provides:")
    print("  1. Statistical features (mean, std, min, max, etc.)")
    print("  2. Temporal features (rolling windows)")
    print("  3. Rate of change features")
    print("  4. Lag features")
    print("  5. Interaction features")
    print("  6. Feature selection")
    print("\nUsage:")
    print("  from src.features.feature_engineering import FeatureEngineer")
    print("  engineer = FeatureEngineer(window_sizes=[10, 30, 60])")
    print("  X_features = engineer.fit_transform(X)")
