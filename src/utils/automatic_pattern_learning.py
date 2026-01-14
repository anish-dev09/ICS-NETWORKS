"""
Automatic Pattern Learning Module for ICS Anomaly Detection
Automatically discovers complex patterns using deep feature learning and AutoML.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers, Model  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler


class DeepFeatureExtractor:
    """
    Automatically learns features using deep autoencoders.
    """
    
    def __init__(self, encoding_dim: int = 32, hidden_layers: List[int] = [64, 32]):
        """
        Initialize deep feature extractor.
        
        Args:
            encoding_dim (int): Dimension of learned features
            hidden_layers (list): Hidden layer sizes
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.autoencoder = None
        self.encoder = None
        self.scaler = StandardScaler()
        
    def build_autoencoder(self, input_dim: int) -> Optional[Tuple]:
        """
        Build autoencoder for feature learning.
        
        Args:
            input_dim (int): Input feature dimension
            
        Returns:
            tuple: (autoencoder, encoder)
        """
        if not TF_AVAILABLE:
            print("TensorFlow not available")
            return None
        
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = input_layer
        
        for hidden_size in self.hidden_layers:
            encoded = layers.Dense(hidden_size, activation='relu')(encoded)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(0.2)(encoded)
        
        # Bottleneck (learned features)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoded')(encoded)
        
        # Decoder
        decoded = encoded
        for hidden_size in reversed(self.hidden_layers):
            decoded = layers.Dense(hidden_size, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
        
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Models
        self.autoencoder = Model(input_layer, decoded, name='autoencoder')
        self.encoder = Model(input_layer, encoded, name='encoder')
        
        return self.autoencoder, self.encoder
    
    def train(self, X: np.ndarray, epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train autoencoder to learn features.
        
        Args:
            X (np.ndarray): Training data
            epochs (int): Training epochs
            batch_size (int): Batch size
            
        Returns:
            dict: Training history
        """
        if not TF_AVAILABLE:
            print("TensorFlow not available")
            return {}
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        if self.autoencoder is None:
            self.build_autoencoder(X.shape[1])
        
        # Compile
        self.autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        return history.history
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract learned features.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Learned features
        """
        if self.encoder is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        features = self.encoder.predict(X_scaled, verbose=0)
        return features


class AutomaticFeatureEngineering:
    """
    Automatically engineer features from raw data.
    """
    
    def __init__(self):
        """Initialize automatic feature engineering."""
        self.feature_names = []
        
    def engineer_statistical_features(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Engineer statistical features automatically.
        
        Args:
            X (np.ndarray): Input data (n_samples, n_features)
            
        Returns:
            tuple: (engineered features, feature names)
        """
        features = []
        names = []
        
        # Original features
        features.append(X)
        names.extend([f'orig_{i}' for i in range(X.shape[1])])
        
        # Statistical transforms
        features.append(np.log1p(np.abs(X)))
        names.extend([f'log_{i}' for i in range(X.shape[1])])
        
        features.append(np.sqrt(np.abs(X)))
        names.extend([f'sqrt_{i}' for i in range(X.shape[1])])
        
        features.append(np.square(X))
        names.extend([f'sq_{i}' for i in range(X.shape[1])])
        
        # Interactions (pairwise products for small feature sets)
        if X.shape[1] <= 10:
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    features.append((X[:, i] * X[:, j]).reshape(-1, 1))
                    names.append(f'interact_{i}_{j}')
        
        # Combine
        engineered = np.hstack(features)
        
        self.feature_names = names
        return engineered, names
    
    def engineer_temporal_features(self, X: np.ndarray, window_size: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        Engineer temporal features (rolling statistics).
        
        Args:
            X (np.ndarray): Time-series data
            window_size (int): Window for rolling statistics
            
        Returns:
            tuple: (temporal features, feature names)
        """
        features = []
        names = []
        
        n_samples, n_features = X.shape
        
        # Rolling mean
        rolling_mean = np.array([
            np.convolve(X[:, i], np.ones(window_size)/window_size, mode='same')
            for i in range(n_features)
        ]).T
        features.append(rolling_mean)
        names.extend([f'rolling_mean_{i}' for i in range(n_features)])
        
        # Rolling std
        rolling_std = np.array([
            np.convolve(X[:, i]**2, np.ones(window_size)/window_size, mode='same') - rolling_mean[:, i]**2
            for i in range(n_features)
        ]).T
        rolling_std = np.sqrt(np.maximum(rolling_std, 0))
        features.append(rolling_std)
        names.extend([f'rolling_std_{i}' for i in range(n_features)])
        
        # Differences
        diff = np.diff(X, axis=0, prepend=X[0:1])
        features.append(diff)
        names.extend([f'diff_{i}' for i in range(n_features)])
        
        # Combine
        engineered = np.hstack(features)
        return engineered, names


class AutomaticPatternDiscovery:
    """
    Automatically discovers patterns using unsupervised learning.
    """
    
    def __init__(self, n_patterns: int = 10):
        """
        Initialize pattern discovery.
        
        Args:
            n_patterns (int): Number of patterns to discover
        """
        self.n_patterns = n_patterns
        self.patterns = []
        self.pattern_labels = None
        
    def discover_kmeans(self, X: np.ndarray) -> np.ndarray:
        """
        Discover patterns using K-Means clustering.
        
        Args:
            X (np.ndarray): Feature data
            
        Returns:
            np.ndarray: Cluster labels
        """
        kmeans = KMeans(n_clusters=self.n_patterns, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        self.patterns = kmeans.cluster_centers_
        self.pattern_labels = labels
        
        return labels
    
    def discover_dbscan(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Discover patterns using DBSCAN (density-based).
        
        Args:
            X (np.ndarray): Feature data
            eps (float): DBSCAN epsilon parameter
            min_samples (int): Minimum samples per cluster
            
        Returns:
            np.ndarray: Cluster labels
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Extract pattern centers
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise
        
        patterns = []
        for label in unique_labels:
            mask = labels == label
            pattern_center = X[mask].mean(axis=0)
            patterns.append(pattern_center)
        
        self.patterns = np.array(patterns)
        self.pattern_labels = labels
        
        return labels
    
    def discover_isolation_forest(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        Discover anomalous patterns using Isolation Forest.
        
        Args:
            X (np.ndarray): Feature data
            contamination (float): Expected proportion of anomalies
            
        Returns:
            np.ndarray: Anomaly scores (-1 = anomaly, 1 = normal)
        """
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        
        return anomaly_labels


class SimpleAutoML:
    """
    Simple AutoML for model selection and hyperparameter tuning.
    """
    
    def __init__(self):
        """Initialize SimpleAutoML."""
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def try_models(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Try multiple models and select best.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            
        Returns:
            dict: Model performance results
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, f1_score
        
        # Model candidates
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                
                results[name] = {
                    'accuracy': acc,
                    'f1_score': f1,
                    'model': model
                }
                
                # Track best
                if f1 > self.best_score:
                    self.best_score = f1
                    self.best_model = model
                
                print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")
                
            except Exception as e:
                print(f"{name} failed: {e}")
                results[name] = {'error': str(e)}
        
        self.models = results
        return results
    
    def get_best_model(self):
        """Get best performing model."""
        return self.best_model


class AutomaticPatternLearner:
    """
    Integrated automatic pattern learning system.
    """
    
    def __init__(self, use_deep_features: bool = True, use_automl: bool = True):
        """
        Initialize automatic pattern learner.
        
        Args:
            use_deep_features (bool): Use deep feature extraction
            use_automl (bool): Use AutoML for model selection
        """
        self.use_deep_features = use_deep_features
        self.use_automl = use_automl
        
        self.deep_extractor = DeepFeatureExtractor() if use_deep_features and TF_AVAILABLE else None
        self.feature_engineer = AutomaticFeatureEngineering()
        self.pattern_discovery = AutomaticPatternDiscovery()
        self.automl = SimpleAutoML() if use_automl else None
        
    def learn_patterns(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict:
        """
        Automatically learn patterns from data.
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Optional labels for supervised learning
            
        Returns:
            dict: Learned patterns and features
        """
        result = {}
        
        # 1. Engineer features
        print("Engineering features...")
        X_engineered, feature_names = self.feature_engineer.engineer_statistical_features(X)
        result['engineered_features'] = X_engineered
        result['feature_names'] = feature_names
        
        # 2. Deep feature learning
        if self.deep_extractor:
            print("Learning deep features...")
            history = self.deep_extractor.train(X_engineered, epochs=30)
            deep_features = self.deep_extractor.extract_features(X_engineered)
            result['deep_features'] = deep_features
            result['deep_training'] = history
        else:
            deep_features = X_engineered
        
        # 3. Pattern discovery
        print("Discovering patterns...")
        pattern_labels = self.pattern_discovery.discover_kmeans(deep_features)
        result['pattern_labels'] = pattern_labels
        result['n_patterns'] = len(set(pattern_labels))
        
        # 4. AutoML (if labels provided)
        if y is not None and self.automl:
            print("Running AutoML...")
            # Split for validation
            n_train = int(len(X) * 0.8)
            X_train, X_val = deep_features[:n_train], deep_features[n_train:]
            y_train, y_val = y[:n_train], y[n_train:]
            
            model_results = self.automl.try_models(X_train, y_train, X_val, y_val)
            result['automl_results'] = model_results
            result['best_model'] = self.automl.get_best_model()
        
        return result


if __name__ == "__main__":
    print("Testing Automatic Pattern Learning Module...")
    
    # Generate synthetic ICS data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Normal patterns
    X_normal = np.random.randn(800, n_features) * 0.5 + 2
    y_normal = np.zeros(800)
    
    # Attack patterns
    X_attack = np.random.randn(200, n_features) * 1.5 + 5
    y_attack = np.ones(200)
    
    X = np.vstack([X_normal, X_attack])
    y = np.hstack([y_normal, y_attack])
    
    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    print("\n" + "="*60)
    print("Test 1: Deep Feature Extraction")
    print("="*60)
    
    if TF_AVAILABLE:
        extractor = DeepFeatureExtractor(encoding_dim=8, hidden_layers=[32, 16])
        history = extractor.train(X, epochs=20, batch_size=32)
        
        print(f"\nTraining completed")
        print(f"Final loss: {history['loss'][-1]:.4f}")
        print(f"Final val_loss: {history['val_loss'][-1]:.4f}")
        
        deep_features = extractor.extract_features(X)
        print(f"Original features: {X.shape}")
        print(f"Deep features: {deep_features.shape}")
    else:
        print("\nTensorFlow not available, skipping deep learning test")
    
    print("\n" + "="*60)
    print("Test 2: Automatic Feature Engineering")
    print("="*60)
    
    engineer = AutomaticFeatureEngineering()
    X_engineered, feature_names = engineer.engineer_statistical_features(X)
    
    print(f"\nOriginal features: {X.shape[1]}")
    print(f"Engineered features: {X_engineered.shape[1]}")
    print(f"Sample feature names: {feature_names[:10]}")
    
    print("\n" + "="*60)
    print("Test 3: Automatic Pattern Discovery")
    print("="*60)
    
    discovery = AutomaticPatternDiscovery(n_patterns=5)
    
    # K-Means
    kmeans_labels = discovery.discover_kmeans(X)
    print(f"\nK-Means discovered {len(set(kmeans_labels))} patterns")
    print(f"Pattern distribution: {np.bincount(kmeans_labels)}")
    
    # DBSCAN
    dbscan_labels = discovery.discover_dbscan(X, eps=2.0, min_samples=10)
    print(f"\nDBSCAN discovered {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)} patterns")
    print(f"Noise samples: {np.sum(dbscan_labels == -1)}")
    
    # Isolation Forest
    anomaly_scores = discovery.discover_isolation_forest(X, contamination=0.2)
    print(f"\nIsolation Forest:")
    print(f"Anomalies detected: {np.sum(anomaly_scores == -1)}")
    print(f"Normal samples: {np.sum(anomaly_scores == 1)}")
    
    print("\n" + "="*60)
    print("Test 4: SimpleAutoML")
    print("="*60)
    
    automl = SimpleAutoML()
    
    # Split data
    n_train = 800
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    print(f"\nTrying multiple models...")
    results = automl.try_models(X_train, y_train, X_val, y_val)
    
    print(f"\nBest model score: {automl.best_score:.4f}")
    
    print("\n" + "="*60)
    print("Test 5: Integrated Pattern Learner")
    print("="*60)
    
    learner = AutomaticPatternLearner(
        use_deep_features=TF_AVAILABLE,
        use_automl=True
    )
    
    result = learner.learn_patterns(X, y)
    
    print(f"\n✓ Pattern learning completed:")
    print(f"  Engineered features: {result['engineered_features'].shape}")
    if 'deep_features' in result:
        print(f"  Deep features: {result['deep_features'].shape}")
    print(f"  Patterns discovered: {result['n_patterns']}")
    if 'automl_results' in result:
        print(f"  AutoML models tested: {len(result['automl_results'])}")
    
    print("\n✓ Automatic pattern learning module working!")
