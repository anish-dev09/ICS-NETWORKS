"""
Deep Learning-Based Anomaly Detectors for Zero-Day Attack Detection
Uses autoencoders and LSTMs for unsupervised behavioral learning.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Deep learning detectors will not work.")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector.
    Learns normal behavior patterns and flags reconstruction errors.
    """
    
    def __init__(self, input_dim, encoding_dim=32, threshold_percentile=95):
        """
        Initialize autoencoder detector.
        
        Args:
            input_dim (int): Number of input features
            encoding_dim (int): Compressed representation size
            threshold_percentile (float): Percentile for anomaly threshold
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for AutoencoderDetector. Install with: pip install tensorflow")
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.fitted = False
        self.history = None
    
    def _build_model(self):
        """Build autoencoder architecture."""
        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,), name='encoder_input')
        encoded = layers.Dense(128, activation='relu', name='encoder_1')(encoder_input)
        encoded = layers.Dropout(0.2, name='dropout_1')(encoded)
        encoded = layers.Dense(64, activation='relu', name='encoder_2')(encoded)
        encoded = layers.Dropout(0.2, name='dropout_2')(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu', name='decoder_1')(encoded)
        decoded = layers.Dropout(0.2, name='dropout_3')(decoded)
        decoded = layers.Dense(128, activation='relu', name='decoder_2')(decoded)
        decoded = layers.Dropout(0.2, name='dropout_4')(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear', name='decoder_output')(decoded)
        
        # Autoencoder model
        autoencoder = models.Model(encoder_input, decoded, name='autoencoder')
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def fit(self, X, epochs=50, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train autoencoder on normal traffic.
        
        Args:
            X (np.ndarray): Training data (normal traffic only)
            epochs (int): Training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            verbose (int): Verbosity level
        
        Returns:
            self
        """
        print(f"\n[AutoencoderDetector] Training on {X.shape[0]} samples with {X.shape[1]} features...")
        
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self.model = self._build_model()
        
        if verbose:
            print(f"[AutoencoderDetector] Model architecture:")
            self.model.summary()
        
        # Train
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=verbose
        )
        
        self.history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        # Calculate reconstruction errors on training data
        reconstructed = self.model.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
        
        # Set threshold based on percentile
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        
        print(f"[AutoencoderDetector] Training complete. Threshold set at {self.threshold:.6f}")
        print(f"[AutoencoderDetector] Mean reconstruction error: {np.mean(reconstruction_errors):.6f}")
        
        self.fitted = True
        return self
    
    def predict(self, X):
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X (np.ndarray): Test data
        
        Returns:
            np.ndarray: Binary predictions (0 = normal, 1 = anomaly)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
        
        # Anomalies have reconstruction error > threshold
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        return predictions
    
    def predict_with_scores(self, X):
        """
        Get predictions with anomaly scores.
        
        Args:
            X (np.ndarray): Test data
        
        Returns:
            tuple: (predictions, scores)
        """
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled, verbose=0)
        scores = np.mean(np.square(X_scaled - reconstructed), axis=1)
        predictions = (scores > self.threshold).astype(int)
        
        return predictions, scores
    
    def evaluate(self, X, y_true):
        """
        Evaluate detector performance.
        
        Args:
            X (np.ndarray): Test data
            y_true (np.ndarray): True labels
        
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'anomaly_rate': np.mean(y_pred)
        }
    
    def save(self, filepath):
        """Save model to disk."""
        if self.model:
            self.model.save(f"{filepath}_model.keras")
            np.savez(f"{filepath}_params.npz",
                    threshold=self.threshold,
                    input_dim=self.input_dim,
                    encoding_dim=self.encoding_dim,
                    threshold_percentile=self.threshold_percentile)
            print(f"[AutoencoderDetector] Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        self.model = keras.models.load_model(f"{filepath}_model.keras")
        params = np.load(f"{filepath}_params.npz")
        self.threshold = params['threshold']
        self.input_dim = params['input_dim']
        self.encoding_dim = params['encoding_dim']
        self.threshold_percentile = params['threshold_percentile']
        self.fitted = True
        print(f"[AutoencoderDetector] Model loaded from {filepath}")


class LSTMDetector:
    """
    LSTM-based temporal anomaly detector.
    Learns temporal sequences to detect unusual patterns over time.
    """
    
    def __init__(self, input_dim, sequence_length=10, lstm_units=64, threshold_percentile=95):
        """
        Initialize LSTM detector.
        
        Args:
            input_dim (int): Number of features
            sequence_length (int): Length of temporal sequences
            lstm_units (int): LSTM hidden units
            threshold_percentile (float): Percentile for anomaly threshold
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMDetector. Install with: pip install tensorflow")
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.threshold_percentile = threshold_percentile
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.fitted = False
        self.history = None
    
    def _create_sequences(self, X):
        """Create temporal sequences from data."""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        return np.array(sequences)
    
    def _build_model(self):
        """Build LSTM autoencoder architecture."""
        # Encoder
        encoder_input = layers.Input(shape=(self.sequence_length, self.input_dim), name='encoder_input')
        encoder_lstm = layers.LSTM(self.lstm_units, return_sequences=False, name='encoder_lstm')(encoder_input)
        encoder_dropout = layers.Dropout(0.2, name='encoder_dropout')(encoder_lstm)
        
        # Repeat vector for decoder
        repeated = layers.RepeatVector(self.sequence_length, name='repeat_vector')(encoder_dropout)
        
        # Decoder
        decoder_lstm = layers.LSTM(self.lstm_units, return_sequences=True, name='decoder_lstm')(repeated)
        decoder_dropout = layers.Dropout(0.2, name='decoder_dropout')(decoder_lstm)
        decoder_output = layers.TimeDistributed(layers.Dense(self.input_dim), name='decoder_output')(decoder_dropout)
        
        # Model
        model = models.Model(encoder_input, decoder_output, name='lstm_autoencoder')
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, X, epochs=50, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train LSTM on temporal sequences.
        
        Args:
            X (np.ndarray): Training data (normal traffic)
            epochs (int): Training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            verbose (int): Verbosity level
        
        Returns:
            self
        """
        print(f"\n[LSTMDetector] Training on {X.shape[0]} samples with {X.shape[1]} features...")
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled)
        print(f"[LSTMDetector] Created {X_seq.shape[0]} sequences of length {self.sequence_length}")
        
        # Build model
        self.model = self._build_model()
        
        if verbose:
            print(f"[LSTMDetector] Model architecture:")
            self.model.summary()
        
        # Train
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=verbose
        )
        
        self.history = self.model.fit(
            X_seq, X_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        # Calculate reconstruction errors
        reconstructed = self.model.predict(X_seq, verbose=0)
        reconstruction_errors = np.mean(np.square(X_seq - reconstructed), axis=(1, 2))
        
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        
        print(f"[LSTMDetector] Training complete. Threshold set at {self.threshold:.6f}")
        print(f"[LSTMDetector] Mean reconstruction error: {np.mean(reconstruction_errors):.6f}")
        
        self.fitted = True
        return self
    
    def predict(self, X):
        """
        Predict anomalies in temporal sequences.
        
        Args:
            X (np.ndarray): Test data
        
        Returns:
            np.ndarray: Binary predictions (0 = normal, 1 = anomaly)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        
        reconstructed = self.model.predict(X_seq, verbose=0)
        reconstruction_errors = np.mean(np.square(X_seq - reconstructed), axis=(1, 2))
        
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        # Pad to match input length
        padded_predictions = np.zeros(len(X), dtype=int)
        padded_predictions[self.sequence_length - 1:] = predictions
        
        return padded_predictions
    
    def predict_with_scores(self, X):
        """
        Get predictions with anomaly scores.
        
        Args:
            X (np.ndarray): Test data
        
        Returns:
            tuple: (predictions, scores)
        """
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        
        reconstructed = self.model.predict(X_seq, verbose=0)
        scores = np.mean(np.square(X_seq - reconstructed), axis=(1, 2))
        predictions = (scores > self.threshold).astype(int)
        
        # Pad scores
        padded_scores = np.zeros(len(X))
        padded_scores[self.sequence_length - 1:] = scores
        
        # Pad predictions
        padded_predictions = np.zeros(len(X), dtype=int)
        padded_predictions[self.sequence_length - 1:] = predictions
        
        return padded_predictions, padded_scores
    
    def evaluate(self, X, y_true):
        """
        Evaluate detector performance.
        
        Args:
            X (np.ndarray): Test data
            y_true (np.ndarray): True labels
        
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'anomaly_rate': np.mean(y_pred)
        }
    
    def save(self, filepath):
        """Save model to disk."""
        if self.model:
            self.model.save(f"{filepath}_model.keras")
            np.savez(f"{filepath}_params.npz",
                    threshold=self.threshold,
                    input_dim=self.input_dim,
                    sequence_length=self.sequence_length,
                    lstm_units=self.lstm_units,
                    threshold_percentile=self.threshold_percentile)
            print(f"[LSTMDetector] Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        self.model = keras.models.load_model(f"{filepath}_model.keras")
        params = np.load(f"{filepath}_params.npz")
        self.threshold = params['threshold']
        self.input_dim = params['input_dim']
        self.sequence_length = params['sequence_length']
        self.lstm_units = params['lstm_units']
        self.threshold_percentile = params['threshold_percentile']
        self.fitted = True
        print(f"[LSTMDetector] Model loaded from {filepath}")


if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("TensorFlow not available. Install with: pip install tensorflow")
        exit(1)
    
    # Test with synthetic data
    print("Testing Deep Learning Detectors...")
    
    # Generate synthetic normal data
    np.random.seed(42)
    X_normal = np.random.randn(1000, 20)
    
    # Generate synthetic anomalies (different distribution)
    X_anomaly = np.random.randn(100, 20) * 3 + 5
    
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(1000), np.ones(100)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Use only normal data for training (unsupervised)
    X_train_normal = X_train[y_train == 0]
    
    print("\n" + "="*60)
    print("Testing Autoencoder Detector")
    print("="*60)
    ae_detector = AutoencoderDetector(input_dim=20, encoding_dim=10)
    ae_detector.fit(X_train_normal, epochs=30, batch_size=32, verbose=0)
    
    ae_metrics = ae_detector.evaluate(X_test, y_test)
    print("\nAutoencoder Results:")
    for metric, value in ae_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*60)
    print("Testing LSTM Detector")
    print("="*60)
    lstm_detector = LSTMDetector(input_dim=20, sequence_length=5, lstm_units=32)
    lstm_detector.fit(X_train_normal, epochs=30, batch_size=32, verbose=0)
    
    lstm_metrics = lstm_detector.evaluate(X_test, y_test)
    print("\nLSTM Results:")
    for metric, value in lstm_metrics.items():
        print(f"  {metric}: {value:.4f}")
