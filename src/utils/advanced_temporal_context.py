"""
Advanced Temporal Context Module for ICS Anomaly Detection
Extends temporal analysis with attention mechanisms and longer sequences.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class SlidingWindowProcessor:
    """
    Processes time-series data with overlapping sliding windows.
    Captures longer temporal dependencies.
    """
    
    def __init__(self, window_size: int = 60, stride: int = 10):
        """
        Initialize sliding window processor.
        
        Args:
            window_size (int): Size of sliding window (seconds/samples)
            stride (int): Step size between windows
        """
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        
    def create_windows(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create overlapping windows from time-series data.
        
        Args:
            data (np.ndarray): Time-series data (n_samples, n_features)
            timestamps (np.ndarray): Optional timestamps
            
        Returns:
            np.ndarray: Windowed data (n_windows, window_size, n_features)
        """
        n_samples = len(data)
        n_features = data.shape[1] if data.ndim > 1 else 1
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        windows = []
        window_times = []
        
        for start in range(0, n_samples - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = data[start:end]
            windows.append(window)
            
            if timestamps is not None:
                window_times.append((timestamps[start], timestamps[end-1]))
        
        self.windows = np.array(windows)
        
        if timestamps is not None:
            return self.windows, window_times
        
        return self.windows
    
    def compute_window_statistics(self, windows: np.ndarray) -> Dict:
        """
        Compute statistical features for each window.
        
        Args:
            windows (np.ndarray): Windowed data
            
        Returns:
            dict: Window statistics
        """
        stats = {
            'mean': np.mean(windows, axis=1),
            'std': np.std(windows, axis=1),
            'min': np.min(windows, axis=1),
            'max': np.max(windows, axis=1),
            'median': np.median(windows, axis=1),
            'q25': np.percentile(windows, 25, axis=1),
            'q75': np.percentile(windows, 75, axis=1)
        }
        
        return stats
    
    def extract_temporal_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract rich temporal features from windows.
        
        Args:
            windows (np.ndarray): Windowed data (n_windows, window_size, n_features)
            
        Returns:
            np.ndarray: Temporal features
        """
        features = []
        
        for window in windows:
            # Statistical features
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            min_val = np.min(window, axis=0)
            max_val = np.max(window, axis=0)
            
            # Trend features
            if len(window) > 1:
                # Linear fit
                x = np.arange(len(window))
                trend = np.polyfit(x, window.T, 1)[0] if window.shape[1] == 1 else np.mean([np.polyfit(x, window[:, i], 1)[0] for i in range(window.shape[1])])
            else:
                trend = 0
            
            # Combine features
            window_features = np.concatenate([
                mean.flatten(),
                std.flatten(),
                [min_val.min(), max_val.max()],
                [trend]
            ])
            
            features.append(window_features)
        
        return np.array(features)


class AttentionLayer:
    """
    Attention mechanism for temporal sequence modeling.
    Learns which time steps are most important.
    """
    
    def __init__(self, hidden_size: int = 64):
        """
        Initialize attention layer.
        
        Args:
            hidden_size (int): Size of attention hidden layer
        """
        self.hidden_size = hidden_size
        self.attention_weights = None
        
    def compute_attention(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention weights for sequences.
        
        Args:
            sequences (np.ndarray): Input sequences (n_samples, seq_len, features)
            
        Returns:
            tuple: Attended sequences, attention weights
        """
        # Simple attention: score each time step based on magnitude
        # In real implementation, this would use learned weights
        
        # Compute attention scores (simple: L2 norm of each time step)
        attention_scores = np.linalg.norm(sequences, axis=2)  # (n_samples, seq_len)
        
        # Softmax normalization
        attention_weights = self._softmax(attention_scores)
        
        # Apply attention
        attended = sequences * attention_weights[:, :, np.newaxis]
        
        self.attention_weights = attention_weights
        return attended, attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def get_keras_layer(self, name: str = 'attention'):
        """Get Keras attention layer."""
        if not TF_AVAILABLE:
            return None
        
        class KerasAttention(layers.Layer):
            def __init__(self, hidden_size=64, **kwargs):
                super().__init__(**kwargs)
                self.hidden_size = hidden_size
                
            def build(self, input_shape):
                self.W = self.add_weight(
                    shape=(input_shape[-1], self.hidden_size),
                    initializer='glorot_uniform',
                    trainable=True,
                    name='attention_W'
                )
                self.b = self.add_weight(
                    shape=(self.hidden_size,),
                    initializer='zeros',
                    trainable=True,
                    name='attention_b'
                )
                self.u = self.add_weight(
                    shape=(self.hidden_size, 1),
                    initializer='glorot_uniform',
                    trainable=True,
                    name='attention_u'
                )
                
            def call(self, inputs):
                # inputs: (batch, seq_len, features)
                score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
                attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
                attended = inputs * attention_weights
                return tf.reduce_sum(attended, axis=1), attention_weights
        
        return KerasAttention(hidden_size=self.hidden_size, name=name)


class LongSequenceModel:
    """
    Model for processing longer sequences with memory-efficient methods.
    Uses chunking and hierarchical processing.
    """
    
    def __init__(self, chunk_size: int = 100, overlap: int = 20):
        """
        Initialize long sequence model.
        
        Args:
            chunk_size (int): Size of each chunk
            overlap (int): Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_sequence(self, sequence: np.ndarray) -> List[np.ndarray]:
        """
        Split long sequence into overlapping chunks.
        
        Args:
            sequence (np.ndarray): Long sequence
            
        Returns:
            list: List of chunks
        """
        chunks = []
        stride = self.chunk_size - self.overlap
        
        for start in range(0, len(sequence) - self.chunk_size + 1, stride):
            end = start + self.chunk_size
            chunks.append(sequence[start:end])
        
        # Add final chunk if needed
        if end < len(sequence):
            chunks.append(sequence[-self.chunk_size:])
        
        return chunks
    
    def hierarchical_process(self, chunks: List[np.ndarray]) -> np.ndarray:
        """
        Process chunks hierarchically.
        
        Args:
            chunks (list): List of chunks
            
        Returns:
            np.ndarray: Aggregated representation
        """
        # Level 1: Process each chunk
        chunk_features = []
        for chunk in chunks:
            # Extract features from chunk
            features = np.array([
                np.mean(chunk, axis=0),
                np.std(chunk, axis=0),
                np.min(chunk, axis=0),
                np.max(chunk, axis=0)
            ]).flatten()
            chunk_features.append(features)
        
        chunk_features = np.array(chunk_features)
        
        # Level 2: Aggregate chunk features
        aggregated = np.array([
            np.mean(chunk_features, axis=0),
            np.std(chunk_features, axis=0),
            chunk_features[0],   # First chunk
            chunk_features[-1]   # Last chunk
        ]).flatten()
        
        return aggregated


class TemporalContextEnhancer:
    """
    Enhanced temporal context processing combining multiple techniques.
    """
    
    def __init__(self, 
                 window_size: int = 60,
                 stride: int = 10,
                 use_attention: bool = True,
                 use_hierarchical: bool = True):
        """
        Initialize temporal context enhancer.
        
        Args:
            window_size (int): Window size for sliding window
            stride (int): Stride for sliding window
            use_attention (bool): Enable attention mechanism
            use_hierarchical (bool): Enable hierarchical processing
        """
        self.window_processor = SlidingWindowProcessor(window_size, stride)
        self.attention = AttentionLayer() if use_attention else None
        self.long_model = LongSequenceModel() if use_hierarchical else None
        
        self.use_attention = use_attention
        self.use_hierarchical = use_hierarchical
        
    def process_sequence(self, data: np.ndarray, 
                         timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        Process time-series with enhanced temporal context.
        
        Args:
            data (np.ndarray): Time-series data
            timestamps (np.ndarray): Optional timestamps
            
        Returns:
            dict: Processed features and metadata
        """
        result = {}
        
        # Create sliding windows
        if timestamps is not None:
            windows, window_times = self.window_processor.create_windows(data, timestamps)
            result['window_times'] = window_times
        else:
            windows = self.window_processor.create_windows(data)
        
        result['windows'] = windows
        result['n_windows'] = len(windows)
        
        # Extract temporal features
        temporal_features = self.window_processor.extract_temporal_features(windows)
        result['temporal_features'] = temporal_features
        
        # Apply attention if enabled
        if self.use_attention and self.attention:
            attended, attention_weights = self.attention.compute_attention(windows)
            result['attended_features'] = attended
            result['attention_weights'] = attention_weights
        
        # Apply hierarchical processing if sequence is long
        if self.use_hierarchical and len(windows) > 100:
            chunks = self.long_model.chunk_sequence(windows)
            hierarchical_features = self.long_model.hierarchical_process(chunks)
            result['hierarchical_features'] = hierarchical_features
        
        return result
    
    def build_lstm_attention_model(self, 
                                   input_shape: Tuple,
                                   n_classes: int = 2) -> Optional[keras.Model]:
        """
        Build LSTM model with attention.
        
        Args:
            input_shape (tuple): Input shape (seq_len, features)
            n_classes (int): Number of output classes
            
        Returns:
            keras.Model: Model with attention
        """
        if not TF_AVAILABLE:
            print("TensorFlow not available")
            return None
        
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        lstm1 = layers.LSTM(128, return_sequences=True)(inputs)
        lstm2 = layers.LSTM(64, return_sequences=True)(lstm1)
        
        # Attention layer
        if self.attention:
            attention_layer = self.attention.get_keras_layer()
            attended, attention_weights = attention_layer(lstm2)
        else:
            attended = layers.GlobalAveragePooling1D()(lstm2)
        
        # Dense layers
        dense1 = layers.Dense(32, activation='relu')(attended)
        dense2 = layers.Dense(16, activation='relu')(dense1)
        outputs = layers.Dense(n_classes, activation='softmax')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model


if __name__ == "__main__":
    print("Testing Advanced Temporal Context Module...")
    
    # Generate sample time-series data
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    # Normal periodic pattern with noise
    t = np.linspace(0, 10*np.pi, n_samples)
    data = np.column_stack([
        np.sin(t) + np.random.randn(n_samples) * 0.1,
        np.cos(t) + np.random.randn(n_samples) * 0.1,
        np.sin(2*t) + np.random.randn(n_samples) * 0.1,
        np.random.randn(n_samples) * 0.5,
        np.random.randn(n_samples) * 0.5
    ])
    
    timestamps = np.arange(n_samples)
    
    print("\n" + "="*60)
    print("Test 1: Sliding Window Processing")
    print("="*60)
    
    window_proc = SlidingWindowProcessor(window_size=60, stride=10)
    windows, window_times = window_proc.create_windows(data, timestamps)
    
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Number of windows: {len(windows)}")
    print(f"Window shape: {windows[0].shape}")
    print(f"First window time range: {window_times[0]}")
    print(f"Last window time range: {window_times[-1]}")
    
    # Compute statistics
    stats = window_proc.compute_window_statistics(windows)
    print(f"\nWindow statistics computed:")
    print(f"  Mean shape: {stats['mean'].shape}")
    print(f"  Std shape: {stats['std'].shape}")
    
    print("\n" + "="*60)
    print("Test 2: Temporal Feature Extraction")
    print("="*60)
    
    temporal_features = window_proc.extract_temporal_features(windows)
    print(f"\nTemporal features shape: {temporal_features.shape}")
    print(f"Features per window: {temporal_features.shape[1]}")
    print(f"Sample features (first window): {temporal_features[0][:5]}")
    
    print("\n" + "="*60)
    print("Test 3: Attention Mechanism")
    print("="*60)
    
    attention = AttentionLayer(hidden_size=64)
    attended, attention_weights = attention.compute_attention(windows)
    
    print(f"\nAttended sequences shape: {attended.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Sample attention weights (first 10): {attention_weights[0][:10]}")
    print(f"Attention weights sum (should be ~1): {attention_weights[0].sum():.4f}")
    
    print("\n" + "="*60)
    print("Test 4: Long Sequence Processing")
    print("="*60)
    
    long_model = LongSequenceModel(chunk_size=100, overlap=20)
    
    # Create longer sequence
    long_data = np.random.randn(1000, 10)
    chunks = long_model.chunk_sequence(long_data)
    
    print(f"\nLong sequence shape: {long_data.shape}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk shape: {chunks[0].shape}")
    
    hierarchical_features = long_model.hierarchical_process(chunks)
    print(f"Hierarchical features shape: {hierarchical_features.shape}")
    
    print("\n" + "="*60)
    print("Test 5: Integrated Temporal Context Enhancer")
    print("="*60)
    
    enhancer = TemporalContextEnhancer(
        window_size=60,
        stride=10,
        use_attention=True,
        use_hierarchical=True
    )
    
    result = enhancer.process_sequence(data, timestamps)
    
    print(f"\nProcessing results:")
    print(f"  Number of windows: {result['n_windows']}")
    print(f"  Temporal features shape: {result['temporal_features'].shape}")
    print(f"  Attended features shape: {result['attended_features'].shape}")
    
    if 'hierarchical_features' in result:
        print(f"  Hierarchical features shape: {result['hierarchical_features'].shape}")
    
    print("\n" + "="*60)
    print("Test 6: LSTM with Attention Model")
    print("="*60)
    
    if TF_AVAILABLE:
        model = enhancer.build_lstm_attention_model(
            input_shape=(60, 5),
            n_classes=2
        )
        
        if model:
            print(f"\nModel created successfully")
            print(f"Total parameters: {model.count_params():,}")
            model.summary()
    else:
        print("\nTensorFlow not available, skipping model test")
    
    print("\n✓ Advanced temporal context module working!")
