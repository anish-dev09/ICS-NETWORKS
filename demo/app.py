"""
ICS Intrusion Detection System - Demo Application
Production-ready Streamlit dashboard for real-time anomaly detection
Integrates CNN, XGBoost, and Random Forest models trained on HAI dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import mock data generator (using mock data for demo)
try:
    from demo.mock_hai_data import generate_mock_hai_data
    MOCK_DATA_AVAILABLE = True
except ImportError as e:
    print(f"Mock data import error: {e}")
    MOCK_DATA_AVAILABLE = False

# Import project modules
try:
    from src.features.feature_engineering import FeatureEngineer
    from src.data.sequence_generator import SequenceGenerator
except ImportError as e:
    # Fallback if imports fail
    print(f"Import error: {e}")  # Debug info
    FeatureEngineer = None  # type: ignore
    SequenceGenerator = None  # type: ignore

# Page configuration
st.set_page_config(
    page_title="ICS Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .safe-box {
        background-color: #00cc00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== MODEL LOADING FUNCTIONS =====
@st.cache_resource
def load_cnn_model():
    """Load trained CNN model with error handling"""
    try:
        import tensorflow as tf
        model_path = project_root / "results" / "models" / "cnn1d_detector.keras"
        if model_path.exists():
            model = tf.keras.models.load_model(model_path)  # type: ignore
            st.sidebar.success("✅ CNN model loaded")
            return model
        else:
            st.sidebar.warning("⚠️ CNN model not found")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ CNN loading error: {str(e)}")
        return None

@st.cache_resource
def load_ml_models():
    """Load trained ML models (XGBoost and Random Forest)"""
    models = {}
    try:
        # Import joblib for loading models (models were saved with joblib, not pickle)
        import joblib
        
        # Try loading XGBoost
        xgb_path = project_root / "results" / "models" / "xgboost_detector.pkl"
        if xgb_path.exists():
            xgb_dict = joblib.load(xgb_path)
            models['xgboost'] = xgb_dict['model'] if isinstance(xgb_dict, dict) else xgb_dict
            st.sidebar.success("✅ XGBoost model loaded")
        
        # Try loading Random Forest
        rf_path = project_root / "results" / "models" / "random_forest_detector.pkl"
        if rf_path.exists():
            rf_dict = joblib.load(rf_path)
            models['random_forest'] = rf_dict['model'] if isinstance(rf_dict, dict) else rf_dict
            st.sidebar.success("✅ Random Forest model loaded")
        
        if not models:
            st.sidebar.warning("⚠️ No ML models found")
    except ImportError as ie:
        st.sidebar.error(f"❌ Import error: {str(ie)}")
    except Exception as e:
        st.sidebar.error(f"❌ ML loading error: {str(e)}")
    
    return models

@st.cache_data
def load_test_data():
    """Load mock HAI test dataset"""
    try:
        if not MOCK_DATA_AVAILABLE:
            st.sidebar.error("❌ Mock data generator not available")
            return None
            
        # Generate mock HAI data
        st.sidebar.info("🔄 Generating mock HAI data...")
        df = generate_mock_hai_data(n_samples=50000, attack_ratio=0.3, random_state=42)
        
        if df is not None and len(df) > 0:
            st.sidebar.success(f"✅ Generated {len(df):,} mock samples (HAI-style)")
            return df
        else:
            st.sidebar.warning("⚠️ No data generated")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Data generation error: {str(e)}")
        return None

@st.cache_data
def load_performance_metrics():
    """Load saved performance metrics"""
    try:
        metrics_path = project_root / "results" / "metrics" / "all_models_comparison.csv"
        if metrics_path.exists():
            return pd.read_csv(metrics_path)
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Metrics loading error: {str(e)}")
        return None

# ===== PREDICTION FUNCTIONS =====
def predict_with_ml(model, data: pd.DataFrame, feature_cols: list) -> Tuple[int, float]:
    """Make prediction with ML model (XGBoost/Random Forest)"""
    try:
        X = data[feature_cols].values.reshape(1, -1)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else 0.5
        return int(prediction), float(probability)
    except Exception as e:
        st.error(f"ML Prediction error: {str(e)}")
        return 0, 0.0

def predict_with_cnn(model, data: np.ndarray) -> Tuple[int, float]:
    """Make prediction with CNN model"""
    try:
        # data should be shape (1, timesteps, features)
        prediction_prob = model.predict(data, verbose=0)[0, 0]
        prediction = 1 if prediction_prob > 0.5 else 0
        return int(prediction), float(prediction_prob)
    except Exception as e:
        st.error(f"CNN Prediction error: {str(e)}")
        return 0, 0.0

def create_gauge_chart(value: float, title: str) -> go.Figure:
    """Create gauge chart for anomaly probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# ===== MAIN APPLICATION =====
def main():
    # ===== SESSION STATE INITIALIZATION (Must be first in main()) =====
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'alert_threshold' not in st.session_state:
        st.session_state.alert_threshold = 0.7
    
    # Header
    st.markdown('<p class="main-header">🛡️ ICS Intrusion Detection System</p>', unsafe_allow_html=True)
    st.markdown("**Real-time anomaly detection for Industrial Control Systems using Deep Learning & ML**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("### Model Loading")
    
    # Add cache clear button
    if st.sidebar.button("🔄 Clear Cache & Reload"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    # Load models
    cnn_model = load_cnn_model()
    ml_models = load_ml_models()
    test_data = load_test_data()
    metrics_df = load_performance_metrics()
    
    # Model selection
    available_models = []
    if cnn_model is not None:
        available_models.append("CNN (1D Convolutional)")
    if 'xgboost' in ml_models:
        available_models.append("XGBoost")
    if 'random_forest' in ml_models:
        available_models.append("Random Forest")
    
    if not available_models:
        st.error("❌ No models loaded! Please train models first.")
        st.stop()
    
    st.sidebar.markdown("### Model Selection")
    selected_model = st.sidebar.selectbox("Choose detection model:", available_models)
    
    st.sidebar.markdown("### Alert Settings")
    st.session_state.alert_threshold = st.sidebar.slider(
        "Alert Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Probability threshold for triggering alerts"
    )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Real-Time Detection",
        "📊 Model Comparison",
        "📈 System Analytics",
        "📜 Detection History"
    ])
    
    # ===== TAB 1: Real-Time Detection =====
    with tab1:
        st.header("Real-Time Intrusion Detection")
        
        if test_data is None:
            st.warning("No test data available for detection")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📡 Sensor Data Input")
                
                # Sample selection
                sample_idx = st.number_input(
                    "Select sample index:",
                    min_value=0,
                    max_value=len(test_data)-1,
                    value=0,
                    step=1
                )
                
                # Get sample
                sample = test_data.iloc[sample_idx]
                actual_label = sample.get('attack', 0)
                
                # Display sensor values
                sensor_cols = [col for col in test_data.columns if col not in ['attack', 'timestamp']]
                sensor_data = sample[sensor_cols]
                
                # Show sensor values in expandable section
                with st.expander("📊 View Sensor Values (83 sensors)"):
                    # Display in 4 columns
                    cols = st.columns(4)
                    for idx, (sensor, value) in enumerate(sensor_data.items()):
                        with cols[idx % 4]:
                            # Convert to float to avoid format errors
                            try:
                                val = float(value)
                                st.metric(str(sensor), f"{val:.4f}")
                            except (ValueError, TypeError):
                                st.metric(str(sensor), str(value))
                
                # Predict button
                if st.button("🔍 Run Detection", type="primary"):
                    with st.spinner("Analyzing sensor data..."):
                        # Initialize variables
                        prediction = 0
                        probability = 0.0
                        
                        # Make prediction based on selected model
                        if selected_model == "CNN (1D Convolutional)":
                            # For CNN, need to create sequence
                            # For demo, we'll use a simplified approach
                            st.info("CNN requires sequential data. Using single sample approximation.")
                            # Create pseudo-sequence by repeating sample
                            sequence = np.repeat(sensor_data.values.reshape(1, -1), 60, axis=0)
                            sequence = sequence.reshape(1, 60, len(sensor_data))
                            prediction, probability = predict_with_cnn(cnn_model, sequence)
                        
                        elif selected_model == "XGBoost":
                            # Feature engineering would normally be applied here
                            prediction, probability = predict_with_ml(
                                ml_models['xgboost'],
                                sample.to_frame().T,
                                sensor_cols
                            )
                        
                        elif selected_model == "Random Forest":
                            prediction, probability = predict_with_ml(
                                ml_models['random_forest'],
                                sample.to_frame().T,
                                sensor_cols
                            )
                        
                        # Store in history
                        st.session_state.detection_history.append({
                            'timestamp': datetime.now(),
                            'sample_idx': sample_idx,
                            'model': selected_model,
                            'prediction': prediction,
                            'probability': probability,
                            'actual': actual_label
                        })
                        
                        # Display results
                        st.success("✅ Detection Complete!")
            
            with col2:
                st.subheader("🎯 Detection Results")
                
                if st.session_state.detection_history:
                    latest = st.session_state.detection_history[-1]
                    
                    # Result display
                    if latest['prediction'] == 1:
                        st.markdown('<div class="alert-box"><h3>⚠️ ATTACK DETECTED</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="safe-box"><h3>✅ NORMAL OPERATION</h3></div>', unsafe_allow_html=True)
                    
                    # Metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Prediction", "Attack" if latest['prediction'] == 1 else "Normal")
                    with col_b:
                        st.metric("Actual", "Attack" if latest['actual'] == 1 else "Normal")
                    
                    # Gauge chart
                    st.plotly_chart(
                        create_gauge_chart(latest['probability'], "Anomaly Probability"),
                        use_container_width=True
                    )
                    
                    # Model info
                    st.info(f"**Model:** {latest['model']}")
                    st.info(f"**Threshold:** {st.session_state.alert_threshold:.2f}")
                    
                    # Alert status
                    if latest['probability'] >= st.session_state.alert_threshold:
                        st.error("🚨 Alert triggered! Probability exceeds threshold.")
                    else:
                        st.success("✅ No alert. System operating normally.")
    
    # ===== TAB 2: Model Comparison =====
    with tab2:
        st.header("📊 Model Performance Comparison")
        
        if metrics_df is not None:
            # Display metrics table
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df, use_container_width=True)
            
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig_acc = px.bar(
                    metrics_df,
                    x='Model',  # Use capital M to match CSV
                    y='Accuracy',  # Use capital A to match CSV
                    title='Model Accuracy Comparison',
                    labels={'Accuracy': 'Accuracy (%)', 'Model': 'Model'},
                    color='Accuracy',
                    color_continuous_scale='blues'
                )
                fig_acc.update_layout(showlegend=False)
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                # F1 Score comparison
                fig_f1 = px.bar(
                    metrics_df,
                    x='Model',  # Use capital M to match CSV
                    y='F1-Score',  # Match exact column name from CSV
                    title='Model F1 Score Comparison',
                    labels={'F1-Score': 'F1 Score', 'Model': 'Model'},
                    color='F1-Score',
                    color_continuous_scale='greens'
                )
                fig_f1.update_layout(showlegend=False)
                st.plotly_chart(fig_f1, use_container_width=True)
            
            # Precision vs Recall
            st.subheader("Precision vs Recall Trade-off")
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=metrics_df['Recall'],  # Use capital R to match CSV
                y=metrics_df['Precision'],  # Use capital P to match CSV
                mode='markers+text',
                text=metrics_df['Model'],  # Use capital M to match CSV
                textposition='top center',
                marker=dict(size=15, color=metrics_df['F1-Score'], colorscale='Viridis', showscale=True),
                name='Models'
            ))
            fig_pr.update_layout(
                xaxis_title='Recall',
                yaxis_title='Precision',
                title='Precision vs Recall (size=F1 Score)',
                hovermode='closest'
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.warning("No performance metrics available. Train models to generate metrics.")
    
    # ===== TAB 3: System Analytics =====
    with tab3:
        st.header("📈 System Monitoring & Analytics")
        
        if test_data is not None:
            # Dataset overview
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", f"{len(test_data):,}")
            with col2:
                attack_count = test_data['attack'].sum() if 'attack' in test_data.columns else 0
                st.metric("Attack Samples", f"{attack_count:,}")
            with col3:
                normal_count = len(test_data) - attack_count
                st.metric("Normal Samples", f"{normal_count:,}")
            with col4:
                attack_ratio = (attack_count / len(test_data) * 100) if len(test_data) > 0 else 0
                st.metric("Attack Ratio", f"{attack_ratio:.2f}%")
            
            # Attack distribution
            if 'attack' in test_data.columns:
                st.subheader("Attack vs Normal Distribution")
                attack_dist = test_data['attack'].value_counts()
                fig_dist = px.pie(
                    values=attack_dist.values,
                    names=['Normal', 'Attack'],
                    title='Class Distribution',
                    color_discrete_sequence=['lightgreen', 'lightcoral']
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Sensor correlation heatmap (sample)
            st.subheader("Sensor Correlation Analysis")
            sensor_cols = [col for col in test_data.columns if col not in ['attack', 'timestamp']]
            
            # Sample first 20 sensors for visualization
            sample_sensors = sensor_cols[:20]
            corr_matrix = test_data[sample_sensors].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title='Sensor Correlation Heatmap (First 20 Sensors)',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Time series visualization
            st.subheader("Sensor Time Series (Sample)")
            num_samples = st.slider("Number of samples to display:", 100, 1000, 500)
            selected_sensors = st.multiselect(
                "Select sensors to visualize:",
                sensor_cols[:10],  # Limit to first 10 for performance
                default=sensor_cols[:3]
            )
            
            if selected_sensors:
                plot_data = test_data.head(num_samples)
                fig_ts = px.line(
                    plot_data,
                    y=selected_sensors,
                    title=f'Sensor Values Over Time (First {num_samples} samples)',
                    labels={'value': 'Sensor Value', 'variable': 'Sensor'}
                )
                st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.warning("No data available for analytics")
    
    # ===== TAB 4: Detection History =====
    with tab4:
        st.header("📜 Detection History & Alerts")
        
        if st.session_state.detection_history:
            # Summary metrics
            history_df = pd.DataFrame(st.session_state.detection_history)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Detections", len(history_df))
            with col2:
                alerts = (history_df['probability'] >= st.session_state.alert_threshold).sum()
                st.metric("Alerts Triggered", alerts)
            with col3:
                correct = (history_df['prediction'] == history_df['actual']).sum()
                accuracy = (correct / len(history_df) * 100) if len(history_df) > 0 else 0
                st.metric("Accuracy", f"{accuracy:.2f}%")
            with col4:
                avg_prob = history_df['probability'].mean()
                st.metric("Avg Probability", f"{avg_prob:.2f}")
            
            # History table
            st.subheader("Detection Log")
            display_df = history_df.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['prediction'] = display_df['prediction'].map({0: 'Normal', 1: 'Attack'})
            display_df['actual'] = display_df['actual'].map({0: 'Normal', 1: 'Attack'})
            display_df['probability'] = display_df['probability'].round(4)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Probability distribution
            st.subheader("Probability Distribution")
            fig_hist = px.histogram(
                history_df,
                x='probability',
                nbins=20,
                title='Distribution of Anomaly Probabilities',
                labels={'probability': 'Anomaly Probability'},
                color_discrete_sequence=['steelblue']
            )
            fig_hist.add_vline(
                x=st.session_state.alert_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Alert Threshold"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.detection_history = []
                st.rerun()
        else:
            st.info("No detection history yet. Run detections in the Real-Time Detection tab.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**ICS Intrusion Detection System** | "
        "Powered by TensorFlow, XGBoost, and Streamlit | "
        "HAI Dataset (21.03)"
    )

if __name__ == "__main__":
    main()
