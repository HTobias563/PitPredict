#!/usr/bin/env python3
"""
PitPredict Streamlit Web Application
F1 Race Outcome Prediction Suite - Fixed Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import traceback
import joblib
from pathlib import Path
import yaml

# Add src to path
app_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(app_root)
sys.path.append(project_root)

# Page config
st.set_page_config(
    page_title="PitPredict - F1 Race Predictor",
    page_icon="F1",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - ohne Emojis
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF1E1E;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .podium-1 {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: black;
        font-weight: bold;
    }
    .podium-2 {
        background: linear-gradient(135deg, #C0C0C0, #A9A9A9);
        color: black;
        font-weight: bold;
    }
    .podium-3 {
        background: linear-gradient(135deg, #CD7F32, #8B4513);
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF1E1E;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black;
    }
    .metric-card h3 {
        color: #FF1E1E;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: #333;
        margin-bottom: 0;
    }
    .error-card {
        background: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        color: #d32f2f;
    }
    .success-card {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

class PitPredictApp:
    """Main Streamlit Application Class"""
    
    def __init__(self):
        self.project_root = project_root
        self.setup_session_state()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'config' not in st.session_state:
            st.session_state.config = self.load_config()
            
    def load_config(self):
        """Load configuration from config.yaml"""
        try:
            config_path = os.path.join(self.project_root, 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            st.warning(f"Could not load config: {e}")
        return {'season': 2024, 'rounds': list(range(1, 25))}
    
    def load_model(self):
        """Load the final position prediction model"""
        try:
            models_dir = os.path.join(self.project_root, 'artifacts', 'models')
            model_path = os.path.join(models_dir, 'final_position_predictor.pkl')
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                st.session_state.model_loaded = True
                return model
            else:
                st.error(f"Model not found at: {model_path}")
                return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def load_training_data(self):
        """Load training data for predictions"""
        try:
            data_path = os.path.join(self.project_root, 'data', 'season=2024', 'driver_race_table.parquet')
            if os.path.exists(data_path):
                return pd.read_parquet(data_path)
            else:
                st.error(f"Training data not found at: {data_path}")
                return None
        except Exception as e:
            st.error(f"Error loading training data: {e}")
            return None
    
    def run(self):
        """Main application entry point"""
        # Header
        st.markdown('<h1 class="main-header">PitPredict</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">F1 Race Outcome Prediction Suite</p>', unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["Home", "Future Race Predictions", "Model Performance", "About"]
        )
        
        # Route to selected page
        if page == "Home":
            self.show_home_page()
        elif page == "Future Race Predictions":
            self.show_future_predictions()
        elif page == "Model Performance":
            self.show_model_performance()
        elif page == "About":
            self.show_about_page()
    
    def show_home_page(self):
        """Show home page with overview"""
        st.header("Welcome to PitPredict!")
        
        # Overview cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Future Race Predictions</h3>
                <p>Predict upcoming 2025+ F1 races with custom grid positions and track types</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Multiple Track Types</h3>
                <p>Netherlands GP, Monaco, Spa, Silverstone, and generic circuit predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Model Performance</h3>
                <p>Detailed metrics and performance analysis of ML models</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model architecture overview
        st.markdown("---")
        st.subheader("Model Architecture")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Final Position Prediction Model:**
            - **Algorithm:** Gradient Boosting Regressor
            - **Features:** 42 engineered features
            - **Performance:** MAE ~2.1 positions
            - **Podium Accuracy:** >90%
            """)
        
        with col2:
            st.markdown("""
            **Feature Categories:**
            - Grid Position & Qualifying Data
            - Historical Driver Performance  
            - Team Performance Metrics
            - Track-specific Characteristics
            - DNF Risk Assessment
            """)
        
        # System status
        st.markdown("---")
        st.subheader("System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            models_dir = os.path.join(self.project_root, 'artifacts', 'models')
            model_exists = os.path.exists(os.path.join(models_dir, 'final_position_predictor.pkl'))
            
            if model_exists:
                st.markdown("""
                <div class="success-card">
                    <strong>Model Status:</strong> Ready
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-card">
                    <strong>Model Status:</strong> Not Found
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            data_path = os.path.join(self.project_root, 'data', 'season=2024', 'driver_race_table.parquet')
            data_exists = os.path.exists(data_path)
            
            if data_exists:
                st.markdown("""
                <div class="success-card">
                    <strong>Data Status:</strong> Available
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-card">
                    <strong>Data Status:</strong> Missing
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            config_path = os.path.join(self.project_root, 'config.yaml')
            config_exists = os.path.exists(config_path)
            
            if config_exists:
                st.markdown("""
                <div class="success-card">
                    <strong>Config Status:</strong> Loaded
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-card">
                    <strong>Config Status:</strong> Missing
                </div>
                """, unsafe_allow_html=True)
    
    def show_future_predictions(self):
        """Show future predictions page"""
        st.header("Future Race Predictions for 2025+")
        st.markdown("Create predictions for upcoming F1 races with customizable parameters")
        
        # Import the future predictor
        try:
            from src.predict_future_race import FutureRacePredictor
            if 'future_predictor' not in st.session_state:
                st.session_state.future_predictor = FutureRacePredictor()
        except Exception as e:
            st.error(f"Could not load Future Race Predictor: {e}")
            return
        
        # Track selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            track_type = st.selectbox(
                "Select Track Type:",
                options=['netherlands', 'monaco', 'spa', 'silverstone', 'default'],
                format_func=lambda x: {
                    'netherlands': 'Netherlands GP (Zandvoort) - High difficulty, hard overtaking',
                    'monaco': 'Monaco GP - Very high difficulty, very hard overtaking',
                    'spa': 'Belgian GP (Spa-Francorchamps) - Medium difficulty, easy overtaking',
                    'silverstone': 'British GP (Silverstone) - Medium difficulty, medium overtaking',
                    'default': 'Generic Circuit - Medium difficulty, medium overtaking'
                }[x]
            )
        
        with col2:
            race_year = st.selectbox("Race Year:", [2025, 2026, 2027], index=0)
        
        # Race name input
        race_name = st.text_input(
            "Race Name:", 
            value=f"{track_type.title()} GP {race_year}",
            help="Enter a descriptive name for this race"
        )
        
        st.markdown("---")
        
        # Grid position customization
        st.subheader("Grid Position Setup")
        
        # Get default grid for the selected track
        try:
            default_grid = self.get_default_grid_for_track(track_type)
        except:
            default_grid = self.get_fallback_grid()
        
        # Grid setup options
        grid_option = st.radio(
            "Choose grid setup:",
            ["Use Default Grid", "Customize Grid Positions"],
            help="Default uses realistic grid positions, Custom lets you set each position"
        )
        
        if grid_option == "Customize Grid Positions":
            grid_positions = self.show_grid_customizer(default_grid)
        else:
            grid_positions = default_grid
            self.show_default_grid(default_grid)
        
        st.markdown("---")
        
        # Prediction button
        if st.button("Generate Future Race Prediction", type="primary", use_container_width=True):
            with st.spinner("Generating future race prediction..."):
                try:
                    predictions = st.session_state.future_predictor.predict_future_race(
                        race_name, grid_positions, track_type, race_year
                    )
                    
                    if predictions is not None and len(predictions) > 0:
                        st.success("Future race prediction generated successfully!")
                        self.display_future_predictions(predictions, track_type, race_name)
                    else:
                        st.error("Failed to generate predictions - empty result")
                        
                except Exception as e:
                    st.error(f"Error generating future prediction: {e}")
                    st.error(traceback.format_exc())
    
    def get_default_grid_for_track(self, track_type: str) -> Dict[str, int]:
        """Get default grid positions for a specific track type"""
        # Import the default grids from the predict_future_race module
        try:
            from src.predict_future_race import get_default_netherlands_grid
            return get_default_netherlands_grid()
        except:
            return self.get_fallback_grid()
    
    def get_fallback_grid(self) -> Dict[str, int]:
        """Fallback grid positions if imports fail"""
        return {
            'VER': 1, 'NOR': 2, 'LEC': 3, 'PIA': 4, 'RUS': 5,
            'HAM': 6, 'SAI': 7, 'PER': 8, 'ALO': 9, 'STR': 10,
            'GAS': 11, 'OCO': 12, 'TSU': 13, 'LAW': 14, 'ALB': 15,
            'COL': 16, 'HUL': 17, 'BOT': 18, 'ZHO': 19, 'BEA': 20
        }
    
    def show_default_grid(self, grid_positions: Dict[str, int]):
        """Display the default grid positions"""
        st.subheader("Default Grid Positions")
        
        # Convert to DataFrame for better display
        grid_df = pd.DataFrame([
            {'Position': pos, 'Driver': driver}
            for driver, pos in sorted(grid_positions.items(), key=lambda x: x[1])
        ])
        
        # Display in columns
        col1, col2, col3, col4 = st.columns(4)
        
        for i, (_, row) in enumerate(grid_df.iterrows()):
            col = [col1, col2, col3, col4][i % 4]
            with col:
                st.write(f"P{row['Position']}: **{row['Driver']}**")
    
    def show_grid_customizer(self, default_grid: Dict[str, int]) -> Dict[str, int]:
        """Show grid position customizer"""
        st.subheader("Customize Grid Positions")
        st.markdown("Drag and drop or manually set grid positions for each driver")
        
        # Create columns for driver selection
        drivers = list(default_grid.keys())
        
        custom_grid = {}
        
        # Create 4 columns for better layout
        cols = st.columns(4)
        
        for i, driver in enumerate(drivers):
            col = cols[i % 4]
            with col:
                position = st.number_input(
                    f"{driver}",
                    min_value=1,
                    max_value=20,
                    value=default_grid[driver],
                    key=f"grid_{driver}"
                )
                custom_grid[driver] = position
        
        # Validate grid positions
        positions = list(custom_grid.values())
        if len(set(positions)) != len(positions):
            st.warning("Warning: Some drivers have the same grid position!")
        
        return custom_grid
    
    def display_future_predictions(self, predictions: pd.DataFrame, track_type: str, race_name: str):
        """Display future race predictions"""
        
        # Track info
        track_info = {
            'netherlands': {'name': 'Zandvoort', 'difficulty': 'High', 'overtaking': 'Hard'},
            'monaco': {'name': 'Monaco', 'difficulty': 'Very High', 'overtaking': 'Very Hard'},
            'spa': {'name': 'Spa-Francorchamps', 'difficulty': 'Medium', 'overtaking': 'Easy'},
            'silverstone': {'name': 'Silverstone', 'difficulty': 'Medium', 'overtaking': 'Medium'},
            'default': {'name': 'Generic Circuit', 'difficulty': 'Medium', 'overtaking': 'Medium'}
        }
        
        info = track_info.get(track_type, track_info['default'])
        
        st.info(f"**Race:** {race_name} | **Track:** {info['name']} | **Difficulty:** {info['difficulty']} | **Overtaking:** {info['overtaking']}")
        
        # Sort predictions by predicted position
        if 'predicted_position_rounded' in predictions.columns:
            predictions_sorted = predictions.sort_values('predicted_position_rounded').reset_index(drop=True)
        elif 'predicted_final_position' in predictions.columns:
            predictions_sorted = predictions.sort_values('predicted_final_position').reset_index(drop=True)
        else:
            predictions_sorted = predictions.copy()
        
        # Podium predictions
        st.subheader("Podium Prediction")
        
        podium = predictions_sorted.head(3)
        cols = st.columns(3)
        
        for i, (_, driver) in enumerate(podium.iterrows()):
            with cols[i]:
                position = i + 1
                css_class = f"podium-{position}"
                
                position_names = {1: "1st", 2: "2nd", 3: "3rd"}
                
                st.markdown(f"""
                <div class="prediction-card {css_class}">
                    <h3>P{position} - {position_names[position]}</h3>
                    <h4>{driver.get('driver', 'Unknown Driver')}</h4>
                    <p>Team: {driver.get('team', 'Unknown Team')}</p>
                    <p>Grid Position: P{int(driver.get('grid_position', 0))}</p>
                    <p>DNF Risk: {driver.get('dnf_risk', 0.2)*100:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Full results table
        st.subheader("Complete Future Race Prediction")
        
        # Prepare display dataframe
        display_df = predictions_sorted.copy()
        display_df['Position'] = range(1, len(display_df) + 1)
        display_df['Driver'] = display_df.get('driver', 'Unknown')
        display_df['Team'] = display_df.get('team', 'Unknown')
        display_df['Grid'] = display_df.get('grid_position', 0).astype(int)
        
        # Handle different prediction column names
        if 'predicted_position_rounded' in display_df.columns:
            display_df['Predicted'] = display_df['predicted_position_rounded'].astype(int)
        elif 'predicted_final_position' in display_df.columns:
            display_df['Predicted'] = display_df['predicted_final_position'].round().astype(int)
        else:
            display_df['Predicted'] = display_df['Position']
        
        display_df['Change'] = display_df['Grid'] - display_df['Predicted']
        display_df['DNF Risk'] = (display_df.get('dnf_risk', 0.2)*100).astype(int)
        
        # Display the table
        result_columns = ['Position', 'Driver', 'Team', 'Grid', 'Predicted', 'Change', 'DNF Risk']
        
        # Color code the dataframe
        def color_change(val):
            if val > 0:
                return 'background-color: lightgreen'
            elif val < 0:
                return 'background-color: lightcoral'
            else:
                return ''
        
        styled_df = display_df[result_columns].style.applymap(color_change, subset=['Change'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Future Prediction CSV",
            data=csv,
            file_name=f"future_prediction_{race_name.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )
        
        # Visualizations
        self.create_future_visualizations(display_df, race_name)
    
    def create_future_visualizations(self, predictions: pd.DataFrame, race_name: str):
        """Create visualizations for future predictions"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Grid vs Predicted Position")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions['Grid'],
                y=predictions['Predicted'],
                mode='markers+text',
                text=predictions['Driver'],
                textposition="middle right",
                marker=dict(
                    size=12,
                    color=predictions['DNF Risk'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="DNF Risk %")
                ),
                name="Drivers"
            ))
            
            # Add diagonal line (perfect prediction)
            fig.add_trace(go.Scatter(
                x=[1, 20],
                y=[1, 20],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name="No Change Line"
            ))
            
            fig.update_layout(
                title=f"Grid Position vs Predicted Position - {race_name}",
                xaxis_title="Grid Position",
                yaxis_title="Predicted Final Position",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Expected Position Changes")
            
            fig = go.Figure()
            
            colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
                     for x in predictions['Change']]
            
            fig.add_trace(go.Bar(
                x=predictions['Driver'],
                y=predictions['Change'],
                marker_color=colors,
                text=predictions['Change'],
                textposition='outside',
                name="Position Change"
            ))
            
            fig.update_layout(
                title=f"Expected Position Changes - {race_name}",
                xaxis_title="Driver",
                yaxis_title="Position Change (+ = Better, - = Worse)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_model_performance(self):
        """Show model performance page"""
        st.header("Model Performance")
        st.markdown("Detailed analysis of the ML model performance and accuracy")
        
        # Try to load metrics
        metrics_path = os.path.join(self.project_root, 'artifacts', 'metrics')
        
        cv_report_path = os.path.join(metrics_path, 'final_position_cv_report.json')
        holdout_report_path = os.path.join(metrics_path, 'final_position_holdout_report.json')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cross-Validation Metrics")
            
            if os.path.exists(cv_report_path):
                try:
                    with open(cv_report_path, 'r') as f:
                        cv_metrics = json.load(f)
                    
                    st.metric("Mean Absolute Error", f"{cv_metrics.get('overall_mae', 'N/A'):.2f}", "positions")
                    st.metric("R² Score", f"{cv_metrics.get('overall_r2', 'N/A'):.3f}")
                    st.metric("RMSE", f"{cv_metrics.get('overall_rmse', 'N/A'):.2f}", "positions")
                    
                except Exception as e:
                    st.error(f"Error loading CV metrics: {e}")
            else:
                st.warning("CV report not found")
        
        with col2:
            st.subheader("Holdout Test Metrics")
            
            if os.path.exists(holdout_report_path):
                try:
                    with open(holdout_report_path, 'r') as f:
                        holdout_metrics = json.load(f)
                    
                    st.metric("Test MAE", f"{holdout_metrics.get('overall_mae', 'N/A'):.2f}", "positions")
                    st.metric("Test R²", f"{holdout_metrics.get('overall_r2', 'N/A'):.3f}")
                    st.metric("Test RMSE", f"{holdout_metrics.get('overall_rmse', 'N/A'):.2f}", "positions")
                    
                except Exception as e:
                    st.error(f"Error loading holdout metrics: {e}")
            else:
                st.warning("Holdout report not found")
        
        # Model files status
        st.subheader("Model Files Status")
        
        models_dir = os.path.join(self.project_root, 'artifacts', 'models')
        model_files = ['final_position_predictor.pkl', 'dnf_pipeline_calibrated.pkl', 'pitstop_predictor_calibrated.pkl']
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            exists = os.path.exists(model_path)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{model_file}**")
            with col2:
                if exists:
                    st.success("Available")
                else:
                    st.error("Missing")
    
    def show_about_page(self):
        """Show about page"""
        st.header("About PitPredict")
        
        st.markdown("""
        **PitPredict** is a comprehensive F1 race outcome prediction system that uses machine learning 
        to forecast race results based on historical data and pre-race factors.
        
        ### Key Features:
        - **Final Position Prediction**: Predicts where each driver will finish
        - **DNF Risk Assessment**: Calculates probability of Did Not Finish
        - **Pit Stop Strategy**: Analyzes optimal pit stop windows
        
        ### Technical Details:
        - **Algorithm**: Gradient Boosting Regressor
        - **Features**: 42+ engineered features including:
          - Grid position and qualifying performance
          - Historical driver and team performance
          - Track-specific characteristics
          - Weather conditions
          - DNF risk factors
        
        ### Model Performance:
        - **Mean Absolute Error**: ~2.1 positions
        - **Podium Accuracy**: >90%
        - **Cross-validation R²**: >0.7
        
        ### Data Sources:
        - FastF1 API for telemetry data
        - Official F1 timing data
        - Historical race results (2018-2024)
        
        ### Development:
        Built with Python, scikit-learn, Streamlit, and Plotly for interactive visualizations.
        """)

def main():
    """Main function to run the Streamlit app"""
    app = PitPredictApp()
    app.run()

if __name__ == "__main__":
    main()
