#!/usr/bin/env python3
"""
PitPredict Streamlit Web Application
F1 Race Outcome Prediction Suite
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

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

class PitPredictApp:
    """Main Streamlit Application Class"""
    
    def __init__(self):
        self.setup_session_state()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'future_predictor' not in st.session_state:
            st.session_state.future_predictor = None
        if 'available_races' not in st.session_state:
            st.session_state.available_races = self.get_available_races()
            
    def get_available_races(self) -> List[str]:
        """Get list of available 2024 race IDs"""
        try:
            # Read config for available races
            config_path = os.path.join(project_root, 'config.yaml')
            if os.path.exists(config_path):
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                rounds = config.get('rounds', range(1, 25))
                season = config.get('season', 2024)
                return [f"{season}_{round:02d}" for round in rounds]
            else:
                return [f"2024_{i:02d}" for i in range(1, 25)]
        except Exception:
            return [f"2024_{i:02d}" for i in range(1, 25)]
    
    def run(self):
        """Main application entry point"""
        # Header
        st.markdown('<h1 class="main-header">🏁 PitPredict</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">F1 Race Outcome Prediction Suite</p>', unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("🏎️ Navigation")
        page = st.sidebar.selectbox(
            "Wähle eine Seite:",
            ["🏠 Home", "🎯 2024 Race Predictions", "🚀 Future Predictions (2025)", "📊 Model Performance", "ℹ️ About"]
        )
        
        # Route to selected page
        if page == "🏠 Home":
            self.show_home_page()
        elif page == "🎯 2024 Race Predictions":
            self.show_2024_predictions()
        elif page == "🚀 Future Predictions (2025)":
            self.show_future_predictions()
        elif page == "📊 Model Performance":
            self.show_model_performance()
        elif page == "ℹ️ About":
            self.show_about_page()
    
    def show_home_page(self):
        """Show home page with overview"""
        st.header("Willkommen bei PitPredict! 🏁")
        
        # Overview cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 2024 Predictions</h3>
                <p>Vorhersagen für alle 2024 F1 Rennen mit echten Daten und Grid-Positionen</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🚀 Future Predictions</h3>
                <p>Hypothetische 2025-Vorhersagen mit benutzerdefinierten Grid-Positionen</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Model Performance</h3>
                <p>Detaillierte Metriken und Leistungsanalyse der ML-Modelle</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model architecture overview
        st.markdown("---")
        st.subheader("🤖 Model Architecture")
        
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
        
        # Quick stats
        if st.button("🔄 Lade aktuelle Statistiken"):
            try:
                # Load model metrics
                metrics_path = os.path.join(project_root, 'artifacts/metrics/final_position_cv_report.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    
                    st.success("✅ Modell-Statistiken geladen!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{metrics.get('overall_mae', 0):.2f}", "positions")
                    with col2:
                        st.metric("R² Score", f"{metrics.get('overall_r2', 0):.3f}")
                    with col3:
                        st.metric("Podium Accuracy", f"{metrics.get('overall_podium_accuracy', 0)*100:.1f}%")
                    with col4:
                        st.metric("Training Samples", f"{metrics.get('n_samples', 0):,}")
                        
                else:
                    st.warning("⚠️ Modell-Metriken nicht gefunden. Trainiere das Modell zuerst.")
            except Exception as e:
                st.error(f"❌ Fehler beim Laden der Statistiken: {e}")
    
    def show_2024_predictions(self):
        """Show 2024 race predictions page"""
        st.header("🎯 2024 F1 Race Predictions")
        st.markdown("Vorhersagen für echte 2024 F1-Rennen basierend auf tatsächlichen Grid-Positionen")
        
        # Race selection
        col1, col2 = st.columns([2, 1])
        with col1:
            race_id = st.selectbox(
                "📅 Wähle ein Rennen:",
                options=st.session_state.available_races,
                format_func=lambda x: f"Round {x.split('_')[1]} - {x}",
                key="race_selector_2024"
            )
        
        with col2:
            predict_button = st.button("🎯 Vorhersage starten", key="predict_2024_btn")
        
        if predict_button:
            self.run_2024_prediction(race_id)
        
        # Show cached predictions if available
        if st.session_state.predictions is not None:
            self.display_2024_predictions(st.session_state.predictions)
    
    def run_2024_prediction(self, race_id: str):
        """Run prediction for 2024 race"""
        with st.spinner(f"🔄 Berechne Vorhersagen für {race_id}..."):
            try:
                # Run prediction
                predictions = predict_race_positions(race_id)
                st.session_state.predictions = predictions
                st.success(f"✅ Vorhersage für {race_id} erfolgreich berechnet!")
                
            except FileNotFoundError as e:
                st.error(f"❌ Daten nicht gefunden: {e}")
                st.info("💡 Stelle sicher, dass das Modell trainiert ist und die Daten verfügbar sind.")
            except Exception as e:
                st.error(f"❌ Fehler bei der Vorhersage: {e}")
    
    def display_2024_predictions(self, predictions: pd.DataFrame):
        """Display 2024 predictions with visualizations"""
        
        # Podium predictions
        st.subheader("🏆 Podium Vorhersage")
        
        podium = predictions.head(3)
        cols = st.columns(3)
        
        for i, (_, driver) in enumerate(podium.iterrows()):
            with cols[i]:
                position = i + 1
                css_class = f"podium-{position}"
                
                st.markdown(f"""
                <div class="prediction-card {css_class}">
                    <h3>P{position}</h3>
                    <h4>{driver['driver']} ({driver['team']})</h4>
                    <p>Grid: P{int(driver['grid_position'])}</p>
                    <p>Confidence: {driver.get('prediction_confidence', 0.5)*100:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Full results table
        st.subheader("📊 Vollständige Vorhersage")
        
        # Prepare display dataframe
        display_df = predictions.copy()
        display_df['Position'] = range(1, len(display_df) + 1)
        display_df['Driver'] = display_df['driver']
        display_df['Team'] = display_df['team']
        display_df['Grid'] = display_df['grid_position'].astype(int)
        display_df['Predicted'] = display_df['predicted_position_rounded'].astype(int)
        display_df['DNF Risk'] = (display_df.get('dnf_risk', 0.2) * 100).round(1)
        display_df['Confidence'] = (display_df.get('prediction_confidence', 0.5) * 100).round(0)
        
        st.dataframe(
            display_df[['Position', 'Driver', 'Team', 'Grid', 'Predicted', 'DNF Risk', 'Confidence']],
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        self.create_2024_visualizations(predictions)
        
        # Export option
        if st.button("💾 Export als CSV"):
            csv = predictions.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"pitpredict_{predictions.iloc[0]['race_id']}.csv",
                mime="text/csv"
            )
    
    def create_2024_visualizations(self, predictions: pd.DataFrame):
        """Create visualizations for 2024 predictions"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Grid vs Predicted Position")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions['grid_position'],
                y=predictions['predicted_final_position'],
                mode='markers+text',
                text=predictions['driver'],
                textposition="middle right",
                marker=dict(
                    size=10,
                    color=predictions.get('dnf_risk', 0.2),
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="DNF Risk")
                ),
                name="Drivers"
            ))
            
            # Add diagonal line (perfect prediction)
            fig.add_trace(go.Scatter(
                x=[1, 20],
                y=[1, 20],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name="Perfect Prediction"
            ))
            
            fig.update_layout(
                title="Grid Position vs Predicted Final Position",
                xaxis_title="Grid Position",
                yaxis_title="Predicted Final Position",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Position Changes")
            
            predictions_viz = predictions.copy()
            predictions_viz['position_change'] = predictions_viz['grid_position'] - predictions_viz['predicted_final_position']
            predictions_viz = predictions_viz.sort_values('position_change', ascending=False)
            
            fig = go.Figure()
            
            colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
                     for x in predictions_viz['position_change']]
            
            fig.add_trace(go.Bar(
                x=predictions_viz['driver'],
                y=predictions_viz['position_change'],
                marker_color=colors,
                text=predictions_viz['position_change'].round(1),
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Expected Position Changes (Grid → Final)",
                xaxis_title="Driver",
                yaxis_title="Position Change (+ = Better, - = Worse)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_future_predictions(self):
        """Show future predictions page"""
        st.header("🚀 Future Race Predictions (2025)")
        st.markdown("Erstelle hypothetische Vorhersagen für 2025-Rennen mit benutzerdefinierten Grid-Positionen")
        
        # Initialize future predictor
        if st.session_state.future_predictor is None:
            st.session_state.future_predictor = FutureRacePredictor()
        
        # Track selection
        col1, col2 = st.columns([2, 1])
        with col1:
            track_type = st.selectbox(
                "🏁 Wähle eine Strecke:",
                options=['netherlands', 'monaco', 'spa', 'silverstone', 'default'],
                format_func=lambda x: {
                    'netherlands': '🇳🇱 Netherlands GP (Zandvoort)',
                    'monaco': '🇲🇨 Monaco GP', 
                    'spa': '🇧🇪 Belgian GP (Spa-Francorchamps)',
                    'silverstone': '🇬🇧 British GP (Silverstone)',
                    'default': '🌍 Generic Circuit'
                }[x]
            )
        
        with col2:
            season = st.number_input("📅 Season", min_value=2025, max_value=2030, value=2025)
        
        # Grid position editor
        st.subheader("🏁 Grid Positions Editor")
        st.markdown("Bearbeite die Grid-Positionen für deine Vorhersage:")
        
        # Get default grid
        default_grid = get_default_netherlands_grid()
        
        # Create grid editor
        grid_positions = self.create_grid_editor(default_grid)
        
        # Prediction button
        if st.button("🚀 Future Prediction starten", key="predict_future_btn"):
            self.run_future_prediction(track_type, grid_positions, season)
        
        # Show future prediction results
        if 'future_results' in st.session_state and st.session_state.future_results is not None:
            self.display_future_predictions(st.session_state.future_results, track_type)
    
    def create_grid_editor(self, default_grid: Dict[str, int]) -> Dict[str, int]:
        """Create interactive grid position editor"""
        
        # Create two columns for driver selection
        drivers = list(default_grid.keys())
        mid = len(drivers) // 2
        
        col1, col2 = st.columns(2)
        
        grid_positions = {}
        
        with col1:
            st.markdown("**Positions 1-10:**")
            for i, driver in enumerate(drivers[:mid]):
                pos = st.number_input(
                    f"{driver}",
                    min_value=1,
                    max_value=20,
                    value=default_grid[driver],
                    key=f"grid_{driver}"
                )
                grid_positions[driver] = pos
        
        with col2:
            st.markdown("**Positions 11-20:**")
            for i, driver in enumerate(drivers[mid:]):
                pos = st.number_input(
                    f"{driver}",
                    min_value=1,
                    max_value=20,
                    value=default_grid[driver],
                    key=f"grid_{driver}"
                )
                grid_positions[driver] = pos
        
        # Quick presets
        st.markdown("**Quick Presets:**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("🏆 Championship Order"):
                # Reset to default championship order
                return default_grid
                
        with preset_col2:
            if st.button("🎲 Random Grid"):
                # Create random grid
                import random
                positions = list(range(1, 21))
                random.shuffle(positions)
                return dict(zip(drivers, positions))
        
        with preset_col3:
            if st.button("🔄 Reset Default"):
                # Reset to default
                return default_grid
        
        return grid_positions
    
    def run_future_prediction(self, track_type: str, grid_positions: Dict[str, int], season: int):
        """Run future prediction"""
        
        race_name = f"{track_type.title()} GP {season}"
        
        with st.spinner(f"🔄 Berechne Future Prediction für {race_name}..."):
            try:
                results = st.session_state.future_predictor.predict_future_race(
                    race_name, grid_positions, track_type, season
                )
                st.session_state.future_results = results
                st.success(f"✅ Future Prediction für {race_name} erfolgreich berechnet!")
                
            except FileNotFoundError as e:
                st.error(f"❌ Daten nicht gefunden: {e}")
                st.info("💡 Stelle sicher, dass das Modell trainiert ist und die 2024-Daten verfügbar sind.")
            except Exception as e:
                st.error(f"❌ Fehler bei der Future Prediction: {e}")
                st.exception(e)
    
    def display_future_predictions(self, results: pd.DataFrame, track_type: str):
        """Display future predictions"""
        
        # Track info
        track_info = {
            'netherlands': {'name': 'Zandvoort', 'difficulty': 'High', 'overtaking': 'Hard'},
            'monaco': {'name': 'Monaco', 'difficulty': 'Very High', 'overtaking': 'Very Hard'},
            'spa': {'name': 'Spa-Francorchamps', 'difficulty': 'Medium', 'overtaking': 'Easy'},
            'silverstone': {'name': 'Silverstone', 'difficulty': 'Medium', 'overtaking': 'Medium'},
            'default': {'name': 'Generic Circuit', 'difficulty': 'Medium', 'overtaking': 'Medium'}
        }
        
        info = track_info.get(track_type, track_info['default'])
        
        st.info(f"🏁 **Track:** {info['name']} | **Difficulty:** {info['difficulty']} | **Overtaking:** {info['overtaking']}")
        
        # Podium
        st.subheader("🏆 Podium Prediction 2025")
        
        podium = results.head(3)
        cols = st.columns(3)
        
        for i, (_, driver) in enumerate(podium.iterrows()):
            with cols[i]:
                position = i + 1
                css_class = f"podium-{position}"
                
                st.markdown(f"""
                <div class="prediction-card {css_class}">
                    <h3>P{position}</h3>
                    <h4>{driver['driver']} ({driver['team']})</h4>
                    <p>Grid: P{int(driver['grid_position'])}</p>
                    <p>DNF Risk: {driver.get('dnf_risk', 0.2)*100:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Full results
        st.subheader("📊 Complete 2025 Prediction")
        
        display_df = results.copy()
        display_df['Pos'] = range(1, len(display_df) + 1)
        display_df['Driver'] = display_df['driver']
        display_df['Team'] = display_df['team']
        display_df['Grid'] = display_df['grid_position'].astype(int)
        display_df['Predicted'] = display_df['predicted_position_rounded'].astype(int)
        display_df['Change'] = display_df['Grid'] - display_df['Predicted']
        
        # Color code the changes
        def color_change(val):
            if val > 0:
                return 'background-color: lightgreen'
            elif val < 0:
                return 'background-color: lightcoral'
            else:
                return ''
        
        styled_df = display_df[['Pos', 'Driver', 'Team', 'Grid', 'Predicted', 'Change']].style.applymap(color_change, subset=['Change'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Championship points projection
        self.show_championship_points(results)
        
        # Visualization
        self.create_future_visualizations(results, track_type)
    
    def show_championship_points(self, results: pd.DataFrame):
        """Show championship points projection"""
        
        st.subheader("🏆 Championship Points Projection")
        
        # F1 points system
        points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        
        points_data = []
        for i, (_, driver) in enumerate(results.iterrows(), 1):
            points = points_system.get(i, 0)
            if points > 0:
                points_data.append({
                    'Driver': driver['driver'],
                    'Team': driver['team'],
                    'Position': i,
                    'Points': points
                })
        
        if points_data:
            points_df = pd.DataFrame(points_data)
            st.dataframe(points_df, use_container_width=True, hide_index=True)
            
            # Points visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=points_df['Driver'],
                y=points_df['Points'],
                marker_color='#FF1E1E',
                text=points_df['Points'],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Championship Points Distribution",
                xaxis_title="Driver",
                yaxis_title="Points",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def create_future_visualizations(self, results: pd.DataFrame, track_type: str):
        """Create visualizations for future predictions"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Position Movement")
            
            results_viz = results.copy()
            results_viz['position_change'] = results_viz['grid_position'] - results_viz['predicted_final_position']
            results_viz = results_viz.sort_values('position_change', ascending=False)
            
            fig = go.Figure()
            
            colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
                     for x in results_viz['position_change']]
            
            fig.add_trace(go.Bar(
                x=results_viz['driver'],
                y=results_viz['position_change'],
                marker_color=colors,
                text=results_viz['position_change'].round(1),
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Expected Position Changes",
                xaxis_title="Driver", 
                yaxis_title="Positions Gained/Lost",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Team Performance")
            
            team_performance = results.groupby('team').agg({
                'predicted_final_position': 'mean',
                'dnf_risk': 'mean'
            }).round(2)
            
            team_performance = team_performance.sort_values('predicted_final_position')
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=team_performance['predicted_final_position'],
                y=team_performance['dnf_risk'],
                mode='markers+text',
                text=team_performance.index,
                textposition="middle right",
                marker=dict(
                    size=15,
                    color='blue',
                    opacity=0.7
                ),
                name="Teams"
            ))
            
            fig.update_layout(
                title="Team Performance vs Reliability",
                xaxis_title="Average Predicted Position",
                yaxis_title="Average DNF Risk",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_model_performance(self):
        """Show model performance page"""
        st.header("📊 Model Performance Analysis")
        
        try:
            # Load metrics
            metrics_path = os.path.join(project_root, 'artifacts/metrics/final_position_cv_report.json')
            holdout_path = os.path.join(project_root, 'artifacts/metrics/final_position_holdout_report.json')
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    cv_metrics = json.load(f)
                
                # Overview metrics
                st.subheader("🎯 Overall Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE (Mean Absolute Error)", f"{cv_metrics.get('overall_mae', 0):.2f}", "positions")
                with col2:
                    st.metric("RMSE", f"{cv_metrics.get('overall_rmse', 0):.2f}", "positions") 
                with col3:
                    st.metric("R² Score", f"{cv_metrics.get('overall_r2', 0):.3f}")
                with col4:
                    st.metric("Podium Accuracy", f"{cv_metrics.get('overall_podium_accuracy', 0)*100:.1f}%")
                
                # Cross-validation results
                if 'fold_scores' in cv_metrics:
                    st.subheader("📈 Cross-Validation Results")
                    
                    fold_df = pd.DataFrame(cv_metrics['fold_scores'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=[f"Fold {i}" for i in fold_df['fold']],
                            y=fold_df['mae'],
                            name='MAE'
                        ))
                        fig.update_layout(title="MAE by Cross-Validation Fold", yaxis_title="MAE")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=[f"Fold {i}" for i in fold_df['fold']],
                            y=fold_df['podium_accuracy'] * 100,
                            name='Podium Accuracy',
                            marker_color='gold'
                        ))
                        fig.update_layout(title="Podium Accuracy by Fold", yaxis_title="Accuracy (%)")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                if 'feature_importance' in cv_metrics and cv_metrics['feature_importance']:
                    st.subheader("🔍 Feature Importance")
                    
                    feature_imp = cv_metrics['feature_importance']
                    top_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:15]
                    
                    if top_features:
                        features_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            y=features_df['Feature'][::-1],
                            x=features_df['Importance'][::-1],
                            orientation='h',
                            marker_color='steelblue'
                        ))
                        fig.update_layout(
                            title="Top 15 Most Important Features",
                            xaxis_title="Importance",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Training info
                st.subheader("ℹ️ Training Information")
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.metric("Training Samples", f"{cv_metrics.get('n_samples', 0):,}")
                with info_col2:
                    st.metric("DNF Rate", f"{cv_metrics.get('dnf_rate', 0)*100:.1f}%")
                with info_col3:
                    st.metric("Total DNFs", f"{cv_metrics.get('n_dnfs', 0):,}")
                
            else:
                st.warning("⚠️ Model-Metriken nicht gefunden. Trainiere das Modell zuerst mit `make train`")
                
        except Exception as e:
            st.error(f"❌ Fehler beim Laden der Performance-Daten: {e}")
    
    def show_about_page(self):
        """Show about page"""
        st.header("ℹ️ About PitPredict")
        
        st.markdown("""
        ## 🏁 PitPredict - F1 Race Outcome Prediction Suite
        
        PitPredict ist ein fortschrittliches Machine Learning-System zur Vorhersage von Formel-1-Rennergebnissen. 
        Es kombiniert drei spezialisierte Modelle zu einem umfassenden Prediction-System:
        
        ### 🤖 Model Architecture
        
        **1. Final Position Prediction Model** 🎯
        - **Algorithmus:** Gradient Boosting Regressor
        - **Features:** 42 engineerte Features
        - **Performance:** MAE ~2.1 Positionen, 96.7% Podium-Accuracy
        - **Zweck:** Hauptmodell für Endplatzierungen
        
        **2. DNF Prediction Model** ⚡
        - **Zweck:** Vorhersage der Ausfallwahrscheinlichkeit
        - **Features:** Fahrer/Team-Zuverlässigkeit, Track-Risiken
        - **Integration:** Als Feature im Final Position Model
        
        **3. Pit Stop Strategy Model** 🔧
        - **Zweck:** Optimale Pit Stop-Strategien (in Entwicklung)
        - **Features:** Reifenverschleiß, Streckencharakteristika
        
        ### 📊 Feature Categories
        
        - **Grid Position & Qualifying:** Startposition, Qualifying-Zeiten, Gap zur Pole
        - **Historical Performance:** Letzte 5 Rennen, Durchschnittswerte, Trends
        - **Team Performance:** Team-Qualifying-Rank, Zuverlässigkeit
        - **Track-specific:** Überholschwierigkeit, Pit Loss, Street Circuit
        - **DNF Risk:** Fahrer- und Team-Ausfallraten
        
        ### 🚀 Unique Features
        
        **Future Race Predictions (2025+):**
        - Extrapolation von 2024-Daten auf hypothetische 2025-Szenarien
        - Berücksichtigung von Fahrer-Transfers (Hamilton → Ferrari)
        - Track-spezifische Anpassungen
        - Benutzer-definierte Grid-Positionen
        
        ### 📈 Performance Metrics
        
        - **MAE:** ~2.1 Positionen (sehr gut für F1)
        - **Podium Accuracy:** >90% 
        - **Points Accuracy:** ~85%
        - **R² Score:** ~0.75
        
        ### 🛠️ Technical Stack
        
        - **ML Framework:** scikit-learn, pandas, numpy
        - **Web Framework:** Streamlit
        - **Visualization:** Plotly
        - **Data:** FastF1 API, historical F1 data
        - **Deployment:** Python package with CLI + Web UI
        
        ### 📚 Data Sources
        
        - **2024 F1 Season:** Komplette Renndaten, Qualifying, Grid-Positionen
        - **Historical Data:** Multi-season performance trends
        - **Real-time Integration:** FastF1 API für aktuelle Daten
        
        ### 🎯 Use Cases
        
        - **Race Predictions:** Vor-Rennen Vorhersagen für Fans und Analysten
        - **Strategy Analysis:** Verständnis von Einflussfaktoren
        - **Future Scenarios:** "Was-wäre-wenn" Analysen für 2025+
        - **Model Research:** Basis für weitere ML-Forschung im Motorsport
        
        ---
        
        **Entwickelt mit ❤️ für die F1-Community**
        
        *Hinweis: Alle Vorhersagen sind statistisch und dienen der Unterhaltung. 
        Die reale Formel 1 ist unvorhersagbar - und das macht sie so spannend! 🏁*
        """)

def main():
    """Main entry point"""
    app = PitPredictApp()
    app.run()

if __name__ == "__main__":
    main()
