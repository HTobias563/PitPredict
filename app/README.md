# PitPredict Streamlit App

A modern web application for F1 race predictions using the PitPredict ML models.

## Features

- 🎯 **2024 Race Predictions**: Real race predictions with actual grid positions
- 🚀 **Future Predictions (2025)**: Hypothetical scenarios with custom grids
- 📊 **Model Performance**: Detailed metrics and analysis
- 📈 **Interactive Visualizations**: Plotly charts and graphs
- 💾 **Export Functionality**: CSV downloads

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r app/requirements.txt
   ```

2. **Run the App:**
   ```bash
   streamlit run app/pitpredict_app.py
   ```

3. **Open Browser:**
   Navigate to `http://localhost:8501`

## Usage

### 2024 Race Predictions
- Select any 2024 F1 race
- Get instant predictions based on real grid positions
- View podium predictions and full results
- Explore interactive visualizations

### Future Predictions (2025)
- Choose from 5 different track types
- Customize grid positions for all 20 drivers
- Account for driver transfers (Hamilton → Ferrari)
- Generate hypothetical race outcomes

### Model Performance
- View cross-validation metrics
- Analyze feature importance
- Track model accuracy over different scenarios

## Technical Architecture

- **Frontend:** Streamlit with custom CSS styling
- **Visualizations:** Plotly for interactive charts
- **Backend:** Direct integration with PitPredict models
- **Data:** Real-time loading from trained models

## Deployment

For production deployment:

```bash
# Local development
streamlit run app/pitpredict_app.py

# Production with specific port
streamlit run app/pitpredict_app.py --server.port 8501 --server.address 0.0.0.0
```

## Screenshots

The app features:
- Modern F1-themed UI with red accent colors
- Responsive grid layouts
- Interactive podium displays with gold/silver/bronze styling  
- Real-time prediction calculations
- Export functionality for all results
