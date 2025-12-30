# 🧠 Dopamine Defense - TikTok Behaviour Analysis

A machine learning-powered behavioral addiction analysis system that transforms TikTok watch history into actionable insights and predictive risk assessments.

## 📊 Overview

**Dopamine Defense** is a clinical-grade behavioral analytics tool that analyzes TikTok usage patterns to identify addictive behaviors, predict relapse risks, and provide AI-powered intervention strategies.  Built with Python and FastAPI, it combines advanced machine learning with Google's Gemini AI to deliver personalized behavioral assessments.

## ✨ Features

### 🔍 Core Analytics
- **Addiction Clock Heatmap**: 24×7 visual representation of usage patterns showing when you're most vulnerable
- **Risk Prediction Model**: Machine learning ensemble (Logistic Regression + Random Forest + XGBoost) predicting tomorrow's relapse probability
- **Behavioral Scoring System**: Multi-factor algorithm evaluating sleep sabotage, work-hour disruption, and total engagement
- **Pattern Recognition**: Identifies critical addiction markers including late-night usage (2-7 AM) and work-hour distractions (9-18 weekdays)

### 🤖 AI-Powered Insights
- **Gemini AI Integration**: Clinical behavioral analysis with personalized intervention recommendations
- **Context-Aware Recommendations**: Takes into account your local timezone, day of week, and historical patterns
- **Three-Part Assessment Framework**:
  - Pattern Recognition:  Identifies concerning behavioral trends
  - Risk Factor Analysis: Explains why specific timeframes are vulnerable
  - Intervention Strategy: Provides concrete 24-hour action plans

### 📈 Visualization Dashboard
- **Interactive HTML Interface**: Clean, responsive web UI with real-time analytics
- **Multi-Chart Display**:
  - Time-series trend graphs with threshold indicators
  - Weekday radar charts showing vulnerability by day
  - Hourly heatmaps revealing peak addiction periods
- **Statistics Summary**: Total events, date ranges, bad day ratios, and risk scores

## 🏗️ Architecture

```
Tiktok-Behaviour-Analysis/
├── main.py                      # FastAPI server with ML prediction pipeline
├── index.html                   # Interactive web dashboard
├── tiktok-analysis.ipynb        # Jupyter notebook for model training & EDA
├── dataset/                     # Sample TikTok watch history files
│   ├── David. txt
│   └── Reynard.txt
├── tiktok_voting_model.pkl      # Trained ensemble classifier
├── tiktok_scaler.pkl            # RobustScaler for feature normalization
├── robust_threshold. pkl         # Smoothed score threshold for "bad habit" classification
├── decision_threshold.pkl       # Optimized F1 decision boundary
└── . env                         # Configuration (Gemini API key)
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib python-dotenv google-genai
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/LouSens/Tiktok-Behaviour-Analysis.git
cd Tiktok-Behaviour-Analysis
```

2. **Configure API Key**
Create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

3. **Run the server**
```bash
python main.py
```

4. **Access the dashboard**
Navigate to `http://localhost:8000`

### Usage

1. **Export TikTok Data**: 
   - Go to TikTok Settings → Privacy → Download your data
   - Extract `Watch History. txt` from the archive

2. **Upload & Analyze**:
   - Click "Upload File" on the dashboard
   - Select your `Watch History.txt`
   - View instant behavioral analysis with AI recommendations

## 🧪 Machine Learning Pipeline

### Data Processing
- **Timestamp Parsing**: Extracts UTC timestamps and converts to user's local timezone (Asia/Kuala_Lumpur)
- **Feature Engineering**:
  - `day_of_week`: Weekday index (0=Monday, 6=Sunday)
  - `prev_score`: Previous day's smoothed habit score
  - `morning_clicks`: Early-day usage triggering daily patterns
  - `volatility`: 5-day rolling standard deviation

### Scoring Algorithm
```python
# Sleep Sabotage:  3x penalty for 2-7 AM usage
sleep_penalty = late_night_clicks * 3.0

# Weekday Penalty: Work-hour disruption (9-18) + baseline
weekday_score = sleep_penalty + (work_hour_clicks * 2.0) + (total_clicks * 0.05)

# Weekend Penalty: Total usage + sleep disruption
weekend_score = sleep_penalty + (total_clicks * 0.2)
```

### Model Training (from Jupyter Notebook)
- **Outlier Handling**: Winsorization at 95th percentile
- **Smoothing**:  Exponential weighted mean (span=3 days)
- **Threshold Definition**: 40th percentile of smoothed scores
- **Ensemble**:  Soft voting classifier combining 3 models
- **Evaluation**:  Precision-recall optimization with F1 maximization

## 📊 API Endpoints

### `GET /`
Returns the HTML dashboard

### `GET /health`
```json
{
  "status":  "healthy",
  "gemini_available": true,
  "model_loaded": true,
  "api_key_configured": true,
  "timestamp": "2025-12-30T12:00:00"
}
```

### `POST /analyze`
**Request**:  Multipart form-data with `Watch History.txt`

**Response**:
```json
{
  "status": "success",
  "forecast": {
    "risk_score": 0.6234,
    "risk_level": "high"
  },
  "statistics": {
    "total_events": 12453,
    "date_range": {"start": "2025-06-01", "end": "2025-12-30"},
    "bad_days": 45,
    "bad_ratio": 0.382
  },
  "charts": {
    "dates": ["2025-06-01", ... ],
    "scores": [120.5, 230.1, ...],
    "threshold": 250. 0,
    "heatmap_z": [[5, 2, 0, ... ], ... ],
    "radar_values": [125.3, 200.4, ...]
  },
  "gemini":  "**Pattern Recognition** .. .",
  "metadata": {
    "gemini_available": true,
    "model_loaded": true,
    "analyzed_at": "2025-12-30T12:00:00"
  }
}
```

## 🔬 Key Technologies

- **Backend**: FastAPI, Uvicorn
- **ML Stack**: scikit-learn, XGBoost, pandas, NumPy
- **AI**: Google Gemini 2.5 Flash Lite
- **Visualization**: Plotly. js (heatmaps, radar charts, time series)
- **Frontend**:  Vanilla HTML/CSS/JavaScript
- **Data Processing**: Regex parsing, timezone conversion, rolling statistics

## 🎯 Use Cases

- Personal behavioral intervention for social media addiction
- Research on digital wellbeing and dopamine-driven behaviors
- Parental monitoring of adolescent screen time patterns
- Corporate productivity analytics for work-hour distractions
- Clinical psychology tools for addiction assessment

## 🛡️ Privacy

All data processing happens **locally on your machine**. TikTok history files are: 
- Never stored on external servers
- Processed in-memory only
- Deleted after analysis completion
- Only statistics (not raw data) are sent to Gemini AI

## 📝 License

This project is available for educational and research purposes. 

## 🤝 Contributing

Contributions welcome! Key areas for improvement:
- Support for additional social media platforms (Instagram, YouTube)
- Advanced time-series forecasting (LSTM, Prophet)
- Mobile app version
- Multi-user comparative analytics

## 📧 Contact

**Author**: LouSens  
**Repository**: [github.com/LouSens/Tiktok-Behaviour-Analysis](https://github.com/LouSens/Tiktok-Behaviour-Analysis)

---

**⚠️ Disclaimer**: This tool is for informational purposes only and does not constitute medical or psychological advice. If you're experiencing serious addiction symptoms, please consult a qualified healthcare professional. 
