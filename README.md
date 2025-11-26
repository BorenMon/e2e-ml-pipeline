# End-to-End Machine Learning Pipeline

Data collection → preprocessing → training → validation → deployment → monitoring.

## Project Overview

This project implements a complete ML pipeline for **Mobile Phone Price Prediction** using regression models. It includes:

1. **Model Training & Evaluation** - 3 models (Linear Regression, Random Forest, XGBoost)
2. **MLflow Experiment Tracking** - Track parameters, metrics, and models
3. **Model Deployment** - FastAPI REST API for serving predictions

## Assignment Requirements Coverage

### ✅ Part 1: Building and Evaluating ML Models
- ✅ Load dataset (CSV file)
- ✅ Split data into training/testing sets (80/20)
- ✅ Train 3 different models:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- ✅ Calculate evaluation metrics:
  - **MSE** (Mean Squared Error)
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **R² Score** (Coefficient of Determination)

### ✅ Part 2: MLflow for Experiment Tracking
- ✅ Log model hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- ✅ Log evaluation metrics (MSE, RMSE, MAE, R² Score)
- ✅ Compare results in MLflow UI

### ✅ Part 3: Model Deployment
- ✅ Save best model using joblib
- ✅ Build FastAPI app to serve predictions
- ✅ Test API with sample requests (Swagger UI available)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models and Track Experiments

Open and run the Jupyter notebook:
```bash
jupyter notebook training/notebooks/machine_learning.ipynb
```

The notebook will:
- Load and preprocess the data
- Train 3 models with MLflow tracking
- Save the best model to `training/models/best_model.joblib`
- Generate comparison tables

### 3. View MLflow Experiments

After training, start MLflow UI to view experiment results:

```bash
mlflow ui
```

Then open your browser to: `http://localhost:5000`

You can:
- Compare all 3 models side-by-side
- View logged parameters and metrics
- Download models from the UI

### 4. Deploy the API

Start the FastAPI server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

The API will be available at: `http://localhost:8000`

### 5. Launch Streamlit Frontend (Optional)

Start the Streamlit app for a user-friendly interface:

```bash
streamlit run streamlit_app.py
```

The Streamlit app will be available at: `http://localhost:8501`

**Note:** Make sure the FastAPI server is running first, as Streamlit connects to it for predictions.

## API Usage

### Interactive Documentation (Swagger UI)

Visit: `http://localhost:8000/docs`

### API Endpoints

#### 1. Root Endpoint
```bash
GET http://localhost:8000/
```

#### 2. Health Check
```bash
GET http://localhost:8000/health
```

#### 3. Predict Price
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "ram": 8,
  "battery_capacity": 4000,
  "mobile_weight": 180,
  "front_camera": 16,
  "back_camera": 48,
  "screen_size": 6.5
}
```

### Testing the API

#### Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ram": 8,
    "battery_capacity": 4000,
    "mobile_weight": 180,
    "front_camera": 16,
    "back_camera": 48,
    "screen_size": 6.5
  }'
```

#### Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "ram": 8,
        "battery_capacity": 4000,
        "mobile_weight": 180,
        "front_camera": 16,
        "back_camera": 48,
        "screen_size": 6.5
    }
)

print(response.json())
```

#### Using Swagger UI:
1. Go to `http://localhost:8000/docs`
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Enter the feature values
5. Click "Execute"

#### Using Streamlit Frontend:
1. Start FastAPI: `python main.py`
2. Start Streamlit: `streamlit run streamlit_app.py`
3. Open `http://localhost:8501` in your browser
4. Use the interactive sliders to input features
5. Click "Predict Price" to get results

## Project Structure

```
E2E ML Pipeline/
├── main.py                          # FastAPI application
├── streamlit_app.py                 # Streamlit frontend
├── test_api.py                      # API testing script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── training/
│   ├── data/
│   │   └── Mobiles Dataset (2025).csv
│   ├── models/                      # Saved models directory
│   │   ├── best_model.joblib
│   │   ├── scaler.joblib
│   │   └── feature_names.joblib
│   ├── notebooks/
│   │   └── machine_learning.ipynb   # Training notebook
│   └── mlruns/                      # MLflow tracking data
└── venv/                            # Virtual environment
```

## Model Performance

Based on the training results:

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **XGBoost** | 108.87 | 69.62 | **92.12%** ⭐ |
| Random Forest | 133.29 | 87.47 | 88.20% |
| Linear Regression | 302.47 | 216.00 | 39.21% |

**Best Model**: XGBoost (automatically saved and deployed)

## Features Used for Prediction

- **RAM** (GB): 1-32
- **Battery Capacity** (mAh): 1000-10000
- **Mobile Weight** (grams): 50-500
- **Front Camera** (MP): 0-100
- **Back Camera** (MP): 0-200
- **Screen Size** (inches): 3-10

## Notes

- The model predicts prices in USD. PKR conversion is included in API response.
- MLflow tracking data is stored locally in `training/mlruns/`
- The best model is automatically selected based on R² score
- All models are logged to MLflow for comparison

## Troubleshooting

1. **Model not found error**: Make sure you've run the training notebook first
2. **MLflow UI not showing experiments**: Check that `training/mlruns/` directory exists
3. **Port already in use**: Change the port in `main.py` or use `--port` flag with uvicorn
