"""
FastAPI Application for Mobile Phone Price Prediction
Deployed model serves predictions via REST API with MLflow tracking
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import numpy as np
from pathlib import Path
import mlflow
import time
from datetime import datetime
from contextlib import asynccontextmanager

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "Predictions"
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Global variables for tracking
prediction_count = 0
total_prediction_time = 0.0
prediction_history = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global model
    
    print("=" * 60)
    print("Starting API with MLflow tracking...")
    
    try:
        # Setup MLflow experiment
        try:
            experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            if experiment is None:
                mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
                print(f"Created MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
            else:
                print(f"Using existing MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
        except Exception as e:
            print(f"Warning: Could not setup MLflow experiment: {str(e)}")
        
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    except Exception as e:
        print(f"Error during startup: {str(e)}")
    
    print("=" * 60)
    yield
    
    # Shutdown - log final metrics
    if prediction_count > 0:
        try:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            avg_prediction_time = total_prediction_time / prediction_count
            with mlflow.start_run(run_name="API_Summary"):
                mlflow.log_metric("total_predictions", prediction_count)
                mlflow.log_metric("avg_prediction_time_ms", avg_prediction_time * 1000)
                mlflow.log_param("api_version", "1.0.0")
            print(f"Logged API summary: {prediction_count} predictions, avg time: {avg_prediction_time*1000:.2f}ms")
        except Exception as e:
            print(f"Warning: Could not log shutdown summary: {str(e)}")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Mobile Phone Price Prediction API",
    description="API for predicting mobile phone prices using trained ML models with MLflow tracking",
    version="1.0.0",
    lifespan=lifespan
)

# Load model and scaler
model_dir = Path(__file__).parent / "training" / "models"

try:
    model = joblib.load(model_dir / "best_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    feature_names = joblib.load(model_dir / "feature_names.joblib")
    print(f"Model loaded successfully from {model_dir}")
    print(f"Model type: {type(model).__name__}")
except FileNotFoundError as e:
    print(f"Warning: Model files not found. Please train the model first. Error: {e}")
    model = None
    scaler = None
    feature_names = None


# Request model for input validation
class PhoneFeatures(BaseModel):
    """Input features for mobile phone price prediction"""
    ram: float = Field(..., description="RAM in GB", ge=1, le=32)
    battery_capacity: float = Field(..., description="Battery capacity in mAh", ge=1000, le=10000)
    mobile_weight: float = Field(..., description="Mobile weight in grams", ge=50, le=500)
    front_camera: float = Field(..., description="Front camera in MP", ge=0, le=100)
    back_camera: float = Field(..., description="Back camera in MP", ge=0, le=200)
    screen_size: float = Field(..., description="Screen size in inches", ge=3, le=10)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ram": 8,
                "battery_capacity": 4000,
                "mobile_weight": 180,
                "front_camera": 16,
                "back_camera": 48,
                "screen_size": 6.5
            }
        }
    )


# Response model
class PricePrediction(BaseModel):
    """Predicted price response"""
    predicted_price_usd: float = Field(..., description="Predicted price in USD")
    predicted_price_pkr: float = Field(..., description="Predicted price in PKR (converted)")
    model_type: str = Field(..., description="Type of model used for prediction")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_price_usd": 450.25,
                "predicted_price_pkr": 125169.5,
                "model_type": "XGBRegressor"
            }
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "mlflow_tracking": True,
        "total_predictions": prediction_count,
        "avg_prediction_time_ms": round((total_prediction_time / prediction_count * 1000) if prediction_count > 0 else 0, 2)
    }


@app.post("/predict", response_model=PricePrediction)
async def predict_price(phone: PhoneFeatures):
    """
    Predict mobile phone price based on features
    
    - **ram**: RAM in GB (1-32)
    - **battery_capacity**: Battery capacity in mAh (1000-10000)
    - **mobile_weight**: Mobile weight in grams (50-500)
    - **front_camera**: Front camera in MP (0-100)
    - **back_camera**: Back camera in MP (0-200)
    - **screen_size**: Screen size in inches (3-10)
    
    Returns predicted price in USD and PKR
    All predictions are logged to MLflow for tracking.
    """
    global prediction_count, total_prediction_time, prediction_history
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using the notebook."
        )
    
    start_time = time.time()
    
    try:
        # Prepare features in the correct order
        features = np.array([[
            phone.ram,
            phone.battery_capacity,
            phone.mobile_weight,
            phone.front_camera,
            phone.back_camera,
            phone.screen_size
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Convert to PKR (using approximate rate)
        PKR_TO_USD_RATE = 278.0
        price_pkr = prediction * PKR_TO_USD_RATE
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        prediction_count += 1
        total_prediction_time += prediction_time
        
        # Log to MLflow
        try:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{prediction_count}"):
                # Log input features
                mlflow.log_param("ram", phone.ram)
                mlflow.log_param("battery_capacity", phone.battery_capacity)
                mlflow.log_param("mobile_weight", phone.mobile_weight)
                mlflow.log_param("front_camera", phone.front_camera)
                mlflow.log_param("back_camera", phone.back_camera)
                mlflow.log_param("screen_size", phone.screen_size)
                
                # Log prediction results
                mlflow.log_metric("predicted_price_usd", round(prediction, 2))
                mlflow.log_metric("predicted_price_pkr", round(price_pkr, 2))
                mlflow.log_metric("prediction_time_ms", prediction_time * 1000)
                
                # Log model info
                mlflow.log_param("model_type", type(model).__name__)
                mlflow.log_param("timestamp", datetime.now().isoformat())
                
                # Log features as JSON artifact (for batch analysis)
                features_dict = {
                    "ram": float(phone.ram),
                    "battery_capacity": float(phone.battery_capacity),
                    "mobile_weight": float(phone.mobile_weight),
                    "front_camera": float(phone.front_camera),
                    "back_camera": float(phone.back_camera),
                    "screen_size": float(phone.screen_size),
                    "predicted_price_usd": round(prediction, 2),
                    "prediction_time_ms": round(prediction_time * 1000, 2)
                }
                mlflow.log_dict(features_dict, "prediction_details.json")
        except Exception as mlflow_error:
            # Don't fail the request if MLflow logging fails
            print(f"Warning: MLflow logging failed: {str(mlflow_error)}")
        
        # Store in history (keep last 100)
        prediction_history.append({
            "timestamp": datetime.now().isoformat(),
            "features": features_dict,
            "prediction_time_ms": round(prediction_time * 1000, 2)
        })
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        return PricePrediction(
            predicted_price_usd=round(prediction, 2),
            predicted_price_pkr=round(price_pkr, 2),
            model_type=type(model).__name__
        )
    
    except Exception as e:
        # Log error to MLflow
        prediction_time = time.time() - start_time
        try:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_metric("prediction_time_ms", prediction_time * 1000)
                mlflow.log_param("error", str(e))
                mlflow.log_param("error_type", type(e).__name__)
                mlflow.log_param("timestamp", datetime.now().isoformat())
        except Exception as mlflow_error:
            print(f"Warning: MLflow error logging failed: {str(mlflow_error)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )
