#!/bin/bash
set -e

# Start MLflow UI in background
mlflow ui --port 5000 --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --allowed-hosts "*" --cors-allowed-origins "*" &

# Start Streamlit in background
streamlit run ui.py --server.port 8501 --server.address 0.0.0.0 &

# Start FastAPI based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    exec fastapi run api.py --host 0.0.0.0 --port 8000 --workers 4
else
    exec fastapi dev api.py --host 0.0.0.0 --port 8000 --reload
fi

