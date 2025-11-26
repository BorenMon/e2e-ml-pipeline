"""
Streamlit Frontend for Mobile Phone Price Prediction
Connects to FastAPI backend for predictions
"""

import streamlit as st
import requests
from typing import Optional

# Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_price(features: dict) -> Optional[dict]:
    """Call FastAPI to get price prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to FastAPI backend. Make sure it's running on http://localhost:8000")
        st.info("Start the API with: `python main.py` or `uvicorn main:app --reload`")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📱 Mobile Phone Price Predictor</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.warning("⚠️ FastAPI backend is not running. Please start it first.")
        st.code("python main.py", language="bash")
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app predicts mobile phone prices based on:
        - RAM capacity
        - Battery capacity
        - Mobile weight
        - Camera specifications
        - Screen size
        
        The prediction is powered by an **XGBoost** machine learning model
        trained on mobile phone data.
        """)
        
        st.header("🔗 API Status")
        if check_api_health():
            st.success("✅ API is running")
            try:
                health = requests.get(f"{API_URL}/health").json()
                st.info(f"Model: {health.get('model_type', 'Unknown')}")
            except:
                pass
        else:
            st.error("❌ API is not running")
        
        st.header("📊 Quick Stats")
        st.markdown("""
        - **Best Model**: XGBoost
        - **R² Score**: 92.12%
        - **RMSE**: 108.87
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Enter Phone Specifications")
        
        # Input fields
        ram = st.slider(
            "RAM (GB)",
            min_value=1,
            max_value=32,
            value=8,
            step=1,
            help="Random Access Memory in Gigabytes"
        )
        
        battery = st.slider(
            "Battery Capacity (mAh)",
            min_value=1000,
            max_value=10000,
            value=4000,
            step=100,
            help="Battery capacity in milliamp hours"
        )
        
        weight = st.slider(
            "Mobile Weight (grams)",
            min_value=50,
            max_value=500,
            value=180,
            step=10,
            help="Weight of the mobile phone in grams"
        )
        
        front_camera = st.slider(
            "Front Camera (MP)",
            min_value=0,
            max_value=100,
            value=16,
            step=1,
            help="Front camera megapixels"
        )
        
        back_camera = st.slider(
            "Back Camera (MP)",
            min_value=0,
            max_value=200,
            value=48,
            step=1,
            help="Back camera megapixels"
        )
        
        screen_size = st.slider(
            "Screen Size (inches)",
            min_value=3.0,
            max_value=10.0,
            value=6.5,
            step=0.1,
            help="Screen size in inches"
        )
        
        # Prepare features
        features = {
            "ram": float(ram),
            "battery_capacity": float(battery),
            "mobile_weight": float(weight),
            "front_camera": float(front_camera),
            "back_camera": float(back_camera),
            "screen_size": float(screen_size)
        }
        
        # Predict button
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    with col2:
        st.header("💰 Price Prediction")
        
        if predict_button:
            with st.spinner("Predicting price..."):
                result = predict_price(features)
                
                if result:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    st.markdown("### Predicted Price")
                    st.markdown(
                        f'<div class="price-display">${result["predicted_price_usd"]:,.2f} USD</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(f"**PKR:** PKR {result['predicted_price_pkr']:,.2f}")
                    st.markdown(f"**Model:** {result['model_type']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show input summary
                    st.markdown("---")
                    st.markdown("### 📋 Input Summary")
                    summary_data = {
                        "RAM": f"{ram} GB",
                        "Battery": f"{battery} mAh",
                        "Weight": f"{weight} g",
                        "Front Camera": f"{front_camera} MP",
                        "Back Camera": f"{back_camera} MP",
                        "Screen Size": f"{screen_size}\""
                    }
                    for key, value in summary_data.items():
                        st.markdown(f"- **{key}:** {value}")
        else:
            st.info("👈 Enter phone specifications and click 'Predict Price' to get started!")
            
            # Show example
            st.markdown("---")
            st.markdown("### 💡 Example")
            st.markdown("""
            Try these values for a mid-range phone:
            - RAM: 8 GB
            - Battery: 4000 mAh
            - Weight: 180 g
            - Front Camera: 16 MP
            - Back Camera: 48 MP
            - Screen Size: 6.5"
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Powered by FastAPI + Streamlit | ML Model: XGBoost"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

