import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("assets/style.css")
except FileNotFoundError:
    pass # Will just use default if CSS is missing

# --- Load Model & Data ---
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/housing_data.csv')

try:
    model = load_model()
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.error("Model or Data not found. Please run the training script first.")

# --- Helper function for UI elements ---
def glass_container(content, title=None):
    if title:
        st.markdown(f'<div class="glass-container"><h3>{title}</h3>{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="glass-container">{content}</div>', unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Price Predictor", "📊 Data Insights (EDA)", "ℹ️ About Project"])

if not data_loaded:
    st.stop()

# ==========================================
# PAGE 1: PRICE PREDICTOR
# ==========================================
if page == "🏠 Price Predictor":
    st.markdown("<h1>Predict Real Estate Value</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;'>Leverage our advanced XGBoost Machine Learning model to get accurate market estimates.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### 📍 Property Details")
        
        location = st.selectbox("Neighborhood / Location", df['location'].unique())
        sqft_area = st.number_input("Square Footage (sqft)", min_value=500, max_value=10000, value=1500, step=100)
        age_of_property = st.slider("Age of Property (Years)", min_value=0, max_value=100, value=5)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### 🛋️ Configuration & Amenities")
        
        col2a, col2b = st.columns(2)
        with col2a:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        with col2b:
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
            
        furnishing_status = st.selectbox("Furnishing Status", df['furnishing_status'].unique())
        
        st.markdown("#### Amenities")
        col2c, col2d, col2e = st.columns(3)
        with col2c:
            has_parking = st.checkbox("Parking", value=True)
        with col2d:
            has_pool = st.checkbox("Swimming Pool")
        with col2e:
            has_gym = st.checkbox("Gym / Fitness")
        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction Button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Calculate Market Value", use_container_width=True)
    
    if predict_btn:
        # Create input dataframe
        input_data = pd.DataFrame({
            'sqft_area': [sqft_area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'age_of_property': [age_of_property],
            'location': [location],
            'furnishing_status': [furnishing_status],
            'has_parking': [1 if has_parking else 0],
            'has_pool': [1 if has_pool else 0],
            'has_gym': [1 if has_gym else 0]
        })
        
        # Make prediction
        with st.spinner("Analyzing market data..."):
            prediction = model.predict(input_data)[0]
        
        # Display Results
        st.markdown(f'''
        <div class="prediction-box">
            <h3 style="margin:0; color:#f8fafc;">Estimated Property Value</h3>
            <div class="prediction-value">₹ {prediction:,.2f}</div>
            <p style="color:#94a3b8; margin:0;">Based on {location} market averages and property specs.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Radar Chart for Property Profile
        st.markdown("<br>### 🎯 Property Profile Profile", unsafe_allow_html=True)
        categories = ['Size', 'Luxury', 'Utility', 'Newness', 'Location Prime']
        
        # Normalize values for radar
        size_score = min(sqft_area / 5000 * 100, 100)
        luxury_score = ((1 if has_pool else 0) + (1 if has_gym else 0) + (1 if furnishing_status=='Fully-Furnished' else 0.5)) / 3 * 100
        utility_score = min((bedrooms + bathrooms) / 10 * 100, 100)
        newness_score = max(100 - age_of_property, 0)
        location_score = 90 if location == 'Downtown' else 80 if location == 'Business District' else 60
        
        fig_radar = go.Figure(data=go.Scatterpolar(
          r=[size_score, luxury_score, utility_score, newness_score, location_score],
          theta=categories,
          fill='toself',
          line_color='#38bdf8'
        ))
        
        fig_radar.update_layout(
          polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)"
          ),
          paper_bgcolor="rgba(0,0,0,0)",
          font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# PAGE 2: DATA INSIGHTS
# ==========================================
elif page == "📊 Data Insights (EDA)":
    st.markdown("<h1>Market Insights & EDA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;'>Interactive visualizations exploring the relationships between property features and prices.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### Price by Location")
        fig1 = px.box(df, x='location', y='price', color='location', template='plotly_dark')
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### Price vs. Square Footage")
        fig2 = px.scatter(df, x='sqft_area', y='price', color='furnishing_status', trendline="ols", template='plotly_dark')
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### Feature Correlations")
    numeric_df = df.select_dtypes(include=['float64', 'int32', 'int64'])
    corr = numeric_df.corr()
    fig3 = px.imshow(corr, text_auto=".2f", aspect="auto", template='plotly_dark', color_continuous_scale='Blues')
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3: ABOUT
# ==========================================
elif page == "ℹ️ About Project":
    st.markdown("<h1>About This Project</h1>", unsafe_allow_html=True)
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("""
    ### 🏢 House Price Prediction System
    
    This project is an end-to-end Machine Learning application designed to predict real estate prices based on various property features. 
    It serves as a **proof of concept** for real estate portals, banks, and investors to generate automated property valuations.
    
    **Tech Stack:**
    * **Frontend:** Streamlit (Custom Glassmorphism UI)
    * **Data Manipulation:** Pandas, NumPy
    * **Machine Learning:** Scikit-Learn, XGBoost
    * **Visualizations:** Plotly Interactive Charts
    
    **Model Performance:**
    The core prediction engine uses an **XGBoost Regressor** wrapped in a Scikit-Learn Pipeline, handling missing values, standardizing numerical features, and one-hot encoding categorical variables seamlessly. The model achieved an impressive **R² score of > 0.98** on the synthetic testing set.
    
    **Designed for Portfolio Showcasing** 🚀
    """)
    st.markdown('</div>', unsafe_allow_html=True)
