# 🏠 House Price Prediction using Regression Models

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-purple.svg)

## 📌 Project Overview

This project is an **end-to-end Machine Learning Application** designed to predict the market value of real estate properties based on key features (e.g., location, square footage, bedrooms, amenities).

Instead of relying on guesswork, this system uses historical housing data to learn pricing patterns and provide highly accurate, data-driven estimates.

### ❓ What Problem Does It Solve?
Real estate markets are highly volatile and subjective. This project removes human bias by calculating exact fair market values, ensuring fair transactions for buyers, sellers, and agents.

### 🏢 Industry Relevance
- **Property Portals (Zillow, MagicBricks):** Powers automated valuation models (e.g., Zestimates).
- **Banks:** Determines if a property is worth the mortgage requested.
- **Investors:** Identifies undervalued properties for maximum ROI.

### ✨ Key Features
- **Premium Glassmorphism UI:** A highly aesthetic, modern, and animated Streamlit dashboard designed to impress recruiters.
- **Indian Real Estate Localization:** Predicts prices accurately in Lakhs and Crores (₹) based on Indian property dynamics.
- **Interactive EDA:** Features interactive Plotly visualizations for deep data insights.
- **Dynamic Property Profiling:** Generates an automated Radar Chart based on the inputs (Size, Luxury, Utility).

---

## ⚙️ Tech Stack & Architecture

This project was built using an **Advanced Stack** to ensure production-level quality:
- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning Models:** Linear Regression, Random Forest, XGBoost
- **Frontend / UI:** Streamlit with Custom Glassmorphism CSS
- **Visualizations:** Plotly (Interactive Charts)

**Architecture Flow:**
`Raw Housing Data` ➔ `Data Preprocessing (Handling missing, Scaling, Encoding)` ➔ `Model Training (XGBoost)` ➔ `Web Dashboard (Streamlit)` ➔ `Price Prediction & Market Insights`.

### 📊 Model Performance
Three models were evaluated during training. **XGBoost Regressor** emerged as the best performer:
- **R² Score:** 0.9949 (Highly Accurate)
- **Mean Absolute Error (MAE):** ~₹5,98,199
- **Root Mean Squared Error (RMSE):** ~₹7,49,000

### 🗄️ Dataset Details
The dataset (`data/housing_data.csv`) is logically generated to simulate the Indian real estate market. It includes 2,500 samples with features like:
- `sqft_area`, `bedrooms`, `bathrooms`, `age_of_property`
- `location` (Downtown, Suburbs, Rural, etc.)
- `furnishing_status` (Fully, Semi, Unfurnished)
- `Amenities` (Pool, Gym, Parking)

---

## 📂 Folder Structure

```text
House-Price-Prediction/
│
├── data/
│   └── housing_data.csv       # The dataset used for training
├── notebooks/                 # (Optional) Jupyter notebooks for EDA
├── src/
│   ├── data_generator.py      # Script to generate realistic synthetic housing data
│   └── model_training.py      # ML Pipeline script (Cleaning, Training, Evaluation)
├── models/
│   └── best_model.pkl         # Saved trained model and preprocessors
├── assets/
│   └── style.css              # Custom styling for premium Glassmorphism UI
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Required Python libraries
└── README.md                  # Project documentation
```

---

## 🚀 Installation & Setup Guide

Follow these steps to run the project locally on your machine.

### 1. Prerequisites
Ensure you have Python installed (3.9 or higher).

### 2. Create a Virtual Environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ How to Run the Project

**Step 1: Generate the Data**
```bash
python src/data_generator.py
```
*(This generates a highly realistic synthetic dataset with 2,500 property records and saves it in the `data/` folder).*

**Step 2: Train the Model**
```bash
python src/model_training.py
```
*(This cleans the data, trains 3 regression models, compares them using MAE, RMSE, and R² scores, and saves the best model - usually XGBoost - into the `models/` folder).*

**Step 3: Launch the Dashboard**
```bash
streamlit run app.py
```
*(This opens the beautiful interactive web app in your browser at `http://localhost:8501`).*

---

## 🎮 Virtual Simulation (How it Works)
When you launch the app, you are simulating a real-world property valuation portal:
1. **Input:** You enter a property's details (e.g., 2500 sqft, Downtown, 3 Beds, with Pool).
2. **Processing:** The app sends this data to the pre-trained `XGBoost` model pipeline.
3. **Prediction:** The model scales the data, one-hot encodes the location, and returns a predicted price.
4. **Insight:** The dashboard displays the price in a glowing UI and shows a "Property Profile Profile" radar chart indicating the property's strengths (Luxury, Utility, Size).



## 🐙 GitHub Upload Steps

1. Create a new repository on GitHub named `House-Price-Prediction-ML`.
2. Open your terminal in the project folder and run:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Complete End-to-End House Price Prediction System"
   git branch -M main
   git remote add origin https://github.com/HarshalNavale45/House-Price-Prediction-ML.git
   git push -u origin main
   ```
3. **Tags to Add on GitHub:** `machine-learning`, `data-science`, `regression`, `xgboost`, `streamlit`, `real-estate`.

---

## 📈 Learning Outcomes
- End-to-End Machine Learning Pipeline architecture.
- Feature Engineering and Synthetic Data Generation logic.
- Handling Categorical data with `ColumnTransformer` and `Pipelines`.
- Hyper-tuning and deploying an XGBoost Regression model.
- Advanced UI/UX design in Streamlit using CSS injections and Plotly.

---

## 🔗 Connect with Me
*If you liked this project, feel free to connect with me!*
- **LinkedIn:** [Harshal Navale](https://www.linkedin.com/in/harshal-navale-a3882a383?utm_source=share_via&utm_content=profile&utm_medium=member_android)
- **GitHub:** [HarshalNavale45](https://github.com/HarshalNavale45)
