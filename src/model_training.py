import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_models():
    print("Loading data...")
    df = pd.read_csv('data/housing_data.csv')
    
    # Feature Engineering (Optional, just an example)
    # E.g., we could add 'price_per_sqft' but that's what we are predicting.
    # Instead, let's keep it simple as requested for a beginner-friendly approach.
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define categorical and numerical features
    numeric_features = ['sqft_area', 'bedrooms', 'bathrooms', 'age_of_property']
    categorical_features = ['location', 'furnishing_status']
    binary_features = ['has_parking', 'has_pool', 'has_gym']
    
    # Preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features) # Already 0/1
        ])
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_pipeline = None
    
    print("\nTraining models and evaluating...")
    print("-" * 50)
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        print(f"Model: {name}")
        print(f"  MAE:  ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  R²:   {r2:.4f}")
        print("-" * 50)
        
        # Save best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = name
            best_pipeline = pipeline

    print(f"\nBest Model: {best_model} with R² = {best_r2:.4f}")
    
    # Save the best model pipeline
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'best_model.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"Saved the best model pipeline to {model_path}")

if __name__ == "__main__":
    train_models()
