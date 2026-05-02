import pandas as pd
import numpy as np
import os

def generate_house_data(num_samples=2500):
    """
    Generates a realistic synthetic housing dataset.
    """
    np.random.seed(42)
    
    # Base features
    locations = ['Downtown', 'Suburbs', 'Uptown', 'Rural', 'Business District']
    furnishing = ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished']
    
    # Generate random features
    sqft_area = np.random.randint(500, 5000, num_samples)
    bedrooms = np.random.randint(1, 6, num_samples)
    bathrooms = np.random.randint(1, 4, num_samples)
    
    # Ensure bedrooms and bathrooms are somewhat correlated with area
    bedrooms = np.where(sqft_area > 3000, np.random.randint(3, 6, num_samples), bedrooms)
    bathrooms = np.where(sqft_area > 3000, np.random.randint(2, 5, num_samples), bathrooms)
    
    age_of_property = np.random.randint(0, 50, num_samples) # in years
    location = np.random.choice(locations, num_samples)
    furnishing_status = np.random.choice(furnishing, num_samples)
    
    # Binary amenities
    has_parking = np.random.choice([0, 1], num_samples, p=[0.2, 0.8])
    has_pool = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
    has_gym = np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
    
    # Calculate Base Price (logical mapping for Indian Real Estate)
    # Base price per sqft (e.g. ₹5500 per sqft)
    base_price = sqft_area * 5500 
    
    # Location multipliers
    location_multiplier = {
        'Downtown': 1.5,
        'Business District': 1.4,
        'Uptown': 1.2,
        'Suburbs': 0.9,
        'Rural': 0.6
    }
    
    # Furnishing multipliers
    furnish_multiplier = {
        'Fully-Furnished': 1.15,
        'Semi-Furnished': 1.05,
        'Unfurnished': 1.0
    }
    
    # Apply logic to generate realistic price
    prices = []
    for i in range(num_samples):
        price = base_price[i]
        
        # Location effect
        price *= location_multiplier[location[i]]
        
        # Furnishing effect
        price *= furnish_multiplier[furnishing_status[i]]
        
        # Amenities effect
        if has_parking[i]: price += 500000
        if has_pool[i]: price += 1500000
        if has_gym[i]: price += 1000000
        
        # Age penalty (depreciation)
        price -= (age_of_property[i] * 50000)
        
        # Extra bedrooms/bathrooms value
        price += (bedrooms[i] * 500000)
        price += (bathrooms[i] * 300000)
        
        # Add some random noise to make it realistic (not perfectly linear)
        noise = np.random.normal(0, 500000)
        price += noise
        
        # Ensure minimum price
        if price < 1500000:
            price = 1500000 + np.random.randint(100000, 500000)
            
        prices.append(round(price, 2))
        
    # Create DataFrame
    df = pd.DataFrame({
        'sqft_area': sqft_area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_of_property': age_of_property,
        'location': location,
        'furnishing_status': furnishing_status,
        'has_parking': has_parking,
        'has_pool': has_pool,
        'has_gym': has_gym,
        'price': prices
    })
    
    # Introduce some realistic missing values to simulate real-world data cleaning needs
    # (Optional: uncomment if you want to teach imputation)
    # df.loc[np.random.choice(df.index, 50), 'age_of_property'] = np.nan
    
    # Save to data folder
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', 'housing_data.csv')
    df.to_csv(file_path, index=False)
    print(f"Successfully generated {num_samples} samples of housing data.")
    print(f"Saved to: {file_path}")
    print("\nDataset Preview:")
    print(df.head())
    
if __name__ == "__main__":
    generate_house_data()
