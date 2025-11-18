import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def load_data():
    """
    Load house price data using relative path
    """
    dataset_path = 'dataset/house_price.csv'

    if not os.path.exists(dataset_path):
        dataset_path = '../dataset/house_price.csv'
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"Successfully loaded data from: {dataset_path}")
        return df
    else:
        raise FileNotFoundError(f"Could not find dataset at: {dataset_path}")

def train_model(df):
    """
    Train a Linear Regression model to predict house prices
    """
    # Prepare features and target
    X = df[['Bedroom', 'Total-Sqft', 'Bathroom']]
    y = df['Price']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def evaluate_model(df, model):
    """
    Make predictions using the trained model
    Returns predictions and evaluation metrics
    """
    # Prepare features
    X = df[['Bedroom', 'Total-Sqft', 'Bathroom']]
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions

def run_house_price_pipeline():
    """
    Complete end-to-end house price prediction pipeline
    """
    df = load_data()
    model = train_model(df)
    predictions = evaluate_model(df, model)
    
    # Calculate evaluation metrics
    r2 = r2_score(df['Price'], predictions)
    mae = mean_absolute_error(df['Price'], predictions)
    
    return {
        'data_shape': df.shape,
        'model_type': type(model).__name__,
        'coefficients': dict(zip(['Bedroom', 'Total-Sqft', 'Bathroom'], model.coef_)),
        'intercept': model.intercept_,
        'r2_score': r2,
        'mae': mae,
        'predictions': predictions,
        'actual_prices': df['Price'].values
    }

def predict_new_house(model, bedroom, sqft, bathroom):
    """
    Predict price for a new house
    """
    new_data = pd.DataFrame({
        'Bedroom': [bedroom],
        'Total-Sqft': [sqft], 
        'Bathroom': [bathroom]
    })
    
    prediction = model.predict(new_data)
    return prediction[0]