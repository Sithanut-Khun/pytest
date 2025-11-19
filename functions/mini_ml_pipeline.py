import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


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
    Train a Linear Regression model using train/test split
    """
    X = df[['Bedroom', 'Total-Sqft', 'Bathroom']]
    y = df['Price']

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Make predictions using the trained model
    Returns predictions and evaluation metrics
    """
    predictions = model.predict(X_test)
    
    return predictions

def run_house_price_pipeline():
    df = load_data()

    model, X_train, X_test, y_train, y_test = train_model(df)
    
    predictions = evaluate_model(model, X_test, y_test)

    # Evaluation on test data only
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return {
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'model_type': type(model).__name__,
        'coefficients': dict(zip(['Bedroom', 'Total-Sqft', 'Bathroom'], model.coef_)),
        'intercept': model.intercept_,
        'r2_score': r2,
        'mae': mae,
        'predictions': predictions,
        'actual_prices': y_test.values
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


if __name__ == "__main__":
    results = run_house_price_pipeline()
    print("Pipeline Results:")
    for key, value in results.items():
        if key in ['predictions', 'actual_prices']:
            print(f"{key}: {value[:5]}...")  # Print only first 5 values
        else:
            print(f"{key}: {value}")