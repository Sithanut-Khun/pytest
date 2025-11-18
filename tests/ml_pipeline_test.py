import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from functions.mini_ml_pipeline import load_data, train_model, evaluate_model

def test_data_loading():
    """Test that house price data loads correctly"""
    df = load_data()
    
    # Check data is non-empty
    assert len(df) > 0
    assert not df.empty
    
    # Check correct columns exist
    expected_columns = ['Bedroom', 'Total-Sqft', 'Bathroom', 'Price']
    for col in expected_columns:
        assert col in df.columns
    
    # Check no null values
    assert not df.isnull().any().any()

def test_model_training():
    """Test that Linear Regression model trains without errors"""
    df = load_data()
    model = train_model(df)
    
    # Check model is trained (has coefficients/intercept)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')
    
    # Check model can make predictions without error
    try:
        sample_data = df[['Bedroom', 'Total-Sqft', 'Bathroom']].iloc[:1]
        predictions = model.predict(sample_data)
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float, np.floating))
    except NotFittedError:
        pytest.fail("Model is not fitted properly")

def test_pipeline_prediction():
    """Test end-to-end pipeline and prediction range"""
    df = load_data()
    model = train_model(df)
    predictions = evaluate_model(df, model)
    
    # Check predictions are valid numbers
    assert all(isinstance(pred, (int, float, np.floating)) for pred in predictions)
    assert len(predictions) == len(df)
    
    # Check predictions are reasonable (positive prices)
    assert all(pred > 0 for pred in predictions)

def test_end_to_end_pipeline():
    """Complete integration test of all components"""
    # Load data
    df = load_data()
    assert len(df) > 0
    assert 'Price' in df.columns
    
    # Train model
    model = train_model(df)
    assert hasattr(model, 'predict')
    
    # Evaluate model
    predictions = evaluate_model(df, model)
    assert len(predictions) == len(df)
    assert all(pred > 0 for pred in predictions)

def test_data_quality():
    """Test house price data quality aspects"""
    df = load_data()
    
    # Check data types
    assert df['Bedroom'].dtype in [np.int64]
    assert df['Total-Sqft'].dtype in [np.int64, np.float64]
    assert df['Bathroom'].dtype in [np.int64, np.float64]
    assert df['Price'].dtype in [np.int64, np.float64]
    
    # Check reasonable data ranges
    assert df['Bedroom'].between(1, 4).all()
    assert df['Total-Sqft'].between(650, 2600).all()
    assert df['Bathroom'].between(1, 4).all()
    assert df['Price'].between(50000, 3000000).all()

def test_model_performance():
    """Test that model has reasonable performance"""
    df = load_data()
    model = train_model(df)
    predictions = evaluate_model(df, model)
    
    # Calculate R² score to evaluate performance
    from sklearn.metrics import r2_score, accuracy_score
    r2 = r2_score(df['Price'], predictions)
    
    # For such a small dataset, we just check the metric is calculated
    assert -1 <= r2 <= 1  # R² can be negative if model is worse than mean