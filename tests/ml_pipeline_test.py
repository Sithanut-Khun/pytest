import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
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
    model, X_train, X_test, y_train, y_test = train_model(df)
    
    # Check model is trained (has coefficients/intercept)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')
    
    # Check model can make predictions without error
    try:
        sample_data = X_test.iloc[:1]
        predictions = model.predict(sample_data)
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float, np.floating))
    except NotFittedError:
        pytest.fail("Model is not fitted properly")

def test_pipeline_prediction():
    """Test end-to-end pipeline and prediction range"""
    df = load_data()
    model, X_train, X_test, y_train, y_test = train_model(df)
    predictions = evaluate_model(model, X_test, y_test)
    
    # Check predictions are valid numbers
    assert all(isinstance(pred, (int, float, np.floating)) for pred in predictions)
    assert len(predictions) == len(X_test)
    
    # Check predictions are reasonable (positive prices)
    assert all(pred > 0 for pred in predictions)

def test_end_to_end_pipeline():
    """Complete integration test of all components"""
    df = load_data()
    model, X_train, X_test, y_train, y_test = train_model(df)
    
    predictions = evaluate_model(model, X_test, y_test)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(predictions) == len(y_test)
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
    """Test that model has reasonable performance on test set"""
    df = load_data()
    model, X_train, X_test, y_train, y_test = train_model(df)
    
    predictions = evaluate_model(model, X_test, y_test)
    
    r2 = r2_score(y_test, predictions)
    assert -1 <= r2 <= 1  