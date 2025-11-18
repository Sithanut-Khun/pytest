import pandas as pd
from clean_data import normalize_column
import numpy as np
import pytest


def test_normalized_values_range():
    """Test that all normalized values are between 0 and 1"""
    df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    normalized_df = normalize_column(df, 'values')
    
    # Check all values are between 0 and 1
    assert (normalized_df['values'] >= 0).all()
    assert (normalized_df['values'] <= 1).all()

def test_column_length_unchanged():
    """Test that output column length matches input"""
    df = pd.DataFrame({'values': [10, 20, 30, 40, 50]})
    original_length = len(df)
    
    normalized_df = normalize_column(df, 'values')
    
    assert len(normalized_df) == original_length
    assert len(normalized_df['values']) == original_length

def test_invalid_column_raises_error():
    """Test that invalid column name raises KeyError"""
    df = pd.DataFrame({'valid_col': [1, 2, 3]})
    
    with pytest.raises(KeyError):
        normalize_column(df, 'invalid_column')

def test_single_value_column():
    """Test normalization with single value (should be 0 or 1)"""
    df = pd.DataFrame({'values': [5]})
    normalized_df = normalize_column(df, 'values')
    
    assert normalized_df['values'].iloc[0] in [0, 1]

def test_constant_column():
    """Test normalization when all values are the same"""
    df = pd.DataFrame({'values': [3, 3, 3, 3]})
    normalized_df = normalize_column(df, 'values')
    
    # All values should be the same after normalization (typically 0)
    assert (normalized_df['values'] == 0).all()

def test_negative_values():
    """Test normalization with negative values"""
    df = pd.DataFrame({'values': [-5, 0, 5]})
    normalized_df = normalize_column(df, 'values')
    
    assert (normalized_df['values'] >= 0).all()
    assert (normalized_df['values'] <= 1).all()

def test_normalization_correctness():
    """Test that normalization produces expected values"""
    df = pd.DataFrame({'values': [0, 5, 10]})
    normalized_df = normalize_column(df, 'values')
    
    # 0 should map to 0, 5 to 0.5, 10 to 1
    expected = [0.0, 0.5, 1.0]
    np.testing.assert_array_almost_equal(normalized_df['values'].values, expected)