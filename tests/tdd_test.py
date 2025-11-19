import pandas as pd
import pytest
from functions.clean_data import normalize_column


@pytest.fixture
def sample_data():
    """Create sample data"""
    data = pd.DataFrame({'score': [45, 2, 58, 60, 78]})
    return data

def test_normalized_values_range(sample_data):
    """Test that all normalized values are between 0 and 1"""
    df = sample_data
    normalized_df = normalize_column(df, 'score')
    
    # Check all values are between 0 and 1
    assert (normalized_df['score'] >= 0).all()
    assert (normalized_df['score'] <= 1).all()

def test_column_length_unchanged(sample_data):
    """Test that output column length matches input"""
    df = sample_data
    original_length = len(df)
    
    normalized_df = normalize_column(df, 'score')
    
    assert len(normalized_df) == original_length
    assert len(normalized_df['score']) == original_length

def test_invalid_column_raises_error(sample_data):
    """Test that invalid column name raises KeyError"""
    df = sample_data
    
    with pytest.raises(KeyError):
         normalize_column(df, 'Age')