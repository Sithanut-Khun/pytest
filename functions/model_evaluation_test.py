import pytest
import numpy as np
from clean_data import evaluate_model

# Tests (written first, before implementation)
def test_perfect_accuracy():
    """Test accuracy = 1.0 for perfect predictions"""
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 1]
    
    results = evaluate_model(y_true, y_pred)
    
    assert results['accuracy'] == 1.0
    assert results['f1_score'] == 1.0

def test_all_wrong_predictions():
    """Test F1 score = 0.0 when all predictions are wrong"""
    y_true = [1, 1, 1, 1]  # All positive
    y_pred = [0, 0, 0, 0]  # All predicted negative
    
    results = evaluate_model(y_true, y_pred)
    
    assert results['f1_score'] == 0.0
    assert results['accuracy'] == 0.0

def test_output_keys():
    """Test output contains both accuracy and f1_score keys"""
    y_true = [1, 0, 1]
    y_pred = [1, 0, 0]
    
    results = evaluate_model(y_true, y_pred)
    
    assert 'accuracy' in results
    assert 'f1_score' in results
    assert len(results) == 2  # Only these two keys

def test_mixed_predictions():
    """Test with mixed correct and incorrect predictions"""
    y_true = [1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 1, 1]  # 2 wrong predictions
    
    results = evaluate_model(y_true, y_pred)
    
    # 4 correct out of 6 = 0.666 accuracy
    expected_accuracy = 4 / 6
    assert results['accuracy'] == pytest.approx(expected_accuracy)
    assert 0 <= results['f1_score'] <= 1

def test_binary_f1_calculation():
    """Test F1 score calculation with known values"""
    y_true = [1, 0, 1, 0, 1, 0]  # 3 positives, 3 negatives
    y_pred = [1, 0, 0, 0, 1, 1]  # TP=2, FP=1, FN=1
    
    results = evaluate_model(y_true, y_pred)
    
    # Manual F1 calculation:
    # Precision = TP/(TP+FP) = 2/(2+1) = 0.666
    # Recall = TP/(TP+FN) = 2/(2+1) = 0.666
    # F1 = 2 * (precision * recall) / (precision + recall) = 0.666
    expected_f1 = 2 / 3
    assert results['f1_score'] == pytest.approx(expected_f1)

def test_multiclass_f1():
    """Test with multiclass labels"""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]  # 2 wrong predictions
    
    results = evaluate_model(y_true, y_pred)
    
    # 4 correct out of 6 = 0.666 accuracy
    expected_accuracy = 4 / 6
    assert results['accuracy'] == pytest.approx(expected_accuracy)
    assert 0 <= results['f1_score'] <= 1

def test_empty_arrays():
    """Test with empty arrays"""
    y_true = []
    y_pred = []
    
    results = evaluate_model(y_true, y_pred)
    
    # Should handle empty case gracefully
    assert results['accuracy'] == 0.0
    assert results['f1_score'] == 0.0

def test_numpy_arrays():
    """Test that function works with numpy arrays"""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    
    results = evaluate_model(y_true, y_pred)
    
    assert 'accuracy' in results
    assert 'f1_score' in results
    assert 0 <= results['accuracy'] <= 1
    assert 0 <= results['f1_score'] <= 1