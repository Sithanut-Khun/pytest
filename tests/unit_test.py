import pytest
import pandas as pd
from functions.clean_data import clean_data

class TestDataCleaningSimple:
    """Simple test"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with only 3 columns"""
        data = {
            'Hours_Studied': [5, 3, 5, 6, None],
            'Attendance': [90, 85, 90, 88, 95],
            'Exam_Score': [88, 75, 88, 82, None]
        }
        return pd.DataFrame(data)

    def test_remove_duplicates(self, sample_data):
        """Test that duplicates are removed correctly"""
        df = sample_data
        cleaned_df = clean_data(df)
        
        assert cleaned_df.duplicated().sum() == 0
        
        hours_studied = cleaned_df['Hours_Studied'].tolist()
        attendence = cleaned_df['Attendance'].tolist()
        exam_score = cleaned_df['Exam_Score'].tolist()
        
        assert hours_studied.count(5) == 1 
        assert attendence.count(90) == 1
        assert exam_score.count(88) == 1

    def test_remove_nulls(self, sample_data):
        """Test that all null values are dropped"""
        df = sample_data
        cleaned_df = clean_data(df)
        
        assert cleaned_df.isnull().sum().sum() == 0

    def test_rows_decrease(self, sample_data):
        """Test that number of rows decreases after cleaning"""
        df = sample_data
        original_rows = len(df)
        cleaned_df = clean_data(df)
        
        assert len(cleaned_df) < original_rows
        
        assert len(cleaned_df) == 3
        