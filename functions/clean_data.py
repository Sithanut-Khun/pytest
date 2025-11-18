
def clean_data(df):
    """
    Simple clean_data function that removes duplicates and nulls
    """
    df_clean = df.drop_duplicates()
    df_clean = df_clean.dropna()
    return df_clean



def normalize_column(df, column):
    """
    Normalize a column to scale values between 0 and 1
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    # Check if column exists
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    # Create a copy to avoid modifying original DataFrame
    df_copy = df.copy()
    
    # Extract the column values
    values = df_copy[column]
    
    # Handle case where all values are the same
    if values.min() == values.max():
        # Set all to 0 (or could be 1, but 0 is more common)
        df_copy[column] = 0.0
    else:
        # Apply min-max normalization: (x - min) / (max - min)
        df_copy[column] = (values - values.min()) / (values.max() - values.min())
    
    return df_copy



from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using accuracy and F1 score
    
    Parameters:
    y_true: array-like of true labels
    y_pred: array-like of predicted labels
    
    Returns:
    dict: Dictionary containing 'accuracy' and 'f1_score'
    """
    # Handle empty arrays case
    if len(y_true) == 0 and len(y_pred) == 0:
        return {'accuracy': 0.0, 'f1_score': 0.0}
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate F1 score
    # Use 'weighted' average for multiclass, 'binary' for binary
    if len(set(y_true)) <= 2:
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'f1_score': float(f1)
    }