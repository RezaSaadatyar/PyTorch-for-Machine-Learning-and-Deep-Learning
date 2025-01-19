# ================================ Presented by: Reza Saadatyar (2023-2024) ====================================
# ================================== E-mail: Reza.Saadatyar@outlook.com ========================================

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def data_normalization(data_train: np.ndarray, data_test: np.ndarray, data_valid: np.ndarray = None, method: str =
                   "StandardScaler") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize training, testing, and optional validation datasets using the specified method.
    
    Parameters:
    - data_train: array-like
        The training dataset.
    - data_test: array-like
        The testing dataset.
    - data_valid: array-like, optional (default=None)
        The validation dataset.
    - method: str, optional (default="StandardScaler")
        The normalization method to use. Options: "StandardScaler" or "MinMaxScaler".
    
    Returns:
    - data_train_norm: array-like
        The normalized training dataset.
    - data_test_norm: array-like
        The normalized testing dataset.
    - data_valid_norm: array-like or None
        The normalized validation dataset, if provided.

    Import module:
    - from Functions.data_normalization import data_normalization

    Example:
    - X_train, X_test, X_valid = data_normalization(x_train, x_test, x_valid, method="StandardScaler")
    """
    # Reshape data if necessary
    data_train = data_train.reshape(-1, 1) if data_train.ndim == 1 else data_train
    data_test = data_test.reshape(-1, 1) if data_test.ndim == 1 else data_test
    if data_valid is not None:
        data_valid = data_valid.reshape(-1, 1) if data_valid.ndim == 1 else data_valid
        
    # Select the normalization method
    if method == "MinMaxScaler":
        norm = MinMaxScaler()
    elif method == "StandardScaler":
        norm = StandardScaler()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'StandardScaler' or 'MinMaxScaler'.")
        
    # Normalize the datasets
    data_train_norm = norm.fit_transform(data_train)
    data_test_norm = norm.transform(data_test)
    data_valid_norm = norm.transform(data_valid) if data_valid is not None else data_valid
    
    return data_train_norm, data_test_norm, data_valid_norm