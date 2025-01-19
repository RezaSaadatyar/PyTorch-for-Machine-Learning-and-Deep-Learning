# ================================ Presented by: Reza Saadatyar (2023-2024) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

import copy
import numpy as np
import pandas as pd
from typing import Union
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class MissingData:
    
    def __init__(self, dataset: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Initialize the MissingData class.

        Parameters:
        - dataset (Union[pd.DataFrame, np.ndarray]): Input dataset as a pandas DataFrame or NumPy array.
        
        Import module:
        - from Functions.data_cleaner import MissingData
        
        Example:
        - obj_miss = MissingData(data)
           1. data_clean = obj_miss.fill_mode()        # Fill missing values using mode
           2. data_clean = obj_miss.fill_knn()         # Fill missing values using K-Nearest Neighbors
           3. data_clean = obj_miss.fill_mean()        # Fill missing values using the mean
           4. data_clean = obj_miss.fill_median()      # Fill missing values using the median
           5. data_clean = obj_miss.fill_forward()     # Fill missing values using forward fill
           6. data_clean = obj_miss.fill_backward()    # Fill missing values using backward fill
           7. data_clean = obj_miss.fill_interpolate() # Fill missing values using interpolation
        """
        
        if isinstance(dataset, np.ndarray): # Convert numpy array to DataFrame
            self.data = pd.DataFrame(dataset)
        else:
            self.data = copy.deepcopy(dataset)
            
        # self.label_encoders = {}  # To store label encoders for categorical columns
        self.all_nan = self.data.columns[self.data.isnull().any()].tolist() # Find all columns with NaN
        
        # Identify numerical and object columns
        self.numerical_cols = self.data.select_dtypes(include=['number']).columns
        
        # Find numerical columns with NaN
        self.numerical_nan = self.numerical_cols[self.data[self.numerical_cols].isnull().any()].tolist()

    def drop_missing(self) -> pd.DataFrame:
        """Remove rows with missing values."""
        if len(self.all_nan) < 1:
            print("There are no missing values. Data remains unchanged.")
            return self.data

        print(f"Removing rows with missing values in columns: {list(self.all_nan)}).")
        return self.data.dropna()

    def fill_mean(self) -> pd.DataFrame:
        """Impute missing values with the mean (for numerical columns)."""
        if len(self.numerical_nan) < 1:
            print("There are no missing values. Data remains unchanged.")
            return self.data

        self.data[self.numerical_cols] = self.data[self.numerical_cols].fillna(self.data[self.numerical_cols].mean())
        print(f"NaN values filld with mean in numerical columns: {list(self.numerical_nan)}")
        return self.data

    def fill_median(self) -> pd.DataFrame:
        """Impute missing values with the median (for numerical columns)."""
        if len(self.numerical_nan) < 1:
            print("There are no missing values. Data remains unchanged.")
            return self.data

        self.data[self.numerical_cols] = self.data[self.numerical_cols].fillna(self.data[self.numerical_cols].median())
        print(f"NaN values filld with median in numerical columns: {list(self.numerical_nan)}")
        return self.data

    def fill_mode(self) -> pd.DataFrame:
        """Impute missing values with the mode (for both numerical and categorical columns)."""
        if len(self.all_nan) < 1:
            print("There are no missing values. Data remains unchanged.")
            return self.data
 
        for col in self.all_nan:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        print(f"NaN values filld with mode in columns: {list(self.all_nan)}")
        return self.data

    def fill_forward(self) -> pd.DataFrame:
        """Forward fill missing values (for both numerical and categorical columns)."""
        if len(self.all_nan) < 1:
            print("There are no missing values. Data remains unchanged.")
            return self.data

        self.data = self.data.ffill()
        print(f"All NaN values have been forward filled in every column.: {list(self.all_nan)}")
        return self.data

    def fill_backward(self) -> pd.DataFrame:
        """Backward fill missing values (for both numerical and categorical columns)."""
        if len(self.all_nan) < 1:
            print("There are no missing values. Data remains unchanged.")
            return self.data

        self.data = self.data.bfill()
        print(f"All NaN values have been backward filled in every column: {list(self.all_nan)}")
        return self.data

    def fill_interpolate(self, method: str = 'linear') -> pd.DataFrame:
        """Interpolate missing values (for numerical columns)."""
        if len(self.numerical_nan) < 1:
            print("There are no missing values. Data remains unchanged.")
            return self.data

        self.data[self.numerical_cols] = self.data[self.numerical_cols].interpolate(method=method)
        print(f"NaN values interpolated in numerical columns: {list(self.numerical_nan)}")
        return self.data

    def fill_knn(self, n_neighbors: int = 2) -> pd.DataFrame:
        """Impute missing values using K-Nearest Neighbors (for numerical columns)."""
        if len(self.numerical_nan) < 1:
            print("There are no missing values. Data remains unchanged.")
            return self.data

        imputer = KNNImputer(n_neighbors=n_neighbors)
        self.data[self.numerical_cols] = imputer.fit_transform(self.data[self.numerical_cols])
        print(f"NaN values filled using KNN in numerical columns: {list(self.numerical_nan)}")
        return self.data

    # def fill_predictive(self, target_column):
    #     """Impute missing values using predictive modeling (Random Forest)."""
    #     # Convert categorical target column to numerical if necessary
    #     if self.data[target_column].dtype == 'object':
    #         le = LabelEncoder()
    #         self.data[target_column] = le.fit_transform(self.data[target_column].astype(str))
    #         self.label_encoders[target_column] = le
    #         print(f"Converted categorical column '{target_column}' to numerical for imputation.")

    #     # Separate data into missing and non-missing
    #     missing_data = self.data[self.data[target_column].isna()]
    #     non_missing_data = self.data.dropna()

    #     # Features and target
    #     X = non_missing_data.drop(columns=[target_column])
    #     y = non_missing_data[target_column]

    #     # Train model
    #     if self.data[target_column].dtype == 'object':
    #         model = RandomForestClassifier()
    #     else:
    #         model = RandomForestRegressor()
    #     model.fit(X, y)

    #     # Predict missing values
    #     X_missing = missing_data.drop(columns=[target_column])
    #     predicted_values = model.predict(X_missing)

    #     # Fill missing values
    #     self.data.loc[self.data[target_column].isna(), target_column] = predicted_values

    #     # Convert numerical target column back to categorical if necessary
    #     if target_column in self.label_encoders:
    #         self.data[target_column] = self.label_encoders[target_column].inverse_transform(self.data[target_column].astype(int))
    #         print(f"Converted numerical column '{target_column}' back to categorical after imputation.")

    #     print(f"NaN values imputed using predictive modeling in column: '{target_column}'")
        # return self.data

class OutlierData:
    """
    A class to handle outliers in pandas DataFrames or NumPy arrays using Z-score.
    """
    def __init__(self, dataset: Union[pd.DataFrame, np.ndarray], column: str = None) -> None:
        """
        Initialize the OutlierData class.
        
        Parameters:
        - dataset: Input data as a pandas DataFrame or NumPy array.
        - column: The column to analyze if `dataset` is a DataFrame. Defaults to None.
        
        Import module:
        - from Functions.data_cleaner import OutlierData
        - from Functions.plot_outlier import plot_outlier
        
        Example:
        - obj_outliers = OutlierData(data_clean, column=None)
          1. data_outlier = obj_outliers.zscore(value_thr=2, threshold_active="on") # Detect outliers using the zscore method 
          2. data_outlier = obj_outliers.iqr(iqr_k=1.5)  # Detect outliers using the IQR method
          - plot_outlier(data_clean, title="Feature distribution without applying outlier methods", figsize=(10, 2.5))
          - plot_outlier(data_outlier, title="Feature distribution using IQR method", figsize=(10, 2.5))
        """
        if isinstance(dataset, np.ndarray):  # Convert NumPy array to DataFrame
            self.data = pd.DataFrame(dataset)
        else:
            self.data = copy.deepcopy(dataset)  # Create a deep copy of the DataFrame
        self.column = column

    def zscore(self, value_thr: Union[int, float] = 3, threshold_active: str = "off") -> pd.DataFrame:
        """
        Detect and optionally remove outliers using Z-score.
        Parameters:
        - value_thr: The Z-score threshold for detecting outliers. Defaults to 3.
        - threshold_active: Whether to remove outliers ("on" or "off"). Defaults to "off".
        Returns:
        - The dataset with outliers optionally removed.
        """
        if self.column is not None:
            # Calculate Z-scores for the specified column
            mean = self.data[self.column].mean(axis=0)
            std = self.data[self.column].std(axis=0)
            self.data[self.column] = (self.data[self.column] - mean) / std
            outliers = np.abs(self.data[self.column]) > value_thr  # Detect outliers
        else:
            # Calculate Z-scores for all columns
            self.data = (self.data - self.data.mean()) / self.data.std()
            outliers = np.abs(self.data) > value_thr  # Detect outliers
        
        if threshold_active.lower() == "on":  # Remove outliers if threshold_active is "on"
            self.data = self.data[~outliers.any(axis=1)]

        return self.data
    
    def iqr(self, iqr_k: Union[int, float] = 1.5) -> pd.DataFrame:
        """
        Handle outliers using the Interquartile Range (IQR) method.
        Parameters:
        - iqr_k: The multiplier for IQR to calculate bounds. Defaults to 1.5.
        Returns:
        - The dataset with outliers handled by clipping them to the IQR bounds.
        """
        if self.column is not None:
            # Calculate quartiles and IQR for the specified column
            q1 = np.quantile(self.data[self.column], 0.25, axis=0)
            q3 = np.quantile(self.data[self.column], 0.75, axis=0)
            iqr_ = q3 - q1

            # Calculate bounds
            lower_bound = q1 - iqr_k * iqr_
            upper_bound = q3 + iqr_k * iqr_

            # Clip the column values to the bounds to handle outliers
            self.data[self.column] = self.data[self.column].clip(lower=lower_bound, upper=upper_bound)
        else:
            # Calculate quartiles and IQR for all columns
            q1 = np.quantile(self.data, 0.25, axis=0)
            q3 = np.quantile(self.data, 0.75, axis=0)
            iqr_ = q3 - q1

            # Calculate bounds
            lower_bound = q1 - iqr_k * iqr_
            upper_bound = q3 + iqr_k * iqr_

            # Clip all column values to the bounds to handle outliers
            self.data = self.data.clip(lower=lower_bound, upper=upper_bound)

        return self.data