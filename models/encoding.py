import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class OneHotEncoding:
    def __init__(self, sparse=False):
        """
        Initialize the encoder.

        Parameters:
        sparse (bool): Whether to return a sparse matrix for the encoded data.
        """
        self.cols_values = None
        self.sparse = sparse

    def fit(self, df):
        """
        Learn the unique values for each categorical column.

        Parameters:
        df (pd.DataFrame): Input DataFrame to fit the encoder.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Select only categorical columns
        self.cols_values = {col: df[col].dropna().unique() for col in df.select_dtypes(include=['object', 'category']).columns}

        if not self.cols_values:
            raise ValueError("No categorical columns found to encode.")

    def transform(self, df):
        """
        Apply one-hot encoding to the DataFrame.

        Parameters:
        df (pd.DataFrame): Input DataFrame to transform.

        Returns:
        pd.DataFrame or scipy.sparse.csr_matrix: Transformed DataFrame or sparse matrix.
        """
        if self.cols_values is None:
            raise RuntimeError("fit() must be called before transform().")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        encoded_data = []
        for col, values in self.cols_values.items():
            if col not in df.columns:
                raise KeyError(f"Column '{col}' is missing in the input DataFrame.")

            # Create one-hot encoded columns
            one_hot = pd.get_dummies(df[col], prefix=col, columns=values)
            encoded_data.append(one_hot)

        # Concatenate all encoded columns with the original DataFrame (excluding the original categorical columns)
        df_encoded = pd.concat([df.drop(columns=self.cols_values.keys()), *encoded_data], axis=1)

        if self.sparse:
            return csr_matrix(df_encoded.values)  # Return sparse matrix
        return df_encoded

    def fit_transform(self, df):
        """
        Fit the encoder and transform the DataFrame.

        Parameters:
        df (pd.DataFrame): Input DataFrame to encode.

        Returns:
        pd.DataFrame or scipy.sparse.csr_matrix: Transformed DataFrame or sparse matrix.
        """
        self.fit(df)
        return self.transform(df)