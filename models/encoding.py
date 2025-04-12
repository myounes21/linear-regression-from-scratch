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
            one_hot = pd.get_dummies(df[col], prefix=col, columns=values).astype(int)

            # Generate expected column names based on what was seen in fit()
            expected_cols = [f"{col}_{v}" for v in values]
            # Reindex to add any missing columns with 0s
            one_hot = one_hot.reindex(columns=expected_cols, fill_value=0)

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




class LabelEncoding:
    def __init__(self):
        self.mapping = {}

    def fit(self, df):
        """
        Fit label encoders for all categorical columns in the DataFrame.

        Parameters:
        df (pd.DataFrame): Input DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # Only encode object or category columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            unique_vals = df[col].dropna().unique()
            self.mapping[col] = {val: idx for idx, val in enumerate(unique_vals)}

    def transform(self, df):
        """
        Transform the categorical columns using the learned encodings.

        Parameters:
        df (pd.DataFrame): Input DataFrame to transform.

        Returns:
        pd.DataFrame: Encoded DataFrame.
        """
        if not self.mapping:
            raise RuntimeError("fit() must be called before transform().")

        df_encoded = df.copy()

        for col, col_map in self.mapping.items():
            if col not in df.columns:
                raise KeyError(f"Column '{col}' is missing in the input DataFrame.")
            df_encoded[col] = df[col].map(col_map).fillna(-1).astype(int)

        return df_encoded

    def fit_transform(self, df):
        """
        Fit and transform in one step.

        Parameters:
        df (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Encoded DataFrame.
        """
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df_encoded):
        """
        Convert encoded DataFrame back to original categories.

        Parameters:
        df_encoded (pd.DataFrame): DataFrame with encoded values.

        Returns:
        pd.DataFrame: Decoded DataFrame.
        """
        df_decoded = df_encoded.copy()

        for col, col_map in self.mapping.items():
            reverse_map = {v: k for k, v in col_map.items()}
            if col in df_encoded.columns:
                df_decoded[col] = df_encoded[col].map(reverse_map)

        return df_decoded
