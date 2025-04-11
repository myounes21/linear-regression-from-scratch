# Project Structure
LinearRegressionProject/
├── data/                # Store dataset files here
├── models/
│   ├── linear_regression.py    # Custom Linear Regression model
│   ├── regularization.py      # Ridge (L2) and Lasso (L1) Regularization
│   ├── preprocessing.py       # Feature scaling and categorical encoding
├── metrics/
│   ├── evaluation.py          # Evaluation metrics (MSE, RMSE, R²)
├── notebook/
│   ├── LinearRegressionDemo.ipynb  # Jupyter Notebook for demo and analysis
├── main.py                   # Script to train and test the model
├── README.md                 # Project explanation and instructions





# Linear Regression from Scratch

This project implements linear regression using various gradient descent methods: **Batch Gradient Descent (GD)**, **Stochastic Gradient Descent (SGD)**, and **Mini-Batch Gradient Descent**. The model is implemented from scratch to gain an understanding of linear regression and optimization techniques.

## Classes

### 1. **BaseLinearRegression**
The base class that provides core functionality such as adding a bias term, making predictions, and setting model coefficients.

- **Methods:**
  - `add_bias(X)`: Adds a bias term (column of ones) to the feature matrix.
  - `predict(X)`: Makes predictions based on the learned coefficients.
  - `set_coefficients()`: Sets the coefficients and intercept after fitting the model.

### 2. **OLS (Ordinary Least Squares)**
Implements the linear regression model using the normal equation.

- **Methods:**
  - `fit(X, y)`: Fits the model using the normal equation to calculate the coefficients directly.

### 3. **BatchGD (Batch Gradient Descent)**
Implements the Batch Gradient Descent algorithm for linear regression.

- **Methods:**
  - `fit(X, y)`: Fits the model using batch gradient descent. The model iterates over the entire dataset to update weights.

### 4. **SGD (Stochastic Gradient Descent)**
Implements the Stochastic Gradient Descent algorithm for linear regression.

- **Methods:**
  - `fit(X, y)`: Fits the model using stochastic gradient descent. It updates weights using a single data point per iteration.

### 5. **MiniBatchGD (Mini-Batch Gradient Descent)**
Implements the Mini-Batch Gradient Descent algorithm for linear regression.

- **Methods:**
  - `fit(X, y)`: Fits the model using mini-batch gradient descent. The model updates weights based on small random subsets of data (mini-batches).

## Methods

### `fit(X, y)`
Fits the model to the data using the corresponding gradient descent method or normal equation.

### `predict(X)`
Makes predictions based on the learned coefficients.

### `add_bias(X)`
Adds a bias term (column of ones) to the feature matrix.

### `set_coefficients()`
Sets the coefficients and intercept after fitting the model.

## How to Use

### Example Code

```python
from models.linear_regression import OLS, BatchGD, SGD, MiniBatchGD

# Load your dataset
# X_train, y_train = ...

# OLS Example
model_ols = OLS()
model_ols.fit(X_train, y_train)
predictions_ols = model_ols.predict(X_test)

# Batch Gradient Descent Example
model_batch_gd = BatchGD(alpha=0.01, max_itr=1000)
model_batch_gd.fit(X_train, y_train)
predictions_batch_gd = model_batch_gd.predict(X_test)

# Stochastic Gradient Descent Example
model_sgd = SGD(alpha=0.01, max_itr=1000)
model_sgd.fit(X_train, y_train)
predictions_sgd = model_sgd.predict(X_test)

# Mini-Batch Gradient Descent Example
model_mini_batch_gd = MiniBatchGD(alpha=0.01, max_itr=1000, batch_size=32)
model_mini_batch_gd.fit(X_train, y_train)
predictions_mini_batch_gd = model_mini_batch_gd.predict(X_test)

