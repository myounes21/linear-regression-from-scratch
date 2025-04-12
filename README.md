# 🧠 Linear Regression from Scratch

This project is a complete implementation of **Linear Regression from scratch using NumPy**, designed to be clean, modular, and educational. It covers everything from preprocessing and encoding to regularization and different gradient descent techniques — all without using any machine learning libraries like scikit-learn for the model itself.

---

## 🚀 Features

- 📈 **Optimization Methods**
  - OLS (Ordinary Least Squares)
  - Batch Gradient Descent
  - Stochastic Gradient Descent
  - Mini-batch Gradient Descent

- 🧩 **Regularization**
  - L1 Regularization (Lasso)
  - L2 Regularization (Ridge)

- 🔧 **Preprocessing**
  - Standardization
  - Normalization

- 🎛️ **Encoding**
  - One-Hot Encoding
  - Label Encoding

- 📏 **Evaluation Metrics**
  - R² Score
  - Mean Squared Error (MSE)

---

## 🧠 Why This Project?

This project is built for **learning** and **experimentation**. Instead of relying on external ML libraries, every step — from optimization to regularization — is written from scratch using only NumPy. It’s a great resource if you're learning:

- The math behind regression
- How regularization affects optimization
- How preprocessing affects learning
- How to structure ML code modularly

---

## 🗂️ Project Structure

├── data/                        # Dataset files (raw or processed)
│
├── models/                      # All model-related code
│   ├── linear_regression.py     # OLS, Batch GD, SGD, Mini-batch GD implementations
│   ├── regularization.py        # L1 (Lasso) and L2 (Ridge) regularization logic
│   ├── scaling.py               # Standardization and Normalization functions
│   └── encoding.py              # One-hot and label encoding
│
├── metrics/                     # Model evaluation metrics
│   └── evaluation.py            # R² and MSE functions
│
├── notebook/                    # Jupyter notebooks for demos
│   └── LinearRegressionDemo.ipynb
│
├── main.py                      # Run training, evaluation, and testing
├── README.md                    # Project documentation

