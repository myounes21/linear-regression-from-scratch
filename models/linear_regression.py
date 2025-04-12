import numpy as np


class BaseLinearRegression:
    def __init__(self, lr=0.01, max_itr=1000, tol=1e-6, batch_size=32):
        self.W = None
        self.coef_ = None
        self.intercept_ = None
        self.lr = lr
        self.max_itr = max_itr
        self.tol = tol
        self.batch_size = batch_size

    def add_bias(self, X):
        """Add bias term (column of ones) to the feature matrix X"""
        return np.c_[np.ones((X.shape[0], 1)), X]

    def predict(self, X):
        """Predict using the learned coefficients"""
        X_bias = self.add_bias(X)  # Ensure bias is added to input features
        return X_bias @ self.W

    def set_coefficients(self):
        """Extract and set the coefficients and intercept"""
        if self.W is not None:
            self.intercept_ = self.W[0]  # First element is the intercept (bias)
            self.coef_ = self.W[1:]  # Remaining elements are the coefficients
        else:
            raise ValueError("Model coefficients (W) are not set. Please fit the model first.")


class OLS(BaseLinearRegression):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """Fit the model using the normal equation"""
        X_bias = self.add_bias(X)

        try:
            # Normal equation: (X.T @ X)^(-1) @ (X.T @ y)
            self.W = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
        except np.linalg.LinAlgError:
            raise ValueError("X.T @ X is singular, can't invert.")

        self.set_coefficients()


# batch gradiant decent
class BatchGD(BaseLinearRegression):
    def fit(self, X, y):
        """Fit the model using gradient descent"""
        X_bias = self.add_bias(X)  # Add bias term to X

        # Initialize weights (W) as zeros
        self.W = np.zeros(X_bias.shape[1])

        prev_loss = float('inf')

        for i in range(self.max_itr):
            # Predictions
            y_hat = self.predict(X)  # Use the predict method here

            # Compute the loss (MSE)
            loss = np.mean((y_hat - y) ** 2)

            # Compute gradients
            dw = (2 / len(X)) * X_bias.T @ (y_hat - y)

            # Update weights
            self.W -= self.lr * dw

            # Early stopping based on loss change
            if abs(prev_loss - loss) < self.tol:
                print(f"Converged in {i} iterations.")
                break

            prev_loss = loss

        self.set_coefficients()


# Stochastic Gradient Descent
class SGD(BaseLinearRegression):
    def fit(self, X, y):
        X_bias = self.add_bias(X)
        n_samples = X_bias.shape[0]
        n_features = X_bias.shape[1]

        # Initialize weights
        self.W = np.zeros(n_features)

        prev_loss = float('inf')

        # Convert target variable to numpy array to avoid issues with pandas index during training
        y = np.array(y)

        for i in range(self.max_itr):
            # Pick a random sample
            random_idx = np.random.randint(0, n_samples)
            xi = X_bias[random_idx]  # shape: (n_features,)
            yi = y[random_idx]      # shape: scalar

            # Prediction
            yi_hat = xi @ self.W

            # Gradient
            dw = 2 * (yi_hat - yi) * xi

            # Update weights
            self.W -= self.lr * dw

            # Compute total loss
            if i % 100 == 0:
                y_hat_all = X_bias @ self.W
                loss = np.mean((y_hat_all - y) ** 2)

                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss

        self.set_coefficients()


# Mini Batch gradiant decent
class MiniBatchGD(BatchGD):
    def fit(self, X, y):
        X_bias = self.add_bias(X)  # This should add an extra column of ones for the bias
        n_samples, n_features = X_bias.shape  # n_features should now include the bias term

        self.W = np.zeros(n_features)  # Initialize weights to match the size of X_bias (including the bias)

        y = np.array(y)  # Convert target variable to numpy array to avoid issues with pandas index during training

        prev_loss = float('inf')

        for i in range(self.max_itr):
            # Shuffle data at the start of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_bias_shuffled = X_bias[indices]
            y_shuffled = y[indices]

            # Mini-batches loop
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                xb = X_bias_shuffled[start:end]
                yb = y_shuffled[start:end]

                # Prediction
                yb_hat = xb @ self.W

                # Gradient
                dw = (2 / len(xb)) * xb.T @ (yb_hat - yb)

                # Update weights
                self.W -= self.lr * dw

            # Calculate loss after each epoch
            y_hat_all = X_bias @ self.W  # Now X_bias and self.W should be compatible
            loss = np.mean((y_hat_all - y) ** 2)

            # Early stopping
            if abs(prev_loss - loss) < self.tol:
                print(f"Converged at epoch {i}")
                break
            prev_loss = loss

        self.set_coefficients()