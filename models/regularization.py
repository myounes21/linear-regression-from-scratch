import numpy as np

class BaseRegularization:
    def __init__(self, lr=0.01, alpha=0.1, batch_size=32, max_iter=1000, tol=1e-6):
        # Initialize hyperparameters and placeholders for weights and bias
        self.lr = lr
        self.alpha = alpha  # Regularization strength
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol  # Tolerance for early stopping
        self.W = None
        self.b = None

    def predict(self, X):
        # Perform predictions using the linear model: y = X @ W + b
        return X @ self.W + self.b


class L2(BaseRegularization):
    def fit(self, X, y):
        y = np.array(y)  # Ensure y is a NumPy array for compatibility
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.W = np.zeros(n_features)
        self.b = 0

        # Initialize previous loss and assign it to infinite to make sure the iteration happens at least onece
        prev_loss = float("inf")

        for epoch in range(self.max_iter):
            # Shuffle the data to ensure randomness in mini-batches
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                y_hat = xb @ self.W + self.b

                # Compute the gradiants
                gradient_W = -(2 / len(xb)) * xb.T @ (yb - y_hat) + 2 * self.alpha * self.W
                gradient_b = -(2 / len(xb)) * np.sum(yb - y_hat)

                # Update weights and bias
                self.W -= self.lr * gradient_W
                self.b -= self.lr * gradient_b

            # Compute the loss for the entire dataset
            y_hat_all = X @ self.W + self.b
            loss = np.mean((y - y_hat_all) ** 2) + self.alpha * np.sum(self.W ** 2)

            # Check for convergence using early stopping
            if abs(prev_loss - loss) < self.tol:
                print(f"Converged at epoch {epoch}")
                break
            prev_loss = loss

class L1(BaseRegularization):
    def fit(self, X, y):
        y = np.array(y)
        n_samples, n_features = X.shape

        self.W = np.zeros(n_features)
        self.b = 0

        prev_loss = float("inf")

        for epoch in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                y_hat = xb @ self.W + self.b

                # Note: L1 regularization uses the sign of weights for the gradient
                gradient_W = -(2 / len(xb)) * xb.T @ (yb - y_hat) + self.alpha * np.sign(self.W)
                gradient_b = -(2 / len(xb)) * np.sum(yb - y_hat)

                self.W -= self.lr * gradient_W
                self.b -= self.lr * gradient_b

            # Note: L1 regularization uses the absolute value of weights in the loss
            y_hat_all = X @ self.W + self.b
            loss = np.mean((y - y_hat_all) ** 2) + self.alpha * np.sum(np.abs(self.W))

            if abs(prev_loss - loss) < self.tol:
                print(f"Converged at epoch {epoch}")
                break
            prev_loss = loss