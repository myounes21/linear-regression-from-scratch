# import numpy as np
#
# class R2Score:
#     def __init__(self):
#         self.r2 = None
#
#     def evaluate(self, y_true, y_pred):
#         """
#         Calculate the R² score.
#
#         Parameters:
#         y_true (array-like): True target values.
#         y_pred (array-like): Predicted target values.
#
#         Returns:
#         float: R² score.
#         """
#         y_true = np.array(y_true)
#         y_pred = np.array(y_pred)
#
#         ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
#         ss_residual = np.sum((y_true - y_pred) ** 2)
#
#         self.r2 = 1 - (ss_residual / ss_total)
#         return self.r2
import numpy as np

def r2_score(y_true, y_pred):
    """
    Calculate the R² score.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns:
    float: R² score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    return 1 - (ss_residual / ss_total)


def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns:
    float: Mean Squared Error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean((y_true - y_pred) ** 2)