import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.model_selection import train_test_split
from models.linear_regression import OLS, BatchGD, SGD, MiniBatchGD
from models.regularization import L2, L1
from models.scaling import StandardizationScaler

df = pd.read_csv("data/house.csv")

X = df.drop("House_Price", axis=1)
y = df["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardizationScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def train_evaluate(name, model):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    score = r2_score(y_test, y_pred)
    print(f"{name} R2 Score: {score}")

train_evaluate("sk_normal equation model", LinearRegression())
train_evaluate("my normal equation model", OLS())

train_evaluate("my batch gradiant decent model", BatchGD(lr=0.1, max_itr=100))

train_evaluate("sklearn SGD", SGDRegressor())
train_evaluate("my SGD", SGD())

train_evaluate("my mini batch model", MiniBatchGD())

train_evaluate("sklearn L2 (ridge) model", Ridge(alpha=0.01))
train_evaluate("my L2 (ridge) model", L2(alpha=0.01))

train_evaluate("sklearn L1 (lasso) model", Lasso())
train_evaluate("my L1 (lasso) model", Lasso())