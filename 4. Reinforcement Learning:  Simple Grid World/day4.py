
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("synthetic_overfit_dataset.csv")
X = data.drop("target", axis=1)
y = data["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------ Linear Regression ------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

print("=== Linear Regression (Overfitted) ===")
print("MSE:", mean_squared_error(y_test, y_pred_lin))
print("R² Score:", r2_score(y_test, y_pred_lin))

# ------------------ L1 Regularization (Lasso) ------------------
lasso = Lasso(alpha=0.5)  # alpha = λ, higher means more regularization
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print("\n=== Lasso Regression (L1 Regularized) ===")
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R² Score:", r2_score(y_test, y_pred_lasso))
print("Selected Coefficients: \n", lasso.coef_)


# Evaluate both on train and test
print("Linear Regression Train R2:", lin_reg.score(X_train, y_train))
print("Linear Regression Test R2:", lin_reg.score(X_test, y_test))

print("Lasso Regression Train R2:", lasso.score(X_train, y_train))
print("Lasso Regression Test R2:", lasso.score(X_test, y_test))


'''Interpretation:
If train R² is high but test R² is low → Overfitting 😓

If train R² ≈ test R² → Generalization good ✅'''