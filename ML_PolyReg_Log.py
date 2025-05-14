import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

# Load data
df = pd.read_csv("china_gdp.csv")
x = df.iloc[:, 0].values.reshape(-1, 1)  # Year
y = df.iloc[:, 1].values                 # GDP (in dollars)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Scale the input features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
x_poly_train = poly.fit_transform(x_train_scaled)
x_poly_test = poly.transform(x_test_scaled)

# 🚀 Log-transform the GDP target
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

# Train model on log-transformed GDP
model = LinearRegression()
model.fit(x_poly_train, y_train_log)

# Predict (still in log scale)
y_pred_log = model.predict(x_poly_test)
y_pred_train_log = model.predict(x_poly_train)

# Convert back to original scale
y_pred = np.exp(y_pred_log)
y_pred_train = np.exp(y_pred_train_log)

# === Evaluation Metrics (on original scale) ===
print("----- Evaluation Metrics -----")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):,.2f}")

print(f"Test MAPE: {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")
print(f"Train MAPE: {mean_absolute_percentage_error(y_train, y_pred_train) * 100:.2f}%")

test_mse = mean_squared_error(y_test, y_pred)
train_mse = mean_squared_error(y_train, y_pred_train)
print(f"Test MSE: {test_mse:,.2f}")
print(f"Train MSE: {train_mse:,.2f}")
print(f"Test RMSE: {np.sqrt(test_mse):,.2f}")
print(f"Train RMSE: {np.sqrt(train_mse):,.2f}") 

# R² Score
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_pred_train)

print(f"Test R²: {r2_test:.4f}")
print(f"Train R²: {r2_train:.4f}")

# === Plot: Prediction Curve ===
x_plot = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_plot_scaled = scaler.transform(x_plot)
x_plot_poly = poly.transform(x_plot_scaled)
y_plot = np.exp(model.predict(x_plot_poly))  # predict in log, convert back

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', alpha=0.5, label='Training Data')
plt.scatter(x_test, y_test, color='green', alpha=1, label='Test Data')
plt.plot(x_plot, y_plot, color='red', lw=2, label='Polynomial Regression (log-transformed)')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('China GDP Polynomial Regression (Log-Transformed Target)')
plt.legend()
plt.grid(True)
plt.show()

# === Plot: Actual vs Predicted (Test Set) ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Actual vs Predicted GDP (Test Set)')
plt.grid(True)
plt.show()
