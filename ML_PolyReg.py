import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("china_gdp.csv")
x = df.iloc[:, 0].values.reshape(-1, 1)  # Year
y = df.iloc[:, 1].values                 # GDP

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Scale data by setting the mean to 0 and standard  deviation to 1
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Helps linear regression learn non-linear relationships
poly = PolynomialFeatures(degree=3)
x_poly_train = poly.fit_transform(x_train_scaled)
x_poly_test = poly.transform(x_test_scaled)

#.fit adjusts/learns the weight coefficients using polynomial features  
model = LinearRegression()
model.fit(x_poly_train, y_train)

# Predictions
y_pred = model.predict(x_poly_test) #evaluate real world performance/predict on test data
y_pred_train = model.predict(x_poly_train) #checks fit quality/predict on training data

# Evaluation
print("Test MAE:", mean_absolute_error(y_test, y_pred))
print("Train MAE:", mean_absolute_error (y_train, y_pred_train))
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.metrics import mean_squared_error
import numpy as np

# Mean Squared Error
mse_test = mean_squared_error(y_test, y_pred)
mse_train = mean_squared_error(y_train, y_pred_train)

# Root Mean Squared Error
rmse_test = np.sqrt(mse_test)
rmse_train = np.sqrt(mse_train)

# Print results
print(f"Test MSE: {mse_test:.2f}")
print(f"Train MSE: {mse_train:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Train RMSE: {rmse_train:.2f}")


from sklearn.metrics import r2_score

# R² Score
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_pred_train)

print(f"Test R²: {r2_test:.4f}")
print(f"Train R²: {r2_train:.4f}")



# Create smooth curve for plotting
x_plot = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_plot_scaled = scaler.transform(x_plot)  # Important: scale the plot data too!
x_plot_poly = poly.transform(x_plot_scaled)
y_plot = model.predict(x_plot_poly)

# Plotting
plt.figure(figsize=(10, 6))

# 1. Plot actual training data
plt.scatter(x_train, y_train, color='blue', alpha=0.5, label='Training Data')

# 2. Plot actual test data
plt.scatter(x_test, y_test, color='green', alpha=1, label='Test Data')

# 3. Plot polynomial regression curve
plt.plot(x_plot, y_plot, color='red', lw=2, label='Polynomial Regression (degree=3)')

plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('China GDP Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()

# Additional: y_test vs y_pred scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.show()