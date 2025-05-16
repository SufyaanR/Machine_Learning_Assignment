import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Load data
df = pd.read_csv("china_gdp.csv")
x = df.iloc[:, 0].values.reshape(-1, 1)  # Year
y = df.iloc[:, 1].values                 # GDP

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Polynomial regression can get unstable with large values, especially higher powers. 
# Scaling makes learning easier and more stable for the model
# Scale data by setting the mean to 0 and standard  deviation to 1
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Helps linear regression learn non-linear relationships
poly = PolynomialFeatures(degree=5)
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

# Mean Squared Error
mse_test = mean_squared_error(y_test, y_pred)
mse_train = mean_squared_error(y_train, y_pred_train)

# Root Mean Squared Error
rmse_test = np.sqrt(mse_test)
rmse_train = np.sqrt(mse_train)

# R² Score
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_pred_train)

# Print results
print(f"Test MSE: {mse_test:.2f}")
print(f"Train MSE: {mse_train:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Test R²: {r2_test:.4f}")
print(f"Train R²: {r2_train:.4f}")

# Create smooth curve for plotting
x_plot = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_plot_scaled = scaler.transform(x_plot)  # scale the plot data too!
x_plot_poly = poly.transform(x_plot_scaled)
y_plot = model.predict(x_plot_poly)

# Plotting
plt.figure(figsize=(10, 6))

# Plot actual training data
plt.scatter(x_train, y_train, color='blue', alpha=0.5, label='Training Data')

# Plot actual test data
plt.scatter(x_test, y_test, color='green', alpha=1, label='Test Data')

# Plot polynomial regression curve
plt.plot(x_plot, y_plot, color='red', lw=2, label='Polynomial Regression (degree=5)')

plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('China GDP Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()

# y_test vs y_pred scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.show()

#Cross Validation using K fold
# Scale the entire dataset
x_scaled_all = scaler.fit_transform(x)

# Create polynomial features
x_poly_all = poly.fit_transform(x_scaled_all)

# Initialize model and K-Fold
model2 = LinearRegression()
kf = KFold(n_splits=5   , shuffle=True, random_state=42)
r2_scores = []

# Perform K-Fold Cross Validation
#This loop runs once for each fold
for train_index, test_index in kf.split(x_poly_all):
    x_train_fold, x_test_fold = x_poly_all[train_index], x_poly_all[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    # The model is trained on this fold’s training data
    model2.fit(x_train_fold, y_train_fold)
    y_pred_fold = model2.predict(x_test_fold)

    r2 = r2_score(y_test_fold, y_pred_fold)
    r2_scores.append(r2)

# Print average R² score
avg_r2 = np.mean(r2_scores)
print(f"\nAverage R² score from 5-Fold Cross-Validation: {avg_r2:.4f}")

#Getting the equation
x_poly_unscaled = poly.fit_transform(x)
model.fit(x_poly_unscaled, y)

# Get model coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_    
# Display polynomial equation
# Loops through each coefficient with its index (which becomes the power of x)
# If i == 0, it’s the intercept (just a number)
# If i > 0, it adds * x^i to make it look like a polynomial term
terms = [f"{coeff:.3f} * x^{i}" if i > 0 else f"{coeff:.3f}" 
             for i, coeff in enumerate(coefficients)]
equation = " + ".join(terms)
print("\nPolynomial Regression Equation:\ny = " + equation)
