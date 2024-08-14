import numpy as np
from sklearn.model_selection import train_test_split

# Dataset
X = np.array([[750], [800], [850], [900], [1000]]) # House sizes
y = np.array([150, 180, 200, 220, 300]) # Prices

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict the prices for the test set
y_pred = model.predict(X_test)

# Print the predictions and the actual values
for i in range(len(y_test)):
  print(f"Actual Price: {y_test[i]}k, Predicted Prices: {y_pred[i]:.2f}k")

# Calculate model's performance
from sklearn.metrics import mean_squared_error, r2_score

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.2f}")

# Plot the Results

# Plot the training data
import matplotlib.pyplot as plt

# Plot the training data
plt.scatter(X_train, y_train, color='blue', label='Training data')

# Plot the testing data
plt.scatter(X_test, y_test, color='green', label='Testing Data')

# Plot the regression line
plt.plot(X, model.predict(X), color='red', label='Regression Line')

# Labels and title
plt.title("House Size vs Price")
plt.xlabel("House Size (sq ft)")
plt.ylabel('Price (k$)')
plt.legend()

# Display the MSE and R-squared on the plot
plt.text(min(X) + 150, max(y) - 10, f'MSE: {mse:.2f}', fontsize=12, color='black')
plt.text(min(X) + 150, max(y) - 25, f'R²: {r2:.2f}', fontsize=12, color='black')

# Show the plot
plt.show()