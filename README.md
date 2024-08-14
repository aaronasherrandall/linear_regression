# Supervised Regression Example: House Price Prediction

This repository contains a simple example of applying linear regression to a small dataset. The goal is to predict house prices based on the size of the house (in square feet).

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Algebraic Method](#algebraic-method)
  - [Linear Combination](#linear-combination)
  - [Example Calculation](#example-calculation)
- [Python Implementation](#python-implementation)
- [Plotting the Data](#plotting-the-data)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

Linear Regression is a fundamental algorithm in machine learning used for predicting continuous target variables. This example demonstrates how linear regression can be applied to predict house prices based on house size using a simple dataset.

## Dataset

The dataset used in this example consists of five data points with one feature:

| House Size (sq ft) | Price (k$) |
|--------------------|------------|
| 750                | 150        |
| 800                | 180        |
| 850                | 200        |
| 900                | 220        |
| 1000               | 300        |

## Algebraic Method

### Linear Combination

The linear combination in linear regression is calculated as:

\[
y = w \cdot x + b
\]

Where:

- \( y \) is the predicted price.
- \( x \) is the input feature (house size).
- \( w \) is the weight (slope).
- \( b \) is the bias (y-intercept).

### Example Calculation

If we assume an initial weight \( w = 0.5 \) and bias \( b = 50 \), the prediction of a house size \( x = 750 \) is:

\[
y = 0.5 \cdot 750 + 50 = 375 + 50 = 425 \text{ k$}
\]

## Python Implementation

Here’s how to implement linear regression using Python's scikit-learn:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Dataset
X = np.array([[750], [800], [850], [900], [1000]])  # House sizes
y = np.array([150, 180, 200, 220, 300])  # Prices

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict the prices for the test set
y_pred = model.predict(X_test)

# Print the learned weights (coefficients) and bias (intercept)
print(f"Weights (slope): {model.coef_}")
print(f"Bias (intercept): {model.intercept_}")
