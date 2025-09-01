# -----------------------------------------
# Stock Price Prediction using Multi-Layer Perceptron (MLP)
# -----------------------------------------
# This script predicts the next day's stock closing price using a fully connected neural network (MLP).
# It includes data loading, preprocessing, exploratory data analysis, outlier removal, 
# feature scaling, MLP model creation, training, evaluation, and baseline comparison.
# -----------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv('/content/drive/MyDrive/A.csv')  # Path to your CSV file

# Quick overview of the dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())  # Check for missing values

# -----------------------------
# 2. Prepare Target Variable
# -----------------------------
# Shift 'Close' column by -1 to predict the next day's closing price
df['target'] = df['Close'].shift(-1)

# Remove rows with NaN values created by shifting
df.dropna(inplace=True)

# Drop unnecessary columns
df = df.drop(['Date', 'Adj Close'], axis=1)

# Remove duplicate rows
df = df.drop_duplicates()

print(df.head())

# -----------------------------
# 3. Exploratory Data Analysis (EDA)
# -----------------------------
# Correlation matrix to see relationships between features
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Histograms and boxplots for each feature
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

for col in feature_cols:
    # Histogram with KDE for distribution
    plt.figure(figsize=(6, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    
    # Boxplot to check for outliers
    plt.figure(figsize=(6, 6))
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# -----------------------------
# 4. Outlier Removal using IQR
# -----------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove rows that have outliers in any feature
df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# Optional: visualize features after outlier removal
for col in feature_cols:
    plt.figure(figsize=(6, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col} after outlier removal')
    plt.show()
    
    plt.figure(figsize=(6, 6))
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col} after outlier removal')
    plt.show()

# -----------------------------
# 5. Feature Scaling
# -----------------------------
X = df[feature_cols]  # Features
y = df['target']      # Target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize features

# -----------------------------
# 6. Time-Series Train/Test Split
# -----------------------------
# Split data into training (first 80%) and testing (last 20%) based on time
split_index = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# -----------------------------
# 7. Build MLP Model
# -----------------------------
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))  # First hidden layer
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dense(16, activation='relu'))  # Third hidden layer
model.add(Dense(8, activation='relu'))   # Fourth hidden layer
model.add(Dense(1, activation='linear')) # Output layer for regression

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# -----------------------------
# 8. Train the Model
# -----------------------------
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# -----------------------------
# 9. Evaluate the Model
# -----------------------------
# Evaluate loss on the test set
loss = model.evaluate(X_test, y_test)
print(f'Test Loss (MSE, MAE): {loss}')

# Make predictions
y_pred = model.predict(X_test)

# Plot Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# -----------------------------
# 10. Baseline Model Comparison
# -----------------------------
# Use a dummy regressor to compare against a naive baseline (predicting mean)
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
baseline_score = dummy.score(X_test, y_test)  # R² score
print("Baseline R²:", baseline_score)
