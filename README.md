# Stock Price Prediction using Multi-Layer Perceptron (MLP)

This project predicts the **next day's stock closing price** using a **Multi-Layer Perceptron (MLP)** neural network implemented in **TensorFlow/Keras**. It demonstrates the complete workflow for a real-world regression problem, including **data preprocessing, exploratory data analysis, feature scaling, time-series train/test split, model training, evaluation, and baseline comparison**.

---

## 🔍 Project Overview

Stock prices are influenced by many factors and are inherently **time-series data**. Predicting stock prices is a challenging regression problem. In this project:

- We use **historical stock data** including `Open`, `High`, `Low`, `Close`, and `Volume`.
- The **target variable** is the next day’s closing price.
- We build a **fully connected neural network (MLP)** with multiple hidden layers.
- The model is trained and evaluated on a **time-series split** (past 80% as training, last 20% as testing) to prevent data leakage.

---

## 📂 Dataset

- The dataset should be a CSV file with columns:
