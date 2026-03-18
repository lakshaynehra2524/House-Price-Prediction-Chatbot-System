# House Price Prediction & Chatbot System

## Overview
This project is a machine learning-based system that predicts house prices and also includes a simple chatbot for user interaction.

The application is built using Streamlit and allows users to input housing features and compare predictions from different regression models.

---

## Features

### 1. House Price Prediction
- Takes user input like:
  - Median Income
  - House Age
  - Rooms, Bedrooms
  - Population
  - Location (Latitude, Longitude)
- Uses trained models to predict house price

### 2. Multiple Model Comparison
- Linear Regression
- Ridge Regression
- Lasso Regression

Users can compare predictions from all three models.

---

### 3. Chatbot
- Answers questions related to:
  - dataset
  - preprocessing
  - regression models
- Uses:
  - TF-IDF Vectorizer
  - Cosine Similarity
  - NLP preprocessing (NLTK)

---

## Tech Stack

- Python
- Streamlit
- scikit-learn
- NLTK
- NumPy
- Joblib / Pickle

---

## Models Used

- Linear Regression
- Ridge Regression
- Lasso Regression

All models are pre-trained and loaded using `.pkl` files.

---

## How It Works

1. User enters housing details  
2. Data is scaled using a saved scaler  
3. Models generate predictions  
4. Results are displayed in UI  

For chatbot:
1. User enters a query  
2. Text is preprocessed  
3. TF-IDF vectorization is applied  
4. Best matching response is returned  

---

## Project Structure
