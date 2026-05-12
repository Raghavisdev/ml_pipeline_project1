# # Online Shoppers Purchase Prediction System

A Machine Learning based project that predicts whether an online user is likely to make a purchase based on browsing behavior and session data.

## Project Overview

This project analyzes user interaction data from an e-commerce website and predicts purchase intent using different Machine Learning models.

The system uses:
- Data preprocessing
- Feature engineering
- Hyperparameter tuning
- Classification algorithms
- Model evaluation metrics

## Features

- Predicts customer purchase behavior
- Handles numerical and categorical data
- Uses preprocessing pipelines
- Includes visualization and evaluation metrics
- Supports real-time prediction using user inputs

## Machine Learning Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

After comparison, Logistic Regression was selected as the final model because it provided the best balance between recall and overall performance.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle

## Project Structure

```text
PBLfinal.ipynb        -> Main model training notebook
PBLtestingf.ipynb    -> Testing notebook
predict.py           -> Prediction script
final_model.pkl      -> Trained ML model
README.md            -> Project documentation