# ML Regression Web App

## Overview
This Streamlit-based web application allows users to upload a dataset, select features and target variables, and perform regression analysis using multiple machine learning models. The application provides model comparison based on performance metrics such as MAE, MSE, RMSE, and R² Score. Users can also fine-tune hyperparameters, visualize model performance, and download predictions.

## Features
- **Upload CSV Dataset**: Users can upload their own dataset in CSV format.
- **Feature Selection**: Select independent variables (features) and the dependent variable (target).
- **Choose Regression Models**: Compare multiple models including:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Extra Trees Regressor
  - Support Vector Regressor
  - Gradient Boosting Regressor
- **Hyperparameter Tuning**:
  - Adjust `alpha` for Ridge and Lasso Regression.
  - Adjust `n_estimators` for ensemble models.
- **Performance Evaluation**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- **Downloadable Results**:
  - Model comparison table.
  - Full dataset with predictions.

## Installation
### Prerequisites
Ensure you have **Python 3.10+** installed. Then, set up a virtual environment:

```sh
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Running the Application
Run the Streamlit app using the following command:
```sh
streamlit run app.py
```
This will launch the app in your default web browser.

## How to Use
1. **Upload Dataset**: Click the file uploader and select a CSV file.
2. **Feature Selection**: Choose feature columns and the target column.
3. **Model Selection**: Select regression models for comparison.
4. **Hyperparameter Tuning**: Adjust `alpha` (for Ridge/Lasso) and `n_estimators` (for ensemble models).
5. **View Results**: The app displays:
   - Model performance table.
   - Recommended metric ranges.
6. **Download Results**:
   - Comparison table (`model_comparison.csv`).
   - Dataset with predictions (`dataset_with_predictions.csv`).



## Dependencies
This project requires:
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`


