# California House Price Prediction
## Live Demo

[Open the California House Price Prediction App](https://california-housing-price-prediction-tuplawnenycmzhxvwappirp.streamlit.app/)
End-to-end machine learning project for predicting California house prices using Python, Scikit-learn, preprocessing pipelines, and Random Forest regression.

## Project Overview

This project builds a regression model to predict median house values from housing and location-related features.

The workflow includes:

- Data loading and exploration
- Train/test split before preprocessing
- Missing-value handling
- One-hot encoding for categorical data
- Linear Regression baseline
- Random Forest regression
- Model comparison
- Deployment-oriented model size optimization
- Final model evaluation
- Residual analysis
- Feature importance analysis

## Dataset

The dataset contains 20,640 housing records.

Features include:

- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- ocean_proximity

Target:

- median_house_value

## Models

Two main models were evaluated:

### Linear Regression
Used as the baseline regression model.

### Random Forest Regressor
Random Forest significantly outperformed Linear Regression.

A smaller 60-tree Random Forest was selected as the deployment model because it retained almost the same predictive performance while reducing the serialized model size.

## Final Model Performance

| Metric | Result |
|---|---:|
| MAE | 31,819.25 |
| RMSE | 48,986.97 |
| R² | 0.8169 |

Final model:

**Random Forest Regressor — 60 trees**

Model file size:

**16.73 MB**

## Preprocessing

The complete Scikit-learn pipeline handles:

- Median imputation for numeric missing values
- Most-frequent imputation for categorical missing values
- One-hot encoding for `ocean_proximity`
- Random Forest regression

Because preprocessing is included inside the saved pipeline, raw input features can be passed directly to the model.

## Files

- `california_house_price_prediction.ipynb` — complete ML workflow
- `house_price_pipeline.joblib` — deployment-ready model pipeline
- `california_house_test.csv` — clean held-out test dataset
- `app.py` — Streamlit application
- `requirements.txt` — dependencies

## Explainability

Feature importance analysis is included to identify the variables contributing most to house-price predictions.

This project will also be reused as the regression case study in the **TrustAI** explainable machine learning platform.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## Deployment

A Streamlit web application will provide interactive house-price predictions using the trained pipeline.
