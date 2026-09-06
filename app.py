import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("house_price_pipeline.joblib")


model = load_model()

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🏠 California House Price Prediction")

st.write(
    "Enter housing and location details below to estimate the median house value."
)

# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------

longitude = st.number_input(
    "Longitude",
    value=-122.23,
    format="%.2f"
)

latitude = st.number_input(
    "Latitude",
    value=37.88,
    format="%.2f"
)

housing_median_age = st.number_input(
    "Housing Median Age",
    min_value=1.0,
    value=30.0
)

total_rooms = st.number_input(
    "Total Rooms",
    min_value=1.0,
    value=2000.0
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    min_value=0.0,
    value=400.0
)

population = st.number_input(
    "Population",
    min_value=1.0,
    value=1000.0
)

households = st.number_input(
    "Households",
    min_value=1.0,
    value=350.0
)

median_income = st.number_input(
    "Median Income",
    min_value=0.0,
    value=4.0,
    format="%.4f"
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Median House Value: ${prediction:,.2f}"
    )

# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

st.divider()

st.subheader("Model Information")

st.write(
    """
    **Model:** Random Forest Regressor  
    **Trees:** 60  
    **R²:** 0.8169  
    **MAE:** 31,819.25  
    **RMSE:** 48,986.97  

    The model pipeline includes missing-value handling,
    categorical encoding, and Random Forest regression.
    """
)
