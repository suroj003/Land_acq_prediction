import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("land_acq_final.pkl")
model_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="LandSetu",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 LandSetu")
st.subheader("Land Compensation Estimation")

st.write(
    "Enter the land details below to estimate the compensation amount."
)

# -----------------------------
# Numerical inputs
# -----------------------------

land_area = st.number_input(
    "Land Area (sq m)",
    min_value=0.0,
    value=1000.0
)

guideline_rate = st.number_input(
    "Guideline Rate per sq m",
    min_value=0.0,
    value=1000.0
)

distance_highway = st.number_input(
    "Distance from Highway (km)",
    min_value=0.0,
    value=5.0
)

distance_main_road = st.number_input(
    "Distance from Main Road (km)",
    min_value=0.0,
    value=2.0
)

distance_city = st.number_input(
    "Distance from City (km)",
    min_value=0.0,
    value=5.0
)

existing_structure = st.number_input(
    "Existing Structure Value",
    min_value=0.0,
    value=0.0
)

# -----------------------------
# Categorical inputs
# -----------------------------

land_type = st.selectbox(
    "Land Type",
    ["Agricultural", "Residential", "Commercial"]
)

land_use = st.selectbox(
    "Land Use",
    ["Farming", "Housing", "Shop", "Industrial"]
)

locality_type = st.selectbox(
    "Locality Type",
    ["Rural", "Urban"]
)

district = st.selectbox(
    "District",
    [
        "Barpeta",
        "Nalbari",
        "Charaideo",
        "Golaghat",
        "Sonitpur",
        "Cachar",
        "Nagaon",
        "Kamrup"
    ]
)

acquisition_purpose = st.selectbox(
    "Acquisition Purpose",
    [
        "Industrial",
        "Public Infrastructure",
        "Railway"
    ]
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("💰 Estimate Compensation"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "land_area_sqm": [land_area],
        "land_type": [land_type],
        "land_use": [land_use],
        "locality_type": [locality_type],
        "district": [district],
        "guideline_rate_per_sqm": [guideline_rate],
        "distance_highway_km": [distance_highway],
        "distance_main_road_km": [distance_main_road],
        "distance_city_km": [distance_city],
        "existing_structure_value": [existing_structure],
        "acquisition_purpose": [acquisition_purpose]
    })

    # One-Hot Encoding
    input_data = pd.get_dummies(
        input_data,
        columns=[
            "land_type",
            "land_use",
            "locality_type",
            "district",
            "acquisition_purpose"
        ]
    )

    # Make columns exactly the same as training data
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)

    compensation = prediction[0]

    st.success(
        f"Estimated Compensation: ₹{compensation:,.2f}"
    )