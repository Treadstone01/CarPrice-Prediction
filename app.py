import streamlit as st
from joblib import load
import numpy as np

# Load the trained model
model = load('artifacts/rfr_predictor.joblib')

# Title and Description
st.title("Car Price Predictor")
st.write("""
    This app predicts the price of a car based on the following features:
    - Wheel Base
    - Length
    - Width
    - Curb Weight
    - Engine Size
    - Horsepower
    - City Fuel Consumption (L/100km)
    - Bore
    - Number of Doors
    - Number of Cylinders
""")

# Input Features
wheel_base = st.number_input("Wheel Base (in cm)", min_value=0.0, step=0.1)
length = st.number_input("Length (in cm)", min_value=0.0, step=0.1)
width = st.number_input("Width (in cm)", min_value=0.0, step=0.1)
curb_weight = st.number_input("Curb Weight (in kg)", min_value=0.0, step=0.1)
engine_size = st.number_input("Engine Size (in cc)", min_value=0.0, step=1.0)
horsepower = st.number_input("Horsepower", min_value=0.0, step=1.0)
city_l_100km = st.number_input("City Fuel Consumption (L/100km)", min_value=0.0, step=0.1)
num_cylinders = st.selectbox("Number of Cylinders", options=[3, 4, 5, 6, 8, 12], index=1)

# Prediction Button
if st.button("Predict Price"):
    # Prepare input data
    input_features = np.array([
        wheel_base, length, width, curb_weight, engine_size,
        horsepower, city_l_100km,num_cylinders
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_features)
    
    # Display the prediction
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
