import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load trained symbolic regression model
with open("sr_bundle_energy_heating.pkl", "rb") as f:
    obj = pickle.load(f)

sr_model = obj["model"]

st.title("🏠 Energy Efficiency Simulation")
st.subheader("Symbolic Regression-Based Heating Load Prediction")

st.markdown("Adjust building parameters to simulate heating load.")

# Sliders for input features
relative_compactness = st.slider("Relative Compactness", 0.6, 1.0, 0.8)
surface_area = st.slider("Surface Area", 500.0, 900.0, 700.0)
wall_area = st.slider("Wall Area", 200.0, 400.0, 300.0)
roof_area = st.slider("Roof Area", 100.0, 300.0, 200.0)
height = st.slider("Overall Height", 3.0, 7.0, 5.0)
orientation = st.slider("Orientation", 2.0, 5.0, 3.0)
glazing_area = st.slider("Glazing Area", 0.0, 0.4, 0.2)
glazing_distribution = st.slider("Glazing Area Distribution", 0.0, 5.0, 2.0)

# Prepare input for model (IMPORTANT: must scale same way as training)
input_values = [
    relative_compactness,
    surface_area,
    wall_area,
    roof_area,
    height,
    orientation,
    glazing_area,
    glazing_distribution
]

input_data = pd.DataFrame([input_values])

# Match exactly what model expects
input_data = input_data.iloc[:, :len(sr_model.feature_names_in_)]
input_data.columns = sr_model.feature_names_in_



# If you used scaling during training, you must load scaler and apply it here.
# For now, assuming X_train was already scaled and sr_model expects scaled input.

# Predict
prediction = sr_model.predict(input_data)

st.success(f" Predicted Heating Load: {prediction[0]:.2f}")
