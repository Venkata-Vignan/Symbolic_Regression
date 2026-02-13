import streamlit as st
import numpy as np

st.title("🏠 Energy Efficiency Simulation")
st.subheader("Symbolic Regression-Based Heating Load Prediction")

st.markdown("Adjust building parameters to simulate heating load.")

# Input sliders (must match feature order used in training)

relative_compactness = st.slider("Relative Compactness", 0.6, 1.0, 0.8)
surface_area = st.slider("Surface Area", 500.0, 900.0, 700.0)
wall_area = st.slider("Wall Area", 200.0, 400.0, 300.0)
roof_area = st.slider("Roof Area", 100.0, 300.0, 200.0)
height = st.slider("Overall Height", 3.0, 7.0, 5.0)
orientation = st.slider("Orientation", 2.0, 5.0, 3.0)
glazing_area = st.slider("Glazing Area", 0.0, 0.4, 0.2)
glazing_distribution = st.slider("Glazing Area Distribution", 0.0, 5.0, 2.0)

# Put all inputs into array (IMPORTANT: same order as training)
input_data = np.array([[ 
    relative_compactness,
    surface_area,
    wall_area,
    roof_area,
    height,
    orientation,
    glazing_area,          # x6
    glazing_distribution
]])

# --------------------------------------------
# Symbolic Regression Equation (Pure NumPy)
# Heating Load = 3.3015666 - 0.30122823*x6**4
# --------------------------------------------

x6 = input_data[:, 6]

heating_load = 3.3015666 - 0.30122823 * (x6 ** 4)

st.success(f"🔥 Predicted Heating Load: {heating_load[0]:.4f}")

st.markdown("### 📐 Discovered Equation")
st.code("Heating Load = 3.3015666 - 0.30122823 * (Glazing Area)^4")
