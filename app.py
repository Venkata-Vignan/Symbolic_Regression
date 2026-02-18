import streamlit as st
import numpy as np

st.title("Energy Efficiency Simulation")
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
# Symbolic Regression Equation
# Heating Load = 3.090366 * sqrt(1 - 0.104708072692815 * x6^4)
# --------------------------------------------

x6 = input_data[:, 6]

heating_load = 3.090366 * np.sqrt(
    np.maximum(0, 1 - 0.104708072692815 * (x6 ** 4))
)


st.success(f"Predicted Heating Load: {heating_load[0]:.4f}")

st.markdown("Discovered Equation")
st.code("Heating Load = 3.3015666 - 0.30122823 * (Glazing Area)^4")


st.markdown("Live Response Curve (Glazing Area Effect)")

x_vals = np.linspace(0, 0.4, 100)
y_vals = 3.090366 * np.sqrt(
    np.maximum(0, 1 - 0.104708072692815 * (x_vals ** 4))
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals)
ax.set_xlabel("Glazing Area")
ax.set_ylabel("Heating Load")
ax.set_title("Heating Load vs Glazing Area")

st.pyplot(fig)

st.markdown("### 🏔 3D Surface Visualization")

import plotly.graph_objects as go

x_range = np.linspace(0, 0.4, 50)
y_range = np.linspace(0, 1, 50)

X, Y = np.meshgrid(x_range, y_range)
Z = 3.090366 * np.sqrt(
    np.maximum(0, 1 - 0.104708072692815 * (X ** 4))
)

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.update_layout(
    scene=dict(
        xaxis_title='Glazing Area',
        yaxis_title='Dummy Variable',
        zaxis_title='Heating Load'
    ),
    height=600
)

st.plotly_chart(fig)

# Example dummy linear model
linear_pred = 2.8 - 0.2 * x6

st.markdown("### ⚖ Model Comparison")

st.write("Symbolic Prediction:", heating_load[0])
st.write("Linear Approximation:", linear_pred[0])


st.markdown("### 🎯 Optimize Glazing Area")

best_x6 = np.linspace(0, 0.4, 1000)
best_y = 3.090366 * np.sqrt(
    np.maximum(0, 1 - 0.104708072692815 * (best_x6 ** 4))
)

optimal_index = np.argmin(best_y)

st.success(
    f"Optimal Glazing Area: {best_x6[optimal_index]:.4f}\n"
    f"Minimum Heating Load: {best_y[optimal_index]:.4f}"
)



