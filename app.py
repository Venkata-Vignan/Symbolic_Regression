import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("Energy Efficiency Simulation")
st.subheader("Symbolic Regression-Based Heating Load Prediction")

st.markdown("Adjust building parameters to simulate heating load.")

# ==============================
# INPUT SLIDERS
# ==============================

relative_compactness = st.slider("Relative Compactness (x1)", 0.6, 1.0, 0.8)
surface_area = st.slider("Surface Area (x2)", 500.0, 900.0, 700.0)
wall_area = st.slider("Wall Area (x3)", 200.0, 400.0, 300.0)
roof_area = st.slider("Roof Area (x4)", 100.0, 300.0, 200.0)
height = st.slider("Overall Height (x5)", 3.0, 7.0, 5.0)
orientation = st.slider("Orientation (x6)", 2.0, 5.0, 3.0)
glazing_area = st.slider("Glazing Area (x7)", 0.0, 0.4, 0.2)
glazing_distribution = st.slider("Glazing Area Distribution (x8)", 0.0, 5.0, 2.0)

# ==============================
# SYMBOLIC REGRESSION EQUATION
# ==============================

x1 = relative_compactness
x2 = surface_area
x4 = roof_area
x6 = orientation

heating_load = (
    x2
    + 28.2412583767502
    * (
        0.188173289911719 * x4
        + 1
        - 0.355119484864328 / (x6 + 2.8750505)
    ) ** 2
    + 0.8409344 / (x1 + 0.54067427)
)

st.success(f"Predicted Heating Load: {heating_load:.4f}")

st.markdown("Discovered Equation")
st.code(
    "x2 + 28.2412583767502*(0.188173289911719*x4 + 1 - "
    "0.355119484864328/(x6 + 2.8750505))**2 + "
    "0.8409344/(x1 + 0.54067427)"
)

# ==============================
# LIVE RESPONSE CURVE (Roof Area Effect)
# ==============================

st.markdown("Live Response Curve (Roof Area Effect)")

x_vals = np.linspace(100, 300, 100)
y_vals = (
    x2
    + 28.2412583767502
    * (
        0.188173289911719 * x_vals
        + 1
        - 0.355119484864328 / (x6 + 2.8750505)
    ) ** 2
    + 0.8409344 / (x1 + 0.54067427)
)

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals)
ax.set_xlabel("Roof Area (x4)")
ax.set_ylabel("Heating Load")
ax.set_title("Heating Load vs Roof Area")

st.pyplot(fig)

# ==============================
# 3D SURFACE VISUALIZATION
# ==============================

st.markdown("3D Surface Visualization (Roof Area vs Orientation)")

roof_range = np.linspace(100, 300, 50)
orientation_range = np.linspace(2, 5, 50)

X, Y = np.meshgrid(roof_range, orientation_range)

Z = (
    x2
    + 28.2412583767502
    * (
        0.188173289911719 * X
        + 1
        - 0.355119484864328 / (Y + 2.8750505)
    ) ** 2
    + 0.8409344 / (x1 + 0.54067427)
)

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.update_layout(
    scene=dict(
        xaxis_title='Roof Area (x4)',
        yaxis_title='Orientation (x6)',
        zaxis_title='Heating Load'
    ),
    height=600
)

st.plotly_chart(fig)

# ==============================
# OPTIMIZATION (Roof Area)
# ==============================

st.markdown("Optimize Roof Area")

search_range = np.linspace(100, 300, 1000)

search_output = (
    x2
    + 28.2412583767502
    * (
        0.188173289911719 * search_range
        + 1
        - 0.355119484864328 / (x6 + 2.8750505)
    ) ** 2
    + 0.8409344 / (x1 + 0.54067427)
)

optimal_index = np.argmin(search_output)

st.success(
    f"Optimal Roof Area: {search_range[optimal_index]:.4f}\n"
    f"Minimum Heating Load: {search_output[optimal_index]:.4f}"
)
