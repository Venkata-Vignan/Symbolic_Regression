import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 5000

# Physical building features
relative_compactness = np.random.uniform(0.6, 1.0, n_samples)
surface_area = np.random.uniform(500, 900, n_samples)
wall_area = np.random.uniform(200, 400, n_samples)
roof_area = np.random.uniform(100, 300, n_samples)
height = np.random.uniform(3, 7, n_samples)
glazing_area = np.random.uniform(0, 0.4, n_samples)

# Temperature difference (winter)
delta_T = np.random.uniform(15, 25, n_samples)

# True underlying physical heating load model
heating_load = (
    0.08 * surface_area * delta_T / relative_compactness
    + 0.5 * wall_area
    - 2.0 * glazing_area * surface_area
)

# Add small realistic noise
heating_load += np.random.normal(0, 50, n_samples)

df_energy = pd.DataFrame({
    "relative_compactness": relative_compactness,
    "surface_area": surface_area,
    "wall_area": wall_area,
    "roof_area": roof_area,
    "height": height,
    "glazing_area": glazing_area,
    "delta_T": delta_T,
    "heating_load": heating_load
})

df_energy.head()
