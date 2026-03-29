# Symbolic Regression for Heating Load Prediction

This project compares traditional machine learning models with symbolic regression for predicting building heating load on the Energy Efficiency dataset. The main goal is to show that symbolic regression can provide a strong balance between predictive performance and interpretability by producing a readable closed-form equation instead of only black-box predictions.

The project includes:
- a training and experimentation notebook
- a Streamlit web app for interactive comparison and presentation
- scripts for bundle generation, result-asset generation, and PySR runtime setup
- saved result images and model artifacts

## Project Goal

The project studies heating load prediction using:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Symbolic Regression with PySR

The key idea is:
- `Random Forest` gives the best predictive accuracy
- `Symbolic Regression` gives the best interpretability, because it produces an explicit mathematical expression that can be analyzed directly

## Current Model Results

These are the current results exported from the latest bundle:

| Model | R2 | RMSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.9118 | 3.0320 | 2.1722 |
| Ridge Regression | 0.9116 | 3.0354 | 2.1908 |
| Lasso Regression | 0.9113 | 3.0409 | 2.2050 |
| Random Forest | 0.9935 | 0.8245 | 0.4714 |
| Symbolic Regression | 0.9708 | 1.7453 | 1.3103 |

Symbolic regression currently has:
- complexity: `15`
- backend in the app bundle: `pysr_model`

## Best Symbolic Equation

The current symbolic model uses the following discovered equation:

```text
X2 + 28.2412583767502 * (0.188173289911719 * X4 + 1 - 0.355119484864328 / (X6 + 2.8750505))**2 + 0.8409344 / (X1 + 0.54067427)
```

Where:
- `X1` = Relative Compactness
- `X2` = Surface Area
- `X3` = Wall Area
- `X4` = Roof Area
- `X5` = Overall Height
- `X6` = Orientation
- `X7` = Glazing Area
- `X8` = Glazing Area Distribution

## Repository Structure

```text
Symbolic_Regression/
|-- app.py
|-- SR.ipynb
|-- scripts/
|   |-- prepare_bundle.py
|   |-- generate_results_assets.py
|-- artifacts/
|   |-- hall_of_fame_archive/
|-- research/
|   |-- Reference Papers/
|   |-- paper_images/
|-- pysr_runtime.py
|-- symbolic_predictor.py
|-- Requirements.txt
|-- README.md
|-- Results/
|-- sr_model_energy_heating.pkl
|-- sr_bundle_energy_heating.pkl
```

## Main Files

- [`SR.ipynb`](e:/OneDrive/Desktop/Symbolic_Regression/SR.ipynb)
  Main experimentation notebook for loading the dataset, preprocessing, training models, and analyzing results.

- [`app.py`](e:/OneDrive/Desktop/Symbolic_Regression/app.py)
  Streamlit application for:
  comparison of traditional and symbolic models,
  interactive prediction,
  visualization,
  optimization,
  and held-out evaluation plots.

- [`prepare_bundle.py`](e:/OneDrive/Desktop/Symbolic_Regression/scripts/prepare_bundle.py)
  Builds the deployment bundle used by the app. It exports:
  scaler, trained models, metrics, valid configurations, and evaluation records.

- [`generate_results_assets.py`](e:/OneDrive/Desktop/Symbolic_Regression/scripts/generate_results_assets.py)
  Regenerates the saved PNG result images in the `Results` folder from the current bundle.

- [`pysr_runtime.py`](e:/OneDrive/Desktop/Symbolic_Regression/pysr_runtime.py)
  Configures the Julia and PySR runtime so the real `PySRRegressor` can load properly in this project.

- [`symbolic_predictor.py`](e:/OneDrive/Desktop/Symbolic_Regression/symbolic_predictor.py)
  Shared symbolic equation utilities used by the app and export scripts.

## Installation

Create and activate a Python virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r Requirements.txt
```

## Requirements

Core Python dependencies:

- `numpy==1.24.4`
- `pandas==2.0.3`
- `scikit-learn==1.3.2`
- `matplotlib==3.7.2`
- `seaborn==0.12.2`
- `pysr==0.12.3`
- `sympy==1.12`
- `streamlit==1.32.2`
- `plotly==5.18.0`

## Julia and PySR Setup

This project uses PySR, which depends on Julia.

Important notes:
- Julia `1.10.10` is currently being used in this project environment.
- The PySR runtime is configured through [`pysr_runtime.py`](e:/OneDrive/Desktop/Symbolic_Regression/pysr_runtime.py).
- A workspace-local Julia depot is used so the symbolic model can load reliably.

If PySR loading fails on another machine, make sure:
- Julia is installed
- the Python environment has `pysr`
- the Julia environment can access the required packages

## How to Run the Project

### 1. Rebuild the Model Bundle

This creates or refreshes the app bundle with:
- scaler
- trained models
- symbolic PySR model
- evaluation records

```powershell
python scripts/prepare_bundle.py --with-symbolic
```

### 2. Regenerate Result Images

This refreshes the charts saved in the `Results` folder:

```powershell
python scripts/generate_results_assets.py
```

### 3. Run the Streamlit App

```powershell
streamlit run app.py
```

## Streamlit App Features

The app includes four major sections:

### Results Comparison
- metric tables for traditional and symbolic models
- R2 comparison chart
- error comparison chart
- radar profile chart
- symbolic equation display

### Visualization
- interactive feature controls
- multi-model response curves
- symbolic regression surface plot
- symbolic contour map
- Random Forest feature importance chart

### Optimization
- model-wise optimal roof area summary
- optimization curves for all models
- difference-vs-symbolic comparison curves

### Prediction
- dataset-backed and custom configuration inputs
- selected-case prediction summary
- symbolic-regression focused comparison
- all-model prediction comparison
- residual chart
- held-out actual-vs-predicted evaluation chart

## Results Folder

The `Results` folder contains saved plots generated from the current model bundle:

- `r^2_comparison.png`
- `error_comparison.png`
- `Act_pred.png`
- `Feature Importance (RF).png`
- `Heating Load Contour Map.png`
- `Optimization Curve.png`
- `3d_visualization.png`
- `complexity_vs_loss_curve.png`

These files can be refreshed using:

```powershell
python scripts/generate_results_assets.py
```

## Workflow Summary

Recommended run order:

```powershell
python scripts/prepare_bundle.py --with-symbolic
python scripts/generate_results_assets.py
streamlit run app.py
```

## Interpretation Summary

This project supports the following conclusion:

- `Random Forest` is the most accurate model for heating load prediction.
- `Symbolic Regression` is the most interpretable model because it provides a readable mathematical expression with competitive performance.
- The app is designed to present symbolic regression as the interpretability-focused model while still comparing it honestly against all other models.

## Notes

- The notebook and scripts should be kept in sync when retraining.
- If the symbolic model is retrained, regenerate the bundle and result images.
- The app depends on the current exported bundle, not on notebook state alone.

## Future Improvements

Possible next improvements:

- add automated tests for bundle integrity and prediction flow
- export notebook results directly as part of the training workflow
- add model explainability notes for each input variable
- add a final executive-summary panel for report or submission screenshots
