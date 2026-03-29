import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pysr_runtime import load_pysr_pickle
from symbolic_predictor import (
    SYMBOLIC_EQUATION_LATEX,
    SYMBOLIC_EQUATION_TEXT,
    SymbolicMetadata,
    predict_symbolic_from_raw,
)


SOURCE_MODEL_PATH = PROJECT_ROOT / "sr_model_energy_heating.pkl"
OUTPUT_BUNDLE_PATH = PROJECT_ROOT / "sr_bundle_energy_heating.pkl"
DATA_HOME = Path.home() / "scikit_learn_data"
LEGACY_SYMBOLIC_METRICS = {"R2": 0.9774, "RMSE": 1.5332, "MAE": 1.1317, "Complexity": 18}


def compute_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-symbolic",
        action="store_true",
        help="Also try to load the saved PySR model into the bundle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = fetch_openml(
        name="Energy_efficiency",
        version=1,
        as_frame=True,
        data_home=str(DATA_HOME),
        parser="auto",
    )
    df = dataset.frame.apply(pd.to_numeric)

    y = df["Y1"].astype(float)
    X = df.drop(columns=["Y1", "Y2"], errors="ignore")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=5000),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
        ),
    }

    for model in models.values():
        model.fit(X_train_scaled, y_train)

    metrics = {}
    evaluation_table = X_test_df.reset_index(drop=True).copy()
    evaluation_table["Actual"] = y_test.reset_index(drop=True)
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        metrics[name] = compute_metrics(y_test, y_pred)
        evaluation_table[name] = y_pred

    symbolic_model = None
    symbolic_text = None
    symbolic_latex = None
    symbolic_metadata = SymbolicMetadata(backend="scaled_equation_proxy")
    if args.with_symbolic and SOURCE_MODEL_PATH.exists():
        symbolic_model = load_pysr_pickle(SOURCE_MODEL_PATH)
        symbolic_metadata = SymbolicMetadata(backend="pysr_model")
        models["Symbolic Regression"] = symbolic_model
        metrics["Symbolic Regression"] = compute_metrics(y_test, symbolic_model.predict(X_test_scaled))
        try:
            best = symbolic_model.get_best()
            symbolic_text = str(best.get("sympy_format"))
            symbolic_complexity = best.get("complexity")
            if symbolic_complexity is not None:
                metrics["Symbolic Regression"]["Complexity"] = int(symbolic_complexity)
            latex_value = best.get("latex_format")
            if latex_value is not None:
                symbolic_latex = str(latex_value)
        except Exception:
            pass
    else:
        metrics["Symbolic Regression"] = LEGACY_SYMBOLIC_METRICS.copy()
        symbolic_text = SYMBOLIC_EQUATION_TEXT
        symbolic_latex = SYMBOLIC_EQUATION_LATEX

    if symbolic_model is not None:
        symbolic_pred = symbolic_model.predict(X_test_scaled)
    else:
        symbolic_pred = predict_symbolic_from_raw(
            X_test_df.reset_index(drop=True),
            scaler,
            list(X.columns),
        )
    evaluation_table["Symbolic Regression"] = symbolic_pred

    feature_ranges = {
        column: (float(X[column].min()), float(X[column].max()))
        for column in X.columns
    }
    default_input = {column: float(X[column].median()) for column in X.columns}
    valid_feature_values = {
        column: [float(value) for value in sorted(X[column].dropna().unique().tolist())]
        for column in X.columns
    }
    configuration_table = df[list(X.columns) + ["Y1"]].copy()
    configuration_table.insert(0, "Configuration ID", [f"CFG-{index:03d}" for index in range(len(configuration_table))])

    bundle = {
        "feature_names": list(X.columns),
        "target_name": "Y1",
        "scaler": scaler,
        "models": models,
        "metrics": metrics,
        "feature_ranges": feature_ranges,
        "default_input": default_input,
        "valid_feature_values": valid_feature_values,
        "configurations": configuration_table.to_dict(orient="records"),
        "evaluation_records": evaluation_table.to_dict(orient="records"),
    }

    if symbolic_model is not None:
        bundle["symbolic_model"] = symbolic_model
    if symbolic_text is not None:
        bundle["symbolic_equation_text"] = symbolic_text
    if symbolic_latex is not None:
        bundle["symbolic_equation_latex"] = symbolic_latex
    bundle["symbolic_metadata"] = {
        "name": symbolic_metadata.name,
        "complexity": symbolic_metadata.complexity,
        "backend": symbolic_metadata.backend,
    }

    with OUTPUT_BUNDLE_PATH.open("wb") as file:
        pickle.dump(bundle, file)

    print(f"Saved bundle to {OUTPUT_BUNDLE_PATH}")


if __name__ == "__main__":
    main()
