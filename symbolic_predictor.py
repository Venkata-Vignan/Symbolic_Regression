from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SYMBOLIC_FEATURES = ("X1", "X2", "X4", "X6")
SYMBOLIC_EQUATION_TEXT = (
    "X2 + 28.2412583767502 * (0.188173289911719 * X4 + 1 - "
    "0.355119484864328 / (X6 + 2.8750505))**2 + 0.8409344 / (X1 + 0.54067427)"
)
SYMBOLIC_EQUATION_LATEX = (
    r"X_{2} + 28.2412583767502 \left(0.188173289911719 X_{4} + 1 - "
    r"\frac{0.355119484864328}{X_{6} + 2.8750505}\right)^{2} + "
    r"\frac{0.8409344}{X_{1} + 0.54067427}"
)


@dataclass(frozen=True)
class SymbolicMetadata:
    name: str = "Symbolic Regression"
    complexity: int = 18
    backend: str = "equation"


def predict_symbolic(feature_frame: pd.DataFrame) -> np.ndarray:
    missing = [feature for feature in SYMBOLIC_FEATURES if feature not in feature_frame.columns]
    if missing:
        raise KeyError(
            "Symbolic predictor is missing required features: " + ", ".join(missing)
        )

    x1 = feature_frame["X1"].to_numpy(dtype=float)
    x2 = feature_frame["X2"].to_numpy(dtype=float)
    x4 = feature_frame["X4"].to_numpy(dtype=float)
    x6 = feature_frame["X6"].to_numpy(dtype=float)

    return (
        x2
        + 28.2412583767502
        * (
            0.188173289911719 * x4
            + 1
            - 0.355119484864328 / (x6 + 2.8750505)
        ) ** 2
        + 0.8409344 / (x1 + 0.54067427)
    )


def predict_symbolic_from_raw(
    feature_frame: pd.DataFrame,
    scaler,
    feature_names,
) -> np.ndarray:
    scaled = pd.DataFrame(
        scaler.transform(feature_frame[list(feature_names)]),
        columns=list(feature_names),
        index=feature_frame.index,
    )
    return predict_symbolic(scaled)
