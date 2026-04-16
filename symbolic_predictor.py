from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SYMBOLIC_FEATURES = ("X1", "X3", "X5", "X7")
SYMBOLIC_EQUATION_TEXT = (
    "X3 + (X5 + 2.5169373) * (X7 + 8.7142315 + 0.24151891 / (0.38839805 - X1))"
)
SYMBOLIC_EQUATION_LATEX = (
    r"X_{3} + \left(X_{5} + 2.5169373\right)\left(X_{7} + 8.7142315 + "
    r"\frac{0.24151891}{0.38839805 - X_{1}}\right)"
)


@dataclass(frozen=True)
class SymbolicMetadata:
    name: str = "Symbolic Regression"
    complexity: int = 15
    backend: str = "equation"


def predict_symbolic(feature_frame: pd.DataFrame) -> np.ndarray:
    missing = [feature for feature in SYMBOLIC_FEATURES if feature not in feature_frame.columns]
    if missing:
        raise KeyError(
            "Symbolic predictor is missing required features: " + ", ".join(missing)
        )

    x1 = feature_frame["X1"].to_numpy(dtype=float)
    x3 = feature_frame["X3"].to_numpy(dtype=float)
    x5 = feature_frame["X5"].to_numpy(dtype=float)
    x7 = feature_frame["X7"].to_numpy(dtype=float)

    return (
        x3
        + (x5 + 2.5169373)
        * (x7 + 8.7142315 + 0.24151891 / (0.38839805 - x1))
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
