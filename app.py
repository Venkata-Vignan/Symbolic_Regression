import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from symbolic_predictor import (
    SYMBOLIC_FEATURES,
    predict_symbolic_from_raw,
)


st.set_page_config(
    page_title="Symbolic Regression for Energy Efficiency Prediction: A Comparative Analysis with Traditional Machine Learning Models",
    layout="wide",
)


BUNDLE_PATH = Path("sr_bundle_energy_heating.pkl")
LEGACY_METRICS = {
    "Linear Regression": {"R2": 0.9118038162167849, "RMSE": 3.0319796709720896, "MAE": 2.17221658932295},
    "Ridge Regression": {"R2": 0.9116028949393403, "RMSE": 3.0354313093878527, "MAE": 2.1907793299161833},
    "Lasso Regression": {"R2": 0.9112858626977701, "RMSE": 3.040869656590189, "MAE": 2.204965713990391},
    "Random Forest": {"R2": 0.9934775756360946, "RMSE": 0.8245289599573794, "MAE": 0.4713533817734691},
    "Symbolic Regression": {"R2": 0.9707777473317559, "RMSE": 1.7452523600481433, "MAE": 1.31030150996022, "Complexity": 15},
}
FEATURE_LABELS = {
    "X1": "Relative Compactness",
    "X2": "Surface Area",
    "X3": "Wall Area",
    "X4": "Roof Area",
    "X5": "Overall Height",
    "X6": "Orientation",
    "X7": "Glazing Area",
    "X8": "Glazing Area Distribution",
}
DEFAULT_RANGES = {
    "X1": (0.5, 1.0, 0.75),
    "X2": (500.0, 900.0, 700.0),
    "X3": (200.0, 450.0, 300.0),
    "X4": (100.0, 250.0, 150.0),
    "X5": (3.0, 7.0, 3.5),
    "X6": (2, 5, 2),
    "X7": (0.0, 0.4, 0.25),
    "X8": (0, 5, 2),
}
MODEL_ORDER = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Random Forest",
    "Symbolic Regression",
]
MODEL_COLORS = {
    "Linear Regression": "#6c8ebf",
    "Ridge Regression": "#9a7cff",
    "Lasso Regression": "#f4a261",
    "Random Forest": "#2ec4b6",
    "Symbolic Regression": "#ff6b6b",
}
MODEL_DASHES = {
    "Linear Regression": "solid",
    "Ridge Regression": "dot",
    "Lasso Regression": "dash",
    "Random Forest": "longdash",
    "Symbolic Regression": "solid",
}
MODEL_WIDTHS = {
    "Linear Regression": 2.4,
    "Ridge Regression": 2.4,
    "Lasso Regression": 2.4,
    "Random Forest": 2.8,
    "Symbolic Regression": 4.5,
}


def configure_plotly_theme():
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,20,27,0.92)",
        font=dict(color="#e6edf3", family="Segoe UI, Arial, sans-serif", size=14),
        title=dict(font=dict(size=22, color="#f8fafc"), x=0.02, xanchor="left"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        margin=dict(l=36, r=24, t=72, b=36),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.12)",
            title_font=dict(size=14),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.12)",
            title_font=dict(size=14),
        ),
        polar=dict(
            bgcolor="rgba(9,20,27,0.92)",
            radialaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.10)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.10)"),
        ),
        scene=dict(
            bgcolor="rgba(9,20,27,0.92)",
            xaxis=dict(
                backgroundcolor="rgba(9,20,27,0.92)",
                gridcolor="rgba(255,255,255,0.08)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.10)",
            ),
            yaxis=dict(
                backgroundcolor="rgba(9,20,27,0.92)",
                gridcolor="rgba(255,255,255,0.08)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.10)",
            ),
            zaxis=dict(
                backgroundcolor="rgba(9,20,27,0.92)",
                gridcolor="rgba(255,255,255,0.08)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.10)",
            ),
        ),
        colorway=[MODEL_COLORS[name] for name in MODEL_ORDER],
    )
    pio.templates["sr_dark"] = template
    pio.templates.default = "sr_dark"


def finalize_figure(fig, height=460):
    fig.update_layout(
        template="sr_dark",
        height=height,
        hoverlabel=dict(
            bgcolor="#10212c",
            bordercolor="rgba(255,255,255,0.12)",
            font=dict(color="#f8fafc"),
        ),
    )
    return fig


def add_model_line_traces(fig, df, x_col, y_col, title, y_axis_title):
    ordered_models = [model for model in MODEL_ORDER if model in set(df["Model"].tolist())]
    for model_name in ordered_models:
        model_df = df[df["Model"] == model_name].sort_values(x_col)
        fig.add_trace(
            go.Scatter(
                x=model_df[x_col],
                y=model_df[y_col],
                mode="lines",
                name=model_name,
                line=dict(
                    color=MODEL_COLORS.get(model_name, "#cccccc"),
                    width=MODEL_WIDTHS.get(model_name, 2.5),
                    dash=MODEL_DASHES.get(model_name, "solid"),
                ),
                opacity=1.0 if model_name == "Symbolic Regression" else 0.92,
            )
        )
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_axis_title)
    return fig


def inject_app_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0b141b 0%, #101b24 100%);
            color: #e6edf3;
        }
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        .hero-shell {
            background: rgba(14, 26, 35, 0.96);
            color: #f8fafc;
            border-radius: 18px;
            padding: 1.35rem 1.5rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
            margin-bottom: 0.85rem;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            opacity: 0.68;
            margin-bottom: 0.35rem;
        }
        .hero-title {
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0;
        }
        .mini-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.8rem 0 1rem 0;
        }
        .mini-card {
            background: rgba(16, 30, 40, 0.94);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 0.85rem 0.9rem 0.8rem 0.9rem;
            box-shadow: none;
        }
        .mini-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9fb3c8;
            margin-bottom: 0.25rem;
        }
        .mini-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.2rem;
        }
        .mini-note {
            color: #bfd1e0;
            font-size: 0.83rem;
            line-height: 1.35;
        }
        .section-banner {
            background: rgba(14, 26, 35, 0.92);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            box-shadow: none;
            margin: 0.2rem 0 0.8rem 0;
        }
        .section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #f4f7fb;
            margin-bottom: 0.2rem;
        }
        .section-copy {
            color: #bfd1e0;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(15, 31, 41, 0.92);
            border-radius: 10px;
            padding: 0.45rem 0.85rem;
            border: 1px solid rgba(255,255,255,0.06);
            color: #d9e5ef;
        }
        .stTabs [aria-selected="true"] {
            background: #1f5f7a !important;
            color: #ffffff !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(16, 30, 40, 0.94);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
            box-shadow: none;
        }
        div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"], div[data-testid="stAlert"] {
            border-radius: 14px;
            overflow: hidden;
        }
        div[data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stSlider"] {
            color: #e6edf3;
        }
        [data-testid="stSidebar"] {
            background: #0d1820;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] * {
            color: #e6edf3;
        }
        .stMarkdown, .stText, label, p, div {
            color: inherit;
        }
        @media (max-width: 900px) {
            .mini-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .hero-title {
                font-size: 1.45rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(bundle, traditional_results, symbolic_results):
    symbolic_row = symbolic_results.iloc[0].to_dict() if not symbolic_results.empty else {}
    best_traditional = (
        traditional_results.sort_values("R2", ascending=False).iloc[0].to_dict()
        if not traditional_results.empty
        else {}
    )
    backend = bundle.get("symbolic_metadata", {}).get("backend", "unknown").replace("_", " ")
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Project Dashboard</div>
            <div class="hero-title">Symbolic Regression for Energy Efficiency Prediction: A Comparative Analysis with Traditional Machine Learning Models</div>
        </div>
        <div class="mini-grid">
            <div class="mini-card">
                <div class="mini-label">Best Traditional Model</div>
                <div class="mini-value">{best_traditional.get("Model", "N/A")}</div>
                <div class="mini-note">Top baseline by R2 with a data-driven deployment bundle.</div>
            </div>
            <div class="mini-card">
                <div class="mini-label">Symbolic Interpretability</div>
                <div class="mini-value">High</div>
                <div class="mini-note">Closed-form equation with direct variable-level reasoning.</div>
            </div>
            <div class="mini-card">
                <div class="mini-label">Symbolic R2</div>
                <div class="mini-value">{symbolic_row.get("R2", float("nan")):.4f}</div>
                <div class="mini-note">Performance strong enough to support the interpretability story.</div>
            </div>
            <div class="mini-card">
                <div class="mini-label">Equation Complexity</div>
                <div class="mini-value">{symbolic_row.get("Complexity", "N/A")}</div>
                <div class="mini-note">Compact enough for explanation while retaining nonlinearity.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        f"Current symbolic backend: `{backend}`"
    )


def render_section_banner(title, copy):
    st.markdown(
        f"""
        <div class="section-banner">
            <div class="section-title">{title}</div>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_symbolic_spotlight(actual_value, predictions):
    if "Symbolic Regression" not in predictions:
        return

    symbolic_error = abs(predictions["Symbolic Regression"] - actual_value)
    ranked = sorted(
        ((name, abs(value - actual_value)) for name, value in predictions.items()),
        key=lambda item: item[1],
    )
    symbolic_rank = next(
        (index + 1 for index, (name, _) in enumerate(ranked) if name == "Symbolic Regression"),
        None,
    )
    closest_model, closest_error = ranked[0]
    message = (
        f"Symbolic Regression is the interpretability highlight of this app because it provides a readable equation, "
        f"not just a prediction. For this selected configuration, its absolute error is `{symbolic_error:.2f}` "
        f"and it ranks `{symbolic_rank}` out of `{len(ranked)}` by closeness to the actual heating load."
    )
    if closest_model == "Symbolic Regression":
        st.success(message + " On this case, it is also the closest model to the actual value.")
    else:
        st.info(
            message
            + f" The closest model on this case is `{closest_model}` with absolute error `{closest_error:.2f}`."
        )


def plot_actual_vs_predicted(bundle):
    evaluation_df = get_evaluation_records(bundle)
    if evaluation_df.empty:
        st.info("Actual-vs-predicted evaluation data is not available in the current bundle.")
        return

    available_models = [name for name in MODEL_ORDER if name in evaluation_df.columns]
    fig = go.Figure()
    for model_name in available_models:
        model_df = evaluation_df[["Actual", model_name]].rename(columns={model_name: "Predicted"})
        fig.add_trace(
            go.Scatter(
                x=model_df["Actual"],
                y=model_df["Predicted"],
                mode="markers",
                name=model_name,
                marker=dict(
                    color=MODEL_COLORS.get(model_name, "#cccccc"),
                    size=9 if model_name == "Symbolic Regression" else 7,
                    opacity=0.88 if model_name == "Symbolic Regression" else 0.68,
                    line=dict(
                        width=1.2 if model_name == "Symbolic Regression" else 0.5,
                        color="rgba(255,255,255,0.35)",
                    ),
                ),
            )
        )

    min_axis = float(
        min(
            evaluation_df["Actual"].min(),
            min(evaluation_df[model_name].min() for model_name in available_models),
        )
    )
    max_axis = float(
        max(
            evaluation_df["Actual"].max(),
            max(evaluation_df[model_name].max() for model_name in available_models),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_axis, max_axis],
            y=[min_axis, max_axis],
            mode="lines",
            name="Perfect Fit",
            line=dict(color="#ffffff", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted Heating Load",
        xaxis_title="Actual Heating Load",
        yaxis_title="Predicted Heating Load",
    )
    finalize_figure(fig, height=540)
    st.plotly_chart(fig, use_container_width=True)


def plot_symbolic_vs_actual(actual_value, predictions):
    if "Symbolic Regression" not in predictions:
        return

    compare_df = pd.DataFrame(
        {
            "Series": ["Actual", "Symbolic Regression"],
            "Heating Load": [actual_value, predictions["Symbolic Regression"]],
        }
    )
    fig = px.bar(
        compare_df,
        x="Series",
        y="Heating Load",
        color="Series",
        color_discrete_map={
            "Actual": "#264653",
            "Symbolic Regression": MODEL_COLORS["Symbolic Regression"],
        },
        title="Actual vs Symbolic Regression for Selected Configuration",
    )
    fig.update_traces(texttemplate="%{y:.2f}", textposition="outside", cliponaxis=False)
    finalize_figure(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)


def load_bundle(bundle_path: Path):
    if not bundle_path.exists():
        return {}, [f"Bundle not found at `{bundle_path}`."]

    with bundle_path.open("rb") as file:
        bundle = pickle.load(file)

    if not isinstance(bundle, dict):
        bundle = {"symbolic_model": bundle}

    if "model" in bundle and "symbolic_model" not in bundle:
        bundle["symbolic_model"] = bundle["model"]

    warnings = []
    required = ["scaler", "feature_names", "models", "metrics"]
    missing = [key for key in required if key not in bundle]
    if missing:
        warnings.append(
            "Bundle is missing "
            + ", ".join(f"`{key}`" for key in missing)
            + ". The app will avoid showing inaccurate live predictions."
        )

    return bundle, warnings


def build_results_tables(bundle):
    metrics = bundle.get("metrics")
    if metrics:
        rows = []
        for model_name, values in metrics.items():
            row = {
                "Model": model_name,
                "R2": values.get("R2"),
                "RMSE": values.get("RMSE"),
                "MAE": values.get("MAE"),
            }
            if "Complexity" in values:
                row["Complexity"] = values["Complexity"]
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            [{"Model": name, **values} for name, values in LEGACY_METRICS.items()]
        )

    traditional = df[df["Model"] != "Symbolic Regression"].copy()
    symbolic = df[df["Model"] == "Symbolic Regression"].copy()
    return traditional, symbolic


def get_feature_names(bundle):
    feature_names = bundle.get("feature_names")
    if not feature_names:
        return list(FEATURE_LABELS.keys())
    return list(feature_names)


def get_feature_ranges(bundle, feature_names):
    ranges = bundle.get("feature_ranges", {})
    defaults = bundle.get("default_input", {})
    resolved = {}
    for name in feature_names:
        if name in ranges:
            low, high = ranges[name]
            default = defaults.get(name, (low + high) / 2)
        else:
            low, high, default = DEFAULT_RANGES.get(name, (0.0, 1.0, 0.5))
        resolved[name] = (low, high, default)
    return resolved


def get_valid_feature_values(bundle):
    return bundle.get("valid_feature_values", {})


def get_configurations(bundle):
    records = bundle.get("configurations", [])
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def get_evaluation_records(bundle):
    records = bundle.get("evaluation_records", [])
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def get_best_symbolic_configuration(bundle):
    configurations = get_configurations(bundle)
    if configurations.empty:
        return None

    feature_names = get_feature_names(bundle)
    required_columns = set(feature_names + ["Y1", "Configuration ID"])
    if not required_columns.issubset(configurations.columns):
        return None

    feature_frame = configurations[feature_names].copy()
    predictions = predict_all_models(bundle, feature_frame, include_symbolic_proxy=True)
    if not predictions or "Symbolic Regression" not in predictions:
        return None

    actual = configurations["Y1"].to_numpy(dtype=float)
    symbolic_pred = np.asarray(predictions["Symbolic Regression"], dtype=float)
    symbolic_error = np.abs(symbolic_pred - actual)

    other_errors = []
    for model_name, values in predictions.items():
        if model_name == "Symbolic Regression":
            continue
        other_errors.append(np.abs(np.asarray(values, dtype=float) - actual))

    if other_errors:
        stacked_other_errors = np.vstack(other_errors)
        symbolic_best_mask = symbolic_error <= stacked_other_errors.min(axis=0)
    else:
        symbolic_best_mask = np.ones_like(symbolic_error, dtype=bool)

    candidate_indices = np.where(symbolic_best_mask)[0]
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(len(configurations))

    best_index = candidate_indices[np.argmin(symbolic_error[candidate_indices])]
    best_row = configurations.iloc[int(best_index)].to_dict()
    best_row["Symbolic Abs Error"] = float(symbolic_error[int(best_index)])
    return best_row


def coerce_to_valid_option(options, value):
    if value in options:
        return value
    numeric_options = [float(option) for option in options]
    return min(numeric_options, key=lambda option: abs(option - float(value)))


def collect_feature_input(bundle, key_prefix="input", initial_values=None):
    feature_names = get_feature_names(bundle)
    feature_ranges = get_feature_ranges(bundle, feature_names)
    valid_feature_values = get_valid_feature_values(bundle)
    values = {}
    initial_values = initial_values or {}

    st.header("Feature Inputs")
    st.markdown(
        "\n".join(
            f"- {name} -> {FEATURE_LABELS.get(name, name)}"
            for name in feature_names
        )
    )

    columns = st.columns(2)
    for index, name in enumerate(feature_names):
        low, high, default = feature_ranges[name]
        column = columns[index % 2]
        label = f"{name} - {FEATURE_LABELS.get(name, name)}"
        selected_default = initial_values.get(name, default)
        with column:
            if name in valid_feature_values and valid_feature_values[name]:
                options = valid_feature_values[name]
                selected_default = coerce_to_valid_option(options, selected_default)
                widget_key = f"{key_prefix}_{name}"
                if widget_key in st.session_state:
                    st.session_state[widget_key] = coerce_to_valid_option(
                        options,
                        st.session_state[widget_key],
                    )
                values[name] = st.select_slider(
                    label,
                    options=options,
                    value=selected_default,
                    key=widget_key,
                )
            elif isinstance(default, (int, np.integer)) and isinstance(low, (int, np.integer)) and isinstance(high, (int, np.integer)):
                values[name] = st.slider(label, int(low), int(high), int(selected_default), key=f"{key_prefix}_{name}")
            else:
                values[name] = st.slider(label, float(low), float(high), float(selected_default), key=f"{key_prefix}_{name}")
    return values


def make_feature_frame(feature_values, feature_names):
    ordered = [feature_values[name] for name in feature_names]
    return pd.DataFrame([ordered], columns=feature_names)


def get_symbolic_backend(bundle):
    return bundle.get("symbolic_metadata", {}).get("backend", "unknown")


def predict_all_models(bundle, feature_frame, include_symbolic_proxy=True):
    scaler = bundle.get("scaler")
    models = bundle.get("models", {})
    if scaler is None or not models:
        return None

    feature_names = get_feature_names(bundle)
    scaled_array = scaler.transform(feature_frame[feature_names])
    predictions = {}

    for name, model in models.items():
        predictions[name] = model.predict(scaled_array)

    if (
        include_symbolic_proxy
        and "Symbolic Regression" not in predictions
        and set(SYMBOLIC_FEATURES).issubset(feature_frame.columns)
    ):
        predictions["Symbolic Regression"] = predict_symbolic_from_raw(
            feature_frame,
            scaler,
            feature_names,
        )

    return predictions


def predict_single(bundle, feature_values, include_symbolic_proxy=True):
    feature_names = get_feature_names(bundle)
    feature_frame = make_feature_frame(feature_values, feature_names)
    predictions = predict_all_models(bundle, feature_frame, include_symbolic_proxy=include_symbolic_proxy)
    if predictions is None:
        return None
    return {name: float(values[0]) for name, values in predictions.items()}


def build_curve_frame(feature_values, varied_values, varied_feature, compare_feature=None, compare_values=None):
    rows = []
    compare_values = [None] if compare_values is None else compare_values
    for compare_value in compare_values:
        for varied_value in varied_values:
            row = feature_values.copy()
            row[varied_feature] = float(varied_value)
            if compare_feature is not None and compare_value is not None:
                row[compare_feature] = float(compare_value)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_prediction_comparison(predictions, title):
    category_order = MODEL_ORDER + ["Actual"]
    prediction_table = pd.DataFrame(
        {
            "Model": list(predictions.keys()),
            "Predicted Heating Load": list(predictions.values()),
        }
    )
    prediction_table["Model"] = pd.Categorical(
        prediction_table["Model"], categories=category_order, ordered=True
    )
    prediction_table = prediction_table.sort_values("Model")
    st.dataframe(prediction_table, use_container_width=True)
    color_map = dict(MODEL_COLORS)
    color_map["Actual"] = "#264653"
    fig_pred = px.bar(
        prediction_table,
        x="Model",
        y="Predicted Heating Load",
        color="Model",
        color_discrete_map=color_map,
        title=title,
    )
    fig_pred.update_traces(
        texttemplate="%{y:.2f}",
        textposition="outside",
        cliponaxis=False,
    )
    finalize_figure(fig_pred, height=440)
    st.plotly_chart(fig_pred, use_container_width=True)

    if "Symbolic Regression" in predictions:
        delta_rows = []
        symbolic_value = predictions["Symbolic Regression"]
        for name, value in predictions.items():
            if name == "Symbolic Regression":
                continue
            delta_rows.append(
                {
                    "Model": name,
                    "Difference vs Symbolic": value - symbolic_value,
                }
            )
        if delta_rows:
            delta_df = pd.DataFrame(delta_rows)
            fig_delta = px.bar(
                delta_df,
                x="Model",
                y="Difference vs Symbolic",
                color="Model",
                color_discrete_map=MODEL_COLORS,
                title="Traditional Model Difference vs Symbolic Regression",
            )
            fig_delta.add_hline(y=0, line_dash="dash")
            fig_delta.update_traces(texttemplate="%{y:.2f}", textposition="outside", cliponaxis=False)
            finalize_figure(fig_delta, height=420)
            st.plotly_chart(fig_delta, use_container_width=True)


def render_kpi_cards(traditional_results, symbolic_results):
    if symbolic_results.empty:
        return

    symbolic_row = symbolic_results.iloc[0].to_dict()
    best_traditional_r2 = traditional_results["R2"].max() if not traditional_results.empty else None
    best_traditional_rmse = traditional_results["RMSE"].min() if not traditional_results.empty else None
    best_traditional_model = (
        traditional_results.sort_values("R2", ascending=False).iloc[0]["Model"]
        if not traditional_results.empty
        else "N/A"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Symbolic R2",
            f"{symbolic_row.get('R2', float('nan')):.4f}",
            None
            if best_traditional_r2 is None
            else f"{symbolic_row.get('R2', 0.0) - best_traditional_r2:+.4f} vs best traditional",
        )
    with col2:
        st.metric(
            "Symbolic RMSE",
            f"{symbolic_row.get('RMSE', float('nan')):.4f}",
            None
            if best_traditional_rmse is None
            else f"{best_traditional_rmse - symbolic_row.get('RMSE', 0.0):+.4f} improvement gap",
        )
    with col3:
        complexity = symbolic_row.get("Complexity", "N/A")
        st.metric("Best Traditional Model", best_traditional_model, f"Complexity {complexity}")


def plot_metrics_radar(traditional_results, symbolic_results):
    metrics_df = pd.concat([traditional_results, symbolic_results], ignore_index=True).copy()
    if metrics_df.empty:
        return

    epsilon = 1e-9
    metrics_df["R2 Score"] = metrics_df["R2"]
    metrics_df["RMSE Score"] = 1.0 / (metrics_df["RMSE"] + epsilon)
    metrics_df["MAE Score"] = 1.0 / (metrics_df["MAE"] + epsilon)

    radar_rows = []
    for _, row in metrics_df.iterrows():
        for label in ["R2 Score", "RMSE Score", "MAE Score"]:
            radar_rows.append(
                {
                    "Model": row["Model"],
                    "Metric": label,
                    "Score": float(row[label]),
                }
            )
    radar_df = pd.DataFrame(radar_rows)
    fig = px.line_polar(
        radar_df,
        r="Score",
        theta="Metric",
        color="Model",
        line_close=True,
        color_discrete_map=MODEL_COLORS,
        title="Model Profile Radar",
    )
    fig.update_traces(fill="toself")
    finalize_figure(fig, height=520)
    st.plotly_chart(fig, use_container_width=True)


def plot_complexity_loss_curve(bundle):
    symbolic_model = bundle.get("models", {}).get("Symbolic Regression")
    equations = getattr(symbolic_model, "equations_", None)
    if equations is None:
        return

    equations_df = pd.DataFrame(equations).copy()
    if equations_df.empty or "complexity" not in equations_df.columns or "loss" not in equations_df.columns:
        return

    equations_df["complexity"] = pd.to_numeric(equations_df["complexity"], errors="coerce")
    equations_df["loss"] = pd.to_numeric(equations_df["loss"], errors="coerce")
    equations_df = equations_df.dropna(subset=["complexity", "loss"]).sort_values("complexity")
    if equations_df.empty:
        return

    fig = px.line(
        equations_df,
        x="complexity",
        y="loss",
        markers=True,
        title="Complexity vs Loss Curve",
    )
    fig.update_traces(
        line=dict(color=MODEL_COLORS["Symbolic Regression"], width=4),
        marker=dict(size=9, color=MODEL_COLORS["Symbolic Regression"]),
    )
    finalize_figure(fig, height=430)
    st.plotly_chart(fig, use_container_width=True)


def plot_visualization(bundle, feature_values):
    feature_names = get_feature_names(bundle)
    feature_ranges = get_feature_ranges(bundle, feature_names)
    if "X4" not in feature_names or "X6" not in feature_names:
        st.info("Visualization expects `X4` and `X6` to be present in the feature set.")
        return

    roof_min, roof_max, _ = feature_ranges["X4"]
    orientation_min, orientation_max, _ = feature_ranges["X6"]

    roof_range = np.linspace(float(roof_min), float(roof_max), 200)
    line_df = build_curve_frame(feature_values, roof_range, "X4")
    line_predictions = predict_all_models(bundle, line_df, include_symbolic_proxy=True)
    if line_predictions is None:
        st.info("Visualization is unavailable until the bundle includes trained models and scaler.")
        return

    curve_rows = []
    for model_name, values in line_predictions.items():
        for roof_value, prediction in zip(roof_range, values):
            curve_rows.append(
                {
                    "Roof Area": roof_value,
                    "Heating Load": float(prediction),
                    "Model": model_name,
                }
            )
    curve_df = pd.DataFrame(curve_rows)
    curve_df["Model"] = pd.Categorical(curve_df["Model"], categories=MODEL_ORDER, ordered=True)

    fig_line = go.Figure()
    add_model_line_traces(
        fig_line,
        curve_df,
        x_col="Roof Area",
        y_col="Heating Load",
        title="Heating Load vs Roof Area Across Models",
        y_axis_title="Heating Load",
    )
    finalize_figure(fig_line, height=460)
    st.plotly_chart(fig_line, use_container_width=True)

    if "Symbolic Regression" in line_predictions:
        diff_rows = []
        symbolic_values = line_predictions["Symbolic Regression"]
        for model_name, values in line_predictions.items():
            if model_name == "Symbolic Regression":
                continue
            for roof_value, prediction, symbolic_value in zip(roof_range, values, symbolic_values):
                diff_rows.append(
                    {
                        "Roof Area": roof_value,
                        "Difference vs Symbolic": float(prediction - symbolic_value),
                        "Model": model_name,
                    }
                )
        diff_df = pd.DataFrame(diff_rows)
        fig_diff = go.Figure()
        add_model_line_traces(
            fig_diff,
            diff_df,
            x_col="Roof Area",
            y_col="Difference vs Symbolic",
            title="Difference from Symbolic Regression Across Roof Area",
            y_axis_title="Difference vs Symbolic",
        )
        fig_diff.add_hline(y=0, line_dash="dash")
        finalize_figure(fig_diff, height=430)
        st.plotly_chart(fig_diff, use_container_width=True)

    orientation_range = np.linspace(float(orientation_min), float(orientation_max), 50)
    roof_grid, orientation_grid = np.meshgrid(roof_range[::4], orientation_range)
    grid_df = build_curve_frame(
        feature_values,
        roof_range[::4],
        "X4",
        compare_feature="X6",
        compare_values=orientation_range,
    )
    grid_predictions = predict_all_models(bundle, grid_df, include_symbolic_proxy=True)
    symbolic_grid = grid_predictions.get("Symbolic Regression")
    if symbolic_grid is None:
        st.info("Symbolic surface view needs the symbolic equation inputs `X1`, `X2`, `X4`, and `X6`.")
        return
    grid_pred = symbolic_grid.reshape(orientation_grid.shape)

    fig_surface = go.Figure(
        data=[
            go.Surface(
                x=roof_grid,
                y=orientation_grid,
                z=grid_pred,
                colorscale="Viridis",
            )
        ]
    )
    fig_surface.update_layout(
        scene=dict(
            xaxis_title="Roof Area (X4)",
            yaxis_title="Orientation (X6)",
            zaxis_title="Heating Load",
        ),
        height=600,
        title="Symbolic Regression Response Surface",
    )
    finalize_figure(fig_surface, height=620)
    st.plotly_chart(fig_surface, use_container_width=True)


def plot_optimization(bundle, feature_values):
    feature_names = get_feature_names(bundle)
    if "X4" not in feature_names:
        st.info("Optimization expects `X4` to be present in the feature set.")
        return

    roof_min, roof_max, _ = get_feature_ranges(bundle, feature_names)["X4"]
    search_range = np.linspace(float(roof_min), float(roof_max), 500)
    search_df = build_curve_frame(feature_values, search_range, "X4")
    curve_predictions = predict_all_models(bundle, search_df, include_symbolic_proxy=True)
    if curve_predictions is None:
        st.info("Optimization is unavailable until the bundle includes trained models and scaler.")
        return

    summary_rows = []
    curve_rows = []
    for model_name, values in curve_predictions.items():
        values = np.asarray(values, dtype=float)
        best_index = int(np.argmin(values))
        summary_rows.append(
            {
                "Model": model_name,
                "Optimal Roof Area": float(search_range[best_index]),
                "Minimum Heating Load": float(values[best_index]),
            }
        )
        for roof_value, prediction in zip(search_range, values):
            curve_rows.append(
                {
                    "Roof Area": roof_value,
                    "Heating Load": float(prediction),
                    "Model": model_name,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["Model"] = pd.Categorical(summary_df["Model"], categories=MODEL_ORDER, ordered=True)
    summary_df = summary_df.sort_values("Model")
    st.subheader("Optimal Roof Area by Model")
    st.dataframe(summary_df, use_container_width=True)

    curve_plot_df = pd.DataFrame(curve_rows).sort_values(["Model", "Roof Area"])
    fig = go.Figure()
    add_model_line_traces(
        fig,
        curve_plot_df,
        x_col="Roof Area",
        y_col="Heating Load",
        title="Optimization Curves Across Models",
        y_axis_title="Heating Load",
    )
    finalize_figure(fig, height=460)
    st.plotly_chart(fig, use_container_width=True)

    if "Symbolic Regression" in curve_predictions:
        symbolic_values = np.asarray(curve_predictions["Symbolic Regression"], dtype=float)
        diff_rows = []
        for model_name, values in curve_predictions.items():
            if model_name == "Symbolic Regression":
                continue
            for roof_value, prediction, symbolic_value in zip(search_range, values, symbolic_values):
                diff_rows.append(
                    {
                        "Roof Area": roof_value,
                        "Difference vs Symbolic": float(prediction - symbolic_value),
                        "Model": model_name,
                    }
                )
        diff_df = pd.DataFrame(diff_rows)
        fig_diff = go.Figure()
        add_model_line_traces(
            fig_diff,
            diff_df,
            x_col="Roof Area",
            y_col="Difference vs Symbolic",
            title="Optimization Difference from Symbolic Regression",
            y_axis_title="Difference vs Symbolic",
        )
        fig_diff.add_hline(y=0, line_dash="dash")
        finalize_figure(fig_diff, height=430)
        st.plotly_chart(fig_diff, use_container_width=True)


bundle, bundle_warnings = load_bundle(BUNDLE_PATH)
traditional_results, symbolic_results = build_results_tables(bundle)
symbolic_backend = get_symbolic_backend(bundle)

inject_app_styles()
configure_plotly_theme()
render_hero(bundle, traditional_results, symbolic_results)

for warning in bundle_warnings:
    st.warning(warning)

if not bundle.get("metrics"):
    st.info(
        "Showing legacy summary metrics because the current bundle does not contain exported evaluation metadata yet."
    )

page_results, page_visualization, page_optimization, page_prediction = st.tabs(
    ["Results Comparison", "Visualization", "Optimization", "Prediction"]
)

with page_results:
    render_section_banner(
        "Results Overview",
        "Compare model accuracy, errors, and the final symbolic equation.",
    )
    render_kpi_cards(traditional_results, symbolic_results)
    tab_summary, tab_charts, tab_equation = st.tabs(["Scorecards", "Comparative Charts", "Equation"])

    with tab_summary:
        st.subheader("Traditional ML Models")
        st.dataframe(traditional_results, use_container_width=True)
        st.subheader("Symbolic Regression")
        st.dataframe(symbolic_results, use_container_width=True)

    with tab_charts:
        if not traditional_results.empty:
            combined = pd.concat(
                [
                    traditional_results[["Model", "R2"]],
                    symbolic_results[["Model", "R2"]],
                ],
                ignore_index=True,
            )
            fig_r2 = px.bar(
                combined,
                x="Model",
                y="R2",
                color="Model",
                color_discrete_map=MODEL_COLORS,
                title="R2 Comparison",
            )
            fig_r2.update_traces(texttemplate="%{y:.4f}", textposition="outside", cliponaxis=False)
            finalize_figure(fig_r2, height=430)
            st.plotly_chart(fig_r2, use_container_width=True)

            error_rows = pd.concat(
                [
                    traditional_results[["Model", "RMSE", "MAE"]],
                    symbolic_results[["Model", "RMSE", "MAE"]],
                ],
                ignore_index=True,
            ).melt(id_vars="Model", var_name="Metric", value_name="Value")
            fig_error = px.bar(
                error_rows,
                x="Model",
                y="Value",
                color="Metric",
                barmode="group",
                title="Error Comparison",
            )
            finalize_figure(fig_error, height=460)
            st.plotly_chart(fig_error, use_container_width=True)
            plot_complexity_loss_curve(bundle)

    with tab_equation:
        equation_text = bundle.get("symbolic_equation_latex") or bundle.get("symbolic_equation_text")
        if equation_text:
            st.subheader("Best Discovered Symbolic Equation")
            st.latex(equation_text)
        st.info(
            "Presentation note: pair this equation with the comparative charts to show not just performance, but also interpretability and deployability."
        )

with page_visualization:
    render_section_banner(
        "Visualization",
        "Explore how heating load changes across the design space.",
    )
    viz_input_tab, viz_chart_tab = st.tabs(["Controls", "Visual Analytics"])
    with viz_input_tab:
        current_input = collect_feature_input(bundle, key_prefix="visualization")
    with viz_chart_tab:
        plot_visualization(bundle, current_input)

        models = bundle.get("models", {})
        rf_model = models.get("Random Forest")
        feature_names = get_feature_names(bundle)
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": rf_model.feature_importances_,
                }
            )
            fig_feat = px.bar(
                importance_df,
                x="Feature",
                y="Importance",
                title="Random Forest Feature Importance",
                color="Feature",
            )
            fig_feat.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)
            finalize_figure(fig_feat, height=430)
            st.plotly_chart(fig_feat, use_container_width=True)

with page_optimization:
    render_section_banner(
        "Optimization",
        "Inspect how roof area affects heating load across models.",
    )
    opt_input_tab, opt_chart_tab = st.tabs(["Scenario Setup", "Optimization Views"])
    with opt_input_tab:
        current_input = collect_feature_input(bundle, key_prefix="optimization")
    with opt_chart_tab:
        plot_optimization(bundle, current_input)

with page_prediction:
    render_section_banner(
        "Prediction",
        "Compare predictions for a selected building configuration.",
    )
    configurations = get_configurations(bundle)
    recommended_symbolic_config = get_best_symbolic_configuration(bundle)
    pred_setup_tab, pred_results_tab = st.tabs(["Scenario Selection", "Prediction Views"])
    with pred_setup_tab:
        input_mode = st.radio(
            "Prediction input mode",
            ["Dataset Configuration", "Custom Configuration"],
            horizontal=True,
        )

        initial_values = None
        actual_value = None
        if input_mode == "Dataset Configuration" and not configurations.empty:
            config_options = [
                f"{row['Configuration ID']} | Y1={row['Y1']:.2f} | X1={row['X1']}, X2={row['X2']}, X4={row['X4']}, X6={row['X6']}"
                for _, row in configurations.iterrows()
            ]
            default_index = 0
            if recommended_symbolic_config is not None:
                recommended_label = (
                    f"{recommended_symbolic_config['Configuration ID']} | "
                    f"Y1={recommended_symbolic_config['Y1']:.2f} | "
                    f"X1={recommended_symbolic_config['X1']}, X2={recommended_symbolic_config['X2']}, "
                    f"X4={recommended_symbolic_config['X4']}, X6={recommended_symbolic_config['X6']}"
                )
                if recommended_label in config_options:
                    default_index = config_options.index(recommended_label)
                st.success(
                    "Recommended symbolic configuration: "
                    f"{recommended_symbolic_config['Configuration ID']} "
                    f"(Symbolic Abs Error = {recommended_symbolic_config['Symbolic Abs Error']:.2f})"
                )
            selected_config_label = st.selectbox(
                "Choose a valid building configuration",
                config_options,
                index=default_index,
            )
            selected_index = config_options.index(selected_config_label)
            selected_row = configurations.iloc[selected_index]
            initial_values = {
                name: float(selected_row[name]) for name in get_feature_names(bundle)
            }
            actual_value = float(selected_row["Y1"])

        current_input = collect_feature_input(bundle, key_prefix="prediction", initial_values=initial_values)

    with pred_results_tab:
        include_symbolic_proxy = True
        predictions = predict_single(bundle, current_input, include_symbolic_proxy=include_symbolic_proxy)
        if predictions is None:
            st.error(
                "Live predictions are disabled because the exported bundle does not yet include the scaler and trained models together."
            )
            st.code("python scripts/prepare_bundle.py --with-symbolic")
        else:
            selected_case_tab, symbolic_focus_tab, all_models_tab, heldout_tab = st.tabs(
                ["Selected Case", "Symbolic Focus", "All Models", "Held-out Evaluation"]
            )

            if actual_value is not None:
                symbolic_error = (
                    abs(predictions["Symbolic Regression"] - actual_value)
                    if "Symbolic Regression" in predictions
                    else None
                )
                best_model = min(predictions, key=lambda name: abs(predictions[name] - actual_value))
                smallest_error = min(abs(value - actual_value) for value in predictions.values())

                with selected_case_tab:
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Actual Heating Load", f"{actual_value:.2f}")
                    with col_b:
                        st.metric("Closest Model", best_model)
                    with col_c:
                        st.metric("Smallest Absolute Error", f"{smallest_error:.2f}")
                    with col_d:
                        if symbolic_error is not None:
                            st.metric("Symbolic Abs Error", f"{symbolic_error:.2f}", "Best interpretable model")
                        else:
                            st.metric("Symbolic Abs Error", "N/A", "Unavailable")

                    residual_rows = []
                    for model_name, value in predictions.items():
                        residual_rows.append(
                            {
                                "Model": model_name,
                                "Residual": float(value - actual_value),
                            }
                        )
                    residual_df = pd.DataFrame(residual_rows)
                    fig_residual = px.bar(
                        residual_df,
                        x="Model",
                        y="Residual",
                        color="Model",
                        color_discrete_map=MODEL_COLORS,
                        title="Prediction Residuals vs Actual Heating Load",
                    )
                    fig_residual.add_hline(y=0, line_dash="dash")
                    fig_residual.update_traces(texttemplate="%{y:.2f}", textposition="outside", cliponaxis=False)
                    finalize_figure(fig_residual, height=430)
                    st.plotly_chart(fig_residual, use_container_width=True)

                with symbolic_focus_tab:
                    render_symbolic_spotlight(actual_value, predictions)
                    plot_symbolic_vs_actual(actual_value, predictions)

                with all_models_tab:
                    actual_comparison = predictions.copy()
                    actual_comparison["Actual"] = actual_value
                    plot_prediction_comparison(actual_comparison, "Prediction Comparison Against Actual Value")

                with heldout_tab:
                    plot_actual_vs_predicted(bundle)
            else:
                with selected_case_tab:
                    st.info("Select a dataset-backed configuration to compare model predictions against the actual heating load.")
                with symbolic_focus_tab:
                    if "Symbolic Regression" in predictions:
                        st.metric("Symbolic Predicted Heating Load", f"{predictions['Symbolic Regression']:.2f}")
                    st.info("Use dataset-backed mode for a full symbolic-vs-actual comparison.")
                with all_models_tab:
                    plot_prediction_comparison(predictions, "Prediction Comparison")
                with heldout_tab:
                    plot_actual_vs_predicted(bundle)

            if symbolic_backend != "pysr_model":
                st.info(
                    "Symbolic regression is shown here using the exported symbolic equation backend because the original PySR model object is not currently loading."
                )

st.markdown("---")
st.markdown("(c) 2026 | Advanced Symbolic Regression Energy Modeling Framework")
