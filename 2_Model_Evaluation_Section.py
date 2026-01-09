import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


PLOTLY_TEMPLATE = "plotly_dark"


_DARK_CSS = """
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
  [data-testid="stSidebar"] { background: #0b1220; }
  [data-testid="stSidebarNav"] { display: none; }
  .panel {
    background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(2,6,23,0.95));
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 14px;
    padding: 14px 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  }
  .section-title { font-size: 1.05rem; font-weight: 700; margin-top: 0.5rem; }
</style>
"""


@dataclass(frozen=True)
class EvalArtifacts:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: Optional[np.ndarray] = None


def _coerce_1d(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    return x


def _try_load_from_csv(path: str) -> Optional[EvalArtifacts]:
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    cols = {c.lower(): c for c in df.columns}
    true_col = cols.get("y_true") or cols.get("true") or cols.get("label")
    pred_col = cols.get("y_pred") or cols.get("pred") or cols.get("prediction")
    score_col = cols.get("y_score") or cols.get("score") or cols.get("prob") or cols.get("proba")

    if true_col is None or pred_col is None:
        return None

    y_true = _coerce_1d(df[true_col].to_numpy())
    y_pred = _coerce_1d(df[pred_col].to_numpy())

    y_score = None
    if score_col is not None:
        y_score = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
        y_score = _coerce_1d(y_score)

    return EvalArtifacts(y_true=y_true, y_pred=y_pred, y_score=y_score)


def _try_load_from_npz(path: str) -> Optional[EvalArtifacts]:
    if not os.path.exists(path):
        return None

    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None

    keys = {k.lower(): k for k in data.files}
    t = keys.get("y_true")
    p = keys.get("y_pred")
    s = keys.get("y_score")

    if t is None or p is None:
        return None

    y_true = _coerce_1d(data[t])
    y_pred = _coerce_1d(data[p])
    y_score = _coerce_1d(data[s]) if s is not None else None

    return EvalArtifacts(y_true=y_true, y_pred=y_pred, y_score=y_score)


def load_eval_artifacts() -> Optional[EvalArtifacts]:
    ss_model = st.session_state.get("model")
    ss_X_test = st.session_state.get("X_test")
    ss_y_test = st.session_state.get("y_test")
    if ss_model is not None and ss_X_test is not None and ss_y_test is not None:
        try:
            y_true = _coerce_1d(np.asarray(ss_y_test))
            y_pred = _coerce_1d(np.asarray(ss_model.predict(ss_X_test)))

            y_score = None
            if hasattr(ss_model, "predict_proba"):
                proba = np.asarray(ss_model.predict_proba(ss_X_test))
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    y_score = _coerce_1d(proba[:, 1])
            if y_score is None and hasattr(ss_model, "decision_function"):
                y_score = _coerce_1d(np.asarray(ss_model.decision_function(ss_X_test)))

            return EvalArtifacts(y_true=y_true, y_pred=y_pred, y_score=y_score)
        except Exception:
            pass

    ss_true = st.session_state.get("y_true")
    ss_pred = st.session_state.get("y_pred")
    ss_score = st.session_state.get("y_score")
    if ss_true is not None and ss_pred is not None:
        y_true = _coerce_1d(np.asarray(ss_true))
        y_pred = _coerce_1d(np.asarray(ss_pred))
        y_score = _coerce_1d(np.asarray(ss_score)) if ss_score is not None else None
        return EvalArtifacts(y_true=y_true, y_pred=y_pred, y_score=y_score)

    env_path = os.getenv("EVAL_ARTIFACTS_PATH")
    if env_path:
        if env_path.lower().endswith(".npz"):
            out = _try_load_from_npz(env_path)
            if out is not None:
                return out
        out = _try_load_from_csv(env_path)
        if out is not None:
            return out

    for p in [
        "eval_artifacts.npz",
        "evaluation_artifacts.npz",
        "eval.csv",
        "evaluation.csv",
        "metrics.csv",
    ]:
        if p.lower().endswith(".npz"):
            out = _try_load_from_npz(p)
        else:
            out = _try_load_from_csv(p)
        if out is not None:
            return out

    return None


def _render_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T
    ordered_cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in df.columns]
    df = df[ordered_cols].copy()
    for c in ["precision", "recall", "f1-score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
    if "support" in df.columns:
        df["support"] = pd.to_numeric(df["support"], errors="coerce").fillna(0).astype(int)
    st.dataframe(df, use_container_width=True)


def _render_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        x=[str(x) for x in labels],
        y=[str(x) for x in labels],
        text_auto=True,
        color_continuous_scale="Blues",
        template=PLOTLY_TEMPLATE,
        aspect="auto",
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=0, r=0, t=40, b=0),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> None:
    y_true = _coerce_1d(y_true)
    y_score = _coerce_1d(y_score)

    classes = np.unique(y_true)
    if len(classes) != 2:
        st.info("ROC Curve is available only for binary classification (exactly 2 classes).")
        return

    pos_label = classes.max()
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        auc = roc_auc_score(y_true, y_score)
    except Exception as e:
        st.info(f"ROC Curve skipped (unable to compute): {e}")
        return

    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    fig = px.line(roc_df, x="FPR", y="TPR", template=PLOTLY_TEMPLATE)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="rgba(226,232,240,0.35)", dash="dash"),
    )
    fig.update_layout(
        title=f"ROC Curve (AUC = {auc:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=0, r=0, t=50, b=0),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Model Evaluation", layout="wide")
    st.markdown(_DARK_CSS, unsafe_allow_html=True)

    st.sidebar.markdown("### Navigation")
    try:
        st.sidebar.page_link("app.py", label="Dashboard")
        st.sidebar.page_link("pages/1_Graphs_Section.py", label="EDA")
        st.sidebar.page_link("pages/2_Model_Evaluation_Section.py", label="Model Evaluation")
    except Exception:
        pass
    st.sidebar.markdown("---")

    st.title("Model Evaluation")

    run = st.button("Run Evaluation")
    if not run:
        st.info("Click 'Run Evaluation' to compute and display metrics.")
        return

    artifacts = load_eval_artifacts()
    if artifacts is None:
        st.warning("Please run prediction first to generate evaluation metrics.")
        return

    y_true = artifacts.y_true
    y_pred = artifacts.y_pred

    if len(y_true) != len(y_pred):
        st.warning(f"y_true length ({len(y_true)}) != y_pred length ({len(y_pred)}).")
        return

    tab_rep, tab_cm, tab_roc = st.tabs(["Classification Report", "Confusion Matrix", "ROC Curve"])

    with tab_rep:
        _render_classification_report(y_true=y_true, y_pred=y_pred)

    with tab_cm:
        _render_confusion_matrix(y_true=y_true, y_pred=y_pred)

    with tab_roc:
        if artifacts.y_score is None:
            st.info("ROC Curve skipped (no `y_score` available).")
            return
        if len(artifacts.y_score) != len(y_true):
            st.info(f"ROC Curve skipped (`y_score` length {len(artifacts.y_score)} != y_true length {len(y_true)}).")
            return
        _render_roc_curve(y_true=y_true, y_score=artifacts.y_score)


if __name__ == "__main__":
    main()
