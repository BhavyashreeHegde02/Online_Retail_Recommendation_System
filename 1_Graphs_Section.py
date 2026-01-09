import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt

from recommender_backend import load_cleaned_retail_excel


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


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    df = load_cleaned_retail_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    if "CustomerID" in df.columns:
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    if "UnitPrice" in df.columns:
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    return df


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _require_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.info(f"This graph needs columns: {cols}. Missing: {missing}.")
        return False
    return True


def _line_revenue(df: pd.DataFrame) -> pd.Series:
    if {"Quantity", "UnitPrice"}.issubset(df.columns):
        q = pd.to_numeric(df["Quantity"], errors="coerce")
        p = pd.to_numeric(df["UnitPrice"], errors="coerce")
        return (q * p).astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def main() -> None:
    st.set_page_config(page_title="EDA", layout="wide")
    st.markdown(_DARK_CSS, unsafe_allow_html=True)

    st.sidebar.markdown("### Navigation")
    try:
        st.sidebar.page_link("app.py", label="Dashboard")
        st.sidebar.page_link("pages/1_Graphs_Section.py", label="EDA")
        st.sidebar.page_link("pages/2_Model_Evaluation_Section.py", label="Model Evaluation")
    except Exception:
        pass
    st.sidebar.markdown("---")

    st.title("Exploratory Data Analysis")

    default_path = os.getenv("RETAIL_DATA_PATH", "OnlineRetail_cleaned.xlsx")
    data_path = os.getenv("GRAPHS_DATA_PATH", default_path)

    if not os.path.exists(data_path):
        st.error(
            "Dataset not found. Set `GRAPHS_DATA_PATH` (or `RETAIL_DATA_PATH`) to a valid file path."
        )
        st.stop()

    df = load_dataset(data_path)
    if df.empty:
        st.error("Dataset loaded but is empty.")
        st.stop()

    base_required = ["Quantity", "UnitPrice", "InvoiceDate", "CustomerID", "Country"]
    if any(c not in df.columns for c in base_required):
        st.warning(
            "This EDA page is built for Online Retail columns: Quantity, UnitPrice, InvoiceDate, CustomerID, Country. "
            "Some plots may be unavailable because required columns are missing."
        )

    graph = st.selectbox(
        "Select graph",
        options=[
            "Top Selling Products",
            "Revenue by Product",
            "Monthly Sales Trend",
            "Top Customers by Spending",
            "Country-wise Sales Distribution",
            "Price Distribution",
            "Order Size Distribution",
            "Correlation Heatmap",
            "Recommendation Outcome Distribution",
        ],
        index=0,
    )

    st.markdown("---")

    if graph == "Top Selling Products":
        if not _require_cols(df, ["UnitPrice", "Quantity"]):
            return

        d = df[["UnitPrice", "Quantity"]].copy()
        d["UnitPrice"] = pd.to_numeric(d["UnitPrice"], errors="coerce")
        d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce")
        d = d.dropna(subset=["UnitPrice", "Quantity"])
        d = d[d["Quantity"] > 0]
        if d.empty:
            st.info("No valid Quantity values to plot.")
            return

        top = (
            d.groupby("UnitPrice", dropna=False)["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(top, x="Quantity", y="UnitPrice", orientation="h", template=PLOTLY_TEMPLATE)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=640)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "Revenue by Product":
        if not _require_cols(df, ["UnitPrice", "Quantity"]):
            return

        d = df[["UnitPrice", "Quantity"]].copy()
        d["UnitPrice"] = pd.to_numeric(d["UnitPrice"], errors="coerce")
        d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce")
        d = d.dropna(subset=["UnitPrice", "Quantity"])
        d = d[(d["Quantity"] > 0) & (d["UnitPrice"] > 0)]
        if d.empty:
            st.info("No valid Quantity/UnitPrice values to compute revenue.")
            return

        rev = (d["Quantity"] * d["UnitPrice"]).astype(float)
        top = (
            d.assign(Revenue=rev)
            .groupby("UnitPrice", dropna=False)["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(top, x="Revenue", y="UnitPrice", orientation="h", template=PLOTLY_TEMPLATE)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=640)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "Monthly Sales Trend":
        if not _require_cols(df, ["InvoiceDate", "Quantity", "UnitPrice"]):
            return

        d = df[["InvoiceDate", "Quantity", "UnitPrice"]].copy()
        d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="coerce")
        d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce")
        d["UnitPrice"] = pd.to_numeric(d["UnitPrice"], errors="coerce")
        d = d.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
        d = d[(d["Quantity"] > 0) & (d["UnitPrice"] > 0)]
        if d.empty:
            st.info("No valid rows to compute monthly sales.")
            return

        d = d.assign(Revenue=(d["Quantity"] * d["UnitPrice"]).astype(float))
        d = d.assign(Month=d["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
        m = d.groupby("Month", dropna=False)["Revenue"].sum().reset_index().sort_values("Month")
        fig = px.line(m, x="Month", y="Revenue", template=PLOTLY_TEMPLATE)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "Top Customers by Spending":
        if not _require_cols(df, ["CustomerID", "Quantity", "UnitPrice"]):
            return

        d = df[["CustomerID", "Quantity", "UnitPrice"]].copy()
        d["CustomerID"] = pd.to_numeric(d["CustomerID"], errors="coerce")
        d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce")
        d["UnitPrice"] = pd.to_numeric(d["UnitPrice"], errors="coerce")
        d = d.dropna(subset=["CustomerID", "Quantity", "UnitPrice"])
        d = d[(d["Quantity"] > 0) & (d["UnitPrice"] > 0)]
        if d.empty:
            st.info("No valid rows to compute customer spending.")
            return

        d = d.assign(Revenue=(d["Quantity"] * d["UnitPrice"]).astype(float))
        top = d.groupby("CustomerID", dropna=False)["Revenue"].sum().sort_values(ascending=False).head(15).reset_index()
        top["CustomerID"] = top["CustomerID"].astype("Int64").astype(str)
        fig = px.bar(top, x="Revenue", y="CustomerID", orientation="h", template=PLOTLY_TEMPLATE)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=640)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "Country-wise Sales Distribution":
        if not _require_cols(df, ["Country", "Quantity", "UnitPrice"]):
            return

        d = df[["Country", "Quantity", "UnitPrice"]].copy()
        d["Country"] = d["Country"].astype(str).str.strip()
        d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce")
        d["UnitPrice"] = pd.to_numeric(d["UnitPrice"], errors="coerce")
        d = d.dropna(subset=["Country", "Quantity", "UnitPrice"])
        d = d[(d["Quantity"] > 0) & (d["UnitPrice"] > 0)]
        if d.empty:
            st.info("No valid rows to compute country sales.")
            return

        d = d.assign(Revenue=(d["Quantity"] * d["UnitPrice"]).astype(float))
        c = d.groupby("Country", dropna=False)["Revenue"].sum().sort_values(ascending=False).head(20).reset_index()
        fig = px.bar(c, x="Revenue", y="Country", orientation="h", template=PLOTLY_TEMPLATE)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=680)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "Price Distribution":
        if not _require_cols(df, ["UnitPrice"]):
            return

        x = pd.to_numeric(df["UnitPrice"], errors="coerce")
        x = x[(x > 0) & x.notna()]
        if x.empty:
            st.info("No valid UnitPrice values to plot.")
            return

        fig = px.histogram(pd.DataFrame({"UnitPrice": x}), x="UnitPrice", nbins=70, template=PLOTLY_TEMPLATE)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "Order Size Distribution":
        if not _require_cols(df, ["CustomerID", "InvoiceDate", "Quantity"]):
            return

        d = df[["CustomerID", "InvoiceDate", "Quantity"]].copy()
        d["CustomerID"] = pd.to_numeric(d["CustomerID"], errors="coerce")
        d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="coerce")
        d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce")
        d = d.dropna(subset=["CustomerID", "InvoiceDate", "Quantity"])
        d = d[d["Quantity"] > 0]
        if d.empty:
            st.info("No valid rows to compute order size distribution.")
            return

        d = d.assign(OrderDate=d["InvoiceDate"].dt.date)
        sizes = (
            d.groupby(["CustomerID", "OrderDate"], dropna=False)["Quantity"]
            .sum()
            .reset_index(name="OrderSize")
        )

        fig = px.histogram(sizes, x="OrderSize", nbins=60, template=PLOTLY_TEMPLATE)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "Correlation Heatmap":
        cols = [c for c in ["Quantity", "UnitPrice", "CustomerID"] if c in df.columns]
        if len(cols) < 2:
            st.info("Correlation heatmap needs at least 2 numeric columns.")
            return

        d = df[cols].copy()
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        if {"Quantity", "UnitPrice"}.issubset(df.columns):
            d["Revenue"] = _line_revenue(df)

        d = d.dropna(how="all")
        num = d.select_dtypes(include=[np.number]).copy()
        if num.shape[1] < 2:
            st.info("Correlation heatmap needs at least 2 numeric columns.")
            return

        corr = num.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=680)
        st.plotly_chart(fig, use_container_width=True)

    elif graph == "Recommendation Outcome Distribution":
        y_pred = st.session_state.get("y_pred")
        if y_pred is None:
            st.info("Please run prediction first to generate evaluation metrics.")
            return

        yp = np.asarray(y_pred).reshape(-1)
        if yp.size == 0:
            st.info("Please run prediction first to generate evaluation metrics.")
            return

        rec_n = int((yp == 1).sum())
        not_rec_n = int((yp == 0).sum())
        if rec_n + not_rec_n == 0:
            st.info("Please run prediction first to generate evaluation metrics.")
            return

        st.subheader("Recommendation Outcome Distribution")

        left, mid, right = st.columns([1, 2, 1])
        with mid:
            fig, ax = plt.subplots(figsize=(3.5, 3.2))

            fig.patch.set_facecolor("#0b1220")
            ax.set_facecolor("#0b1220")
            ax.set_title("Recommendation Outcome Distribution", color="#e2e8f0")

            labels = ["Recommended", "Not Recommended"]
            sizes = [rec_n, not_rec_n]
            colors = ["#22c55e", "#64748b"]

            ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                radius=0.8,
                textprops={"color": "#e2e8f0"},
            )
            ax.axis("equal")
            st.pyplot(fig)


if __name__ == "__main__":
    main()
