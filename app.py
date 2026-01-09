import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import normalize

from recommender_backend import (
    RecommenderConfig,
    build_item_user_matrix,
    load_cleaned_retail_excel,
    recommend_top_n,
)


PLOTLY_TEMPLATE = "plotly_dark"


st.set_page_config(
    page_title="Retail Recommender (SaaS Dashboard)",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


_DARK_CSS = """
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
  [data-testid="stSidebar"] { background: #0b1220; }
  [data-testid="stSidebarNav"] { display: none; }
  .kpi-card {
    background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(2,6,23,0.95));
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 14px;
    padding: 14px 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  }
  .kpi-title { color: rgba(226,232,240,0.78); font-size: 0.85rem; margin-bottom: 6px; }
  .kpi-value { color: #e2e8f0; font-size: 1.55rem; font-weight: 700; line-height: 1.2; }
  .kpi-sub { color: rgba(226,232,240,0.65); font-size: 0.8rem; margin-top: 6px; }
  .section-title { font-size: 1.05rem; font-weight: 700; margin-top: 0.5rem; }
  .badge {
    display:inline-block; padding: 0.15rem 0.5rem; border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.25); color: rgba(226,232,240,0.9);
    background: rgba(15,23,42,0.45);
    font-size: 0.78rem;
  }
</style>
"""


def _money(x: float) -> str:
    if np.isnan(x):
        return "-"
    if abs(x) >= 1_000_000:
        return f"£{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"£{x/1_000:.2f}K"
    return f"£{x:,.2f}"


def _kpi_card(title: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    df = load_cleaned_retail_excel(path)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    if "CustomerID" in df.columns:
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")

    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    if "UnitPrice" in df.columns:
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    if "TotalPrice" not in df.columns:
        if "Revenue" in df.columns:
            df["TotalPrice"] = pd.to_numeric(df["Revenue"], errors="coerce")
        elif {"Quantity", "UnitPrice"}.issubset(df.columns):
            df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    df["TotalPrice"] = pd.to_numeric(df["TotalPrice"], errors="coerce")

    if "Description" in df.columns:
        df["Description"] = df["Description"].astype(str).str.strip()

    keep = (
        df.get("CustomerID").notna()
        & df.get("InvoiceNo").notna()
        & df.get("Description").notna()
    )
    df = df.loc[keep].copy()

    if "Quantity" in df.columns:
        df = df[pd.to_numeric(df["Quantity"], errors="coerce").fillna(0) > 0].copy()
    if "UnitPrice" in df.columns:
        df = df[pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0) > 0].copy()

    if "InvoiceDate" in df.columns and df["InvoiceDate"].notna().any():
        df["Year"] = df["InvoiceDate"].dt.year.astype("Int64")
        df["Month"] = df["InvoiceDate"].dt.month.astype("Int64")
        df["Day"] = df["InvoiceDate"].dt.day.astype("Int64")
        df["Hour"] = df["InvoiceDate"].dt.hour.astype("Int64")

    return df


@st.cache_resource(show_spinner=False)
def build_model(
    df: pd.DataFrame,
    min_support_customers: int,
    use_binary: bool,
) -> Tuple[object, object, dict, np.ndarray]:
    cfg = RecommenderConfig(
        min_item_support_customers=min_support_customers,
        use_binary_interactions=use_binary,
    )
    return build_item_user_matrix(df, cfg=cfg)


def main() -> None:
    st.markdown(_DARK_CSS, unsafe_allow_html=True)

    st.title("Online Retail Recommendation System")
    st.caption("Item-Based Collaborative Filtering • Cosine Similarity • Analysis-ready Excel")

    st.sidebar.markdown("### Navigation")
    try:
        st.sidebar.page_link("app.py", label="Dashboard")
        st.sidebar.page_link("pages/1_Graphs_Section.py", label="EDA")
        st.sidebar.page_link("pages/2_Model_Evaluation_Section.py", label="Model Evaluation")
    except Exception:
        pass
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Settings")
    default_path = os.getenv("RETAIL_DATA_PATH", "OnlineRetail_cleaned.xlsx")
    data_path = st.sidebar.text_input("Cleaned Excel path", value=default_path)

    st.sidebar.markdown("### Recommender")
    min_support = st.sidebar.slider("Min customers per product", min_value=1, max_value=50, value=5, step=1)
    binary = st.sidebar.toggle("Binary interactions (recommended)", value=True)
    top_n = st.sidebar.slider("Top-N recommendations", min_value=3, max_value=20, value=8, step=1)

    if not os.path.exists(data_path):
        st.error(
            "Cleaned dataset not found. Place `OnlineRetail_cleaned.xlsx` in the project folder "
            "or update the path in the sidebar."
        )
        st.stop()

    df = load_dataset(data_path)
    if df.empty:
        st.error("Dataset loaded but no valid rows remain after basic validation.")
        st.stop()

    revenue = float(df["TotalPrice"].sum(skipna=True))
    customers = int(df["CustomerID"].nunique(dropna=True))
    products = int(df["Description"].nunique(dropna=True))
    orders = int(df["InvoiceNo"].nunique(dropna=True))

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Revenue", _money(revenue), "Sum of TotalPrice")
    with k2:
        _kpi_card("Customers", f"{customers:,}", "Unique CustomerID")
    with k3:
        _kpi_card("Products", f"{products:,}", "Unique descriptions")
    with k4:
        _kpi_card("Orders", f"{orders:,}", "Unique invoices")

    st.markdown("---")

    tab_overview, tab_reco, tab_explain, tab_arch = st.tabs(
        ["Dashboard", "Recommendations", "Explainability", "Architecture"]
    )

    with tab_overview:
        left, right = st.columns([1.1, 0.9])

        with left:
            st.markdown("<div class='section-title'>Revenue trend</div>", unsafe_allow_html=True)
            if "InvoiceDate" in df.columns and df["InvoiceDate"].notna().any():
                ts = (
                    df.assign(date=df["InvoiceDate"].dt.date)
                    .groupby("date", dropna=False)["TotalPrice"]
                    .sum()
                    .reset_index()
                    .sort_values("date")
                )
                fig = px.line(ts, x="date", y="TotalPrice", template=PLOTLY_TEMPLATE)
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("InvoiceDate is missing or invalid; revenue trend is unavailable.")

        with right:
            st.markdown("<div class='section-title'>Top countries by revenue</div>", unsafe_allow_html=True)
            if "Country" in df.columns:
                ctry = (
                    df.groupby("Country", dropna=False)["TotalPrice"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(12)
                    .reset_index()
                )
                fig = px.bar(ctry, x="TotalPrice", y="Country", orientation="h", template=PLOTLY_TEMPLATE)
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Country column not found.")

        st.markdown("<div class='section-title'>Top products (by revenue)</div>", unsafe_allow_html=True)
        top_prod = (
            df.groupby("Description", dropna=False)["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(top_prod, x="TotalPrice", y="Description", orientation="h", template=PLOTLY_TEMPLATE)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=480)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>Data preview</div>", unsafe_allow_html=True)
        st.dataframe(df.head(25), use_container_width=True)

    with tab_reco:
        st.markdown(
            "<span class='badge'>Backend: Item-Item Cosine Similarity</span> "
            "<span class='badge'>Matrix: Items × Customers</span>",
            unsafe_allow_html=True,
        )

        X, X_bin, item_index, items = build_model(df, min_support_customers=min_support, use_binary=binary)

        all_products = sorted(list(item_index.keys()))
        default_product = all_products[0] if all_products else ""
        product = st.selectbox("Choose a product", options=all_products, index=0 if default_product else 0)

        st.markdown("<div class='section-title'>Selected product</div>", unsafe_allow_html=True)
        st.write(product)

        run_pred = st.button("Run Prediction")
        if not run_pred:
            st.info("Run prediction to generate recommendations and evaluation metrics.")
        else:
            st.markdown("<div class='section-title'>Recommended products</div>", unsafe_allow_html=True)
            recs = recommend_top_n(
                product=product,
                X_item_user=X,
                X_item_user_bin=X_bin,
                item_index=item_index,
                items=items,
                top_n=int(top_n),
            )

            if recs.empty:
                st.warning("No recommendations found. Try lowering 'Min customers per product'.")
            else:
                st.dataframe(recs[["Product", "Similarity", "CoPurchaseCustomers"]], use_container_width=True)

            try:
                Xn = normalize(X, axis=1)
                i = item_index.get(product)
                if i is not None:
                    sims = (Xn @ Xn[i].T).toarray().ravel()
                    sims[i] = -1.0

                    cop = (X_bin @ X_bin[i].T).toarray().ravel().astype(int)
                    cop[i] = 0

                    mask = np.arange(len(items)) != i
                    items_other = items[mask]

                    y_true = (cop[mask] > 0).astype(int)
                    y_score = sims[mask].astype(float)
                    rec_set = set(recs["Product"].tolist()) if not recs.empty else set()
                    y_pred = np.asarray([1 if it in rec_set else 0 for it in items_other], dtype=int)

                    st.session_state["y_true"] = y_true
                    st.session_state["y_pred"] = y_pred
                    st.session_state["y_score"] = y_score
            except Exception:
                pass

    with tab_explain:
        st.markdown("### Why these recommendations?")
        st.write(
            "We treat each product as a vector over customers (1 if a customer bought it, else 0 by default). "
            "Cosine similarity between two product vectors measures how often they are purchased by the same customers."
        )

        st.markdown("#### Interpret the scores")
        st.markdown(
            "- **Similarity close to 1**: products are frequently bought by the same customers\n"
            "- **CoPurchaseCustomers**: how many customers bought both products (simple, business-friendly explanation)"
        )

        st.markdown("#### Practical use")
        st.markdown(
            "- Cross-sell on product pages\n"
            "- Basket building\n"
            "- Similar-items fallback for search"
        )

    with tab_arch:
        st.markdown("### Architecture workflow")
        st.markdown(
            "1. **Excel ingestion** (`OnlineRetail_cleaned.xlsx`)\n"
            "2. **Validation** (positive quantity/price, known customer, non-empty product)\n"
            "3. **Item×Customer matrix** (binary or quantity-weighted)\n"
            "4. **Cosine similarity** (item-to-item)\n"
            "5. **Top-N retrieval** + co-purchase counts\n"
            "6. **Streamlit SaaS UI** (KPIs, analytics, recommendations, explainability)"
        )

        try:
            import graphviz

            dot = graphviz.Digraph()
            dot.attr(rankdir="LR")
            dot.node("A", "OnlineRetail_cleaned.xlsx")
            dot.node("B", "Load + Validate")
            dot.node("C", "Item×Customer Matrix")
            dot.node("D", "Cosine Similarity")
            dot.node("E", "Top-N Recommendations")
            dot.node("F", "Streamlit Dashboard")
            dot.edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")])
            st.graphviz_chart(dot, use_container_width=True)
        except Exception:
            st.info("Graphviz is optional. Install `graphviz` to render the architecture diagram.")


if __name__ == "__main__":
    main()
