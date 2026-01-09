import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Retail Recommender", layout="wide")

# ------------------ TITLE ------------------
st.title("🛒 Online Retail Recommendation System")


# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    if os.path.exists("OnlineRetail_cleaned.xlsx"):
        df = pd.read_excel("OnlineRetail_cleaned.xlsx", sheet_name="cleaned")
        return df

    df = pd.read_excel("OnlineRetail.xlsx")
    return df

df = load_data()

# ------------------ DATA CLEANING ------------------
if "DescriptionClean" in df.columns:
    df["Description"] = df["DescriptionClean"]

if "Revenue" not in df.columns and {"Quantity", "UnitPrice"}.issubset(df.columns):
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

if "CustomerID" in df.columns:
    df = df.dropna(subset=["CustomerID"])

if "InvoiceNo" in df.columns:
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

if "Quantity" in df.columns:
    df = df[df["Quantity"] > 0]

if "UnitPrice" in df.columns:
    df = df[df["UnitPrice"] > 0]

# ------------------ SIDEBAR ------------------
st.sidebar.header("🔍 Navigation")
option = st.sidebar.radio(
    "Choose Section",
    ["Dataset Overview", "Visual Analysis", "Product Recommendation"]
)

# =================================================
# SECTION 1: DATASET OVERVIEW
# =================================================
if option == "Dataset Overview":
    st.subheader("📊 Dataset Overview")

    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.head(10))

    st.markdown("**Columns Description:**")
    st.markdown("""
    - InvoiceNo: Transaction ID  
    - StockCode: Product ID  
    - Description: Product Name  
    - Quantity: Items Purchased  
    - InvoiceDate: Date of Purchase  
    - UnitPrice: Price per item  
    - CustomerID: Customer Identifier  
    - Country: Country of Purchase
    """)

# =================================================
# SECTION 2: VISUAL ANALYSIS
# =================================================
elif option == "Visual Analysis":
    st.subheader("📈 Visual Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top 10 Countries by Transactions")
        top_countries = df['Country'].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax)
        ax.set_xlabel("Transactions")
        ax.set_ylabel("Country")
        st.pyplot(fig)

    with col2:
        st.markdown("### Top 10 Most Sold Products")
        top_products = df['Description'].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_products.values, y=top_products.index, ax=ax)
        ax.set_xlabel("Sales Count")
        ax.set_ylabel("Product")
        st.pyplot(fig)

# =================================================
# SECTION 3: RECOMMENDATION SYSTEM
# =================================================
elif option == "Product Recommendation":
    st.subheader("🤖 Product Recommendation Engine")

    st.markdown("**Algorithm Used:** Item-Based Collaborative Filtering (Unsupervised Learning)")

    # Create basket matrix
    basket = df.pivot_table(
        index='InvoiceNo',
        columns='Description',
        values='Quantity',
        aggfunc='sum'
    ).fillna(0)

    basket[basket > 0] = 1

    # Similarity calculation
    item_matrix = basket.T
    similarity = cosine_similarity(item_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=item_matrix.index,
        columns=item_matrix.index
    )

    product_list = similarity_df.columns.tolist()

    selected_product = st.selectbox("Select a Product", product_list)

    if st.button("Recommend Products"):
        recommendations = similarity_df[selected_product] \
            .sort_values(ascending=False)[1:6]

        st.success("Recommended Products:")
        st.table(recommendations)
