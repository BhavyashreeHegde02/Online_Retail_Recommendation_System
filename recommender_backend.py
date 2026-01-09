from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class RecommenderConfig:
    min_item_support_customers: int = 5
    use_binary_interactions: bool = True


def load_cleaned_retail_excel(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet = "cleaned" if "cleaned" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def prepare_retail_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "InvoiceDate" in out.columns:
        out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")

    for c in ["Quantity", "UnitPrice", "CustomerID"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "CustomerID" in out.columns:
        out["CustomerID"] = out["CustomerID"].astype("Int64")

    if "TotalPrice" not in out.columns:
        if "Revenue" in out.columns:
            out["TotalPrice"] = pd.to_numeric(out["Revenue"], errors="coerce")
        elif {"Quantity", "UnitPrice"}.issubset(out.columns):
            out["TotalPrice"] = out["Quantity"] * out["UnitPrice"]

    if "InvoiceDate" in out.columns and out["InvoiceDate"].notna().any():
        if "Year" not in out.columns:
            out["Year"] = out["InvoiceDate"].dt.year.astype("Int64")
        if "Month" not in out.columns:
            out["Month"] = out["InvoiceDate"].dt.month.astype("Int64")
        if "Day" not in out.columns:
            out["Day"] = out["InvoiceDate"].dt.day.astype("Int64")
        if "Hour" not in out.columns:
            out["Hour"] = out["InvoiceDate"].dt.hour.astype("Int64")

    if "Description" in out.columns:
        out["Description"] = out["Description"].astype(str).str.strip()

    if "Country" in out.columns:
        out["Country"] = out["Country"].astype(str).str.strip()

    required = {"CustomerID", "InvoiceNo", "Quantity"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns for recommender: {sorted(missing)}")

    out = out.dropna(subset=["CustomerID", "InvoiceNo"]).copy()

    if "Quantity" in out.columns:
        out = out[pd.to_numeric(out["Quantity"], errors="coerce").fillna(0) > 0].copy()

    if "UnitPrice" in out.columns:
        out = out[pd.to_numeric(out["UnitPrice"], errors="coerce").fillna(0) > 0].copy()

    item_col = "Description" if "Description" in out.columns else "StockCode"
    out[item_col] = out[item_col].astype(str).str.strip()
    out = out[out[item_col].notna() & (out[item_col] != "")].copy()

    out["_item"] = out[item_col]
    out["_qty"] = pd.to_numeric(out["Quantity"], errors="coerce").fillna(0).astype(float)

    return out


def build_item_user_matrix(
    df: pd.DataFrame,
    cfg: Optional[RecommenderConfig] = None,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict[str, int], np.ndarray]:
    cfg = cfg or RecommenderConfig()

    d = prepare_retail_df(df)

    items = d["_item"].astype(str).unique()
    users = d["CustomerID"].astype(int).unique()

    item_index = {it: i for i, it in enumerate(items)}
    user_index = {u: j for j, u in enumerate(users)}

    rows = d["_item"].map(item_index).astype(int).to_numpy()
    cols = d["CustomerID"].astype(int).map(user_index).astype(int).to_numpy()

    if cfg.use_binary_interactions:
        data = np.ones(len(d), dtype=np.float32)
    else:
        data = d["_qty"].to_numpy(dtype=np.float32)

    X = sparse.coo_matrix((data, (rows, cols)), shape=(len(items), len(users))).tocsr()

    X_bin = X.copy()
    X_bin.data = np.ones_like(X_bin.data)

    support = np.asarray(X_bin.sum(axis=1)).ravel()
    keep_mask = support >= cfg.min_item_support_customers

    if keep_mask.sum() < 2:
        raise ValueError(
            "Not enough items after support filtering. "
            "Lower min_item_support_customers in RecommenderConfig."
        )

    X = X[keep_mask]
    X_bin = X_bin[keep_mask]
    items_kept = items[keep_mask]

    item_index_kept = {it: i for i, it in enumerate(items_kept)}

    return X, X_bin, item_index_kept, items_kept


def recommend_top_n(
    product: str,
    X_item_user: sparse.csr_matrix,
    X_item_user_bin: sparse.csr_matrix,
    item_index: Dict[str, int],
    items: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    if product not in item_index:
        return pd.DataFrame(columns=["Product", "Similarity", "CoPurchaseCustomers"])

    Xn = normalize(X_item_user, axis=1)
    i = item_index[product]
    sims = (Xn @ Xn[i].T).toarray().ravel()
    sims[i] = -1.0

    cop = (X_item_user_bin @ X_item_user_bin[i].T).toarray().ravel().astype(int)
    cop[i] = 0

    top_idx = np.argsort(sims)[::-1][:top_n]

    rec = pd.DataFrame(
        {
            "Product": items[top_idx],
            "Similarity": sims[top_idx],
            "CoPurchaseCustomers": cop[top_idx],
        }
    )

    rec = rec[(rec["Similarity"] > 0) & (rec["CoPurchaseCustomers"] > 0)].copy()
    rec.reset_index(drop=True, inplace=True)
    return rec
