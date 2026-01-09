from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningConfig:
    min_unit_price: float = 0.01
    min_quantity: int = 1

    min_customer_invoices: int = 2
    min_customer_distinct_products: int = 2
    max_customer_cancel_rate: float = 0.6

    clip_lower_q: float = 0.001
    clip_upper_q: float = 0.999
    invoice_total_mad_z: float = 6.0


_REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


def load_online_retail_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = sorted(_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["InvoiceNo"] = out["InvoiceNo"].astype(str).str.strip()
    out["StockCode"] = out["StockCode"].astype(str).str.strip()

    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce")

    out["CustomerID"] = pd.to_numeric(out["CustomerID"], errors="coerce").astype("Int64")

    out["Country"] = out["Country"].astype(str).str.strip()
    out["Description"] = out["Description"].astype(str)

    return out


def _normalize_country(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace(
        {
            "EIRE": "Ireland",
            "UK": "United Kingdom",
            "U.K.": "United Kingdom",
            "USA": "United States",
        }
    )
    return s


def _normalize_description(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.upper()
    s = s.str.replace(r"[^A-Z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.strip()
    s = s.replace({"": pd.NA, "NAN": pd.NA})
    return s


def _flag_cancellations_and_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_cancelled_invoice"] = out["InvoiceNo"].str.startswith("C", na=False)
    out["is_return_line"] = (out["Quantity"] < 0) | (out["UnitPrice"] < 0)
    out["line_revenue"] = out["Quantity"] * out["UnitPrice"]
    return out


def _filter_non_merchandise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    desc = out["Description_norm"]
    stock = out["StockCode"].astype(str)

    non_merch = (
        desc.isna()
        | desc.str.fullmatch(r"\?+", na=False)
        | desc.str.contains(r"\bPOSTAGE\b", na=False)
        | desc.str.contains(r"\bCARRIAGE\b", na=False)
        | desc.str.contains(r"\bBANK CHARGES\b", na=False)
        | desc.str.contains(r"\bDOTCOM\b", na=False)
        | desc.str.contains(r"\bADJUST\b", na=False)
        | desc.str.contains(r"\bMANUAL\b", na=False)
        | desc.str.contains(r"\bDISCOUNT\b", na=False)
        | stock.isin({"POST", "D", "M", "DOT"})
    )

    return out.loc[~non_merch].copy()


def _invoice_level_validation(df_sales: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inv = df_sales.groupby("InvoiceNo", dropna=False).agg(
        invoice_date_min=("InvoiceDate", "min"),
        invoice_date_max=("InvoiceDate", "max"),
        customer_n=("CustomerID", pd.Series.nunique),
        country_n=("Country", pd.Series.nunique),
        n_lines=("StockCode", "size"),
    )

    valid = (
        inv["invoice_date_min"].notna()
        & (inv["customer_n"] == 1)
        & (inv["country_n"] == 1)
        & (inv["n_lines"] >= 1)
        & ((inv["invoice_date_max"] - inv["invoice_date_min"]) <= pd.Timedelta(days=1))
    )

    valid_invoices = inv.index[valid]
    df_valid = df_sales[df_sales["InvoiceNo"].isin(valid_invoices)].copy()

    inv_valid = inv.loc[valid].copy()
    inv_valid["invoice_date"] = inv_valid["invoice_date_min"]
    inv_valid.drop(columns=["invoice_date_min", "invoice_date_max"], inplace=True)

    return df_valid, inv_valid


def _clip_outliers(df_sales: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    out = df_sales.copy()

    for col in ["Quantity", "UnitPrice", "line_revenue"]:
        if col not in out.columns:
            continue
        lo = out[col].quantile(cfg.clip_lower_q)
        hi = out[col].quantile(cfg.clip_upper_q)
        out[col] = out[col].clip(lower=lo, upper=hi)

    out["line_revenue"] = out["Quantity"] * out["UnitPrice"]
    return out


def _invoice_outlier_filter(df_sales: pd.DataFrame, cfg: CleaningConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inv = df_sales.groupby("InvoiceNo", dropna=False).agg(
        invoice_revenue=("line_revenue", "sum"),
        invoice_quantity=("Quantity", "sum"),
        n_lines=("StockCode", "size"),
        n_distinct_products=("StockCode", pd.Series.nunique),
        customer_id=("CustomerID", "first"),
        country=("Country", "first"),
        invoice_date=("InvoiceDate", "min"),
    )

    x = np.log1p(inv["invoice_revenue"].clip(lower=0))
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if mad == 0 or np.isnan(mad):
        inv["is_invoice_outlier"] = False
        return df_sales, inv

    z = 0.6745 * (x - med) / mad
    inv["is_invoice_outlier"] = np.abs(z) > cfg.invoice_total_mad_z

    keep = inv.index[~inv["is_invoice_outlier"]]
    df_kept = df_sales[df_sales["InvoiceNo"].isin(keep)].copy()
    return df_kept, inv


def _compute_customer_features(df_all: pd.DataFrame, df_sales: pd.DataFrame) -> pd.DataFrame:
    ref_date = df_sales["InvoiceDate"].max()

    inv_sales = df_sales.groupby(["CustomerID", "InvoiceNo"], dropna=False).agg(
        invoice_revenue=("line_revenue", "sum"),
        invoice_date=("InvoiceDate", "min"),
        n_lines=("StockCode", "size"),
        n_distinct_products=("StockCode", pd.Series.nunique),
        invoice_quantity=("Quantity", "sum"),
    )

    cust = inv_sales.groupby("CustomerID", dropna=False).agg(
        n_invoices=("invoice_revenue", "size"),
        total_revenue=("invoice_revenue", "sum"),
        avg_invoice_revenue=("invoice_revenue", "mean"),
        total_quantity=("invoice_quantity", "sum"),
        avg_distinct_products_per_invoice=("n_distinct_products", "mean"),
        avg_lines_per_invoice=("n_lines", "mean"),
        last_purchase_date=("invoice_date", "max"),
        first_purchase_date=("invoice_date", "min"),
    )

    cust["recency_days"] = (ref_date - cust["last_purchase_date"]).dt.days.astype("Int64")
    cust["tenure_days"] = (cust["last_purchase_date"] - cust["first_purchase_date"]).dt.days.astype("Int64")

    inv_all = df_all.groupby(["CustomerID", "InvoiceNo"], dropna=False).agg(
        is_cancelled_invoice=("is_cancelled_invoice", "max"),
        has_return_line=("is_return_line", "max"),
    )

    cancel_rate = inv_all.groupby("CustomerID", dropna=False)["is_cancelled_invoice"].mean().rename("cancel_rate")
    return_rate = inv_all.groupby("CustomerID", dropna=False)["has_return_line"].mean().rename("return_rate")

    cust = cust.join(cancel_rate, how="left").join(return_rate, how="left")
    cust["cancel_rate"] = cust["cancel_rate"].fillna(0.0)
    cust["return_rate"] = cust["return_rate"].fillna(0.0)

    cust.reset_index(inplace=True)
    return cust


def _customer_behavior_filter(df_sales: pd.DataFrame, cust_features: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    eligible = cust_features[
        (cust_features["n_invoices"] >= cfg.min_customer_invoices)
        & (cust_features["avg_distinct_products_per_invoice"] >= 1)
        & (cust_features["cancel_rate"] <= cfg.max_customer_cancel_rate)
    ]["CustomerID"]

    out = df_sales[df_sales["CustomerID"].isin(set(eligible.dropna().tolist()))].copy()

    prod_cnt = out.groupby("CustomerID")["StockCode"].nunique().rename("n_distinct_products")
    out = out.join(prod_cnt, on="CustomerID")
    out = out[out["n_distinct_products"] >= cfg.min_customer_distinct_products].copy()
    out.drop(columns=["n_distinct_products"], inplace=True)

    return out


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["InvoiceDate"]
    out["invoice_year"] = dt.dt.year.astype("Int64")
    out["invoice_month"] = dt.dt.month.astype("Int64")
    out["invoice_day"] = dt.dt.day.astype("Int64")
    out["invoice_dayofweek"] = dt.dt.dayofweek.astype("Int64")
    out["invoice_hour"] = dt.dt.hour.astype("Int64")
    out["is_weekend"] = out["invoice_dayofweek"].isin([5, 6])
    return out


def _country_sanity_table(df_sales: pd.DataFrame) -> pd.DataFrame:
    g = df_sales.groupby("Country", dropna=False).agg(
        n_rows=("InvoiceNo", "size"),
        n_invoices=("InvoiceNo", pd.Series.nunique),
        n_customers=("CustomerID", pd.Series.nunique),
        revenue_sum=("line_revenue", "sum"),
        unitprice_median=("UnitPrice", "median"),
        unitprice_p95=("UnitPrice", lambda s: s.quantile(0.95)),
        qty_median=("Quantity", "median"),
        qty_p95=("Quantity", lambda s: s.quantile(0.95)),
    )
    g["revenue_share"] = g["revenue_sum"] / g["revenue_sum"].sum() if g["revenue_sum"].sum() != 0 else 0.0
    g["flag_low_volume"] = g["n_invoices"] < 10
    g["flag_suspicious_price_level"] = (g["unitprice_median"] <= 0) | (g["unitprice_p95"] > g["unitprice_median"] * 50)
    g.reset_index(inplace=True)
    g.sort_values(["revenue_sum", "n_invoices"], ascending=False, inplace=True)
    return g


def _quality_report(metrics: Dict[str, float | int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def clean_online_retail(
    input_excel_path: str,
    cfg: CleaningConfig | None = None,
) -> Dict[str, pd.DataFrame]:
    cfg = cfg or CleaningConfig()

    raw = load_online_retail_excel(input_excel_path)
    raw_rows = len(raw)

    df = _coerce_types(raw)
    df = df.drop_duplicates().copy()

    df["Country"] = _normalize_country(df["Country"])
    df["Description_norm"] = _normalize_description(df["Description"])

    df = _flag_cancellations_and_returns(df)

    df_all = df.copy()

    df_sales = df_all[
        (~df_all["is_cancelled_invoice"])
        & (~df_all["is_return_line"])
        & (df_all["Quantity"] >= cfg.min_quantity)
        & (df_all["UnitPrice"] >= cfg.min_unit_price)
        & (df_all["CustomerID"].notna())
        & (df_all["InvoiceDate"].notna())
    ].copy()

    df_sales = _filter_non_merchandise(df_sales)

    df_sales, inv_valid = _invoice_level_validation(df_sales)

    df_sales["line_revenue"] = df_sales["Quantity"] * df_sales["UnitPrice"]
    df_sales = df_sales[df_sales["line_revenue"].notna()].copy()

    df_sales = _clip_outliers(df_sales, cfg)
    df_sales, invoice_features = _invoice_outlier_filter(df_sales, cfg)

    cust_features = _compute_customer_features(df_all, df_sales)
    df_sales = _customer_behavior_filter(df_sales, cust_features, cfg)

    df_sales = _add_time_features(df_sales)

    cleaned = df_sales.copy()
    cleaned.rename(columns={"line_revenue": "Revenue", "Description_norm": "DescriptionClean"}, inplace=True)

    cleaned = cleaned[
        [
            "InvoiceNo",
            "StockCode",
            "Description",
            "DescriptionClean",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "Revenue",
            "CustomerID",
            "Country",
            "invoice_year",
            "invoice_month",
            "invoice_day",
            "invoice_dayofweek",
            "invoice_hour",
            "is_weekend",
        ]
    ].copy()

    cleaned["CustomerID"] = cleaned["CustomerID"].astype("Int64")

    country_sanity = _country_sanity_table(df_sales)

    revenue_bad_rows = int((cleaned["Revenue"] < 0).sum())
    revenue_zero_rows = int((cleaned["Revenue"] == 0).sum())

    metrics = {
        "raw_rows": raw_rows,
        "rows_after_dedup": int(len(df)),
        "sales_rows_after_rules": int(len(df_sales)),
        "n_customers_clean": int(cleaned["CustomerID"].nunique(dropna=True)),
        "n_invoices_clean": int(cleaned["InvoiceNo"].nunique(dropna=True)),
        "n_products_clean": int(cleaned["StockCode"].nunique(dropna=True)),
        "revenue_sum_clean": float(cleaned["Revenue"].sum()),
        "revenue_negative_rows": revenue_bad_rows,
        "revenue_zero_rows": revenue_zero_rows,
        "n_countries_clean": int(cleaned["Country"].nunique(dropna=True)),
        "invoice_outliers_removed": int(invoice_features["is_invoice_outlier"].sum()) if "is_invoice_outlier" in invoice_features.columns else 0,
        "invalid_invoices_removed": int(len(raw["InvoiceNo"].unique())) - int(len(inv_valid)),
    }

    report = _quality_report(metrics)

    invoice_features_reset = invoice_features.reset_index().copy()

    return {
        "cleaned": cleaned,
        "invoice_features": invoice_features_reset,
        "customer_features": cust_features,
        "quality_report": report,
        "country_sanity": country_sanity,
    }


def save_cleaned_to_excel(outputs: Dict[str, pd.DataFrame], output_excel_path: str) -> None:
    try:
        writer = pd.ExcelWriter(output_excel_path, engine="openpyxl")
    except Exception:
        writer = pd.ExcelWriter(output_excel_path)

    with writer:
        for sheet, df in outputs.items():
            safe_sheet = str(sheet)[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)
