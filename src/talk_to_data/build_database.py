"""
Build Database Module for Credit Risk Analytics
===============================================
Extracts, engineers, and loads the Home Credit Default Risk dataset from an
external directory into the compact SQLite database (data/credit_risk_analytics.db).

Preserves the exact feature definitions and aggregation logic established across
all six historical tables in the research notebook:
1. bureau.csv
2. bureau_balance.csv
3. previous_application.csv
4. POS_CASH_balance.csv
5. credit_card_balance.csv
6. installments_payments.csv
"""

import gc
import os
import sqlite3
import sys
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.talk_to_data.database import DEFAULT_DB_PATH, DEFAULT_SCHEMA_PATH, init_database
from src.talk_to_data.schema import TABLE_NAME, get_allowed_columns

DEFAULT_RAW_DATA_DIR = os.environ.get(
    "HOME_CREDIT_DATA_DIR",
    r"C:\Users\hp\Desktop\home-credit-data"
)


def load_and_aggregate_bureau(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load bureau.csv, compute applicant-level debt/credit counts, and extract
    SK_ID_CURR <-> SK_ID_BUREAU links for bureau_balance.csv joining.
    Notebook Reference: Cell 29 & Cell 34.
    """
    bureau_path = os.path.join(data_dir, "bureau.csv")
    if not os.path.exists(bureau_path):
        print(f"Warning: {bureau_path} not found. Skipping bureau features.")
        return pd.DataFrame(columns=["SK_ID_CURR", "bureau_credit_count", "bureau_total_debt"]), pd.DataFrame()

    print("Processing bureau.csv...")
    cols = ["SK_ID_CURR", "SK_ID_BUREAU", "AMT_CREDIT_SUM_DEBT"]
    df_bureau = pd.read_csv(bureau_path, usecols=cols)

    # Applicant-level aggregation
    agg_bureau = df_bureau.groupby("SK_ID_CURR").agg(
        bureau_credit_count=("SK_ID_BUREAU", "count"),
        bureau_total_debt=("AMT_CREDIT_SUM_DEBT", "sum")
    ).reset_index()

    # Retain link for bureau_balance
    links_df = df_bureau[["SK_ID_CURR", "SK_ID_BUREAU"]].copy()

    del df_bureau
    gc.collect()
    print(f"  Bureau aggregated for {len(agg_bureau):,} applicants.")
    return agg_bureau, links_df


def load_and_aggregate_bureau_balance(data_dir: str, bureau_links: pd.DataFrame) -> pd.DataFrame:
    """
    Load bureau_balance.csv in chunks, aggregate delinquency by SK_ID_BUREAU,
    then map to SK_ID_CURR via bureau_links.
    Notebook Reference: Cells 32-35.
    """
    bb_path = os.path.join(data_dir, "bureau_balance.csv")
    if not os.path.exists(bb_path) or bureau_links.empty:
        print(f"Warning: {bb_path} not found or no bureau links. Skipping bureau_balance.")
        return pd.DataFrame(columns=["SK_ID_CURR", "bureau_total_overdue_months"])

    print("Processing bureau_balance.csv (chunked)...")
    cols = ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]
    
    chunk_list = []
    for chunk in pd.read_csv(bb_path, usecols=cols, chunksize=1_000_000):
        # Notebook Cell 32: Status 1-5 is overdue
        chunk["IS_OVERDUE"] = chunk["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
        part = chunk.groupby("SK_ID_BUREAU").agg(
            month_count=("MONTHS_BALANCE", "count"),
            overdue_months=("IS_OVERDUE", "sum")
        ).reset_index()
        chunk_list.append(part)

    combined_bb = pd.concat(chunk_list, ignore_index=True)
    del chunk_list
    gc.collect()

    # Aggregate by SK_ID_BUREAU
    bb_by_bureau = combined_bb.groupby("SK_ID_BUREAU").agg(
        total_months=("month_count", "sum"),
        total_overdue_months=("overdue_months", "sum")
    ).reset_index()

    del combined_bb
    gc.collect()

    # Merge with bureau_links to map to SK_ID_CURR (Notebook Cell 34-35)
    merged_bb = bureau_links.merge(bb_by_bureau, on="SK_ID_BUREAU", how="inner")
    del bb_by_bureau
    gc.collect()

    applicant_bb = merged_bb.groupby("SK_ID_CURR").agg(
        bureau_total_overdue_months=("total_overdue_months", "sum")
    ).reset_index()

    del merged_bb
    gc.collect()
    print(f"  Bureau Balance aggregated for {len(applicant_bb):,} applicants.")
    return applicant_bb


def load_and_aggregate_previous_applications(data_dir: str) -> pd.DataFrame:
    """
    Load previous_application.csv and aggregate application count and refusal rate by SK_ID_CURR.
    Notebook Reference: Cell 39.
    """
    prev_path = os.path.join(data_dir, "previous_application.csv")
    if not os.path.exists(prev_path):
        print(f"Warning: {prev_path} not found. Skipping previous application features.")
        return pd.DataFrame(columns=["SK_ID_CURR", "prev_app_count", "prev_refusal_rate"])

    print("Processing previous_application.csv...")
    cols = ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS"]
    df_prev = pd.read_csv(prev_path, usecols=cols)

    # Notebook Cell 39: IS_REFUSED
    df_prev["IS_REFUSED"] = (df_prev["NAME_CONTRACT_STATUS"] == "Refused").astype("int8")

    agg_df = df_prev.groupby("SK_ID_CURR").agg(
        prev_app_count=("SK_ID_PREV", "count"),
        prev_refused_count=("IS_REFUSED", "sum")
    ).reset_index()

    agg_df["prev_refusal_rate"] = np.where(
        agg_df["prev_app_count"] > 0,
        (agg_df["prev_refused_count"] / agg_df["prev_app_count"]).round(4),
        0.0
    )
    agg_df = agg_df.drop(columns=["prev_refused_count"])

    del df_prev
    gc.collect()
    print(f"  Previous applications aggregated for {len(agg_df):,} applicants.")
    return agg_df


def load_and_aggregate_pos_cash(data_dir: str) -> pd.DataFrame:
    """
    Load POS_CASH_balance.csv in chunks and aggregate POS loan counts and delinquency by SK_ID_CURR.
    Notebook Reference: Cell 136.
    """
    pos_path = os.path.join(data_dir, "POS_CASH_balance.csv")
    if not os.path.exists(pos_path):
        print(f"Warning: {pos_path} not found. Skipping POS_CASH features.")
        return pd.DataFrame(columns=["SK_ID_CURR", "pos_contract_count", "pos_dpd_ratio"])

    print("Processing POS_CASH_balance.csv (chunked)...")
    cols = ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD"]
    
    chunk_list = []
    for chunk in pd.read_csv(pos_path, usecols=cols, chunksize=1_000_000):
        # Notebook Cell 136: HAS_DPD = (SK_DPD > 0)
        chunk["HAS_DPD"] = (chunk["SK_DPD"] > 0).astype("int8")
        agg_chunk = chunk.groupby("SK_ID_CURR").agg(
            prev_contracts=("SK_ID_PREV", "nunique"),
            month_count=("MONTHS_BALANCE", "count"),
            dpd_months=("HAS_DPD", "sum")
        ).reset_index()
        chunk_list.append(agg_chunk)

    combined_pos = pd.concat(chunk_list, ignore_index=True)
    del chunk_list
    gc.collect()

    final_pos = combined_pos.groupby("SK_ID_CURR").agg(
        pos_contract_count=("prev_contracts", "max"),
        total_months=("month_count", "sum"),
        total_dpd_months=("dpd_months", "sum")
    ).reset_index()

    final_pos["pos_dpd_ratio"] = np.where(
        final_pos["total_months"] > 0,
        (final_pos["total_dpd_months"] / final_pos["total_months"]).round(4),
        0.0
    )
    final_pos = final_pos.drop(columns=["total_months", "total_dpd_months"])

    del combined_pos
    gc.collect()
    print(f"  POS/CASH aggregated for {len(final_pos):,} applicants.")
    return final_pos


def load_and_aggregate_credit_card(data_dir: str) -> pd.DataFrame:
    """
    Load credit_card_balance.csv in chunks and aggregate CC balance, utilization, and payments.
    Notebook Reference: Cell 137.
    """
    cc_path = os.path.join(data_dir, "credit_card_balance.csv")
    if not os.path.exists(cc_path):
        print(f"Warning: {cc_path} not found. Skipping Credit Card features.")
        return pd.DataFrame(columns=["SK_ID_CURR", "cc_contract_count", "cc_avg_utilization"])

    print("Processing credit_card_balance.csv (chunked)...")
    cols = ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_PAYMENT_CURRENT", "SK_DPD"]

    chunk_list = []
    for chunk in pd.read_csv(cc_path, usecols=cols, chunksize=500_000):
        # Notebook Cell 137: Utilization & Indicators
        chunk["UTILIZATION"] = np.where(
            chunk["AMT_CREDIT_LIMIT_ACTUAL"] > 0,
            chunk["AMT_BALANCE"] / chunk["AMT_CREDIT_LIMIT_ACTUAL"],
            np.nan
        )
        chunk["HAS_DPD"] = (chunk["SK_DPD"] > 0).astype("int8")

        agg_chunk = chunk.groupby("SK_ID_CURR").agg(
            cc_contracts=("SK_ID_PREV", "nunique"),
            month_count=("MONTHS_BALANCE", "count"),
            sum_util=("UTILIZATION", "sum"),
            count_util=("UTILIZATION", "count"),
            sum_payment=("AMT_PAYMENT_CURRENT", "sum"),
            dpd_months=("HAS_DPD", "sum")
        ).reset_index()
        chunk_list.append(agg_chunk)

    combined_cc = pd.concat(chunk_list, ignore_index=True)
    del chunk_list
    gc.collect()

    final_cc = combined_cc.groupby("SK_ID_CURR").agg(
        cc_contract_count=("cc_contracts", "max"),
        total_months=("month_count", "sum"),
        total_util_sum=("sum_util", "sum"),
        total_util_count=("count_util", "sum"),
        cc_total_payment=("sum_payment", "sum"),
        total_dpd_months=("dpd_months", "sum")
    ).reset_index()

    final_cc["cc_avg_utilization"] = np.where(
        final_cc["total_util_count"] > 0,
        (final_cc["total_util_sum"] / final_cc["total_util_count"]).round(4),
        0.0
    )
    final_cc["cc_dpd_ratio"] = np.where(
        final_cc["total_months"] > 0,
        (final_cc["total_dpd_months"] / final_cc["total_months"]).round(4),
        0.0
    )
    final_cc = final_cc.drop(columns=["total_months", "total_util_sum", "total_util_count", "total_dpd_months"])

    del combined_cc
    gc.collect()
    print(f"  Credit Card aggregated for {len(final_cc):,} applicants.")
    return final_cc


def load_and_aggregate_installments(data_dir: str) -> pd.DataFrame:
    """
    Load installments_payments.csv in chunks to aggregate total payment amount
    and late payment ratio by SK_ID_CURR.
    Notebook Reference: Cell 138.
    """
    inst_path = os.path.join(data_dir, "installments_payments.csv")
    if not os.path.exists(inst_path):
        print(f"Warning: {inst_path} not found. Skipping installments features.")
        return pd.DataFrame(columns=["SK_ID_CURR", "total_payment_amount", "late_payment_ratio"])

    print("Processing installments_payments.csv (chunked)...")
    cols = ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"]

    # Notebook Cell 138: Payment delay and late flag
    chunk_list = []
    for chunk in pd.read_csv(inst_path, usecols=cols, chunksize=1_000_000):
        chunk["PAYMENT_DELAY"] = chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]
        chunk["IS_LATE"] = (chunk["PAYMENT_DELAY"] > 0).astype("int8")
        agg_chunk = chunk.groupby("SK_ID_CURR").agg(
            inst_count=("IS_LATE", "count"),
            late_count=("IS_LATE", "sum"),
            chunk_payment=("AMT_PAYMENT", "sum")
        ).reset_index()
        chunk_list.append(agg_chunk)

    combined = pd.concat(chunk_list, ignore_index=True)
    del chunk_list
    gc.collect()

    final_inst = combined.groupby("SK_ID_CURR").agg(
        total_installments=("inst_count", "sum"),
        total_late=("late_count", "sum"),
        total_payment_amount=("chunk_payment", "sum")
    ).reset_index()

    final_inst["late_payment_ratio"] = np.where(
        final_inst["total_installments"] > 0,
        (final_inst["total_late"] / final_inst["total_installments"]).round(4),
        0.0
    )
    final_inst["total_payment_amount"] = final_inst["total_payment_amount"].round(2)
    final_inst = final_inst.drop(columns=["total_installments", "total_late"])

    del combined
    gc.collect()
    print(f"  Installments aggregated for {len(final_inst):,} applicants.")
    return final_inst


def build_analytics_database(
    raw_data_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    schema_path: Optional[str] = None,
    limit: Optional[int] = None
) -> int:
    """
    Extract raw Home Credit data from all six historical sources, apply
    feature engineering, and bulk load into the SQLite analytics database.

    Args:
        raw_data_dir: Directory containing raw CSV files.
        db_path: Destination SQLite database path.
        schema_path: Path to sql/schema.sql.
        limit: Optional maximum rows to load (None for complete dataset).

    Returns:
        Number of applicant records inserted.
    """
    source_dir = raw_data_dir or DEFAULT_RAW_DATA_DIR
    target_db = db_path or DEFAULT_DB_PATH
    target_schema = schema_path or DEFAULT_SCHEMA_PATH

    print("=" * 75)
    print("BUILDING COMPLETE REAL CREDIT RISK ANALYTICS DATABASE")
    print("=" * 75)
    print(f"Source Raw Data Dir : {source_dir}")
    print(f"Target Database File: {target_db}")

    if not os.path.exists(source_dir):
        raise FileNotFoundError(
            f"Source raw data directory not found at: {source_dir}\n"
            "Please ensure HOME_CREDIT_DATA_DIR is configured."
        )

    app_train_path = os.path.join(source_dir, "application_train.csv")
    if not os.path.exists(app_train_path):
        raise FileNotFoundError(f"application_train.csv not found at: {app_train_path}")

    start_time = time.time()

    # Step 1: Reinitialize Database Schema (drops existing and recreates clean tables)
    print("\n1. Initializing clean database schema...")
    init_database(db_path=target_db, schema_path=target_schema)

    # Step 2: Load and Aggregate All Six Historical Datasets
    print("\n2. Processing all 6 historical credit records matching research notebook...")
    df_bureau, bureau_links = load_and_aggregate_bureau(source_dir)
    df_bureau_bal = load_and_aggregate_bureau_balance(source_dir, bureau_links)
    del bureau_links
    gc.collect()

    df_prev = load_and_aggregate_previous_applications(source_dir)
    df_pos = load_and_aggregate_pos_cash(source_dir)
    df_cc = load_and_aggregate_credit_card(source_dir)
    df_inst = load_and_aggregate_installments(source_dir)

    # Step 3: Load Main Application Dataset
    print("\n3. Loading application_train.csv...")
    app_cols = [
        "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
        "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
    ]
    df_app = pd.read_csv(app_train_path, usecols=app_cols, nrows=limit)
    print(f"  Loaded {len(df_app):,} application records.")

    # Step 4: Merge Historical Features
    print("\n4. Merging historical aggregates...")
    df_merged = df_app.merge(df_bureau, on="SK_ID_CURR", how="left")
    df_merged = df_merged.merge(df_bureau_bal, on="SK_ID_CURR", how="left")
    df_merged = df_merged.merge(df_prev, on="SK_ID_CURR", how="left")
    df_merged = df_merged.merge(df_pos, on="SK_ID_CURR", how="left")
    df_merged = df_merged.merge(df_cc, on="SK_ID_CURR", how="left")
    df_merged = df_merged.merge(df_inst, on="SK_ID_CURR", how="left")

    del df_app, df_bureau, df_bureau_bal, df_prev, df_pos, df_cc, df_inst
    gc.collect()

    # Step 5: Feature Engineering Matching Research Notebook
    print("\n5. Applying business ratios and feature engineering...")
    # Fill defaults for historical metrics where applicant had no history
    df_merged["bureau_credit_count"] = df_merged["bureau_credit_count"].fillna(0).astype("int32")
    df_merged["bureau_total_debt"] = df_merged["bureau_total_debt"].fillna(0.0).round(2)
    df_merged["prev_app_count"] = df_merged["prev_app_count"].fillna(0).astype("int32")
    df_merged["prev_refusal_rate"] = df_merged["prev_refusal_rate"].fillna(0.0).round(4)
    df_merged["total_payment_amount"] = df_merged["total_payment_amount"].fillna(0.0).round(2)
    df_merged["late_payment_ratio"] = df_merged["late_payment_ratio"].fillna(0.0).round(4)

    # Age and Employment (Notebook Cell 141)
    df_merged["age_years"] = ((-df_merged["DAYS_BIRTH"]) / 365.25).round(1)
    days_emp = df_merged["DAYS_EMPLOYED"].replace(365243, np.nan)
    df_merged["employment_years"] = ((-days_emp) / 365.25).round(1)

    # Financial Ratios (Notebook Cell 141)
    income = df_merged["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df_merged["credit_income_ratio"] = (df_merged["AMT_CREDIT"] / income).round(2)
    df_merged["annuity_income_ratio"] = (df_merged["AMT_ANNUITY"] / income).round(2)
    df_merged["credit_goods_ratio"] = (
        df_merged["AMT_CREDIT"] / df_merged["AMT_GOODS_PRICE"].replace(0, np.nan)
    ).round(2)

    # Clean infinite values
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan)

    # Column name alignment to schema
    df_merged.columns = [c.lower() for c in df_merged.columns]

    allowed_cols = get_allowed_columns()
    df_final = df_merged[allowed_cols]

    # Step 6: Bulk Insert into SQLite Database
    print("\n6. Bulk inserting into SQLite database...")
    with sqlite3.connect(target_db) as conn:
        df_final.to_sql(TABLE_NAME, conn, if_exists="append", index=False, chunksize=10000)

        # Bulk insert into relational summary tables
        app_cols = get_allowed_columns("applicants")
        df_final[app_cols].to_sql("applicants", conn, if_exists="append", index=False, chunksize=10000)
        df_final[["sk_id_curr", "bureau_credit_count", "bureau_total_debt"]].to_sql(
            "bureau_summary", conn, if_exists="append", index=False, chunksize=10000
        )
        df_final[["sk_id_curr", "prev_app_count", "prev_refusal_rate"]].to_sql(
            "previous_application_summary", conn, if_exists="append", index=False, chunksize=10000
        )
        df_final[["sk_id_curr", "total_payment_amount", "late_payment_ratio"]].to_sql(
            "installment_summary", conn, if_exists="append", index=False, chunksize=10000
        )

        conn.execute("ANALYZE;")
        conn.commit()

    # Step 7: Build model-predicted risk scores (requires feature store to be populated first)
    print("\n7. Building applicant_risk_scores from CatBoost model predictions...")
    feature_store_path = os.path.join("data", "model_features.db")
    if os.path.exists(feature_store_path):
        try:
            from src.talk_to_data.build_risk_scores import build_risk_scores
            risk_count = build_risk_scores(db_path=target_db, batch_size=10000)
            print(f"   Risk scores populated for {risk_count:,} applicants.")
        except Exception as e:
            print(f"   WARNING: Could not build risk scores: {e}")
            print("   Run: python -m src.talk_to_data.build_risk_scores")
    else:
        print("   WARNING: Feature store not found. Run build_risk_scores separately after populating feature store.")

    total_time = time.time() - start_time
    print(f"\nDatabase built successfully in {total_time:.1f}s.")
    print(f"Total Rows Inserted: {len(df_final):,}")
    return len(df_final)


if __name__ == "__main__":
    data_directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW_DATA_DIR
    build_analytics_database(raw_data_dir=data_directory)

