"""
Build Script for Compact ML Inference Feature Store
===================================================
Constructs `data/model_features.db` containing the exact 231 engineered features
required for offline, zero-raw-CSV runtime inference by the CatBoost model.

Source: HOME_CREDIT_DATA_DIR (Build-time only).
Target: data/model_features.db (applicant_model_features table).
"""

import gc
import os
import pickle
import sqlite3
import time
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_SOURCE_DIR = os.getenv("HOME_CREDIT_DATA_DIR", r"C:\Users\hp\Desktop\home-credit-data")
DEFAULT_TARGET_DB = os.path.join("data", "model_features.db")
DEFAULT_FEATURE_COLUMNS_PATH = os.path.join("artifacts", "feature_columns.pkl")


def load_authoritative_features(features_path: str = DEFAULT_FEATURE_COLUMNS_PATH):
    with open(features_path, "rb") as f:
        return pickle.load(f)


def build_model_feature_store(
    source_dir: Optional[str] = None,
    target_db: Optional[str] = None,
    features_path: Optional[str] = None
) -> int:
    start_total = time.time()
    data_path = source_dir or DEFAULT_SOURCE_DIR
    out_db = target_db or DEFAULT_TARGET_DB
    f_path = features_path or DEFAULT_FEATURE_COLUMNS_PATH

    print("=" * 80)
    print("STARTING ML INFERENCE FEATURE STORE BUILD")
    print(f"Source Directory : {data_path}")
    print(f"Target Database  : {out_db}")
    print("=" * 80)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source raw data directory not found: {data_path}")

    authoritative_cols = load_authoritative_features(f_path)
    print(f"Loaded {len(authoritative_cols)} authoritative model feature columns.")

    # -------------------------------------------------------------------------
    # 1. APPLICATION TRAIN
    # -------------------------------------------------------------------------
    t0 = time.time()
    print("\n[1/7] Loading application_train.csv...")
    app_train_path = os.path.join(data_path, "application_train.csv")
    df_app = pd.read_csv(app_train_path)
    n_applicants = len(df_app)
    print(f"Loaded {n_applicants:,} applicants in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # 2. BUREAU & BUREAU BALANCE AGGREGATIONS
    # -------------------------------------------------------------------------
    t0 = time.time()
    print("\n[2/7] Aggregating bureau.csv and bureau_balance.csv...")
    bureau_path = os.path.join(data_path, "bureau.csv")
    bureau_df = pd.read_csv(bureau_path)

    # Vectorized indicator creation for 100x faster aggregation
    bureau_df["IS_ACTIVE"] = (bureau_df["CREDIT_ACTIVE"] == "Active").astype("int8")
    bureau_df["IS_CLOSED"] = (bureau_df["CREDIT_ACTIVE"] == "Closed").astype("int8")
    bureau_df["IS_SOLD"] = (bureau_df["CREDIT_ACTIVE"] == "Sold").astype("int8")
    bureau_df["IS_OVERDUE"] = (bureau_df["AMT_CREDIT_SUM_OVERDUE"] > 0).astype("int8")

    bureau_features = bureau_df.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_CLOSED_COUNT=("IS_CLOSED", "sum"),
        BUREAU_SOLD_COUNT=("IS_SOLD", "sum"),
        BUREAU_AVG_DAYS_CREDIT=("DAYS_CREDIT", "mean"),
        BUREAU_MIN_DAYS_CREDIT=("DAYS_CREDIT", "min"),
        BUREAU_TOTAL_CREDIT=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AVG_CREDIT=("AMT_CREDIT_SUM", "mean"),
        BUREAU_MAX_CREDIT=("AMT_CREDIT_SUM", "max"),
        BUREAU_TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_AVG_DEBT=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_TOTAL_LIMIT=("AMT_CREDIT_SUM_LIMIT", "sum"),
        BUREAU_AVG_LIMIT=("AMT_CREDIT_SUM_LIMIT", "mean"),
        BUREAU_TOTAL_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_MAX_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "max"),
        BUREAU_OVERDUE_COUNT=("IS_OVERDUE", "sum"),
        BUREAU_DAYS_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_TOTAL_PROLONG=("CNT_CREDIT_PROLONG", "sum"),
        BUREAU_MAX_PROLONG=("CNT_CREDIT_PROLONG", "max")
    ).reset_index()

    # Process bureau_balance in filtered chunks
    bureau_balance_path = os.path.join(data_path, "bureau_balance.csv")
    valid_bureau_set = set(bureau_df["SK_ID_BUREAU"])
    bb_parts = []
    for chunk in pd.read_csv(bureau_balance_path, chunksize=2_500_000):
        chunk_filtered = chunk[chunk["SK_ID_BUREAU"].isin(valid_bureau_set)].copy()
        if chunk_filtered.empty:
            continue
        chunk_filtered["IS_OVERDUE"] = chunk_filtered["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
        chunk_filtered["IS_SEVERE_OVERDUE"] = chunk_filtered["STATUS"].isin(["3", "4", "5"]).astype("int8")
        chunk_filtered["IS_NORMAL"] = chunk_filtered["STATUS"].isin(["0", "C"]).astype("int8")

        part = chunk_filtered.groupby("SK_ID_BUREAU").agg(
            BUREAU_MONTH_COUNT=("MONTHS_BALANCE", "count"),
            BUREAU_OVERDUE_MONTHS=("IS_OVERDUE", "sum"),
            BUREAU_SEVERE_OVERDUE_MONTHS=("IS_SEVERE_OVERDUE", "sum"),
            BUREAU_NORMAL_MONTHS=("IS_NORMAL", "sum"),
            BUREAU_OLDEST_MONTH=("MONTHS_BALANCE", "min"),
            BUREAU_LATEST_MONTH=("MONTHS_BALANCE", "max")
        )
        bb_parts.append(part)

    bb_combined = pd.concat(bb_parts).groupby(level=0).agg({
        "BUREAU_MONTH_COUNT": "sum",
        "BUREAU_OVERDUE_MONTHS": "sum",
        "BUREAU_SEVERE_OVERDUE_MONTHS": "sum",
        "BUREAU_NORMAL_MONTHS": "sum",
        "BUREAU_OLDEST_MONTH": "min",
        "BUREAU_LATEST_MONTH": "max"
    })
    bb_combined["BUREAU_OVERDUE_RATIO"] = bb_combined["BUREAU_OVERDUE_MONTHS"] / bb_combined["BUREAU_MONTH_COUNT"]
    bb_combined["BUREAU_SEVERE_OVERDUE_RATIO"] = bb_combined["BUREAU_SEVERE_OVERDUE_MONTHS"] / bb_combined["BUREAU_MONTH_COUNT"]

    # Link bureau balance with SK_ID_CURR through bureau table
    bureau_with_bb = bureau_df[["SK_ID_CURR", "SK_ID_BUREAU"]].merge(bb_combined, on="SK_ID_BUREAU", how="inner")
    bureau_balance_applicant = bureau_with_bb.groupby("SK_ID_CURR").agg(
        BUREAU_TOTAL_MONTHS=("BUREAU_MONTH_COUNT", "sum"),
        BUREAU_TOTAL_OVERDUE_MONTHS=("BUREAU_OVERDUE_MONTHS", "sum"),
        BUREAU_TOTAL_SEVERE_OVERDUE_MONTHS=("BUREAU_SEVERE_OVERDUE_MONTHS", "sum"),
        BUREAU_AVG_OVERDUE_RATIO=("BUREAU_OVERDUE_RATIO", "mean"),
        BUREAU_MAX_OVERDUE_RATIO=("BUREAU_OVERDUE_RATIO", "max"),
        BUREAU_AVG_SEVERE_OVERDUE_RATIO=("BUREAU_SEVERE_OVERDUE_RATIO", "mean")
    ).reset_index()

    del bureau_df, bb_parts, bb_combined, bureau_with_bb, valid_bureau_set
    gc.collect()
    print(f"Bureau features completed in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # 3. PREVIOUS APPLICATION AGGREGATIONS
    # -------------------------------------------------------------------------
    t0 = time.time()
    print("\n[3/7] Aggregating previous_application.csv...")
    prev_path = os.path.join(data_path, "previous_application.csv")
    prev_df = pd.read_csv(prev_path)

    prev_df["CREDIT_TO_APPLICATION_RATIO"] = prev_df["AMT_CREDIT"] / prev_df["AMT_APPLICATION"].replace(0, np.nan)
    prev_df["HAS_DOWN_PAYMENT"] = (prev_df["AMT_DOWN_PAYMENT"].fillna(0) > 0).astype("int8")
    prev_df["IS_APPROVED"] = (prev_df["NAME_CONTRACT_STATUS"] == "Approved").astype("int8")
    prev_df["IS_REFUSED"] = (prev_df["NAME_CONTRACT_STATUS"] == "Refused").astype("int8")
    prev_df["IS_CANCELLED"] = (prev_df["NAME_CONTRACT_STATUS"] == "Canceled").astype("int8")

    previous_features = prev_df.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
        PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
        PREV_CANCELLED_COUNT=("IS_CANCELLED", "sum"),
        PREV_AVG_APPLICATION=("AMT_APPLICATION", "mean"),
        PREV_AVG_CREDIT=("AMT_CREDIT", "mean"),
        PREV_TOTAL_CREDIT=("AMT_CREDIT", "sum"),
        PREV_MAX_CREDIT=("AMT_CREDIT", "max"),
        PREV_AVG_ANNUITY=("AMT_ANNUITY", "mean"),
        PREV_AVG_DOWN_PAYMENT=("AMT_DOWN_PAYMENT", "mean"),
        PREV_AVG_CREDIT_APPLICATION_RATIO=("CREDIT_TO_APPLICATION_RATIO", "mean"),
        PREV_DOWN_PAYMENT_COUNT=("HAS_DOWN_PAYMENT", "sum")
    ).reset_index()

    previous_features["PREV_APPROVAL_RATE"] = previous_features["PREV_APPROVED_COUNT"] / previous_features["PREV_APP_COUNT"]
    previous_features["PREV_REFUSAL_RATE"] = previous_features["PREV_REFUSED_COUNT"] / previous_features["PREV_APP_COUNT"]

    del prev_df
    gc.collect()
    print(f"Previous applications aggregated in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # 4. POS_CASH_BALANCE AGGREGATIONS
    # -------------------------------------------------------------------------
    t0 = time.time()
    print("\n[4/7] Aggregating POS_CASH_balance.csv...")
    pos_path = os.path.join(data_path, "POS_CASH_balance.csv")
    pos_df = pd.read_csv(pos_path)

    pos_df["HAS_DPD"] = (pos_df["SK_DPD"] > 0).astype("int8")
    pos_df["HAS_DPD_DEF"] = (pos_df["SK_DPD_DEF"] > 0).astype("int8")
    pos_df["IS_ACTIVE"] = (pos_df["NAME_CONTRACT_STATUS"] == "Active").astype("int8")
    pos_df["IS_COMPLETED"] = (pos_df["NAME_CONTRACT_STATUS"] == "Completed").astype("int8")

    pos_features = pos_df.groupby("SK_ID_CURR").agg(
        POS_CONTRACT_COUNT=("SK_ID_PREV", "nunique"),
        POS_MONTH_COUNT=("MONTHS_BALANCE", "count"),
        POS_AVG_DPD=("SK_DPD", "mean"),
        POS_MAX_DPD=("SK_DPD", "max"),
        POS_DPD_MONTHS=("HAS_DPD", "sum"),
        POS_AVG_DPD_DEF=("SK_DPD_DEF", "mean"),
        POS_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
        POS_DPD_DEF_MONTHS=("HAS_DPD_DEF", "sum"),
        POS_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        POS_COMPLETED_COUNT=("IS_COMPLETED", "sum"),
        POS_AVG_INSTALLMENTS=("CNT_INSTALMENT", "mean"),
        POS_AVG_FUTURE_INSTALLMENTS=("CNT_INSTALMENT_FUTURE", "mean"),
        POS_MIN_FUTURE_INSTALLMENTS=("CNT_INSTALMENT_FUTURE", "min")
    ).reset_index()

    pos_features["POS_DPD_RATIO"] = pos_features["POS_DPD_MONTHS"] / pos_features["POS_MONTH_COUNT"]
    pos_features["POS_DPD_DEF_RATIO"] = pos_features["POS_DPD_DEF_MONTHS"] / pos_features["POS_MONTH_COUNT"]
    pos_features["POS_ACTIVE_RATIO"] = pos_features["POS_ACTIVE_COUNT"] / pos_features["POS_MONTH_COUNT"]
    pos_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    del pos_df
    gc.collect()
    print(f"POS/CASH aggregated in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # 5. CREDIT_CARD_BALANCE AGGREGATIONS
    # -------------------------------------------------------------------------
    t0 = time.time()
    print("\n[5/7] Aggregating credit_card_balance.csv...")
    cc_path = os.path.join(data_path, "credit_card_balance.csv")
    cc_df = pd.read_csv(cc_path)

    cc_df["UTILIZATION"] = np.where(
        cc_df["AMT_CREDIT_LIMIT_ACTUAL"] > 0,
        cc_df["AMT_BALANCE"] / cc_df["AMT_CREDIT_LIMIT_ACTUAL"],
        np.nan
    )
    cc_df["HAS_DPD"] = (cc_df["SK_DPD"] > 0).astype("int8")
    cc_df["HAS_DPD_DEF"] = (cc_df["SK_DPD_DEF"] > 0).astype("int8")
    cc_df["HAS_PAYMENT"] = (cc_df["AMT_PAYMENT_CURRENT"].notna()).astype("int8")

    cc_features = cc_df.groupby("SK_ID_CURR").agg(
        CC_CONTRACT_COUNT=("SK_ID_PREV", "nunique"),
        CC_MONTH_COUNT=("MONTHS_BALANCE", "count"),
        CC_AVG_BALANCE=("AMT_BALANCE", "mean"),
        CC_MAX_BALANCE=("AMT_BALANCE", "max"),
        CC_AVG_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_MAX_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
        CC_AVG_UTILIZATION=("UTILIZATION", "mean"),
        CC_MAX_UTILIZATION=("UTILIZATION", "max"),
        CC_TOTAL_DRAWINGS=("AMT_DRAWINGS_CURRENT", "sum"),
        CC_AVG_DRAWINGS=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_TOTAL_DRAWING_COUNT=("CNT_DRAWINGS_CURRENT", "sum"),
        CC_TOTAL_PAYMENT=("AMT_PAYMENT_CURRENT", "sum"),
        CC_AVG_PAYMENT=("AMT_PAYMENT_CURRENT", "mean"),
        CC_PAYMENT_MONTHS=("HAS_PAYMENT", "sum"),
        CC_AVG_MIN_PAYMENT=("AMT_INST_MIN_REGULARITY", "mean"),
        CC_MAX_MATURE_INSTALLMENTS=("CNT_INSTALMENT_MATURE_CUM", "max"),
        CC_AVG_DPD=("SK_DPD", "mean"),
        CC_MAX_DPD=("SK_DPD", "max"),
        CC_DPD_MONTHS=("HAS_DPD", "sum"),
        CC_AVG_DPD_DEF=("SK_DPD_DEF", "mean"),
        CC_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
        CC_DPD_DEF_MONTHS=("HAS_DPD_DEF", "sum")
    ).reset_index()

    cc_features["CC_DPD_RATIO"] = cc_features["CC_DPD_MONTHS"] / cc_features["CC_MONTH_COUNT"]
    cc_features["CC_DPD_DEF_RATIO"] = cc_features["CC_DPD_DEF_MONTHS"] / cc_features["CC_MONTH_COUNT"]
    cc_features["CC_PAYMENT_MONTH_RATIO"] = cc_features["CC_PAYMENT_MONTHS"] / cc_features["CC_MONTH_COUNT"]
    cc_features["CC_PAYMENT_TO_MIN_RATIO"] = cc_features["CC_TOTAL_PAYMENT"] / cc_features["CC_AVG_MIN_PAYMENT"]
    cc_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    del cc_df
    gc.collect()
    print(f"Credit card aggregated in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # 6. INSTALLMENTS_PAYMENTS AGGREGATIONS
    # -------------------------------------------------------------------------
    t0 = time.time()
    print("\n[6/7] Aggregating installments_payments.csv...")
    inst_path = os.path.join(data_path, "installments_payments.csv")
    inst_df = pd.read_csv(inst_path)

    inst_df["PAYMENT_DELAY"] = inst_df["DAYS_ENTRY_PAYMENT"] - inst_df["DAYS_INSTALMENT"]
    inst_df["IS_LATE"] = (inst_df["PAYMENT_DELAY"] > 0).astype("int8")
    inst_df["PAYMENT_RATIO"] = np.where(inst_df["AMT_INSTALMENT"] > 0, inst_df["AMT_PAYMENT"] / inst_df["AMT_INSTALMENT"], np.nan)
    inst_df["IS_UNDERPAID"] = np.where(
        inst_df["AMT_PAYMENT"].notna() & inst_df["AMT_INSTALMENT"].notna(),
        (inst_df["AMT_PAYMENT"] < inst_df["AMT_INSTALMENT"]).astype("int8"),
        np.nan
    )

    inst_features = inst_df.groupby("SK_ID_CURR").agg(
        INSTALLMENT_COUNT=("SK_ID_PREV", "count"),
        AVG_PAYMENT_DELAY=("PAYMENT_DELAY", "mean"),
        MAX_PAYMENT_DELAY=("PAYMENT_DELAY", "max"),
        LATE_PAYMENT_COUNT=("IS_LATE", "sum"),
        UNDERPAYMENT_COUNT=("IS_UNDERPAID", "sum"),
        AVG_PAYMENT_RATIO=("PAYMENT_RATIO", "mean"),
        TOTAL_INSTALLMENT_AMOUNT=("AMT_INSTALMENT", "sum"),
        TOTAL_PAYMENT_AMOUNT=("AMT_PAYMENT", "sum"),
        AVG_INSTALLMENT_AMOUNT=("AMT_INSTALMENT", "mean"),
        AVG_PAYMENT_AMOUNT=("AMT_PAYMENT", "mean"),
        LATE_PAYMENT_RATIO=("IS_LATE", "mean"),
        UNDERPAYMENT_RATIO=("IS_UNDERPAID", "mean")
    ).reset_index()

    inst_features["OVERALL_PAYMENT_RATIO"] = np.where(
        inst_features["TOTAL_INSTALLMENT_AMOUNT"] > 0,
        inst_features["TOTAL_PAYMENT_AMOUNT"] / inst_features["TOTAL_INSTALLMENT_AMOUNT"],
        np.nan
    )
    inst_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    del inst_df
    gc.collect()
    print(f"Installments aggregated in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # 7. MERGE AND FINAL FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    t0 = time.time()
    print("\n[7/7] Merging historical aggregations and engineering 231 final features...")

    # Drop TARGET if present in app_train
    if "TARGET" in df_app.columns:
        df_merged = df_app.drop(columns=["TARGET"]).copy()
    else:
        df_merged = df_app.copy()

    df_merged = df_merged.merge(bureau_features, on="SK_ID_CURR", how="left")
    df_merged = df_merged.merge(bureau_balance_applicant, on="SK_ID_CURR", how="left")
    df_merged = df_merged.merge(previous_features, on="SK_ID_CURR", how="left")
    df_merged = df_merged.merge(pos_features, on="SK_ID_CURR", how="left")
    df_merged = df_merged.merge(cc_features, on="SK_ID_CURR", how="left")
    df_merged = df_merged.merge(inst_features, on="SK_ID_CURR", how="left")

    del bureau_features, bureau_balance_applicant, previous_features, pos_features, cc_features, inst_features
    gc.collect()

    # Ratios
    income = df_merged["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df_merged["CREDIT_INCOME_RATIO"] = df_merged["AMT_CREDIT"] / income
    df_merged["ANNUITY_INCOME_RATIO"] = df_merged["AMT_ANNUITY"] / income
    df_merged["GOODS_INCOME_RATIO"] = df_merged["AMT_GOODS_PRICE"] / income
    df_merged["ANNUITY_CREDIT_RATIO"] = df_merged["AMT_ANNUITY"] / df_merged["AMT_CREDIT"].replace(0, np.nan)
    df_merged["CREDIT_GOODS_RATIO"] = df_merged["AMT_CREDIT"] / df_merged["AMT_GOODS_PRICE"].replace(0, np.nan)

    # Age and Employment
    df_merged["AGE_YEARS"] = (-df_merged["DAYS_BIRTH"]) / 365.25
    df_merged["DAYS_EMPLOYED"] = df_merged["DAYS_EMPLOYED"].replace(365243, np.nan)
    df_merged["EMPLOYMENT_YEARS"] = (-df_merged["DAYS_EMPLOYED"]) / 365.25

    # Family features
    df_merged["INCOME_PER_FAMILY_MEMBER"] = df_merged["AMT_INCOME_TOTAL"] / df_merged["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df_merged["INCOME_PER_CHILD"] = df_merged["AMT_INCOME_TOTAL"] / (df_merged["CNT_CHILDREN"] + 1)

    # Document count
    doc_cols = [c for c in df_merged.columns if c.startswith("FLAG_DOCUMENT_")]
    df_merged["DOCUMENTS_PROVIDED_COUNT"] = df_merged[doc_cols].sum(axis=1)

    # Historical flags
    df_merged["HAS_CC_HISTORY"] = df_merged["CC_CONTRACT_COUNT"].notna().astype("int8")
    df_merged["HAS_BUREAU_HISTORY"] = df_merged["BUREAU_CREDIT_COUNT"].notna().astype("int8")
    df_merged["HAS_PREVIOUS_APPLICATION"] = df_merged["PREV_APP_COUNT"].notna().astype("int8")
    df_merged["HAS_POS_HISTORY"] = df_merged["POS_CONTRACT_COUNT"].notna().astype("int8")
    df_merged["HAS_INSTALLMENT_HISTORY"] = df_merged["INSTALLMENT_COUNT"].notna().astype("int8")

    # Bureau risk ratios
    df_merged["BUREAU_DEBT_CREDIT_RATIO"] = df_merged["BUREAU_TOTAL_DEBT"] / df_merged["BUREAU_TOTAL_CREDIT"].replace(0, np.nan)
    df_merged["BUREAU_OVERDUE_CREDIT_RATIO"] = df_merged["BUREAU_TOTAL_OVERDUE"] / df_merged["BUREAU_TOTAL_CREDIT"].replace(0, np.nan)

    # Replace infinities
    df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Categorical handling: fillna("Missing") and convert to str
    cat_cols = df_merged.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        df_merged[c] = df_merged[c].fillna("Missing").astype(str)

    # Ensure SK_ID_CURR is preserved
    sk_id_series = df_merged["SK_ID_CURR"].astype(int)

    # Select and strictly order columns matching authoritative 231 features
    final_df = df_merged[["SK_ID_CURR"] + authoritative_cols].copy()
    print(f"Final feature dataset constructed with shape: {final_df.shape} in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # WRITE TO SQLITE DATABASE
    # -------------------------------------------------------------------------
    t0 = time.time()
    os.makedirs(os.path.dirname(out_db), exist_ok=True)
    if os.path.exists(out_db):
        os.remove(out_db)

    print(f"\nWriting to SQLite database: {out_db}...")
    with sqlite3.connect(out_db) as conn:
        final_df.to_sql("applicant_model_features", conn, index=False, if_exists="replace")
        print("Creating index on SK_ID_CURR...")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_model_features_sk_id ON applicant_model_features (SK_ID_CURR);")
        conn.commit()

    db_size_mb = os.path.getsize(out_db) / (1024 * 1024)
    print(f"Database written successfully! File size: {db_size_mb:.2f} MB in {time.time() - t0:.2f}s.")
    print(f"Total build time: {time.time() - start_total:.2f}s.")
    return len(final_df)


if __name__ == "__main__":
    build_model_feature_store()
