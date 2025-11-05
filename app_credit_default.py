# ===============================================================
# app_credit_default.py
# Streamlit app for Credit Default Prediction + simple EDA
#
# Cara run (di terminal / Anaconda Prompt):
#   streamlit run app_credit_default.py
#
# Syarat:
#   - file ini ada di folder yang sama dengan:
#       creditdefault.py
#       best_credit_default_model.pkl
# ===============================================================

import os
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import creditdefault as cd  # pakai fungsi yang sudah kamu buat


# ---------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Credit Default Explorer",
    layout="wide",
)


# ---------------------------------------------------------------
# Helper: clean ID columns for display
# ---------------------------------------------------------------
def clean_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    id_cols = ["customer_id", "application_id", "loan_id", "payment_id"]
    for col in id_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype(str)
    if "dpd" in df.columns:
        df["dpd"] = pd.to_numeric(df["dpd"], errors="coerce")
    return df


# ---------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------
st.sidebar.title("Settings")

threshold = st.sidebar.slider(
    "Prediction threshold (default = 0.5)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

default_output_name = st.sidebar.text_input(
    "Default output CSV file name",
    value="credit_default_predictions.csv",
)

show_feature_importance = st.sidebar.checkbox(
    "Show feature importance", value=True
)


# ---------------------------------------------------------------
# Main title
# ---------------------------------------------------------------
st.title("Credit Default Prediction Dashboard")

st.markdown(
    """
This app takes raw payment-level data (combined_df-style), 
shows basic EDA, and then runs the credit default model 
you trained in your notebook.
"""
)


# ---------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------
st.header("1. Upload raw payment-level CSV")

uploaded_file = st.file_uploader(
    "Upload combined_df-style CSV",
    type=["csv"],
    help="File should contain customer_id, loan_id, payment_id, loan_amount, installment_amount, paid_amount, due_date, dpd, etc.",
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# Read and clean
raw_df = pd.read_csv(uploaded_file)
raw_df = clean_id_columns(raw_df)

st.subheader("Raw data preview")
st.write(f"Shape: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
st.dataframe(raw_df.head(), use_container_width=True)


# ---------------------------------------------------------------
# 2. Simple EDA
# ---------------------------------------------------------------
st.header("2. Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic summary (numeric columns)")
    st.dataframe(raw_df.describe().T)

with col2:
    st.subheader("Columns overview")
    st.write(pd.DataFrame({
        "column": raw_df.columns,
        "dtype": [str(t) for t in raw_df.dtypes],
        "n_missing": raw_df.isna().sum().values,
    }))

st.markdown("---")

eda_col1, eda_col2 = st.columns(2)

# DPD distribution
with eda_col1:
    if "dpd" in raw_df.columns:
        st.subheader("DPD distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        raw_df["dpd"].dropna().hist(bins=30, ax=ax)
        ax.set_xlabel("dpd")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("Column 'dpd' not found in data.")

# Province or loan_purpose distribution
with eda_col2:
    if "address_provinsi" in raw_df.columns:
        st.subheader("Top provinces by number of rows")
        top_prov = (
            raw_df["address_provinsi"]
            .value_counts()
            .head(10)
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        top_prov.plot(kind="barh", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("address_provinsi")
        st.pyplot(fig)
    elif "loan_purpose" in raw_df.columns:
        st.subheader("Top loan purposes")
        top_purpose = (
            raw_df["loan_purpose"]
            .value_counts()
            .head(10)
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        top_purpose.plot(kind="barh", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("loan_purpose")
        st.pyplot(fig)
    else:
        st.info("No 'address_provinsi' or 'loan_purpose' column found for EDA bar chart.")


# ---------------------------------------------------------------
# 3. Run Prediction
# ---------------------------------------------------------------
st.header("3. Run credit default prediction")

if not os.path.exists(cd.MODEL_FILENAME):
    st.error(
        f"Model file '{cd.MODEL_FILENAME}' was not found in the current folder. "
        "Please make sure the .pkl is in the same directory as this app."
    )
    st.stop()


run_button = st.button("Run model on uploaded data")

if not run_button:
    st.stop()

with st.spinner("Running model and generating predictions..."):
    # Load model using your helper (silent)
    model = cd.load_model_silent(cd.MODEL_FILENAME)

    # Run pipeline for prediction
    pred_df = cd.predict_raw_data(model, raw_df, threshold=threshold)

st.subheader("Prediction output (head)")
st.dataframe(pred_df.head(), use_container_width=True)

st.write(f"Total customers scored: {len(pred_df):,}")


# ---------------------------------------------------------------
# 4. Prediction distribution charts
# ---------------------------------------------------------------
st.subheader("Prediction distribution")

counts = pred_df["pred_default"].value_counts().sort_index()
labels = ["Non-default (0)", "Default (1)"]
non_def = counts.get(0, 0)
def_ = counts.get(1, 0)
values = [non_def, def_]
total = non_def + def_
percentages = [v / total * 100 if total > 0 else 0 for v in values]

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("Counts by predicted class")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values, color=["green", "orange"])
    ax.set_ylabel("Number of customers")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    st.pyplot(fig)

with chart_col2:
    st.write("Percentage by predicted class")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(
        percentages,
        labels=labels,
        colors=["green", "orange"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.axis("equal")
    st.pyplot(fig)


# ---------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------
if show_feature_importance:
    st.subheader("Feature importance from the model")

    fi_df = cd.get_feature_importance(model, cd.SELECTED_FEATURES)
    if fi_df is None:
        st.info("Feature importance is not available for this model type.")
    else:
        st.dataframe(fi_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(fi_df["feature"], fi_df["importance"], color="orange")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.invert_yaxis()
        st.pyplot(fig)


# ---------------------------------------------------------------
# 6. Download predictions
# ---------------------------------------------------------------
st.subheader("Download predictions")

final_name = default_output_name.strip()
if not final_name:
    final_name = "credit_default_predictions.csv"
if not final_name.lower().endswith(".csv"):
    final_name += ".csv"

csv_buffer = io.StringIO()
pred_df.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode("utf-8")

st.download_button(
    label=f"Download predictions as '{final_name}'",
    data=csv_bytes,
    file_name=final_name,
    mime="text/csv",
)

st.success("Predictions are ready. You can explore them above or download as CSV.")
