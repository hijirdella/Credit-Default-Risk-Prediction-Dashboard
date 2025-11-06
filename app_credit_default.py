# app_credit_default.py
# Streamlit app for Credit Default Prediction + rich EDA
#
# Cara run (di terminal / Anaconda Prompt):
#   streamlit run app_credit_default.py
#
# Syarat:
#   - file ini ada di folder yang sama dengan:
#       creditdefault.py
#       best_credit_default_model.pkl


import os
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

import creditdefault as cd  # pakai fungsi yang sudah kamu buat


# Streamlit page config
st.set_page_config(
    page_title="Credit Default Explorer",
    layout="wide",
)


# Helper: enforce schema / dtypes
def clean_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan dan set tipe data supaya konsisten dengan schema combined_df:
      - ID -> string
      - tanggal -> datetime (sebagian UTC)
      - dpd -> Int64
      - numeric lain -> float
    """
    df = df.copy()

    # ID columns as pandas StringDtype
    id_cols = ["application_id", "customer_id", "loan_id", "payment_id"]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Numeric columns (float)
    num_cols = [
        "loan_amount",
        "loan_duration",
        "installment_amount",
        "paid_amount",
        "dependent",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # DPD as nullable integer
    if "dpd" in df.columns:
        df["dpd"] = pd.to_numeric(df["dpd"], errors="coerce").astype("Int64")

    # Datetime columns with UTC
    if "cdate" in df.columns:
        df["cdate"] = pd.to_datetime(df["cdate"], errors="coerce", utc=True)
    if "fund_transfer_ts" in df.columns:
        df["fund_transfer_ts"] = pd.to_datetime(
            df["fund_transfer_ts"], errors="coerce", utc=True
        )

    # Datetime columns without timezone
    for col in ["dob", "due_date", "paid_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Kolom kategorikal lainnya biarkan apa adanya (object)
    return df


# Sidebar controls
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


# Main title
st.title("Credit Default Prediction Dashboard")

st.markdown(
    """
This app takes raw payment-level data (combined-style),
shows exploratory data analysis (EDA), and then runs the
credit default model.
"""
)


# 1. File uploader
st.header("1. Upload raw payment-level CSV")

uploaded_file = st.file_uploader(
    "Upload combined_df-style CSV",
    type=["csv"],
    help=(
        "File should contain customer_id, loan_id, payment_id, loan_amount, "
        "installment_amount, paid_amount, due_date, dpd, etc."
    ),
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


# 2. Exploratory Data Analysis (≈15 charts)
st.header("2. Exploratory Data Analysis")

# Orange–green palette
GREEN = "#2ecc71"
ORANGE = "#e67e22"
PALETTE = [GREEN, ORANGE]


# 2.1 Overview tables
col1, col2 = st.columns(2)

with col1:
    st.subheader("Numeric summary")
    st.dataframe(raw_df.describe().T)

with col2:
    st.subheader("Columns overview")
    overview_df = pd.DataFrame({
        "column": raw_df.columns,
        "dtype": [str(t) for t in raw_df.dtypes],
        "n_missing": raw_df.isna().sum().values,
        "missing_pct": (raw_df.isna().mean() * 100).round(2).values,
    })
    st.dataframe(overview_df, use_container_width=True)

st.markdown("---")


# Helper plotting functions
def plot_hist(series, title, xlabel):
    fig, ax = plt.subplots(figsize=(5, 3))
    series.dropna().hist(bins=30, ax=ax, color=GREEN, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig)


def plot_bar(counts, title, xlabel, ylabel, horizontal=False, color=ORANGE):
    fig, ax = plt.subplots(figsize=(6, 3))
    if horizontal:
        counts.plot(kind="barh", ax=ax, color=color)
    else:
        counts.plot(kind="bar", ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    st.pyplot(fig)


# 2.2 Numeric distributions
st.subheader("2.2 Numeric distributions")

num_cols = ["loan_amount", "installment_amount", "paid_amount", "dpd"]
for i in range(0, len(num_cols), 2):
    c1, c2 = st.columns(2)
    for j, col in enumerate(num_cols[i:i + 2]):
        if col in raw_df.columns:
            with (c1 if j == 0 else c2):
                plot_hist(raw_df[col], f"Distribution of {col}", col)


# 2.3 Categorical distributions
st.subheader("2.3 Categorical distributions")

cat_cols = [
    "address_provinsi",
    "loan_purpose",
    "marital_status",
    "job_type",
    "job_industry",
    "dependent",
]

for col in cat_cols:
    if col in raw_df.columns:
        counts = raw_df[col].value_counts().head(10)
        with st.expander(f"Distribution of {col}", expanded=False):
            plot_bar(
                counts.sort_values(ascending=True),
                f"{col} (top 10)",
                col,
                "Count",
                horizontal=True,
                color=ORANGE,
            )


# 2.4 Relationships between variables
st.subheader("2.4 Relationships between variables")

r1, r2 = st.columns(2)
with r1:
    if {"loan_amount", "installment_amount"} <= set(raw_df.columns):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(
            raw_df["loan_amount"],
            raw_df["installment_amount"],
            alpha=0.5,
            color=GREEN,
            s=10,
        )
        ax.set_title("Loan vs Installment Amount")
        ax.set_xlabel("loan_amount")
        ax.set_ylabel("installment_amount")
        fig.tight_layout()
        st.pyplot(fig)
with r2:
    if {"installment_amount", "paid_amount"} <= set(raw_df.columns):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(
            raw_df["installment_amount"],
            raw_df["paid_amount"],
            alpha=0.5,
            color=ORANGE,
            s=10,
        )
        ax.set_title("Installment vs Paid Amount")
        ax.set_xlabel("install_amount")
        ax.set_ylabel("paid_amount")
        fig.tight_layout()
        st.pyplot(fig)

r3, r4 = st.columns(2)
with r3:
    if {"dpd", "loan_amount"} <= set(raw_df.columns):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(
            raw_df["loan_amount"],
            raw_df["dpd"],
            alpha=0.4,
            color=ORANGE,
            s=10,
        )
        ax.set_title("DPD vs Loan Amount")
        ax.set_xlabel("loan_amount")
        ax.set_ylabel("dpd")
        fig.tight_layout()
        st.pyplot(fig)

with r4:
    if {"loan_amount", "marital_status"} <= set(raw_df.columns):
        fig, ax = plt.subplots(figsize=(6, 3))
        top_cat = raw_df["marital_status"].value_counts().head(5).index
        subset = raw_df[raw_df["marital_status"].isin(top_cat)]
        subset.boxplot(
            column="loan_amount",
            by="marital_status",
            ax=ax,
            grid=False,
            patch_artist=True,
            boxprops=dict(facecolor=GREEN, alpha=0.5),
        )
        plt.suptitle("")
        ax.set_title("Loan Amount by Marital Status")
        ax.set_xlabel("marital_status")
        ax.set_ylabel("loan_amount")
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=20,
            ha="right",
            fontsize=8,
        )
        fig.tight_layout()
        st.pyplot(fig)

r5, r6 = st.columns(2)
with r5:
    if {"loan_amount", "loan_purpose"} <= set(raw_df.columns):
        fig, ax = plt.subplots(figsize=(7, 3))
        top_purpose = raw_df["loan_purpose"].value_counts().head(5).index
        subset = raw_df[raw_df["loan_purpose"].isin(top_purpose)]
        subset.boxplot(
            column="loan_amount",
            by="loan_purpose",
            ax=ax,
            grid=False,
            patch_artist=True,
            boxprops=dict(facecolor=ORANGE, alpha=0.5),
        )
        plt.suptitle("")
        ax.set_title("Loan Amount by Loan Purpose (top 5)")
        ax.set_xlabel("loan_purpose")
        ax.set_ylabel("loan_amount")
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=20,
            ha="right",
            fontsize=8,
        )
        fig.tight_layout()
        st.pyplot(fig)

with r6:
    if {"dpd", "address_provinsi"} <= set(raw_df.columns):
        tmp = raw_df.copy()
        tmp["is_late"] = (tmp["dpd"] > 0).astype(int)
        prov = (
            tmp.groupby("address_provinsi")["is_late"]
            .mean()
            .sort_values()
            .tail(10)
        )
        plot_bar(
            prov,
            "Late Payment Rate by Province (top 10)",
            "Province",
            "Rate",
            horizontal=True,
            color=ORANGE,
        )

st.markdown("---")

# 2.5 Correlation heatmap
st.subheader("2.5 Correlation heatmap")

numcols = raw_df.select_dtypes(include=["number"]).columns
if len(numcols) > 1:
    corr = raw_df[numcols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = cm.get_cmap("RdYlGn_r")  # orange–green-like gradient
    im = ax.imshow(corr, cmap=cmap, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation heatmap (numeric features)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.info("Not enough numeric columns to compute correlation matrix.")


# 3. Run Prediction
st.header("3. Run credit default prediction")

if not os.path.exists(cd.MODEL_FILENAME):
    st.error(
        f"Model file '{cd.MODEL_FILENAME}' was not found. "
        "Please make sure the .pkl is in the same directory as this app."
    )
    st.stop()

run_button = st.button("Run model on uploaded data")

if not run_button:
    st.stop()

with st.spinner("Running model and generating predictions..."):
    model = cd.load_model_silent(cd.MODEL_FILENAME)
    pred_df = cd.predict_raw_data(model, raw_df, threshold=threshold)

st.subheader("Prediction output (head)")
st.dataframe(pred_df.head(), use_container_width=True)
st.write(f"Total customers scored: {len(pred_df):,}")


# 4. Prediction distribution charts
st.subheader("Prediction distribution")

counts = pred_df["pred_default"].value_counts().sort_index()
labels = ["Non-default (0)", "Default (1)"]
non_def = counts.get(0, 0)
def_ = counts.get(1, 0)
values = [non_def, def_]
total = non_def + def_
percentages = [v / total * 100 if total > 0 else 0 for v in values]

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values, color=PALETTE)
    ax.set_ylabel("Number of customers")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(
        percentages,
        labels=labels,
        colors=PALETTE,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.axis("equal")
    fig.tight_layout()
    st.pyplot(fig)


# 5. Feature importance
if show_feature_importance:
    st.subheader("Feature importance from the model")

    fi_df = cd.get_feature_importance(model, cd.SELECTED_FEATURES)
    if fi_df is not None:
        st.dataframe(fi_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(fi_df["feature"], fi_df["importance"], color=ORANGE)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model type.")


# 6. Download predictions
st.subheader("Download predictions")

final_name = default_output_name.strip()
if not final_name.lower().endswith(".csv"):
    final_name += ".csv"

csv_buf = io.StringIO()
pred_df.to_csv(csv_buf, index=False)

st.download_button(
    label=f"Download predictions as '{final_name}'",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name=final_name,
    mime="text/csv",
)

st.success("Predictions are ready. Explore above or download as CSV.")
