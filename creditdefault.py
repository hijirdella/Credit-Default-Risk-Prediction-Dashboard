# ===============================================================
# creditdefault.py
# Simple credit default prediction for business users (clean display)
#
# Jupyter usage:
#   import importlib, creditdefault as cd
#   importlib.reload(cd)
#   cd.run_credit_default()
#
# Minimal coding for user.
# ===============================================================

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from contextlib import redirect_stdout, redirect_stderr
import io
import os
import sys
import matplotlib.pyplot as plt
import warnings

# Suppress noisy loky warning (CPU count) and limit core fallback
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

MODEL_FILENAME = "best_credit_default_model.pkl"
SELECTED_FEATURES = [
    "early_payment_ratio",
    "ontime_payment_ratio",
    "late_payment_ratio",
    "total_on_time",
    "repay_installment_ratio",
]

# Display setup for Jupyter
try:
    from IPython.display import display as ipy_display

    def show(df):
        ipy_display(df)
except Exception:
    def show(df):
        print(df)


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def load_model_silent(model_path: str):
    """Load model silently."""
    f_out, f_err = io.StringIO(), io.StringIO()
    with redirect_stdout(f_out), redirect_stderr(f_err):
        model = joblib.load(model_path)
    return model


def get_feature_importance(model, feature_names):
    """Extract feature importance or coefficients."""
    est = model
    if hasattr(model, "steps"):
        est = model.steps[-1][1]

    importances = None
    if hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
    elif hasattr(est, "coef_"):
        coef = est.coef_
        if coef.ndim == 2:
            coef = coef[0]
        importances = np.abs(coef)
    else:
        return None

    if importances is None or len(importances) != len(feature_names):
        return None

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------
def build_customer_features_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df["dpd"] = pd.to_numeric(df.get("dpd"), errors="coerce")
    df["paid_amount"] = pd.to_numeric(df.get("paid_amount"), errors="coerce")
    df["installment_amount"] = pd.to_numeric(df.get("installment_amount"), errors="coerce")
    df["loan_amount"] = pd.to_numeric(df.get("loan_amount"), errors="coerce")
    df["due_date"] = pd.to_datetime(df.get("due_date"), errors="coerce")

    # NA-safe flags (fillna(False) before astype(int))
    df["is_early"] = (df["dpd"] < 0).fillna(False).astype(int)
    df["is_on_time"] = (df["dpd"] == 0).fillna(False).astype(int)
    df["is_late"] = (df["dpd"] > 0).fillna(False).astype(int)
    df["is_dpd30"] = (df["dpd"] >= 30).fillna(False).astype(int)
    df["is_dpd60"] = (df["dpd"] >= 60).fillna(False).astype(int)
    df["is_dpd90"] = (df["dpd"] >= 90).fillna(False).astype(int)

    loan_level = (
        df.groupby(["customer_id", "loan_id"], dropna=False)
        .agg(
            loan_amount=("loan_amount", "max"),
            n_instalments=("payment_id", "count"),
            n_paid_instalments=("paid_amount", lambda s: (s > 0).sum()),
            total_billed_amount=("installment_amount", "sum"),
            total_paid_amount=("paid_amount", lambda s: s.fillna(0).sum()),
            worst_dpd_loan=("dpd", "max"),
            avg_dpd_loan=("dpd", "mean"),
            n_early=("is_early", "sum"),
            n_on_time=("is_on_time", "sum"),
            n_late=("is_late", "sum"),
            n_dpd30=("is_dpd30", "sum"),
            n_dpd60=("is_dpd60", "sum"),
            n_dpd90=("is_dpd90", "sum"),
            first_due=("due_date", "min"),
            last_due=("due_date", "max"),
        )
        .reset_index()
    )

    ca = (
        loan_level.groupby("customer_id", dropna=False)
        .agg(
            n_loans=("loan_id", "nunique"),
            total_instalments=("n_instalments", "sum"),
            total_paid_instalments=("n_paid_instalments", "sum"),
            total_principal=("loan_amount", "sum"),
            total_billed_amount=("total_billed_amount", "sum"),
            total_paid_amount=("total_paid_amount", "sum"),
            total_early=("n_early", "sum"),
            total_on_time=("n_on_time", "sum"),
            total_late=("n_late", "sum"),
            worst_dpd=("worst_dpd_loan", "max"),
            avg_dpd=("avg_dpd_loan", "mean"),
            total_dpd30=("n_dpd30", "sum"),
            total_dpd60=("n_dpd60", "sum"),
            total_dpd90=("n_dpd90", "sum"),
            first_due_overall=("first_due", "min"),
            last_due_overall=("last_due", "max"),
        )
        .reset_index()
    )

    total_inst = ca["total_instalments"].replace(0, np.nan)
    total_paid_inst = ca["total_paid_instalments"].replace(0, np.nan)
    total_billed_amt = ca["total_billed_amount"].replace(0, np.nan)
    total_paid_amt = ca["total_paid_amount"].replace(0, np.nan)
    total_principal = ca["total_principal"].replace(0, np.nan)

    features = pd.DataFrame(
        {
            "customer_id": ca["customer_id"],
            "n_loans": ca["n_loans"],
            "total_instalments": ca["total_instalments"],
            "total_paid_instalments": ca["total_paid_instalments"],
            "total_principal": ca["total_principal"],
            "avg_instalments_per_loan": ca["total_instalments"]
            / ca["n_loans"].replace(0, np.nan),
            "active_span_days": (
                ca["last_due_overall"] - ca["first_due_overall"]
            ).dt.days,
            "repay_installment_ratio": ca["total_paid_instalments"] / total_inst,
            "repay_amount_ratio": total_paid_amt / total_billed_amt,
            "principal_repayment_ratio": total_paid_amt / total_principal,
            "total_early": ca["total_early"],
            "total_on_time": ca["total_on_time"],
            "total_late": ca["total_late"],
            "early_payment_ratio": ca["total_early"] / total_paid_inst,
            "ontime_payment_ratio": ca["total_on_time"] / total_paid_inst,
            "late_payment_ratio": ca["total_late"] / total_paid_inst,
            "worst_dpd": ca["worst_dpd"],
            "avg_dpd": ca["avg_dpd"],
            "share_instalments_dpd30_plus": ca["total_dpd30"] / total_inst,
            "share_instalments_dpd60_plus": ca["total_dpd60"] / total_inst,
            "share_instalments_dpd90_plus": ca["total_dpd90"] / total_inst,
        }
    )

    # Display customer_id cleanly (no .000000)
    if pd.api.types.is_float_dtype(features["customer_id"]):
        features["customer_id"] = (
            features["customer_id"].astype("Int64").astype(str)
        )

    return features


# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
def predict_raw_data(model, raw_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    features = build_customer_features_from_raw(raw_df)
    X = features.set_index("customer_id")[SELECTED_FEATURES]

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X), columns=SELECTED_FEATURES, index=X.index
    )

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_imp)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X_imp)
        probs = (s - s.min()) / (s.max() - s.min() + 1e-9)
    else:
        probs = model.predict(X_imp)

    preds = (probs >= threshold).astype(int)
    features = features.set_index("customer_id")
    features["prob_default"] = probs
    features["pred_default"] = preds

    return features.reset_index()


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------
def run_credit_default(
    csv_path: str | None = None,
    output_csv: str | None = None,
    threshold: float = 0.5,
):
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found.")

    # Ask for input CSV if not provided
    if csv_path is None:
        csv_path = input("Enter input CSV filename (default: combined_df.csv): ").strip()
        if not csv_path:
            csv_path = "combined_df.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load and format display
    raw_df = pd.read_csv(csv_path)

    for col in ["customer_id", "application_id", "loan_id", "payment_id"]:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").astype("Int64").astype(str)
    if "dpd" in raw_df.columns:
        raw_df["dpd"] = pd.to_numeric(raw_df["dpd"], errors="coerce")

    print("Input data (head):")
    show(raw_df.head())

    model = load_model_silent(MODEL_FILENAME)
    result = predict_raw_data(model, raw_df, threshold=threshold)

    print("\nPrediction output (head):")
    show(result.head())

    # Charts
    counts = result["pred_default"].value_counts().sort_index()
    labels = ["Non-default (0)", "Default (1)"]
    non_def, def_ = counts.get(0, 0), counts.get(1, 0)
    total = non_def + def_
    percentages = [v / total * 100 if total > 0 else 0 for v in [non_def, def_]]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(labels, [non_def, def_], color=["green", "orange"])
    ax[0].set_title("Prediction Counts by Class")
    ax[0].set_ylabel("Number of Customers")

    ax[1].pie(
        percentages,
        labels=labels,
        colors=["green", "orange"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax[1].set_title("Prediction Percentages by Class")

    plt.tight_layout()
    plt.show()

    # Ask for output CSV name
    if output_csv is None:
        output_csv = input(
            "Enter output CSV filename (default: credit_default_predictions.csv): "
        ).strip()
        if not output_csv:
            output_csv = "credit_default_predictions.csv"

    if not output_csv.lower().endswith(".csv"):
        output_csv += ".csv"

    result.to_csv(output_csv, index=False)
    print(f"\nPredictions saved to: {output_csv}")

    fi_df = get_feature_importance(model, SELECTED_FEATURES)
    if fi_df is not None:
        fi_df.to_csv("feature_importance.csv", index=False)
        print("Feature importance (top rows):")
        show(fi_df.head())
        print("Feature importance saved to: feature_importance.csv")
    else:
        print("Feature importance not available.")


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) >= 2 else None
    out_arg = sys.argv[2] if len(sys.argv) >= 3 else None
    run_credit_default(csv_path=csv_arg, output_csv=out_arg)
