"""
Phase 10/11 - chronological split feasibility check + baseline classifiers
(Logistic Regression, Random Forest, XGBoost), per commodity.

LEAKAGE RULES ENFORCED HERE:
- No shuffling anywhere - train/test split is a strict chronological cut.
- Scalers are fit on TRAIN ONLY, applied to test via transform().
- Feature columns explicitly EXCLUDE next_price/next_date/
  pct_change_to_next/days_to_next_report/spike - Phase 9's label-
  construction columns, contain t+1 information by definition.
- scale_pos_weight (XGBoost's imbalance handling) is computed from
  TRAIN y only, per commodity - never from test or the full dataset.

Output: models/*.pkl, models/*_scaler.pkl, models/training_metadata.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from config import PROJECT_ROOT

INPUT_PATH = (PROJECT_ROOT / "data/interim/price_sentiment_with_target.csv")
MODELS_DIR = (PROJECT_ROOT / "models")

MIN_TEST_SPIKES_FOR_STABLE_METRICS = 5

NON_FEATURE_COLS = {
    "commodity", "date", "dominant_sentiment",
    "next_price", "next_date", "pct_change_to_next", "days_to_next_report",
    "spike",
}

# Every factory takes y_train, even when it doesn't need it - keeps the
# training loop identical for all three models rather than special-casing
# XGBoost. scale_pos_weight is computed FROM y_train, per commodity -
# same imbalance-handling role as class_weight="balanced" for the other two,
# just XGBoost's API wants a number instead of a string.
MODELS = [
    ("Logistic Regression",
     lambda y_train: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
     True),
    ("Random Forest",
     lambda y_train: RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42),
     False),
    ("XGBoost",
     lambda y_train: XGBClassifier(
         n_estimators=200,
         scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
         eval_metric="logloss",
         random_state=42,
     ),
     False),
]


# ---------- split feasibility ----------

def evaluate_split(dates: pd.Series, spikes: pd.Series, train_frac: float) -> dict:
    n = len(dates)
    split_idx = int(n * train_frac)
    return {
        "train_frac": train_frac, "split_idx": split_idx,
        "train_rows": split_idx, "test_rows": n - split_idx,
        "train_spikes": int(spikes.iloc[:split_idx].sum()),
        "test_spikes": int(spikes.iloc[split_idx:].sum()),
        "train_start": dates.iloc[0], "train_end": dates.iloc[split_idx - 1],
        "test_start": dates.iloc[split_idx], "test_end": dates.iloc[-1],
    }


def choose_split(dates: pd.Series, spikes: pd.Series):
    r80 = evaluate_split(dates, spikes, 0.8)
    r70 = evaluate_split(dates, spikes, 0.7)

    if r80["test_spikes"] >= MIN_TEST_SPIKES_FOR_STABLE_METRICS:
        return r80, "80/20", (f"80/20 already yields {r80['test_spikes']} test spikes "
                               f"(>= threshold of {MIN_TEST_SPIKES_FOR_STABLE_METRICS}).")
    if r70["test_spikes"] > r80["test_spikes"]:
        return r70, "70/30", (f"80/20 only yields {r80['test_spikes']} test spikes; 70/30 improves this to "
                               f"{r70['test_spikes']} - still below threshold, but meaningfully better.")
    return r80, "80/20", (f"Neither split reaches {MIN_TEST_SPIKES_FOR_STABLE_METRICS} test spikes "
                           f"(80/20: {r80['test_spikes']}, 70/30: {r70['test_spikes']}) - defaulting to 80/20 "
                           f"to preserve training data. Metrics for this commodity are UNSTABLE regardless.")


def print_feasibility_report(commodity: str, dates: pd.Series, spikes: pd.Series):
    spike_dates = dates[spikes == 1]
    print(f"\n{'='*70}\n{commodity} - CHRONOLOGICAL SPLIT FEASIBILITY CHECK\n{'='*70}")
    print(f"Total valid rows: {len(dates)} | Total spikes: {int(spikes.sum())}")
    print(f"First spike: {spike_dates.min() if len(spike_dates) else 'N/A'} | "
          f"Last spike: {spike_dates.max() if len(spike_dates) else 'N/A'}")
    print("Spikes by year:")
    print(dates[spikes == 1].dt.year.value_counts().sort_index().to_string())

    chosen, label, reason = choose_split(dates, spikes)
    r80, r70 = evaluate_split(dates, spikes, 0.8), evaluate_split(dates, spikes, 0.7)
    for name, r in [("80/20", r80), ("70/30", r70)]:
        print(f"{name} -> train {r['train_start'].date()}-{r['train_end'].date()} "
              f"({r['train_rows']} rows, {r['train_spikes']} spikes) | "
              f"test {r['test_start'].date()}-{r['test_end'].date()} "
              f"({r['test_rows']} rows, {r['test_spikes']} spikes)")

    print(f"\n>>> CHOSEN SPLIT: {label}  |  REASON: {reason}")
    if chosen["test_spikes"] <= 3:
        print(f"*** WARNING: only {chosen['test_spikes']} positive example(s) in test set. "
              f"Precision/Recall/F1 are HIGHLY UNSTABLE - report with this caveat, not as standalone figures. ***")
    return chosen, label, reason


# ---------- model training / evaluation ----------

def evaluate_model(model, X_test, y_test, model_name: str, commodity: str) -> dict:
    y_pred = model.predict(X_test)
    n_test_pos = int(y_test.sum())

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if y_test.nunique() < 2:
        roc_auc, roc_note = None, "UNDEFINED - test set contains only one class."
    else:
        roc_auc, roc_note = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), None

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"\n--- {commodity} | {model_name} ---")
    print(f"Test set: {len(y_test)} rows, {n_test_pos} positive"
          + (" *** UNSTABLE - see warning above ***" if n_test_pos <= 3 else ""))
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | "
          f"ROC-AUC: {roc_auc:.4f}" if roc_auc is not None else f"ROC-AUC: {roc_note}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "commodity": commodity, "model": model_name,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": roc_auc, "roc_auc_note": roc_note,
        "n_test": len(y_test), "n_test_positive": n_test_pos,
        "confusion_matrix": cm.tolist(),
    }


def train_one_model(name, build_model, needs_scaling, X_train, X_test, y_train, y_test, commodity):
    if needs_scaling:
        scaler = StandardScaler().fit(X_train)
        X_train_input, X_test_input = scaler.transform(X_train), scaler.transform(X_test)
        joblib.dump(scaler, MODELS_DIR / f"{commodity.lower()}_{name.lower().replace(' ', '_')}_scaler.pkl")
    else:
        X_train_input, X_test_input = X_train, X_test

    model = build_model(y_train)   # <- y_train now passed to every factory
    model.fit(X_train_input, y_train)
    metrics = evaluate_model(model, X_test_input, y_test, name, commodity)

    joblib.dump(model, MODELS_DIR / f"{commodity.lower()}_{name.lower().replace(' ', '_')}.pkl")
    return metrics


# ---------- main ----------

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values(["commodity", "date"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"Feature count: {len(feature_cols)}\nFeatures: {feature_cols}")

    all_metrics, all_metadata = [], {}

    for commodity in df["commodity"].unique():
        sub = df[df["commodity"] == commodity].sort_values("date").reset_index(drop=True)
        sub_model = sub.dropna(subset=feature_cols + ["spike"]).reset_index(drop=True)
        print(f"\n{commodity}: {len(sub)} -> {len(sub_model)} rows after dropping warm-up/undated rows")

        dates, spikes = sub_model["date"], sub_model["spike"].astype(int)
        chosen, split_label, reason = print_feasibility_report(commodity, dates, spikes)

        split_idx = chosen["split_idx"]
        X, y = sub_model[feature_cols], spikes
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        meta = {
            "chosen_split": split_label, "split_reason": reason,
            "train_date_range": [str(chosen["train_start"].date()), str(chosen["train_end"].date())],
            "test_date_range": [str(chosen["test_start"].date()), str(chosen["test_end"].date())],
            "train_rows": len(X_train), "test_rows": len(X_test),
            "train_positive": int(y_train.sum()), "test_positive": int(y_test.sum()),
            "feature_count": len(feature_cols), "features": feature_cols, "models": {},
        }

        if y_train.sum() == 0:
            print(f"*** {commodity}: zero positive examples in training set - skipping model training. ***")
            all_metadata[commodity] = meta
            continue

        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        print(f"XGBoost scale_pos_weight for {commodity}: {pos_weight:.2f} "
              f"({(y_train == 0).sum()} negative / {(y_train == 1).sum()} positive in train)")

        for name, build_model, needs_scaling in MODELS:
            metrics = train_one_model(name, build_model, needs_scaling, X_train, X_test, y_train, y_test, commodity)
            all_metrics.append(metrics)
            meta["models"][name.lower().replace(" ", "_")] = metrics

        all_metadata[commodity] = meta

    with open(MODELS_DIR / "training_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2, default=str)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(pd.DataFrame(all_metrics)[
        ["commodity", "model", "n_test", "n_test_positive", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ].to_string(index=False))
    print(f"\nModels and metadata saved to {MODELS_DIR}")