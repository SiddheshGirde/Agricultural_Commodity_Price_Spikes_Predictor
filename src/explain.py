"""
Phase 13 - SHAP explainability for the winning model (Random Forest)
on the two commodities with a viable model: Onion and Soyabean.
Wheat is excluded - all three Phase 11 baselines predicted zero spikes
on it, so there is nothing meaningful to explain.

Reuses the EXACT split/feature logic from train.py (imported, not
re-implemented) so the explained test set is guaranteed identical to
what was actually evaluated in Phase 10/11.

Explains predictions on the TEST set specifically - this reflects how
the deployed model reasons about genuinely unseen data, matching what
the Phase 14 dashboard will actually show users, rather than explaining
what the model memorized from training data.

Output: reports/shap/{commodity}_shap_bar.png, printed importance tables
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from train import INPUT_PATH, MODELS_DIR, NON_FEATURE_COLS, choose_split
from config import PROJECT_ROOT

REPORTS_DIR = (PROJECT_ROOT / "reports/shap")
COMMODITIES_TO_EXPLAIN = ["Onion", "Soyabean"]  # Wheat deliberately excluded - see module docstring


def get_class1_shap_values(explainer, X):
    """Handle both SHAP API shapes across versions: older versions return
    a list [class0_array, class1_array]; newer versions return a single
    (n_samples, n_features, n_classes) array. Either way, we want class 1
    (spike=1) specifically."""
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return raw[1]
    if raw.ndim == 3:
        return raw[:, :, 1]
    return raw


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values(["commodity", "date"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    for commodity in COMMODITIES_TO_EXPLAIN:
        print(f"\n{'='*70}\n{commodity} - SHAP EXPLAINABILITY (Random Forest)\n{'='*70}")

        sub = df[df["commodity"] == commodity].sort_values("date").reset_index(drop=True)
        sub_model = sub.dropna(subset=feature_cols + ["spike"]).reset_index(drop=True)

        dates, spikes = sub_model["date"], sub_model["spike"].astype(int)
        chosen, split_label, reason = choose_split(dates, spikes)
        print(f"Using {split_label} split (same as Phase 10/11): "
              f"test = {chosen['test_start'].date()} to {chosen['test_end'].date()}, "
              f"{chosen['test_spikes']} positive")

        split_idx = chosen["split_idx"]
        X_test = sub_model[feature_cols].iloc[split_idx:].reset_index(drop=True)

        model_path = MODELS_DIR / f"{commodity.lower()}_random_forest.pkl"
        model = joblib.load(model_path)
        print(f"Loaded {model_path.name}")

        explainer = shap.TreeExplainer(model)
        shap_values = get_class1_shap_values(explainer, X_test)

        mean_abs_importance = pd.Series(
            np.abs(shap_values).mean(axis=0), index=feature_cols
        ).sort_values(ascending=False)

        mean_signed_direction = pd.Series(
            shap_values.mean(axis=0), index=feature_cols
        )

        print("\nTop 10 features by mean |SHAP value| (overall importance):")
        for feat in mean_abs_importance.head(10).index:
            direction = "pushes toward SPIKE on average" if mean_signed_direction[feat] > 0 else "pushes toward NO SPIKE on average"
            print(f"  {feat:30s} importance={mean_abs_importance[feat]:.4f}  ({direction}, mean signed SHAP={mean_signed_direction[feat]:+.4f})")

        # Bar plot - saved to file, not shown interactively (no GUI assumed in this environment)
        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f"{commodity} - SHAP Feature Importance (Random Forest)")
        plt.tight_layout()
        out_path = REPORTS_DIR / f"{commodity.lower()}_shap_bar.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\nSaved plot: {out_path}")