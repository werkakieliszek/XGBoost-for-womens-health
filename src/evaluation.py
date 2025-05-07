import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, classification_report, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt

# Set this to your latest run directory
run_dir = Path("model_artifacts/20250508_000742")  # update as needed

illnesses = ["trich", "bv", "ct", "gc"]

for illness in illnesses:
    print(f"\n{'='*30}\nEvaluating {illness.upper()}\n{'='*30}")
    illness_dir = run_dir / illness

    # Load artifacts
    model = joblib.load(illness_dir / "model.joblib")
    with open(illness_dir / "features.json") as f:
        features = json.load(f)
    test_df = pd.read_csv(illness_dir / "test_data.csv")

    X_test = test_df[features]
    y_test = test_df["true_label"]

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"MCC: {matthews_corrcoef(y_test, y_pred):.3f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{illness.upper()} (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {illness.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f"{illness.upper()}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {illness.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.show()
