import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from .config import DATA_DIR

def main():
    test = pd.read_csv(f"{DATA_DIR}/interim/test.csv")
    model = joblib.load("models/baseline/model.joblib")
    proba = model.predict_proba(test["content"])[:,1]

    os.makedirs("reports/figures", exist_ok=True)

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(model, test["content"], test["label"])
    plt.title("Baseline Confusion Matrix")
    plt.savefig("reports/figures/cm_baseline.png")
    plt.close()

    # Precision-recall curve
    p, r, _ = precision_recall_curve(test["label"], proba)
    ap = average_precision_score(test["label"], proba)
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (AP={ap:.3f})")
    plt.savefig("reports/figures/pr_baseline.png")
    plt.close()

    print(f"Average Precision: {ap:.3f}")

if __name__ == "__main__":
    main()
