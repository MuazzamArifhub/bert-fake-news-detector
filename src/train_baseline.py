import os, json, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from .config import DATA_DIR, MODEL_DIR

def main():
    train = pd.read_csv(f"{DATA_DIR}/interim/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/interim/test.csv")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])

    pipe.fit(train["content"], train["label"])
    preds = pipe.predict(test["content"])
    report = classification_report(test["label"], preds, output_dict=True)

    os.makedirs(f"{MODEL_DIR}/baseline", exist_ok=True)
    joblib.dump(pipe, f"{MODEL_DIR}/baseline/model.joblib")

    os.makedirs("reports", exist_ok=True)
    metrics_path = "reports/metrics.json"
    try:
        with open(metrics_path) as f: metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
    metrics["baseline"] = report
    with open(metrics_path,"w") as f: json.dump(metrics, f, indent=2)

    print("Saved baseline model and metrics.")

if __name__ == "__main__":
    main()
