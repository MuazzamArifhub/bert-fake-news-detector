from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="Fake News Detector")

def get_predict_fn():
    # Try BERT first
    if os.path.isdir("models/bert"):
        try:
            from src.inference import BertClassifier
            clf = BertClassifier()
            return lambda text: clf.predict(text)
        except Exception as e:
            print(f"[api] BERT load failed, falling back to baseline: {e}")

    # Fallback: baseline scikit-learn pipeline
    import joblib
    pipe = joblib.load("models/baseline/model.joblib")
    def predict_baseline(text: str):
        proba = float(pipe.predict_proba([text])[0, 1])
        return {"fake_prob": proba, "label": int(proba >= 0.5)}
    return predict_baseline

predict_fn = get_predict_fn()

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    return predict_fn(item.text)