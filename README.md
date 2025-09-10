# Fake News Detector

This project detects fake news using two approaches:  
1. A baseline model (TF-IDF + Logistic Regression with scikit-learn)  
2. An optional transformer model (BERT, fine-tuned with Hugging Face Transformers)  

The project includes data preprocessing, model training, evaluation, and a REST API built with FastAPI.

---

## Features
- Load local CSVs (`Fake.csv`, `True.csv`) from `data/raw/`
- Preprocess and clean text automatically
- Train/test split saved in `data/interim/`
- Baseline model: TF-IDF + Logistic Regression
- Evaluation with confusion matrix and precision-recall curve
- FastAPI service with `/predict` endpoint for real-time classification
- Interactive API documentation available at `/docs`

---

## Data
Place two CSVs into `data/raw/`:

- **Fake.csv** → fake news articles (must include a `text` column)  
- **True.csv** → real news articles (must include a `text` column)  

Example (`Fake.csv`):
```csv
text
"Aliens invade Canada in broad daylight"
"Secret cure discovered overnight"
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare train/test splits
```bash
export PYTHONPATH=$PWD
python - <<'PY'
from src.load_data import load_data, split_save
df = load_data("data/raw/Fake.csv", "data/raw/True.csv")
split_save(df)
PY
```

### 3. Train the baseline model
```bash
PYTHONPATH=$PWD python -m src.train_baseline
```

### 4. Evaluate the model
```bash
PYTHONPATH=$PWD python -m src.eval
```

- Confusion matrix and precision-recall curve are saved in `reports/figures/`  
- Metrics are saved in `reports/metrics.json`

---

## Results (Baseline Example)
- Confusion Matrix and Precision-Recall Curve are generated during evaluation  
- Example Average Precision: **1.00** (on a toy dataset)

---

## Run API

### Start the server
```bash
export PYTHONPATH=$PWD
uvicorn api.app:app --reload --host 0.0.0.0 --port 8080
```

### Open the Swagger UI
In your browser:
```
https://<your-forwarded-url>/docs
```

### Example request with curl
```bash
curl -X POST "<your-forwarded-url>/predict"   -H "Content-Type: application/json"   -d '{"text":"Aliens invade Canada in broad daylight"}'
```

### Example response
```json
{"fake_prob": 0.92, "label": 1}
```

---

## Next Steps
- Fine-tune BERT with `src/train_bert.py` for higher accuracy  
- Deploy the API with Docker (`api/Dockerfile`) to platforms like Heroku, Render, or Hugging Face Spaces  
- Expand the dataset for more realistic performance  

---

## Author
**Muazzam Arif**  
GitHub: [@MuazzamArifhub](https://github.com/MuazzamArifhub)