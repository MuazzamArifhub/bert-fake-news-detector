import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .text_clean import clean_text
from .config import DATA_DIR, TEST_SIZE, SEED

def load_kaggle(fake_path, real_path):
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    def guess_text_col(df):
        for cand in ["text","content","article","body","title"]:
            if cand in df.columns:
                return cand
        for c in df.columns:
            if df[c].dtype == "object":
                return c
        return df.columns[0]

    fake = fake.rename(columns={guess_text_col(fake): "content"})
    real = real.rename(columns={guess_text_col(real): "content"})
    fake["label"] = 1  # fake
    real["label"] = 0  # real

    df = pd.concat([fake[["content","label"]], real[["content","label"]]], ignore_index=True)
    df["content"] = df["content"].astype(str).map(clean_text)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df

def split_save(df):
    os.makedirs(f"{DATA_DIR}/interim", exist_ok=True)
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"])
    train.to_csv(f"{DATA_DIR}/interim/train.csv", index=False)
    test.to_csv(f"{DATA_DIR}/interim/test.csv", index=False)
