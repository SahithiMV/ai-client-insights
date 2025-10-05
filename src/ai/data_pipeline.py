import pandas as pd
import re
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data"

def clean_text(text: str) -> str:
    """Basic cleanup: lowercase, strip extra spaces, remove punctuation repetition"""
    text = text.lower().strip()
    text = re.sub(r"[!?.]{2,}", ".", text)       # collapse repeats
    text = re.sub(r"\s{2,}", " ", text)          # extra spaces
    return text

def load_and_clean_feedback(filename="feedback_raw.csv"):
    """Load raw CSV, clean feedback, and return DataFrame"""
    df = pd.read_csv(DATA_PATH / filename)
    df["clean_feedback"] = df["feedback"].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_and_clean_feedback()
    print(df.head())

