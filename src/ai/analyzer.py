# src/ai/analyzer.py
import re
import os
from openai import OpenAI
from functools import lru_cache
from typing import List, Dict
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# ---- Sentiment (works already) ----
@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    return pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU
    )

def analyze_sentiment(text: str) -> Dict:
    res = get_sentiment_pipeline()(text)[0]
    return {"label": res["label"].lower(), "score": float(res["score"])}

# ---- Extractive summarizer (fast, no downloads) ----
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    return parts

def summarize_texts(texts: List[str], max_sentences: int = 2) -> str:
    """Summarize list of cleaned feedback into a concise paragraph."""
    if not texts:
        return "No feedback available."

    # Combine feedback into one text block
    doc = " ".join(texts)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc) if s.strip()]
    if not sentences:
        return "No feedback available."

    # Compute TF-IDF across all sentences
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()

    # Select top N informative sentences
    top_idx = np.argsort(-scores)[:max_sentences]
    top_sorted = sorted(top_idx.tolist())
    summary_sentences = [sentences[i] for i in top_sorted]

    # Compactify repetitive or filler words
    summary = " ".join(summary_sentences)
    summary = re.sub(r"\s+", " ", summary)
    summary = summary.capitalize()

    # Heuristic rewrite to merge small sentences
    if len(summary.split()) < 25:
        summary = (
            summary
            .replace(" .", ".")
            .replace(" but ", ", but ")
            .replace(" and ", ", and ")
        )

    return summary
def humanize_summary_llm(summary: str, max_words: int = 40) -> str:
    """
    If OPENAI_API_KEY is set, rewrite the summary to be more natural and concise.
    Falls back to the original summary on any error.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not summary.strip():
        return summary  # no-op

    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "Rewrite the following feedback summary into 1–2 natural, concise sentences. "
            "Keep the meaning the same, do not add new facts, and aim for a positive, neutral tone. "
            f"Limit to ~{max_words} words.\n\nSummary:\n{summary}"
        )
        # Lightweight, low-cost model is fine here; change if you prefer.
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a concise product analyst."},
                {"role": "user", "content": prompt},
            ],
            timeout=10,
        )
        text = resp.choices[0].message.content.strip()
        # Tiny cleanup
        return text.replace("  ", " ").strip()
    except Exception:
        return summary

def summarize_texts_humanized(texts: list[str], max_sentences: int = 2, max_words: int = 40) -> str:
    base = summarize_texts(texts, max_sentences=max_sentences)
    return humanize_summary_llm(base, max_words=max_words)

