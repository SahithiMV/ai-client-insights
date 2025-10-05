from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
from src.ai.data_pipeline import load_and_clean_feedback
from src.ai.analyzer import analyze_sentiment, summarize_texts, summarize_texts_humanized
load_dotenv()
app = FastAPI(title="AI Client Insight API", version="0.1.0")


# ---------- Models ----------
class Health(BaseModel):
    status: str = "ok"


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    label: str
    score: float


# ---------- Routes ----------
@app.get("/", tags=["meta"])
def root():
    return {"message": "AI Client Insight API is running"}


@app.get("/health", response_model=Health, tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/feedback", tags=["data"])
def get_feedback():
    """Return cleaned feedback records from sample CSV."""
    try:
        df = load_and_clean_feedback()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_feedback", response_model=AnalyzeResponse, tags=["ai"])
def analyze_endpoint(req: AnalyzeRequest):
    """Sentiment on a single piece of text."""
    try:
        return analyze_sentiment(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary", tags=["ai"])
def summary_endpoint(humanize: bool = Query(False, description="Use OpenAI to rewrite summary if API key is set"),
                     max_sentences: int = 2,
                     max_words: int = 40):
    """Summarize all cleaned feedback; optionally humanize with GPT."""
    try:
        df = load_and_clean_feedback()
        texts = df["clean_feedback"].astype(str).tolist()
        if humanize:
            return {"summary": summarize_texts_humanized(texts, max_sentences=max_sentences, max_words=max_words)}
        else:
            return {"summary": summarize_texts(texts, max_sentences=max_sentences)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

