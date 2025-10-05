#  AI Client Insight Dashboard

A lightweight **FastAPI** application that demonstrates how to ingest messy customer feedback, clean it, analyze sentiment, and summarize insights — all within a single deployable backend.  
Built to showcase practical use of **FastAPI**, **Hugging Face**, **TF-IDF summarization**, and optional **OpenAI-powered rewording** in under 8 hours.

---

##  Quick Start

### Clone and Setup
```bash
git clone <your-repo-url>
cd ai-client-insights
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Run in Docker

### Build the image

make build

### Run the container
make serve

### Test API
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/feedback | jq
curl -s -X POST http://127.0.0.1:8000/analyze_feedback \
  -H "Content-Type: application/json" \
  -d '{"text": "delivery was slow but driver polite"}' | jq
curl -s http://127.0.0.1:8000/summary | jq

## API Endpoints
Endpoint	Method	Description
/	GET	Root endpoint — returns app info
/health	GET	Health check
/feedback	GET	Loads and cleans sample feedback from src/data/feedback_raw.csv
/analyze_feedback	POST	Sentiment analysis using Hugging Face (distilbert-base-uncased-finetuned-sst-2-english)
/summary	GET	Extractive summary using TF-IDF (fast, no model downloads)
/summary?humanize=1	GET	Rewrites the summary into natural English using GPT (if OPENAI_API_KEY is set)
