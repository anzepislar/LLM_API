# 🧠 LLM_API — Lightweight Text Intelligence API

A Flask-based microservice providing text summarization, sentiment analysis, entity extraction, zero-shot topic classification, and keyword extraction using HuggingFace Transformers and KeyBERT.

## 🚀 Features
- Summarization (`/summarize`)
- Sentiment Analysis (`/sentiment`)
- Named Entity Recognition (`/ner`)
- Topic Classification (`/classify`)
- Keyword Extraction (`/keywords`)
- Production-ready JSON responses
- Clean modular architecture

## 📦 Installation

```bash
git clone https://github.com/<your-username>/LLM_API.git
cd LLM_API
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
