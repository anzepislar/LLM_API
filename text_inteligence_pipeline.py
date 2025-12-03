# -*- coding: utf-8 -*-
## Summarize text
from transformers import pipeline
from keybert import KeyBERT

model = "sshleifer/distilbart-cnn-6-6"
kw_model = KeyBERT()

summarizer = pipeline("summarization", model=model)
sentiment_pipe = pipeline("sentiment-analysis")
ner_pipe = pipeline("ner")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def summarize(text: str):
    summary = summarizer(text, min_length=50, max_length=100, do_sample=False, truncation=True)

    return summary[0]["summary_text"]

"""## Sentiment Analyzer"""

def sentiment_analyze(text: str):
    return sentiment_pipe(text)

"""## Name Entity Recognision"""

def ner(text: str):
    return ner_pipe(text)


"""## Topic Classification"""

def classify_topic(text: str, candidate_labels: list[str]):
    result = classifier(
        text,
        candidate_labels=candidate_labels,
        multi_label=False
    )
    # Just return label + score for simplicity
    return {
        "label": result["labels"][0],
        "score": result["scores"][0],
        "all": result,
    }

"""## Keyword Extraction"""
def extract_keywords(text: str):

    keywords = kw_model.extract_keywords(text)
    return [{"keyword": kw, "score": float(score)} for kw, score in keywords]

def full_text_analysis(text: str):
    """
    Example combined pipeline: summarize -> keywords -> sentiment.
    You can customize this however you want.
    """
    summary = summarize(text)
    keywords = extract_keywords(summary)
    sentiment = sentiment_analyze(summary)

    return {
        "summary": summary,
        "keywords": keywords,
        "sentiment": sentiment,
    }


# ---------- Local test ----------

# if __name__ == "__main__":
#     demo_text = (
#         "Over the past year, major technology companies such as Google, Meta, "
#         "and OpenAI have shifted their focus from building the largest possible AI models..."
#     )
#     print(full_text_analysis(demo_text))
