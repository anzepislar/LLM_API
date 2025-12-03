from flask import Flask, request, jsonify
from text_inteligence_pipeline import summarize, sentiment_analyze, ner, classify_topic, extract_keywords

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "LLM API running"})

@app.route("/summarize", methods=["POST"])
def summarize_endpoint():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing data"})
    
    text = data["text"]
    
    summary = summarize(text)

    return jsonify({"summary": summary}), 200

@app.route("/sentiment", methods=["POST"])
def sentiment_endpoint():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing data"})
    
    text = data["text"]
    
    response = sentiment_analyze(text)
    
    return jsonify({"sentiment": response}), 200

@app.route("/ner", methods=["POST"])
def ner_endpoint():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing data"})
    
    text = data["text"]
    
    response = ner(text)
    
    return jsonify({"NER Analysis": response}), 200

@app.route("/classify", methods=["POST"])
def classify_endpoint():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing data"})
    
    text = data["text"]
    candidate_labels = data.get("candidate_labels")
    
    if not text or not candidate_labels:
        return jsonify({"error": "Both 'text' and 'candidate_labels' are required"}), 400
    
    response = classify_topic(text, candidate_labels)
    
    return jsonify({"topic": response}), 200

@app.route("/keywords", methods=["POST"])
def keywords_endpoint():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing data"})
    
    text = data["text"]
    
    response = extract_keywords(text)
    
    return jsonify({"keywords": response}), 200

if __name__ == "__main__":
    app.run(debug=True)
