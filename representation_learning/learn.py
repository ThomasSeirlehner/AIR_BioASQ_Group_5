import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

def load_document_corpus(documents_json, train_json):
    """Load document texts (title + abstract) from documents.json for the training data."""
    with open(documents_json, "r") as f:
        documents_data = json.load(f)
    
    # Build lookup dictionary: {url: title + abstract}
    document_lookup = {doc["url"]: f"{doc['title']} {doc['abstract']}".strip() for doc in documents_data}

    # Collect relevant documents from training dataset
    with open(train_json, "r") as f:
        train_data = json.load(f)

    documents = {}
    for q in tqdm(train_data["questions"], desc="Collecting document texts"):
        for doc_url in q.get("documents", []):
            if doc_url not in documents and doc_url in document_lookup:
                documents[doc_url] = document_lookup[doc_url]

    return documents

def build_index(documents):
    """Create TF-IDF index from document text."""
    doc_urls = list(documents.keys())
    doc_texts = list(documents.values())
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    
    return vectorizer, tfidf_matrix, doc_urls

def recommend_documents(question_text, vectorizer, tfidf_matrix, doc_urls, top_k=10):
    """Given a question, recommend top_k documents based on TF-IDF cosine similarity."""
    question_vec = vectorizer.transform([question_text])
    sims = cosine_similarity(question_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    return [doc_urls[i] for i in top_indices]

def predict_for_questions(test_json, vectorizer, tfidf_matrix, doc_urls, output_path):
    """Generate prediction file with recommended documents."""
    with open(test_json, "r") as f:
        test_data = json.load(f)

    output = {"questions": []}

    for q in tqdm(test_data["questions"], desc="Predicting"):
        recommended_docs = recommend_documents(
            q["body"], vectorizer, tfidf_matrix, doc_urls
        )
        output["questions"].append({
            "id": q["id"],
            "documents": recommended_docs
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Predictions saved to {output_path}")

# ---------- Script usage ------------

if __name__ == "__main__":
    # Paths to input/output files
    documents_json_path = "../documents.json"  # Your collected documents
    train_json_path = "../../training13b.json"
    test_json_path = "../../BioASQ-task13bPhaseA-testset1.json"
    output_json_path = "../../predictions.json"

    # Step 1: Prepare corpus from training data using pre-fetched documents
    doc_texts = load_document_corpus(documents_json_path, train_json_path)

    # Step 2: Build TF-IDF index
    vectorizer, tfidf_matrix, doc_urls = build_index(doc_texts)

    # Step 3: Predict documents for test questions
    predict_for_questions(test_json_path, vectorizer, tfidf_matrix, doc_urls, output_json_path)
