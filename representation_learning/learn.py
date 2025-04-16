import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.tokenize import PunktSentenceTokenizer

nltk.download('punkt')

def load_document_corpus(documents_json):
    with open(documents_json, "r") as f:
        documents_data = json.load(f)
    return {doc["url"]: f"{doc['title']} {doc['abstract']}".strip() for doc in documents_data}

def build_index(documents, model, device):
    doc_urls = list(documents.keys())
    doc_texts = list(documents.values())

    print("Encoding document texts...")
    embeddings = model.encode(
        doc_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=True,
        batch_size=32
    ).to(device)

    return embeddings, doc_urls

def predict_and_extract(test_json, documents_json, model, doc_embeddings, doc_urls, output_path, device, top_k_docs=10, top_k_snippets=10):
    with open(test_json, "r") as f:
        test_data = json.load(f)

    doc_lookup = load_document_corpus(documents_json)

    question_texts = [q["body"] for q in test_data["questions"]]
    question_ids = [q["id"] for q in test_data["questions"]]

    print("Encoding questions...")
    question_embeddings = model.encode(
        question_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=True
    ).to(device)

    print("Computing similarities and extracting predictions/snippets...")
    similarities = torch.matmul(question_embeddings, doc_embeddings.T)  # [num_questions, num_docs]

    output = {"questions": []}
    tokenizer = PunktSentenceTokenizer()

    for i, (q_id, q_text) in enumerate(tqdm(zip(question_ids, question_texts), desc="Processing questions", total=len(question_ids))):
        q_embedding = question_embeddings[i]

        top_k_indices = torch.topk(similarities[i], k=top_k_docs).indices.tolist()
        top_docs = [doc_urls[idx] for idx in top_k_indices]

        snippets = []

        for doc_url in top_docs:
            if doc_url not in doc_lookup:
                continue

            doc_text = doc_lookup[doc_url]
            sentences = tokenizer.tokenize(doc_text)

            if not sentences:
                continue

            sent_embeddings = model.encode(sentences, convert_to_tensor=True, device=device)
            cosine_scores = util.cos_sim(q_embedding, sent_embeddings)[0]

            top_indices = torch.topk(cosine_scores, k=min(top_k_snippets, len(sentences))).indices.tolist()

            for idx in top_indices:
                score = cosine_scores[idx].item()
                snippet_text = sentences[idx]
                offset_start = doc_text.find(snippet_text)
                offset_end = offset_start + len(snippet_text)

                snippet = {
                    "document": doc_url,
                    "text": snippet_text,
                    "offsetInBeginSection": offset_start,
                    "offsetInEndSection": offset_end,
                    "beginSection": "sections.0",
                    "endSection": "sections.0"
                }
                snippets.append((score, snippet))

        # Keep top overall snippets across all retrieved documents
        snippets = sorted(snippets, key=lambda x: x[0], reverse=True)[:top_k_snippets]

        output["questions"].append({
            "id": q_id,
            "documents": top_docs,
            "snippets": [s[1] for s in snippets]
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Predictions and snippets saved to {output_path}")

if __name__ == "__main__":
    documents_json_path = "../documents.json"
    test_json_path = "../../BioASQ-TaskB-testData/BioASQ-TaskB-testData/phaseA_12b_04.json"
    output_json_path = "../../results_biobert_phaseA_12b_04.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading BioLinkBERT model...")
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    doc_texts = load_document_corpus(documents_json_path)
    doc_embeddings, doc_urls = build_index(doc_texts, model, device)

    predict_and_extract(
        test_json_path,
        documents_json_path,
        model,
        doc_embeddings,
        doc_urls,
        output_json_path,
        device
    )
