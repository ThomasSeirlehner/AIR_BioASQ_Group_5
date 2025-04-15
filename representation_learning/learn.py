import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

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
        batch_size=32  # cosine similarity works better with normalized vectors
    ).to(device)

    return embeddings, doc_urls

def predict_for_questions(test_json, model, doc_embeddings, doc_urls, output_path, device, top_k=10):
    with open(test_json, "r") as f:
        test_data = json.load(f)

    output = {"questions": []}
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

    print("Computing similarities and predicting...")
    similarities = torch.matmul(question_embeddings, doc_embeddings.T)  # [num_questions, num_docs]

    top_k_indices = torch.topk(similarities, k=top_k, dim=1).indices  # [num_questions, top_k]

    for i, q_id in enumerate(tqdm(question_ids, desc="Writing predictions")):
        top_docs = [doc_urls[idx] for idx in top_k_indices[i].tolist()]
        output["questions"].append({
            "id": q_id,
            "documents": top_docs
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    documents_json_path = "../documents.json"
    train_json_path = "../../training13b.json"
    test_json_path = "../../BioASQ-task13bPhaseA-testset1.json"
    output_json_path = "../../predictions_biolinkbert_fast.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading BioLinkBERT model...")

    model = SentenceTransformer("./fine_tuned_biobert")

    doc_texts = load_document_corpus(documents_json_path, train_json_path)
    doc_embeddings, doc_urls = build_index(doc_texts, model, device)
    predict_for_questions(test_json_path, model, doc_embeddings, doc_urls, output_json_path, device)
