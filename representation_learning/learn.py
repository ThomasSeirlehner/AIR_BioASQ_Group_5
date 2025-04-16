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

def load_document_corpus_snippeds(documents_json):
    with open(documents_json, "r") as f:
        documents_data = json.load(f)
    return {doc["url"]: {"title": doc["title"], "abstract": doc["abstract"]} for doc in documents_data}


def build_index(documents, model, device):
    doc_urls = list(documents.keys())
    doc_texts = list(documents.values())

    print("Encoding document texts (for retrieval)...")
    embeddings = model.encode(
        doc_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=True,
        batch_size=32
    ).to(device)

    return embeddings, doc_urls


def predict_documents(test_data, doc_retrieval_model, doc_embeddings, doc_urls, device, top_k_docs=10):
    question_texts = [q["body"] for q in test_data["questions"]]
    question_ids = [q["id"] for q in test_data["questions"]]

    print("Encoding questions (for retrieval)...")
    question_embeddings = doc_retrieval_model.encode(
        question_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=True
    ).to(device)

    print("Computing document similarities...")
    similarities = torch.matmul(question_embeddings, doc_embeddings.T)  # [num_questions, num_docs]

    predictions = []
    for i, q_id in enumerate(question_ids):
        top_k_indices = torch.topk(similarities[i], k=top_k_docs).indices.tolist()
        top_docs = [doc_urls[idx] for idx in top_k_indices]
        predictions.append({
            "id": q_id,
            "question": question_texts[i],
            "documents": top_docs
        })

    return predictions


def extract_snippets(predictions, documents_json, snippet_model, device, top_k_snippets=10):
    doc_lookup = load_document_corpus_snippeds(documents_json)
    tokenizer = PunktSentenceTokenizer()

    for prediction in tqdm(predictions, desc="Extracting snippets"):
        q_text = prediction["question"]
        q_id = prediction["id"]
        top_docs = prediction["documents"]

        snippets = []

        for doc_url in top_docs:
            if doc_url not in doc_lookup:
                continue

            doc_sections = doc_lookup[doc_url]

            section_sentences = {}
            for section_name, section_text in doc_sections.items():
                if section_text.strip():  # skip empty
                    sentences = tokenizer.tokenize(section_text)
                    section_sentences[section_name] = sentences

            candidate_snippets = []

            for section_name, sentences in section_sentences.items():
                window_sizes = [1, 2, 3]

                for window_size in window_sizes:
                    if len(sentences) < window_size:
                        continue
                    for j in range(len(sentences) - window_size + 1):
                        snippet_text = ' '.join(sentences[j:j + window_size])
                        offset_start = doc_sections[section_name].find(sentences[j])
                        offset_end = doc_sections[section_name].find(sentences[j + window_size - 1]) + len(sentences[j + window_size - 1])
                        candidate_snippets.append({
                            "text": snippet_text,
                            "offsetInBeginSection": offset_start,
                            "offsetInEndSection": offset_end,
                            "beginSection": section_name,
                            "endSection": section_name,
                            "document": doc_url
                        })

            if candidate_snippets:
                q_embedding_snippet = snippet_model.encode(
                    q_text, convert_to_tensor=True, device=device, normalize_embeddings=True
                )

                snippet_texts = [snip["text"] for snip in candidate_snippets]
                snippet_embeddings = snippet_model.encode(
                    snippet_texts, convert_to_tensor=True, device=device, normalize_embeddings=True
                )

                cosine_scores = util.cos_sim(q_embedding_snippet, snippet_embeddings)[0]

                top_indices = torch.topk(cosine_scores, k=min(top_k_snippets, len(candidate_snippets))).indices.tolist()

                for idx in top_indices:
                    score = cosine_scores[idx].item()
                    snippet = candidate_snippets[idx]
                    snippets.append((score, snippet))

        snippets = sorted(snippets, key=lambda x: x[0], reverse=True)[:top_k_snippets]
        prediction["snippets"] = [s[1] for s in snippets]

    return predictions


if __name__ == "__main__":
    documents_json_path = "../documents.json"
    test_json_path = "../../BioASQ-TaskB-testData/BioASQ-TaskB-testData/phaseA_12b_04.json"
    output_json_path = "../../results_biobert_phaseA_12b_04.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading document retrieval model...")
    doc_retrieval_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    print("Loading fine-tuned snippet selection model...")
    snippet_model = SentenceTransformer("./fine_tuned_biobert_snippets_triplet")

    with open(test_json_path, "r") as f:
        test_data = json.load(f)

    doc_texts = load_document_corpus(documents_json_path)
    doc_embeddings, doc_urls = build_index(doc_texts, doc_retrieval_model, device)

    # Document retrieval step
    predictions = predict_documents(
        test_data,
        doc_retrieval_model,
        doc_embeddings,
        doc_urls,
        device
    )

    # Snippet selection step
    final_predictions = extract_snippets(
        predictions,
        documents_json_path,
        snippet_model,
        device
    )

    # Format for BioASQ output
    output = {"questions": []}
    for pred in final_predictions:
        output["questions"].append({
            "id": pred["id"],
            "documents": pred["documents"],
            "snippets": pred["snippets"]
        })

    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Predictions and snippets saved to {output_json_path}")
