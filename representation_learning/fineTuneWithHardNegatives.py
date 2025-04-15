import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from sentence_transformers.util import cos_sim
import numpy as np

def build_triplet_dataset(train_json_path, documents_json_path, model, num_hard_negatives=1):
    with open(documents_json_path, "r") as f:
        documents_data = json.load(f)
    doc_lookup = {doc["url"]: f"{doc['title']} {doc['abstract']}".strip() for doc in documents_data}

    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    doc_urls = list(doc_lookup.keys())
    doc_texts = [doc_lookup[url] for url in doc_urls]

    # Precompute document embeddings
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)

    triplets = []

    for q in train_data["questions"]:
        question = q["body"]
        positive_urls = q.get("documents", [])
        positive_docs = [doc_lookup[url] for url in positive_urls if url in doc_lookup]

        if not positive_docs:
            continue

        # Embed question
        question_embedding = model.encode(question, convert_to_tensor=True)

        # Compute similarity with all documents
        similarities = cos_sim(question_embedding, doc_embeddings)[0].cpu().numpy()

        # Sort by similarity, descending
        sorted_indices = np.argsort(-similarities)

        # For each positive doc, select hard negatives (top similar docs not in positives)
        for pos_doc in positive_docs:
            hard_negatives = []
            for idx in sorted_indices:
                negative_url = doc_urls[idx]
                if negative_url not in positive_urls:
                    hard_negatives.append(doc_lookup[negative_url])
                if len(hard_negatives) >= num_hard_negatives:
                    break

            for neg_doc in hard_negatives:
                triplets.append(InputExample(texts=[question, pos_doc, neg_doc]))

    return triplets

if __name__ == "__main__":

    documents_json_path = "../documents.json"
    train_json_path = "../../training13b.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    # Build dataset
    triplet_data = build_triplet_dataset(train_json_path, documents_json_path, model, num_hard_negatives=1)
    train_dataloader = DataLoader(triplet_data, shuffle=True, batch_size=64)

    # Define loss
    train_loss = losses.TripletLoss(model=model)

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        warmup_steps=100,
        show_progress_bar=True,
        output_path="./fine_tuned_biobert_hard_negatives",
        use_amp=True
    )