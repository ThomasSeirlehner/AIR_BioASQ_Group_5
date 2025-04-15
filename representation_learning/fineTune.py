import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import random

def build_triplet_dataset(train_json_path, documents_json_path):
    with open(documents_json_path, "r") as f:
        documents_data = json.load(f)
    doc_lookup = {doc["url"]: f"{doc['title']} {doc['abstract']}".strip() for doc in documents_data}

    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    triplets = []
    doc_urls = list(doc_lookup.keys())

    for q in train_data["questions"]:
        question = q["body"]
        positive_urls = q.get("documents", [])
        positive_docs = [doc_lookup[url] for url in positive_urls if url in doc_lookup]

        if not positive_docs:
            continue

        for pos_doc in positive_docs:
            # Random negative sampling
            negative_url = random.choice(doc_urls)
            while negative_url in positive_urls:
                negative_url = random.choice(doc_urls)
            negative_doc = doc_lookup[negative_url]

            triplets.append(InputExample(texts=[question, pos_doc, negative_doc]))

    return triplets

if __name__ == "__main__":

    documents_json_path = "../documents.json"
    train_json_path = "../../training13b.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    # Build dataset
    triplet_data = build_triplet_dataset(train_json_path, documents_json_path)
    train_dataloader = DataLoader(triplet_data, shuffle=True, batch_size=64)

    # Define loss
    train_loss = losses.TripletLoss(model=model)

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        warmup_steps=100,
        show_progress_bar=True,
        output_path="./fine_tuned_biobert",
        use_amp=True
    )