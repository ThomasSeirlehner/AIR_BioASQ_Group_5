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

def build_pair_dataset(train_json_path, documents_json_path):
    with open(documents_json_path, "r") as f:
        documents_data = json.load(f)
    doc_lookup = {doc["url"]: f"{doc['title']} {doc['abstract']}".strip() for doc in documents_data}

    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    pairs = []
    for q in train_data["questions"]:
        question = q["body"]
        positive_urls = q.get("documents", [])
        for url in positive_urls:
            if url in doc_lookup:
                pos_doc = doc_lookup[url]
                # Create InputExample with just two texts: anchor (question) and positive document.
                pairs.append(InputExample(texts=[question, pos_doc]))
    return pairs

if __name__ == "__main__":

    documents_json_path = "C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/documents.json"
    train_json_path = "C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/BioASQ-training12b/training12b_new.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    
    pair_data = build_pair_dataset(train_json_path, documents_json_path)
    #triplet_data = build_triplet_dataset(train_json_path, documents_json_path)
    pair_dataloader = DataLoader(pair_data, shuffle=True, batch_size=32)
    #triplet_dataloader = DataLoader(triplet_data, shuffle=True, batch_size=32)
    loss_pair = losses.MultipleNegativesRankingLoss(model=model, scale=30)
    #loss_triplet = losses.TripletLoss(model=model, triplet_margin=0.3)
    num_epochs = 5
    total_steps = len(pair_dataloader) * num_epochs

    model.fit(
        train_objectives=[(pair_dataloader, loss_pair)],#,(triplet_dataloader, loss_triplet)],
        epochs=num_epochs,
        warmup_steps=100,#int(0.1 * total_steps),
        #optimizer_params={'lr': 2e-5},
        show_progress_bar=True,
        output_path="./fine_tuned_biobert_TripleDouble_03_30",
        use_amp=True
    )

    

