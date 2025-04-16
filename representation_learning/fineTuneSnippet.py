import json
import random
import nltk
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm

# Download NLTK punkt tokenizer if not already present
nltk.download('punkt')

def load_training_triplets(training_json, documents_json):
    with open(training_json, "r") as f:
        train_data = json.load(f)
    with open(documents_json, "r") as f:
        documents = json.load(f)

    doc_lookup = {doc["url"]: f"{doc['title']} {doc['abstract']}".strip() for doc in documents}

    # Collect all possible negative candidate sentences (sentences or sentence groups)
    all_sentences = []
    for text in doc_lookup.values():
        sentences = nltk.sent_tokenize(text, language='english')
        all_sentences.append(sentences)

    train_examples = []
    for q in tqdm(train_data["questions"], desc="Preparing triplets"):
        q_text = q["body"]

        for snippet in q.get("snippets", []):
            pos_text = snippet["text"]
            pos_sent_count = len(nltk.sent_tokenize(pos_text, language='english'))

            # Sample a negative: select random sentences group with roughly same number of sentences
            while True:
                candidate_sentences = random.choice(all_sentences)
                if len(candidate_sentences) < pos_sent_count:
                    continue
                start_idx = random.randint(0, len(candidate_sentences) - pos_sent_count)
                neg_text = ' '.join(candidate_sentences[start_idx:start_idx + pos_sent_count])

                # Ensure it's different from the positive snippet text
                if neg_text.strip() != pos_text.strip():
                    break

            train_examples.append(InputExample(texts=[q_text, pos_text, neg_text]))

    return train_examples

if __name__ == "__main__":
    training_json_path = "../../training12b.json"
    documents_json_path = "../documents.json"
    model_save_path = "./fine_tuned_biobert_snippets_triplet_multi"

    print("Loading training data...")
    train_examples = load_training_triplets(training_json_path, documents_json_path)

    print(f"Total triplets: {len(train_examples)}")

    print("Loading base BioBERT model...")
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

    # Triplet loss (anchor, positive, negative)
    train_loss = losses.TripletLoss(model)

    print("Starting fine-tuning with Triplet Loss...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=100,
        output_path=model_save_path,
        show_progress_bar=True,
        use_amp=True
    )
    print(f"Model saved to {model_save_path}")
