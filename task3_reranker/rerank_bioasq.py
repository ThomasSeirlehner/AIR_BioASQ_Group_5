import json
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Load the BioASQ task 1 results file
with open('results_traditionalIR_phaseA.json', 'r') as f:
    task1_results = json.load(f)

# Load the re-ranking model (BioBERT, SciBERT, etc.)
model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

reranked_results = []

for question in tqdm(task1_results['questions']):
    q_id = question['id']
    q_text = question.get('body', '')

    # Extract candidate snippets or fallback to documents if empty
    candidates = [s['text'] for s in question.get('snippets', [])]

    if not candidates:
        print(f"[WARNING] No snippets found for question {q_id}, skipping...")
        continue

    # Encode the question and candidates on the same device
    q_emb = model.encode(q_text, convert_to_tensor=True).unsqueeze(0).to(device)
    c_embs = model.encode(candidates, convert_to_tensor=True).to(device)

    # Compute cosine similarity
    scores = util.cos_sim(q_emb, c_embs)[0]

    # Sort candidates by score
    sorted_indices = torch.argsort(scores, descending=True)
    reranked = [candidates[idx] for idx in sorted_indices]

    # Save reranked results
    reranked_results.append({
        'id': q_id,
        'reranked_snippets': reranked
    })

# Save reranked results to a file
with open('reranked_results.json', 'w') as f:
    json.dump({'questions': reranked_results}, f, indent=2)

print("✅ Reranking complete. Results saved to reranked_results.json")
