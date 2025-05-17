import json
import math
import nltk
from collections import defaultdict, Counter

# Ensure NLTK uses the correct data path
nltk.download('punkt_tab')

# BM25 params
k1 = 1.5
b = 0.75

class BM25Index:
    def __init__(self, documents):
        self.documents = documents
        self.N = len(documents)
        self.avgdl = sum(len(doc['text'].split()) for doc in documents) / self.N
        self.inverted_index = defaultdict(list)
        self.doc_freqs = defaultdict(int)
        self.doc_lens = []
        self.build_index()

    def build_index(self):
        for doc_id, doc in enumerate(self.documents):
            tokens = nltk.word_tokenize(doc['text'].lower())
            self.doc_lens.append(len(tokens))
            tf = Counter(tokens)
            for term in tf:
                self.inverted_index[term].append((doc_id, tf[term]))
            for term in set(tokens):
                self.doc_freqs[term] += 1

    def score(self, query):
        tokens = nltk.word_tokenize(query.lower())
        scores = [0] * self.N
        for term in tokens:
            if term not in self.inverted_index:
                continue
            df = self.doc_freqs[term]
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            postings = self.inverted_index[term]
            for doc_id, tf in postings:
                dl = self.doc_lens[doc_id]
                denom = tf + k1 * (1 - b + b * dl / self.avgdl)
                score = idf * tf * (k1 + 1) / denom
                scores[doc_id] += score
        return scores

def main():
    # Load local corpus: list of dicts with pmid and text (abstract)
    with open(r"BioASQ-corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Load questions
    with open("BioASQ-task13bPhaseA-testset4.json", "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    bm25 = BM25Index(corpus)

    results = {"questions": []}

    for q in questions:
        qid = q["id"]
        query = q["body"]
        scores = bm25.score(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]

        docs = [f"http://www.ncbi.nlm.nih.gov/pubmed/{corpus[i]['pmid']}" for i in ranked]

        snippets = []
        for i in ranked:
            text = corpus[i]["text"]
            snippets.append({
                "text": text[:300],
                "offsetInBeginSection": 0,
                "offsetInEndSection": min(300, len(text)),
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": f"http://www.ncbi.nlm.nih.gov/pubmed/{corpus[i]['pmid']}"
            })

        results["questions"].append({
            "id": qid,
            "documents": docs,
            "snippets": snippets
        })

    with open("results_traditionalIR_local.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Results saved in results_traditionalIR_local.json")

if __name__ == "__main__":
    main()
