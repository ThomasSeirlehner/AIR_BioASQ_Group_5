import json
import requests
from xml.etree import ElementTree as ET

# === CONFIG ===
INPUT_FILE = "BioASQ-task13bPhaseA-testset4.json"  # your test file
OUTPUT_FILE = "results_traditionalIR_phaseA.json"
TOP_K = 10  # top documents per query

# === API URLs ===
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# === Helper functions ===

def fetch_top_pmids(query, retmax=10):
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax
    }
    response = requests.get(ESEARCH_URL, params=params)
    response.raise_for_status()
    return response.json()["esearchresult"]["idlist"]

def fetch_abstracts(pmids):
    ids = ",".join(pmids)
    params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml"
    }
    response = requests.get(EFETCH_URL, params=params)
    response.raise_for_status()

    abstracts = {}
    root = ET.fromstring(response.text)
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        abstract_text = " ".join(
            [elem.text for elem in article.findall(".//AbstractText") if elem.text]
        )
        abstracts[pmid] = abstract_text
    return abstracts

# === Main logic ===

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    output = {"questions": []}

    for q in test_data["questions"]:
        qid = q["id"]
        qtext = q["body"]
        print(f"Processing: {qid}")

        # 1. Fetch top PMIDs from PubMed
        pmids = fetch_top_pmids(qtext, TOP_K)
        documents = [f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" for pmid in pmids]

        # 2. Fetch abstract text for snippets
        abstracts = fetch_abstracts(pmids)
        snippets = []
        for pmid, text in abstracts.items():
            if not text:
                continue
            snippet = {
                "text": text[:300],  # First 300 chars
                "offsetInBeginSection": 0,
                "offsetInEndSection": min(300, len(text)),
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
            }
            snippets.append(snippet)

        output["questions"].append({
            "id": qid,
            "documents": documents,
            "snippets": snippets
        })

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Done! Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()