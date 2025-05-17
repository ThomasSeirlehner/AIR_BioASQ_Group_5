import json
import requests
from xml.etree import ElementTree as ET
import os
print("Files in directory:", os.listdir())

# Load your BioASQ question file
with open(r"C:\Users\User\Desktop\BioASQ-TraditionalIR-Local\BioASQ-training13b.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract unique PMIDs
pmids = set()
for q in data["questions"]:
    for doc in q.get("documents", []):
        if "pubmed" in doc:
            pmids.add(doc.split("/")[-1])

pmids = list(pmids)
print(f"Found {len(pmids)} unique PMIDs.")

# Fetch abstracts from PubMed
corpus = []
batch_size = 100
for i in range(0, len(pmids), batch_size):
    batch = pmids[i:i+batch_size]
    ids = ",".join(batch)
    params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml"
    }
    print(f"Fetching {i+1}-{i+len(batch)}...")
    r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params)
    root = ET.fromstring(r.text)
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle") or ""
        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join([a.text for a in abstract_parts if a.text]) if abstract_parts else ""
        full_text = f"{title.strip()} {abstract.strip()}".strip()
        if full_text:
            corpus.append({"pmid": pmid, "text": full_text})

# Save to file
with open("BioASQ-corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, indent=2)

print("✅ Saved to BioASQ-corpus.json")
