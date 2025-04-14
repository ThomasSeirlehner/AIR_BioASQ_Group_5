import json
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

def fetch_pubmed_title_abstract(url):
    """Fetch title and abstract from a PubMed article via scraping."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title_tag = soup.find("h1", class_="heading-title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Extract abstract
        abstract_tag = soup.find("div", class_="abstract-content selected")
        abstract = abstract_tag.get_text(strip=True) if abstract_tag else ""

        return title, abstract
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return "", ""

def repair_documents_json(json_path):
    """Fix documents with missing title or abstract in the given JSON."""
    # Load the existing document data
    with open(json_path, "r") as f:
        docs = json.load(f)

    updated_count = 0

    for doc in tqdm(docs, desc="Repairing documents"):
        if not doc["title"] or not doc["abstract"]:
            title, abstract = fetch_pubmed_title_abstract(doc["url"])

            # Only update if something new was fetched
            if title or abstract:
                doc["title"] = title
                doc["abstract"] = abstract
                updated_count += 1

    # Save the repaired JSON
    with open(json_path, "w") as f:
        json.dump(docs, f, indent=2)

    print(f"\nDone. {updated_count} documents were updated.")

# --------------------------- Run ---------------------------
if __name__ == "__main__":
    documents_json_path = "C:/Users/user/Documents/4.Semester/AIR/AIR_BioASQ_Group_5/documents.json"  # Your existing JSON
    repair_documents_json(documents_json_path)
