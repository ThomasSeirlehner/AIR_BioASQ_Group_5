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

def extract_all_document_urls(bioasq_path):
    """Collect all unique document URLs from the BioASQ dataset."""
    with open(bioasq_path, "r") as f:
        data = json.load(f)

    url_set = set()
    for q in data["questions"]:
        for url in q.get("documents", []):
            url_set.add(url)
        for snippet in q.get("snippets", []):
            url_set.add(snippet.get("document", ""))

    return sorted(url_set)

def load_existing_documents(existing_json):
    """Load existing documents from a JSON file if it exists."""
    if not existing_json:
        return {}
    
    try:
        with open(existing_json, "r") as f:
            existing_data = json.load(f)
            existing_documents = {doc['url']: doc for doc in existing_data}
            return existing_documents
    except FileNotFoundError:
        print(f"Warning: {existing_json} not found. Starting fresh.")
        return {}

def build_document_dataset(bioasq_path, output_json, existing_json=None):
    """Build a document dataset from the BioASQ dataset and save the results."""
    # Load existing documents if any
    existing_documents = load_existing_documents(existing_json)
    urls = extract_all_document_urls(bioasq_path)
    docs = list(existing_documents.values())  # Start with the existing documents
    count = len(docs)  # Start count at the size of existing documents

    for url in tqdm(urls, desc="Fetching papers"):
        if not url.strip():
            continue

        # Check if the document is already in the existing dataset
        if url in existing_documents:
            continue

        # Fetch new document details
        title, abstract = fetch_pubmed_title_abstract(url)
        docs.append({
            "url": url,
            "title": title,
            "abstract": abstract
        })

        count += 1
        if count % 100 == 0:
            # Save intermediate version
            temp_path = output_json.replace(".json", f"_partial_{count}.json")
            with open(temp_path, "w") as f:
                json.dump(docs, f, indent=2)
            print(f"Saved {count} documents to {temp_path}")

    # Final save
    with open(output_json, "w") as f:
        json.dump(docs, f, indent=2)
    print(f"\nFinal save: {len(docs)} documents saved to {output_json}")

# --------------------------- Run ---------------------------
if __name__ == "__main__":
    input_json_path = "C:/Users/user/Documents/4.Semester/AIR/AIR_BioASQ_Group_5/training13b.json"  # BioASQ dataset file
    output_json_path = "C:/Users/user/Documents/4.Semester/AIR/AIR_BioASQ_Group_5/documents.json"     # Output file path
    existing_json_path = "C:/Users/user/Documents/4.Semester/AIR/AIR_BioASQ_Group_5/documents_partial_38800.json"   # Path to already downloaded JSON (optional)

    build_document_dataset(input_json_path, output_json_path, existing_json=existing_json_path)
