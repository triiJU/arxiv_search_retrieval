# Optimized ArXiv Paper Fetching & Indexing

import os
import arxiv
import chromadb
import requests
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

os.makedirs("models", exist_ok=True)

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="arxiv_papers")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_all_categories():
    """Scrape arXiv to get all available categories dynamically."""
    base_url = "https://arxiv.org"
    response = requests.get(f"{base_url}/archive")
    soup = BeautifulSoup(response.text, "html.parser")
    
    categories = []
    for link in soup.select("a[href^='/list/']"):
        category = link["href"].split("/")[-1]
        categories.append(category)
    
    return categories

def fetch_category_papers(category, max_results_per_category):
    """Fetch papers for a single arXiv category."""
    search = arxiv.Search(
        query=f"cat:{category}", max_results=max_results_per_category, sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "pdf_url": result.pdf_url
        })
    return papers

def fetch_all_arxiv_papers(max_results_per_category=5):
    """Fetch papers from all arXiv categories dynamically using multi-threading."""
    categories = get_all_categories()
    all_papers = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(lambda cat: fetch_category_papers(cat, max_results_per_category), categories)
    
    for paper_list in results:
        all_papers.extend(paper_list)
    
    return all_papers

def store_papers(papers):
    """Store paper embeddings in ChromaDB."""
    for paper in papers:
        embedding = model.encode(paper['summary']).tolist()
        collection.add(
            ids=[paper['title']], embeddings=[embedding], metadatas=[{"title": paper['title'], "url": paper['pdf_url']}]
        )
    print("✅ All arXiv papers stored successfully!")

if __name__ == "__main__":
    max_results = int(input("Enter number of papers per category to fetch: ") or 5)
    papers = fetch_all_arxiv_papers(max_results)
    store_papers(papers)