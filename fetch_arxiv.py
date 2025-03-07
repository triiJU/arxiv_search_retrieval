#Here this serves as the backend logic for my project

import os
import arxiv
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

DATA_PATH = "data"
CHROMA_DB_PATH = os.path.join(DATA_PATH, "chroma_db")
os.makedirs(DATA_PATH, exist_ok=True)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(name="arxiv_papers")

# Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def fetch_all_arxiv_papers(max_results=50):
    search = arxiv.Search(
        query="machine learning",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    documents = []
    metadata = []
    
    for result in search.results():
        text = f"Title: {result.title}\nAbstract: {result.summary}\n"
        documents.append(text)
        metadata.append({"title": result.title, "url": result.pdf_url})

    return documents, metadata

def store_papers(documents, metadata):
    embeddings = embedding_model.encode(documents)
    for i, doc in enumerate(documents):
        chroma_collection.add(
            ids=[str(i)],
            embeddings=[embeddings[i].tolist()],
            documents=[doc],
            metadatas=[metadata[i]]
        )
    print("✅ Papers stored successfully in ChromaDB!")

if __name__ == "__main__":
    docs, meta = fetch_all_arxiv_papers()
    store_papers(docs, meta)
