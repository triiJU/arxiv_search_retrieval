import os
import chromadb
from sentence_transformers import SentenceTransformer

DATA_PATH = "data"
CHROMA_DB_PATH = os.path.join(DATA_PATH, "chroma_db")

# Ensure ChromaDB exists before querying
if not os.path.exists(CHROMA_DB_PATH):
    print("⚠️ ChromaDB index missing! Run `fetch_arxiv.py` first.")
    exit()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = chroma_client.get_collection(name="arxiv_papers")

# Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def query_papers(query, top_k=3):
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = chroma_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if not results["documents"][0]:
        return [], "No relevant papers found."

    docs = [{"title": meta["title"], "url": meta["url"]} for meta in results["metadatas"][0]]
    return docs, "Results retrieved successfully!"

if __name__ == "__main__":
    query = input("Enter your search query: ")
    papers, message = query_papers(query)
    print("\n📄 **Top Research Papers:**")
    for paper in papers:
        print(f"- {paper['title']} ( [PDF]({paper['url']}) )")
