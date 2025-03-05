#Querying and answering

import os
import chromadb
import hashlib
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import streamlit as st

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

@st.cache_resource()
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Loading {model_name}...")
    
    model_path = os.path.join("models", model_name.replace("/", "_"))
    os.makedirs(model_path, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=500)
    return pipe

llm = load_llm()

@st.cache_data()
def query_papers(query, top_k=3):
    """Search ChromaDB for relevant research papers and generate answers."""
    docs = db.similarity_search(query, k=top_k)
    
    if not docs:
        return [], "No relevant papers found."
    
    cache_key = hashlib.sha256(query.encode()).hexdigest()

    context = "\n\n".join([doc.page_content for doc in docs])
    llm_input = f"Context: {context}\n\nQuestion: {query}"
    response = llm(llm_input)[0]['generated_text']
    
    return docs, response

if __name__ == "__main__":
    user_query = input("Enter your research topic: ")
    query_papers(user_query)
