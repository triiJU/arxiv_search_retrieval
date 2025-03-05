import os
import streamlit as st
from fetch_arxiv import fetch_all_arxiv_papers, store_papers
from query_chroma import query_papers
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import chromadb

os.environ["PYTHONWARNINGS"] = "ignore" 

st.set_page_config(page_title="Arxiv-RAG", layout="wide")
st.title("📚 Arxiv-RAG: AI-Powered Research Paper Search")



# Sidebar for Fetching New Papers
st.sidebar.header("🔄 Fetch New Papers")
max_results = st.sidebar.slider("Max papers per category", min_value=1, max_value=50, value=5)
fetch_papers = st.sidebar.button("Fetch Latest arXiv Papers")

if fetch_papers:
    with st.spinner("Fetching papers..."):
        papers = fetch_all_arxiv_papers(max_results)
        store_papers(papers)
    st.sidebar.success("✅ Papers successfully fetched & stored!")

# Search Query Section
st.header(" Search Research Papers")
query = st.text_input("Enter your research query:")
top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)
search = st.button("Search")

if search and query:
    with st.spinner("Searching and generating response..."):
        docs, response = query_papers(query, top_k)
    
    if docs:
        st.subheader("📄 Relevant Papers:")
        for doc in docs:
            st.markdown(f"- [{doc.metadata['title']}]({doc.metadata['url']})")
    else:
        st.warning("No relevant papers found.")
    
    st.subheader(" AI-Generated Summary:")
    st.write(response if response else "No summary available.")
