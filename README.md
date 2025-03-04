Arxiv-RAG: a retrieval-augmented generation (RAG) system that leverages **Mistral-7B & LLaMA-2** to search and summarize research papers from **arXiv**. It uses **ChromaDB** for efficient storage and retrieval.

sdk: streamlit

sdk_version: 5.20.0

app_file: streamlitapp.py

short_description: ArXiv research retrieval system



Features:
- Dynamic arXiv Paper Retrieval from all categories
- AI-Powered Summaries using Mistral-7B (or LLaMA-2)
- Fast Search & Storage via ChromaDB
- Interactive UI built with Streamlit


# Installation

git clone https://github.com/triiJU/arxivrag.git && cd arxivrag

conda create -n arxiv-rag python=3.10 -y && conda activate arxiv-rag

pip install -r requirements.txt


# Usage

### Fetch & Index Papers

python fetch_arxiv.py

### Start Streamlit UI

streamlit run streamlitapp.py

### Search & Generate Answers

python query_chroma.py


## Example Queries
|          Query         |          Expected Output       |
|------------------------|--------------------------------|
| "Transformer models"   |   List of papers + summaries   |
|"Reinforcement learning"| Papers + AI-generated insights |


## Troubleshooting
pip check 

pip install --force-reinstall -r requirements.txt  

rm -rf chroma_db/ && python fetch_arxiv.py  



## License
MIT License.
