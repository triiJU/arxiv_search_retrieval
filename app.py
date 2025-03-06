from flask import Flask, render_template, request, jsonify
from fetch_arxiv import fetch_all_arxiv_papers, store_papers
from query_chroma import query_papers

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Loads the UI

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if query:
        docs, response = query_papers(query, top_k=3)
        return jsonify({"docs": docs, "summary": response})
    return jsonify({"error": "No query provided"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
