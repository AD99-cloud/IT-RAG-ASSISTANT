from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

INDEX_PATH = Path("outputs/index.faiss")
META_PATH = Path("outputs/meta.txt")
DOCS_PATH = Path("data/raw_docs")

TOP_K = 5

def load_metadata():
    return META_PATH.read_text().splitlines()

def load_faiss_index():
    return faiss.read_index(str(INDEX_PATH))

def embed_query(model, query: str):
    emb = model.encode([query])
    return np.array(emb).astype("float32")

def load_chunk_text(doc_name: str, chunk_id: int, chunk_size: int = 500):
    doc_file = DOCS_PATH / doc_name
    text = doc_file.read_text(errors="ignore")
    start = chunk_id * chunk_size
    end = start + chunk_size
    return text[start:end]

def main():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index not found. Run scripts/index_docs.py first.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = load_faiss_index()
    meta = load_metadata()

    while True:
        query = input("\nAsk an IT question (or type 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        q_emb = embed_query(model, query)
        distances, indices = index.search(q_emb, TOP_K)

        print("\nTop matches (with citations):")
        for rank, idx in enumerate(indices[0], start=1):
            if idx < 0:
                continue
            citation = meta[idx]  # format: file::chunk_x
            doc_name, chunk_part = citation.split("::")
            chunk_id = int(chunk_part.replace("chunk_", ""))
            snippet = load_chunk_text(doc_name, chunk_id).strip().replace("\n", " ")
            snippet = snippet[:220] + ("..." if len(snippet) > 220 else "")

            print(f"\n{rank}) [{citation}]")
            print(f"   {snippet}")

if __name__ == "__main__":
    main()
