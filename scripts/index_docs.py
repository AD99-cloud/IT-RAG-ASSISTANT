from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Paths
DOCS_PATH = Path("data/raw_docs")
OUTPUTS_PATH = Path("outputs")
INDEX_PATH = OUTPUTS_PATH / "index.faiss"
META_PATH = OUTPUTS_PATH / "meta.txt"

# Load embedding model (free, local)
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
metadata = []

# Read documents
for file in DOCS_PATH.glob("*"):
    if file.suffix.lower() not in [".txt", ".md"]:
        continue

    text = file.read_text(errors="ignore")

    # Simple chunking
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    for idx, chunk in enumerate(chunks):
        if chunk.strip():
            texts.append(chunk)
            metadata.append(f"{file.name}::chunk_{idx}")

if not texts:
    raise ValueError("No documents found. Check data/raw_docs.")

# Create embeddings
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save outputs
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(INDEX_PATH))

with open(META_PATH, "w") as f:
    for m in metadata:
        f.write(m + "\n")

print(f"Indexed {len(texts)} chunks from {len(set(m.split('::')[0] for m in metadata))} documents.")
