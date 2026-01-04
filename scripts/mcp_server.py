import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

# -----------------------------
# Config
# -----------------------------
INDEX_PATH = Path("outputs/index.faiss")
META_PATH = Path("outputs/meta.txt")
DOCS_PATH = Path("data/raw_docs")

TOP_K = 6
CHUNK_SIZE = 500

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("it-rag-mcp")

# -----------------------------
# Load retrieval assets (startup)
# -----------------------------
def load_metadata() -> List[str]:
    return META_PATH.read_text().splitlines()

def load_faiss_index():
    return faiss.read_index(str(INDEX_PATH))

def load_chunk_text(doc_name: str, chunk_id: int) -> str:
    doc_file = DOCS_PATH / doc_name
    text = doc_file.read_text(errors="ignore")
    start = chunk_id * CHUNK_SIZE
    end = start + CHUNK_SIZE
    return text[start:end]

def parse_citation(citation: str) -> Tuple[str, int]:
    # citation format: "file.md::chunk_3"
    doc_name, chunk_part = citation.split("::")
    chunk_id = int(chunk_part.replace("chunk_", ""))
    return doc_name, chunk_id

# SentenceTransformers model (same as your scripts)
model = SentenceTransformer("all-MiniLM-L6-v2")
index = load_faiss_index()
meta = load_metadata()

# -----------------------------
# MCP server
# -----------------------------
server_instructions = """
This MCP server provides local IT document search (RAG retrieval) using a FAISS index.
Use `search` to find relevant chunks, then `fetch` to retrieve the full chunk text.
"""

def _clean_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(("©", "Cookie", "Privacy"))]
    return lines

def _extract_steps(lines: list[str]) -> list[str]:
    step_patterns = [
        r"^\d+[\).\s]+",         # 1) 1. 1
        r"^[a-zA-Z][\).\s]+",    # a) a. A)
        r"^[-•]\s+",             # bullets
    ]

    steps = []
    for ln in lines:
        if any(re.match(p, ln) for p in step_patterns):
            steps.append(ln)

    if len(steps) < 3:
        action_verbs = ("check", "restart", "open", "verify", "ensure", "disable", "enable", "run", "update", "reset")
        for ln in lines:
            if ln.lower().startswith(action_verbs) or any(v in ln.lower() for v in action_verbs):
                steps.append(ln)
            if len(steps) >= 10:
                break

    seen = set()
    out = []
    for s in steps:
        key = s.lower()
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out[:10]

def _format_runbook_answer(question: str, evidence_blocks: list[tuple[str, str]]) -> dict:
    all_lines = []
    citations_used = []

    for citation, chunk in evidence_blocks:
        citations_used.append(citation)
        all_lines.extend(_clean_lines(chunk))

    steps = _extract_steps(all_lines)

    likely_causes = []
    for ln in all_lines:
        if len(likely_causes) >= 5:
            break
        if len(ln) < 140 and not re.match(r"^(\d+|[-•]|[a-zA-Z][\).\s])", ln):
            if any(word in ln.lower() for word in ["because", "cause", "issue", "problem", "may", "might", "if"]):
                likely_causes.append(ln)

    unique_citations = list(dict.fromkeys(citations_used))

    if len(steps) < 2 and len(all_lines) < 5:
        answer_text = (
            "I couldn't find enough relevant steps in the current knowledge base.\n"
            "Try rephrasing the question or add more docs."
        )
        return {
            "question": question,
            "answer": answer_text,
            "citations": unique_citations,
            "steps": [],
            "likely_causes": [],
        }

    parts = []
    parts.append("Runbook Answer (grounded in sources)\n")

    if likely_causes:
        parts.append("Likely causes (from docs):")
        for c in likely_causes[:4]:
            parts.append(f"- {c}")
        parts.append("")

    parts.append("Recommended steps:")
    for i, s in enumerate(steps, start=1):
        parts.append(f"{i}. {s}")

    parts.append("\nCitations:")
    for c in unique_citations:
        parts.append(f"- {c}")

    return {
        "question": question,
        "answer": "\n".join(parts),
        "citations": unique_citations,
        "steps": steps,
        "likely_causes": likely_causes[:4],
    }

mcp = FastMCP(name="IT-RAG-Assistant", instructions=server_instructions)

@mcp.tool()
async def search(query: str) -> List[Dict[str, Any]]:
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    distances, indices = index.search(q_emb, TOP_K)

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < 0:
            continue

        citation = meta[idx]
        doc_name, chunk_id = parse_citation(citation)
        chunk_text = load_chunk_text(doc_name, chunk_id).strip().replace("\n", " ")
        snippet = chunk_text[:220] + ("..." if len(chunk_text) > 220 else "")

        results.append(
            {
                "id": citation,
                "title": f"{doc_name} (chunk {chunk_id})",
                "url": f"local://{doc_name}#chunk_{chunk_id}",
                "snippet": snippet,
                "rank": rank,
            }
        )

    return results

@mcp.tool()
async def fetch(id: str) -> Dict[str, Any]:
    doc_name, chunk_id = parse_citation(id)
    text = load_chunk_text(doc_name, chunk_id)

    return {
        "id": id,
        "title": f"{doc_name} (chunk {chunk_id})",
        "url": f"local://{doc_name}#chunk_{chunk_id}",
        "text": text,
    }

@mcp.tool()
async def answer_question(query: str) -> dict:
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    distances, indices = index.search(q_emb, TOP_K)

    evidence = []
    for idx in indices[0]:
        if idx < 0:
            continue
        citation = meta[idx]
        doc_name, chunk_id = parse_citation(citation)
        chunk_text = load_chunk_text(doc_name, chunk_id)
        evidence.append((citation, chunk_text))

    return _format_runbook_answer(query, evidence)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
