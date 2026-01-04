from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

INDEX_PATH = Path("outputs/index.faiss")
META_PATH = Path("outputs/meta.txt")
DOCS_PATH = Path("data/raw_docs")

TOP_K = 6
CHUNK_SIZE = 500


def load_metadata() -> list[str]:
    return META_PATH.read_text().splitlines()


def load_faiss_index():
    return faiss.read_index(str(INDEX_PATH))


def embed_query(model, query: str) -> np.ndarray:
    emb = model.encode([query])
    return np.array(emb).astype("float32")


def load_chunk_text(doc_name: str, chunk_id: int) -> str:
    doc_file = DOCS_PATH / doc_name
    text = doc_file.read_text(errors="ignore")
    start = chunk_id * CHUNK_SIZE
    end = start + CHUNK_SIZE
    return text[start:end]


def clean_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(("©", "Cookie", "Privacy"))]
    return lines


def extract_steps(lines: list[str]) -> list[str]:
    """
    Heuristic step extractor: prefers numbered/bulleted/lettered lines.
    """
    step_patterns = [
        r"^\d+[\).\s]+",   # 1) 1. 1
        r"^[a-zA-Z][\).\s]+",  # a) a. A)
        r"^[-•]\s+",       # bullets
    ]

    steps = []
    for ln in lines:
        if any(re.match(p, ln) for p in step_patterns):
            steps.append(ln)

    # If we didn’t capture much, fall back to the most “action-like” lines
    if len(steps) < 3:
        action_verbs = ("check", "restart", "open", "verify", "ensure", "disable", "enable", "run", "update", "reset")
        for ln in lines:
            if ln.lower().startswith(action_verbs) or any(v in ln.lower() for v in action_verbs):
                steps.append(ln)
            if len(steps) >= 8:
                break

    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in steps:
        key = s.lower()
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out[:10]


def format_runbook_answer(question: str, evidence_blocks: list[tuple[str, str]]) -> str:
    """
    evidence_blocks: [(citation, chunk_text), ...]
    """
    # Combine lines from all chunks
    all_lines = []
    citations_used = []
    for citation, chunk in evidence_blocks:
        citations_used.append(citation)
        all_lines.extend(clean_lines(chunk))

    steps = extract_steps(all_lines)

    # Simple "likely causes" heuristic: take a few short non-step lines
    likely_causes = []
    for ln in all_lines:
        if len(likely_causes) >= 5:
            break
        if len(ln) < 140 and not re.match(r"^(\d+|[-•]|[a-zA-Z][\).\s])", ln):
            # avoid duplicating the question-ish lines
            if any(word in ln.lower() for word in ["because", "cause", "issue", "problem", "may", "might", "if"]):
                likely_causes.append(ln)

    # Guardrail: if we couldn’t extract anything useful, say so
    if len(steps) < 2 and len(all_lines) < 5:
        return (
            f"Question: {question}\n\n"
            "Runbook Answer:\n"
            "- I couldn't find enough relevant steps in the current knowledge base.\n\n"
            f"Citations checked: {', '.join(citations_used)}"
        )

    out = []
    out.append(f"Question: {question}\n")
    out.append("Runbook Answer (grounded in sources)\n")

    if likely_causes:
        out.append("Likely causes (from docs):")
        for c in likely_causes[:4]:
            out.append(f"- {c}")
        out.append("")

    out.append("Recommended steps:")
    if steps:
        for i, s in enumerate(steps, start=1):
            out.append(f"{i}. {s}")
    else:
        out.append("- (No explicit step list found; see citations below.)")

    out.append("")
    out.append("Citations:")
    for c in dict.fromkeys(citations_used):  # unique preserve order
        out.append(f"- {c}")

    return "\n".join(out)


def main():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index not found. Run scripts/index_docs.py first.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = load_faiss_index()
    meta = load_metadata()

    while True:
        question = input("\nAsk an IT question (or type 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        q_emb = embed_query(model, question)
        distances, indices = index.search(q_emb, TOP_K)

        evidence = []
        for idx in indices[0]:
            if idx < 0:
                continue
            citation = meta[idx]  # file::chunk_x
            doc_name, chunk_part = citation.split("::")
            chunk_id = int(chunk_part.replace("chunk_", ""))
            chunk_text = load_chunk_text(doc_name, chunk_id)
            evidence.append((citation, chunk_text))

        answer = format_runbook_answer(question, evidence)
        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    main()
