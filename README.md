# IT RAG Assistant (FAISS + MCP)

A local, source-grounded AI assistant for IT troubleshooting built using **Retrieval-Augmented Generation (RAG)** and exposed through the **Model Context Protocol (MCP)**.

This system retrieves answers strictly from internal IT documentation and returns **runbook-style responses with citations**, avoiding hallucinations and external data exposure.

---

## Problem

In most IT organizations, troubleshooting knowledge exists across:
- runbooks  
- KB articles  
- markdown files  
- internal documentation  

Engineers often spend significant time manually searching these resources.  
Generic LLM solutions are risky because they can hallucinate steps, lack traceability, and require sending proprietary data to external services.

---

## Solution

This project implements a **practical, enterprise-ready RAG pipeline**:

- All documents remain **local**
- Retrieval is performed using **FAISS vector search**
- Answers are constructed **only from retrieved content**
- Every response includes **citations**
- The system is exposed as **MCP tools**, making it agent-ready

The result is an **automated runbook assistant**, not a free-form chatbot.

---

## How It Works

1. IT troubleshooting documents are ingested and chunked  
2. Chunks are embedded using SentenceTransformers  
3. Embeddings are stored in a FAISS index  
4. User queries retrieve the most relevant chunks  
5. Retrieved content is structured into runbook-style answers  
6. All functionality is exposed via MCP tools  

---

## MCP Tools

The MCP server exposes the following tools:

- **`search(query)`**  
  Returns the most relevant document chunks.

- **`fetch(id)`**  
  Returns the full text of a specific document chunk.

- **`answer_question(query)`**  
  Produces a complete runbook-style answer grounded in retrieved documents, including citations.

These tools allow the system to be integrated into MCP-compatible agents or copilots.

