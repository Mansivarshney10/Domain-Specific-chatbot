import os, json
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llama_cpp import Llama

INDEX_DIR = os.environ.get("INDEX_DIR", "rag_store")
GGUF_MODEL_PATH = os.environ.get("GGUF_MODEL_PATH", None)
TOP_K = int(os.environ.get("TOP_K", 4))

if GGUF_MODEL_PATH is None or not os.path.exists(GGUF_MODEL_PATH):
    raise RuntimeError("Set GGUF_MODEL_PATH to your local .gguf model path")

# Load FAISS + metadata
faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)
CHUNKS = meta["chunks"]
EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load local LLM
llm = Llama(model_path=GGUF_MODEL_PATH, n_ctx=4096, n_threads=os.cpu_count() or 4)

app = FastAPI(title="Domain Chatbot API")

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []  # [{"role":"user"/"assistant", "content": "..."}]

def retrieve(query: str, k=TOP_K):
    q = EMB.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(q, k)
    ctx = [CHUNKS[i] for i in I[0] if i >= 0]
    return ctx

SYSTEM_PROMPT = """You are a precise domain assistant. Use the provided CONTEXT to answer the USER question with citations as [source i]. 
If the answer is not in the context, say you don't have enough information. Be concise and factual.
"""

def build_prompt(query: str, ctx_chunks: List[str]) -> str:
    ctx = "\n\n".join([f"[source {i+1}] {c}" for i, c in enumerate(ctx_chunks)])
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{ctx}\n\nUSER: {query}\nASSISTANT:"

@app.post("/chat")
def chat(req: ChatRequest):
    ctx_chunks = retrieve(req.query)
    prompt = build_prompt(req.query, ctx_chunks)
    out = llm(
        prompt,
        max_tokens=512,
        temperature=0.2,
        top_p=0.9,
        stop=["USER:", "ASSISTANT:", "###"]
    )
    text = out["choices"][0]["text"].strip()
    return {"answer": text, "context_count": len(ctx_chunks)}
