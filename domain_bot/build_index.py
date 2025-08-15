import argparse, os, json, glob, hashlib
from typing import List, Dict
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

def read_docs(input_dir: str) -> List[Dict]:
    docs = []
    for path in glob.glob(os.path.join(input_dir, "**", "*"), recursive=True):
        if os.path.isdir(path): 
            continue
        if path.lower().endswith(".pdf"):
            try:
                reader = PdfReader(path)
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception as e:
                print(f"[WARN] Failed to read {path}: {e}")
                continue
        elif path.lower().endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            continue
        if text.strip():
            docs.append({"path": path, "text": text})
    return docs

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
        if i <= 0: break
    return chunks

def hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--index_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.index_dir, exist_ok=True)

    print("[*] Reading documents...")
    docs = read_docs(args.input_dir)
    if not docs:
        print("[!] No documents found.")
        return

    print("[*] Chunking...")
    chunks, meta = [], []
    for d in docs:
        for ch in chunk_text(d["text"]):
            chunks.append(ch)
            meta.append({"source": d["path"], "id": hash_str(ch)})

    print("[*] Embedding with sentence-transformers (all-MiniLM-L6-v2)...")
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = emb_model.encode(chunks, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, os.path.join(args.index_dir, "faiss.index"))
    with open(os.path.join(args.index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "meta": meta}, f, ensure_ascii=False, indent=2)
    print(f"[*] Saved index to {args.index_dir} ({len(chunks)} chunks).")

if __name__ == "__main__":
    main()
