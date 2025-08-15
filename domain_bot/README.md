# Domain-Specific Chatbot (LLM + RAG + Optional QLoRA Fine-Tuning)

An end-to-end **local** chatbot specialized for your domain (legal/medical/finance/etc.).  
It supports:
- **RAG** with FAISS (PDF/text ingest → chunk → embeddings → retrieval)
- **Local LLM inference** using `llama-cpp-python` (GGUF models)
- **Optional QLoRA fine-tuning** with Hugging Face Transformers + PEFT
- **FastAPI** backend + **Streamlit** chat UI
- **Simple evaluation** script for accuracy on a small test set

---

## 0) Setup

```bash
# create venv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install requirements
pip install -r requirements.txt
```

### Choose a local GGUF model
Download a quantized `.gguf` model compatible with llama.cpp (e.g. 7B q4_K_M):
- Search on Hugging Face for: `Llama-3-Instruct-GGUF`, `Mistral-7B-Instruct-GGUF`, `Phi-3-mini-4k-instruct-Q4_K_M.gguf` (lightweight).

Set an env var pointing to it:
```bash
export GGUF_MODEL_PATH="/absolute/path/to/your/model.gguf"   # Windows (PowerShell): $env:GGUF_MODEL_PATH="C:\path\model.gguf"
```

> Tip: For low RAM/VRAM machines, start with smaller models like **Phi-3-mini** or **Qwen2-1.5B-Instruct GGUF**.

---

## 1) Ingest & Build RAG Index

Put your domain PDFs or `.txt` files into `data/` then run:

```bash
python build_index.py --input_dir data --index_dir rag_store
```

This will create a FAISS index and a `meta.json` with chunk metadata.

---

## 2) Run the API server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Test:
```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"query":"Explain consideration in contract law."}'
```

---

## 3) Launch the UI (Streamlit)

```bash
streamlit run ui.py
```

Open the local URL printed by Streamlit and chat.

---

## 4) (Optional) QLoRA Fine-Tuning

If you have a GPU with ~12GB VRAM (or more), you can fine-tune an HF model.
Set your HF token (if needed):

```bash
huggingface-cli login
```

Then run:

```bash
python train_qlora.py   --base_model mistralai/Mistral-7B-Instruct-v0.2   --dataset_path data/sample_supervised.jsonl   --output_dir outputs/lora-mistral
```

This trains LoRA adapters only. To use with inference, keep the llama-cpp path for base inference and use the RAG + prompting; or (advanced) export/merge LoRA back to HF weights and convert to GGUF (outside the scope here).

> If you cannot fine-tune, skip this step. RAG + strong prompts on a good instruct GGUF model is already very effective.

---

## 5) Evaluate

Create a small JSONL (`data/eval.jsonl`) with fields `question`, `answer`.
Run:
```bash
python eval.py --eval_file data/eval.jsonl
```

It computes basic relevance metrics (heuristic + BLEU).

---

## 6) Files

- `build_index.py` – ingest PDFs/TXT, chunk, embed (sentence-transformers), build FAISS.
- `api.py` – FastAPI server with retrieval-augmented generation via local GGUF model.
- `ui.py` – Streamlit chat UI.
- `train_qlora.py` – optional QLoRA finetuning script (HF Transformers + PEFT).
- `eval.py` – quick evaluation.
- `requirements.txt` – dependencies.
- `data/sample_supervised.jsonl` – example SFT format.
- `data/sample.txt` – example domain content.

---

## 7) Resume bullets (examples)

- Built and deployed a **local domain-specific LLM chatbot** using FAISS-based RAG and `llama-cpp-python`, achieving high factual accuracy on a curated benchmark.
- Implemented **optional QLoRA** fine-tuning with PEFT to adapt a 7B instruct model to domain data.
- Shipped a **FastAPI** backend and **Streamlit** UI for interactive demos.
