import argparse, json, os, nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import requests

def main():
    nltk.download("punkt", quiet=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--api_url", default="http://localhost:8000/chat")
    args = ap.parse_args()

    with open(args.eval_file, "r", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]

    smooth = SmoothingFunction().method1
    bleus, exacts = [], []

    for r in rows:
        q, gold = r["question"], r["answer"]
        resp = requests.post(args.api_url, json={"query": q, "history": []})
        pred = resp.json().get("answer","")
        bleus.append(sentence_bleu([gold.split()], pred.split(), smoothing_function=smooth))
        exacts.append(int(gold.strip().lower() in pred.strip().lower()))

    print(f"Mean BLEU: {np.mean(bleus):.4f}")
    print(f"Containment@1 (exact substring): {np.mean(exacts):.4f}")

if __name__ == "__main__":
    main()
