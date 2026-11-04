"""Evaluate retrieval against manually labelled WordPress post IDs.
Example: python evaluate_rag.py --dataset eval_dataset.example.json --site https://techcrunch.com --pages 2 --k 3
Metrics are calculated over unique posts (not duplicate chunks from the same post).
"""
import argparse, json, math, re
import numpy as np, requests, faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

def clean(s):
    return re.sub(r"\s+", " ", BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)).strip()

def split(text, size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + int(size * .7), end)
            if boundary > start: end = boundary
        if text[start:end].strip(): out.append(text[start:end].strip())
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return out

def fetch(site, pages):
    docs = []
    for page in range(1, pages + 1):
        response = requests.get(site.rstrip("/") + "/wp-json/wp/v2/posts",
            params={"page": page, "per_page": 100, "_fields": "id,title,content"}, timeout=20)
        if response.status_code == 400 and page > 1: break
        response.raise_for_status()
        batch = response.json()
        for post in batch:
            title = clean(post.get("title", {}).get("rendered", ""))
            body = clean(post.get("content", {}).get("rendered", ""))
            for chunk in split((title + "\n\n" + body).strip()):
                docs.append({"post_id": int(post["id"]), "text": chunk})
        if len(batch) < 100: break
    return docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    with open(args.dataset, encoding="utf-8") as f: cases = json.load(f)
    docs = fetch(args.site, args.pages)
    if not docs: raise SystemExit("No posts fetched.")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = np.asarray(model.encode([d["text"] for d in docs], normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
    totals = {"hit_rate": 0., "recall": 0., "mrr": 0., "ndcg": 0.}; count = 0
    for case in cases:
        relevant = set(map(int, case["relevant_post_ids"]))
        if not relevant: continue
        query = np.asarray(model.encode([case["question"]], normalize_embeddings=True), dtype="float32")
        _, ids = index.search(query, min(max(1, args.k * 5), index.ntotal))
        # Collapse multiple retrieved chunks from the same post, preserving first/best rank.
        ranked, seen = [], set()
        for idx in ids[0]:
            if idx < 0 or idx >= len(docs): continue
            post_id = docs[int(idx)]["post_id"]
            if post_id not in seen:
                ranked.append(post_id); seen.add(post_id)
            if len(ranked) == args.k: break
        hit = float(any(pid in relevant for pid in ranked))
        recall = len(set(ranked) & relevant) / len(relevant)
        mrr = next((1. / (rank + 1) for rank, pid in enumerate(ranked) if pid in relevant), 0.)
        dcg = sum(1. / math.log2(rank + 2) for rank, pid in enumerate(ranked) if pid in relevant)
        ideal = sum(1. / math.log2(rank + 2) for rank in range(min(len(relevant), args.k)))
        ndcg = dcg / ideal if ideal else 0.
        values = {"hit_rate": hit, "recall": recall, "mrr": mrr, "ndcg": ndcg}
        for name, value in values.items(): totals[name] += value
        count += 1
        print(case["question"], {name: round(value, 3) for name, value in values.items()})
    if not count: raise SystemExit("No labelled evaluation cases.")
    print(f"\nMacro averages over {count} queries (k={args.k}):")
    for name, value in totals.items(): print(f"{name}@{args.k}: {value / count:.4f}")
    print("Retrieval metrics only; they do not measure generation faithfulness or reasoning correctness.")

if __name__ == "__main__":
    main()
