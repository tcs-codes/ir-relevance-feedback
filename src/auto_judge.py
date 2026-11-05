import os
import sys
import json
import math
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import setup, tokenise, parse_topics, load_corpus

ROOT = os.path.join(os.path.dirname(__file__), "..")
POOL_PATH = os.path.join(ROOT, "runs", "pool.txt")
DOCS_PATH = os.path.join(ROOT, "corpus", "documents.jsonl")
TOPICS_PATH = os.path.join(ROOT, "topics", "topics.txt")
BM25_PATH = os.path.join(ROOT, "runs", "bm25.txt")
QRELS_DIR = os.path.join(ROOT, "qrels")
os.makedirs(QRELS_DIR, exist_ok=True)


def load_pool(path):
    pool = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                qid, doc_id = parts
                pool[qid].append(doc_id)
    return pool


def load_bm25_scores(path):
    scores = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, doc_id, rank, score, _ = parts[:6]
                scores[qid][doc_id] = float(score)
    return scores


def overlap(topic_terms, text_tokens):
    if not topic_terms:
        return 0.0
    return len(topic_terms & set(text_tokens)) / len(topic_terms)


if __name__ == "__main__":
    setup()
    docs_list = load_corpus(DOCS_PATH)
    docs_by_id = {d["doc_id"]: d for d in docs_list}
    topics = parse_topics(TOPICS_PATH)
    pool = load_pool(POOL_PATH)
    bm25_scores = load_bm25_scores(BM25_PATH)

    topic_terms = {}
    for t in topics:
        qid = t["num"]
        raw = t["title"] + " " + t.get("desc", "")
        topic_terms[qid] = set(tokenise(raw, stem=True))

    qrels = []
    for qid, doc_ids in pool.items():
        t_terms = topic_terms.get(qid, set())
        t_scores = bm25_scores.get(qid, {})
        max_s = max(t_scores.values()) if t_scores else 1.0
        min_s = min(t_scores.values()) if t_scores else 0.0
        span = max_s - min_s if max_s > min_s else 1.0

        for doc_id in doc_ids:
            doc = docs_by_id.get(doc_id)
            if doc is None:
                qrels.append((qid, doc_id, 0))
                continue

            title_toks = tokenise(doc.get("title", ""), stem=True)
            sig_title = overlap(t_terms, title_toks)

            body_snippet = doc.get("text", "")[:600]
            body_toks = tokenise(body_snippet, stem=True)
            sig_body = overlap(t_terms, body_toks)

            raw_s = t_scores.get(doc_id, 0.0)
            sig_bm = (raw_s - min_s) / span

            combined = 0.45 * sig_title + 0.35 * sig_body + 0.20 * sig_bm

            if combined >= 0.40:
                rel = 2
            elif combined >= 0.15:
                rel = 1
            else:
                rel = 0

            qrels.append((qid, doc_id, rel))

    qrels.sort(key=lambda x: (int(x[0]), x[1]))
    out_path = os.path.join(QRELS_DIR, "qrels.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, doc_id, rel in qrels:
            f.write(f"{qid} 0 {doc_id} {rel}\n")

    print(f"Generated {out_path} with {len(qrels)} judgments.")
