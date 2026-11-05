import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import setup, parse_topics, load_corpus
from retrieve_baselines import write_run
from evaluate import load_qrels, evaluate_run

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS_PATH = os.path.join(ROOT, "corpus", "documents.jsonl")
TOPICS_PATH = os.path.join(ROOT, "topics", "topics.txt")
RUNS_DIR = os.path.join(ROOT, "runs")
RESULTS_DIR = os.path.join(ROOT, "results")
QRELS_PATH = os.path.join(ROOT, "qrels", "qrels.txt")
RUN_OUT = os.path.join(RUNS_DIR, "dense_faiss.txt")


def run_dense_retrieval():
    setup()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    docs = load_corpus(DOCS_PATH)
    topics = parse_topics(TOPICS_PATH)

    doc_ids = [d["doc_id"] for d in docs]
    doc_texts = [(d.get("title", "") + ". " + d.get("text", "")).strip() for d in docs]

    doc_embeddings = model.encode(doc_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    doc_embeddings = np.ascontiguousarray(doc_embeddings, dtype=np.float32)

    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)

    results = {}
    for t in topics:
        qid = t["num"]
        q_text = (t["title"] + " " + t["desc"]).strip()
        q_emb = model.encode([q_text], normalize_embeddings=True)
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)

        scores, indices = index.search(q_emb, k=len(doc_ids))
        scores = scores[0]
        indices = indices[0]

        ranked = [(doc_ids[idx], float(scores[rank_idx])) for rank_idx, idx in enumerate(indices) if idx >= 0]
        results[qid] = ranked

    write_run(RUN_OUT, "DENSE_FAISS", results)

    if os.path.exists(QRELS_PATH):
        qrels = load_qrels(QRELS_PATH)
        summary = evaluate_run(RUN_OUT, qrels)
        print(f"Dense FAISS: MAP={summary['MAP']:.4f} nDCG@10={summary['nDCG@10']:.4f} P@10={summary['P@10']:.4f}")
        return summary
    return None


if __name__ == "__main__":
    run_dense_retrieval()
