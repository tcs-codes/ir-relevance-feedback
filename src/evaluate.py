import os
import sys
import math
import glob
import csv
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
QRELS_PATH = os.path.join(ROOT, "qrels", "qrels.txt")
RUNS_DIR = os.path.join(ROOT, "runs")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, doc_id, grade = parts[:4]
                qrels[qid][doc_id] = int(grade)
    return qrels


def load_run(path):
    run = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, doc_id, rank, score, _ = parts[:6]
                run[qid].append((doc_id, float(score)))
    return run


def binary_rel(grade):
    return 1 if grade >= 1 else 0


def average_precision(ranked_docs, qrels_q):
    n_rel = sum(1 for g in qrels_q.values() if binary_rel(g))
    if n_rel == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for k, doc_id in enumerate(ranked_docs, 1):
        rel = binary_rel(qrels_q.get(doc_id, 0))
        if rel:
            hits += 1
            ap += hits / k
    return ap / n_rel


def ndcg_at_k(ranked_docs, qrels_q, k=10):
    def dcg(rels):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

    actual_rels = [qrels_q.get(doc_id, 0) for doc_id in ranked_docs[:k]]
    ideal_rels = sorted(qrels_q.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_rels)
    if ideal_dcg == 0:
        return 0.0
    return dcg(actual_rels) / ideal_dcg


def precision_at_k(ranked_docs, qrels_q, k=10):
    hits = sum(binary_rel(qrels_q.get(d, 0)) for d in ranked_docs[:k])
    return hits / k


def evaluate_run(run_path, qrels):
    run = load_run(run_path)
    run_name = os.path.splitext(os.path.basename(run_path))[0]

    per_query = {}
    for qid, hits in run.items():
        if qid not in qrels:
            continue
        ranked = [doc_id for doc_id, _ in hits]
        qrels_q = qrels[qid]
        per_query[qid] = {
            "AP": average_precision(ranked, qrels_q),
            "nDCG@10": ndcg_at_k(ranked, qrels_q, k=10),
            "P@10": precision_at_k(ranked, qrels_q, k=10),
        }

    n = max(len(per_query), 1)
    summary = {
        "run": run_name,
        "MAP": sum(v["AP"] for v in per_query.values()) / n,
        "nDCG@10": sum(v["nDCG@10"] for v in per_query.values()) / n,
        "P@10": sum(v["P@10"] for v in per_query.values()) / n,
        "n_topics": len(per_query),
        "per_query": per_query,
    }
    return summary


def classify_queries(bm25_pq, other_pq, metric="AP", threshold=0.05):
    improved = 0
    degraded = 0
    unchanged = 0
    for qid in bm25_pq:
        if qid not in other_pq:
            continue
        delta = other_pq[qid][metric] - bm25_pq[qid][metric]
        if delta > threshold:
            improved += 1
        elif delta < -threshold:
            degraded += 1
        else:
            unchanged += 1
    return {"improved": improved, "unchanged": unchanged, "degraded": degraded}


if __name__ == "__main__":
    if not os.path.exists(QRELS_PATH):
        sys.exit(1)

    qrels = load_qrels(QRELS_PATH)

    if len(sys.argv) > 1:
        run_files = sys.argv[1:]
    else:
        run_files = sorted(glob.glob(os.path.join(RUNS_DIR, "*.txt")))
        run_files = [r for r in run_files if "pool" not in r]

    summaries = []
    for rf in run_files:
        s = evaluate_run(rf, qrels)
        summaries.append(s)
        print(f"[{s['run']:20s}] MAP={s['MAP']:.4f} nDCG@10={s['nDCG@10']:.4f} P@10={s['P@10']:.4f}")

    csv_path = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "MAP", "nDCG@10", "P@10", "n_topics"])
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s[k] for k in ["run", "MAP", "nDCG@10", "P@10", "n_topics"]})

    pq_path = os.path.join(RESULTS_DIR, "per_query_results.csv")
    with open(pq_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "qid", "AP", "nDCG@10", "P@10"])
        for s in summaries:
            for qid, m in sorted(s["per_query"].items(), key=lambda x: int(x[0])):
                writer.writerow([s["run"], qid, f"{m['AP']:.4f}", f"{m['nDCG@10']:.4f}", f"{m['P@10']:.4f}"])
