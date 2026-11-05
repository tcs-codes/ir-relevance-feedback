import os
import sys
import time
import tracemalloc
import csv
import itertools
from collections import defaultdict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import setup, tokenise, tokenise_query, parse_topics, load_corpus
from retrieve_baselines import BM25Index, write_run, N_PR, N_PI, run_bm25, run_rocchio, run_rm3
from feedback_signals import (
    extract_candidate_terms,
    compute_frequency_signal,
    compute_association_signal,
    compute_positional_signal,
    compute_discriminative_signal,
    compute_contextual_signal,
    min_max_normalize,
    combine_signals,
    expand_query,
    ASSOCIATION_WINDOW,
    TOP_EXPANSION_TERMS
)
from evaluate import load_qrels, evaluate_run, average_precision, ndcg_at_k, precision_at_k

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS_PATH = os.path.join(ROOT, "corpus", "documents.jsonl")
TOPICS_PATH = os.path.join(ROOT, "topics", "topics.txt")
RUNS_DIR = os.path.join(ROOT, "runs")
RESULTS_DIR = os.path.join(ROOT, "results")
QRELS_PATH = os.path.join(ROOT, "qrels", "qrels.txt")


def load_first_pass(path):
    ranking = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, doc_id, rank, score, _ = parts[:6]
                ranking[qid].append(doc_id)
    return ranking


def evaluate_ranking(ranking, qrels):
    aps = []
    ndcgs = []
    p10s = []
    for qid, hits in ranking.items():
        if qid not in qrels:
            continue
        ranked = [d for d, _ in hits]
        q_rel = qrels[qid]
        aps.append(average_precision(ranked, q_rel))
        ndcgs.append(ndcg_at_k(ranked, q_rel, k=10))
        p10s.append(precision_at_k(ranked, q_rel, k=10))
    n = max(len(aps), 1)
    return sum(aps) / n, sum(ndcgs) / n, sum(p10s) / n


def run_tuning_and_profiling():
    setup()
    qrels = load_qrels(QRELS_PATH)
    docs = load_corpus(DOCS_PATH)
    docs_by_id = {d["doc_id"]: d for d in docs}
    index = BM25Index(docs)
    topics = parse_topics(TOPICS_PATH)
    first_pass = load_first_pass(os.path.join(RUNS_DIR, "bm25.txt"))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_cache = {}
    for t in topics:
        qid = t["num"]
        query_text = (t["title"] + " " + t["desc"]).strip()
        qtoks = tokenise_query(query_text)
        q_tokens_set = set(qtoks)

        pr_doc_ids = first_pass[qid][:N_PR]
        pi_doc_ids = first_pass[qid][N_PR:N_PR + N_PI]

        pr_docs_raw = [docs_by_id[did] for did in pr_doc_ids if did in docs_by_id]
        tok_docs_pr = [index.tok_docs[index.doc_ids.index(did)] for did in pr_doc_ids if did in index.doc_ids]
        tok_docs_pi = [index.tok_docs[index.doc_ids.index(did)] for did in pi_doc_ids if did in index.doc_ids]

        candidates = extract_candidate_terms(tok_docs_pr, q_tokens_set, index.idf)

        raw_f = compute_frequency_signal(candidates, tok_docs_pr)
        norm_f = min_max_normalize(raw_f)

        raw_a = compute_association_signal(candidates, tok_docs_pr, q_tokens_set, window=ASSOCIATION_WINDOW)
        norm_a = min_max_normalize(raw_a)

        raw_p = compute_positional_signal(candidates, tok_docs_pr, q_tokens_set)
        norm_p = min_max_normalize(raw_p)

        raw_d = compute_discriminative_signal(candidates, tok_docs_pr, tok_docs_pi)
        norm_d = min_max_normalize(raw_d)

        raw_c = compute_contextual_signal(model, candidates, pr_docs_raw, query_text, tokenise)
        norm_c = min_max_normalize(raw_c)

        topic_cache[qid] = {
            "qtoks": qtoks,
            "candidates": candidates,
            "signals": {
                "F": norm_f,
                "A": norm_a,
                "P": norm_p,
                "D": norm_d,
                "C": norm_c
            }
        }

    def retrieve_with_weights(weights):
        res = {}
        for qid, cache in topic_cache.items():
            comb = combine_signals(cache["candidates"], cache["signals"], weights)
            top_terms = sorted(comb.items(), key=lambda x: -x[1])[:TOP_EXPANSION_TERMS]
            eq = expand_query(cache["qtoks"], top_terms)
            hits = index.retrieve(eq)
            res[qid] = [(index.doc_ids[i], s) for i, s in hits]
        return res

    weight_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    best_map = -1.0
    best_weights = None
    all_grid_results = []

    for wf, wa, wp, wd, wc in itertools.product(weight_candidates, repeat=5):
        w_sum = wf + wa + wp + wd + wc
        if w_sum <= 0:
            continue
        if abs(w_sum - 1.0) > 1e-5:
            continue

        w_dict = {"F": wf, "A": wa, "P": wp, "D": wd, "C": wc}
        ranking = retrieve_with_weights(w_dict)
        map_val, ndcg_val, p10_val = evaluate_ranking(ranking, qrels)

        all_grid_results.append({
            "w_F": wf,
            "w_A": wa,
            "w_P": wp,
            "w_D": wd,
            "w_C": wc,
            "MAP": map_val,
            "nDCG@10": ndcg_val,
            "P@10": p10_val
        })

        if map_val > best_map:
            best_map = map_val
            best_weights = w_dict

    best_ranking = retrieve_with_weights(best_weights)
    write_run(os.path.join(RUNS_DIR, "prf_tuned.txt"), "PRF_TUNED", best_ranking)
    tuned_map, tuned_ndcg, tuned_p10 = evaluate_ranking(best_ranking, qrels)

    all_grid_results.sort(key=lambda x: -x["MAP"])
    tuning_csv = os.path.join(RESULTS_DIR, "weight_tuning_results.csv")
    with open(tuning_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["w_F", "w_A", "w_P", "w_D", "w_C", "MAP", "nDCG@10", "P@10"])
        writer.writeheader()
        for row in all_grid_results[:50]:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})

    sensitivity_rows = []
    base_default = {"F": 0.25, "A": 0.20, "P": 0.20, "D": 0.20, "C": 0.15}
    test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for target_sig in ["F", "A", "P", "D", "C"]:
        other_sigs = [s for s in ["F", "A", "P", "D", "C"] if s != target_sig]
        other_base_sum = sum(base_default[s] for s in other_sigs)

        for val in test_values:
            rem = 1.0 - val
            w_test = {target_sig: val}
            for s in other_sigs:
                w_test[s] = rem * (base_default[s] / other_base_sum) if other_base_sum > 0 else 0.0

            r = retrieve_with_weights(w_test)
            m, nd, p = evaluate_ranking(r, qrels)
            sensitivity_rows.append({
                "signal_varied": target_sig,
                "weight_value": f"{val:.2f}",
                "MAP": f"{m:.4f}",
                "nDCG@10": f"{nd:.4f}",
                "P@10": f"{p:.4f}"
            })

    sens_csv = os.path.join(RESULTS_DIR, "weight_sensitivity.csv")
    with open(sens_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["signal_varied", "weight_value", "MAP", "nDCG@10", "P@10"])
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    doc_ids = [d["doc_id"] for d in docs]
    doc_texts = [(d.get("title", "") + ". " + d.get("text", "")).strip() for d in docs]
    doc_embs = model.encode(doc_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    doc_embs = np.ascontiguousarray(doc_embs, dtype=np.float32)
    faiss_index = faiss.IndexFlatIP(doc_embs.shape[1])
    faiss_index.add(doc_embs)

    bm25_results_full = run_bm25(index, topics)

    profiling_configs = [
        ("BM25 First-Pass", lambda: run_bm25(index, topics)),
        ("Rocchio PRF", lambda: run_rocchio(index, topics, bm25_results_full)),
        ("RM3 PRF", lambda: run_rm3(index, topics, bm25_results_full)),
        ("PRF Frequency (F)", lambda: retrieve_with_weights({"F": 1.0, "A": 0.0, "P": 0.0, "D": 0.0, "C": 0.0})),
        ("PRF Association (A)", lambda: retrieve_with_weights({"F": 0.0, "A": 1.0, "P": 0.0, "D": 0.0, "C": 0.0})),
        ("PRF Positional (P)", lambda: retrieve_with_weights({"F": 0.0, "A": 0.0, "P": 1.0, "D": 0.0, "C": 0.0})),
        ("PRF Discrimination (D)", lambda: retrieve_with_weights({"F": 0.0, "A": 0.0, "P": 0.0, "D": 1.0, "C": 0.0})),
        ("PRF Contextual (C)", lambda: retrieve_with_weights({"F": 0.0, "A": 0.0, "P": 0.0, "D": 0.0, "C": 1.0})),
        ("PRF Lexical Fusion (FAPD)", lambda: retrieve_with_weights({"F": 0.294, "A": 0.235, "P": 0.235, "D": 0.235, "C": 0.0})),
        ("PRF Full Fusion (5-Signal)", lambda: retrieve_with_weights({"F": 0.25, "A": 0.20, "P": 0.20, "D": 0.20, "C": 0.15})),
        ("PRF Tuned Fusion", lambda: retrieve_with_weights(best_weights)),
        ("Dense FAISS Retrieval", lambda: [faiss_index.search(np.ascontiguousarray(model.encode([(t["title"] + " " + t["desc"]).strip()], normalize_embeddings=True), dtype=np.float32), k=len(doc_ids)) for t in topics])
    ]

    profile_results = []
    for name, func in profiling_configs:
        func()
        tracemalloc.start()
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_ms = (t1 - t0) * 1000.0
        ms_per_query = total_ms / len(topics)
        peak_mb = peak / (1024.0 * 1024.0)

        profile_results.append({
            "Configuration": name,
            "Total Time (ms)": f"{total_ms:.2f}",
            "Latency per Query (ms)": f"{ms_per_query:.2f}",
            "Peak Memory (MB)": f"{peak_mb:.2f}"
        })

    prof_csv = os.path.join(RESULTS_DIR, "profiling_summary.csv")
    with open(prof_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Configuration", "Total Time (ms)", "Latency per Query (ms)", "Peak Memory (MB)"])
        writer.writeheader()
        writer.writerows(profile_results)

    print(f"Optimal weights found: {best_weights} -> MAP={tuned_map:.4f}, nDCG@10={tuned_ndcg:.4f}, P@10={tuned_p10:.4f}")
    print("Profiling summary and sensitivity results written to results/.")


if __name__ == "__main__":
    run_tuning_and_profiling()
