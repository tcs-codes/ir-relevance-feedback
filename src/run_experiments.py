import os
import sys
import csv
from collections import defaultdict
from scipy import stats
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import setup, tokenise, tokenise_query, parse_topics, load_corpus
from retrieve_baselines import BM25Index, write_run, N_PR, N_PI
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
from evaluate import load_qrels, evaluate_run

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS_PATH = os.path.join(ROOT, "corpus", "documents.jsonl")
TOPICS_PATH = os.path.join(ROOT, "topics", "topics.txt")
RUNS_DIR = os.path.join(ROOT, "runs")
RESULTS_DIR = os.path.join(ROOT, "results")
BM25_RUN_PATH = os.path.join(RUNS_DIR, "bm25.txt")
QRELS_PATH = os.path.join(ROOT, "qrels", "qrels.txt")


def load_first_pass_ranking(path):
    ranking = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, doc_id, rank, score, _ = parts[:6]
                ranking[qid].append(doc_id)
    return ranking


def run_pipeline():
    setup()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    docs = load_corpus(DOCS_PATH)
    docs_by_id = {d["doc_id"]: d for d in docs}
    index = BM25Index(docs)
    topics = parse_topics(TOPICS_PATH)
    first_pass = load_first_pass_ranking(BM25_RUN_PATH)

    configs = {
        "prf_frequency": {"F": 1.0},
        "prf_association": {"A": 1.0},
        "prf_fa": {"F": 0.55, "A": 0.45},
        "prf_positional": {"P": 1.0},
        "prf_discriminative": {"D": 1.0},
        "prf_fapd": {"F": 0.294, "A": 0.235, "P": 0.235, "D": 0.235},
        "prf_contextual": {"C": 1.0},
        "prf_full": {"F": 0.25, "A": 0.20, "P": 0.20, "D": 0.20, "C": 0.15},
        "ablation_no_F": {"A": 0.20, "P": 0.20, "D": 0.20, "C": 0.15},
        "ablation_no_A": {"F": 0.25, "P": 0.20, "D": 0.20, "C": 0.15},
        "ablation_no_P": {"F": 0.25, "A": 0.20, "D": 0.20, "C": 0.15},
        "ablation_no_D": {"F": 0.25, "A": 0.20, "P": 0.20, "C": 0.15},
        "ablation_no_C": {"F": 0.25, "A": 0.20, "P": 0.20, "D": 0.20},
    }

    run_results = {cfg_name: {} for cfg_name in configs}
    expansion_terms_log = {cfg_name: {} for cfg_name in configs}

    for t in topics:
        qid = t["num"]
        query_text = t["title"] + " " + t["desc"]
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

        signals_map = {
            "F": norm_f,
            "A": norm_a,
            "P": norm_p,
            "D": norm_d,
            "C": norm_c
        }

        for cfg_name, w_dict in configs.items():
            combined_scores = combine_signals(candidates, signals_map, w_dict)
            top_terms = sorted(combined_scores.items(), key=lambda x: -x[1])[:TOP_EXPANSION_TERMS]
            expansion_terms_log[cfg_name][qid] = [k for k, _ in top_terms]

            exp_query = expand_query(qtoks, top_terms)
            hits = index.retrieve(exp_query)
            run_results[cfg_name][qid] = [(index.doc_ids[i], s) for i, s in hits]

    for cfg_name, res in run_results.items():
        out_path = os.path.join(RUNS_DIR, f"{cfg_name}.txt")
        write_run(out_path, cfg_name.upper(), res)

    terms_csv = os.path.join(RESULTS_DIR, "expansion_terms_analysis.csv")
    with open(terms_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "title", "full_top10_terms", "fapd_top10_terms"])
        for t in topics:
            qid = t["num"]
            writer.writerow([
                qid,
                t["title"],
                ", ".join(expansion_terms_log["prf_full"][qid]),
                ", ".join(expansion_terms_log["prf_fapd"][qid])
            ])

    run_eval_and_stats()


def run_eval_and_stats():
    qrels = load_qrels(QRELS_PATH)
    all_runs = [
        "bm25",
        "rocchio",
        "rm3",
        "prf_frequency",
        "prf_association",
        "prf_fa",
        "prf_positional",
        "prf_discriminative",
        "prf_fapd",
        "prf_contextual",
        "prf_full",
        "ablation_no_F",
        "ablation_no_A",
        "ablation_no_P",
        "ablation_no_D",
        "ablation_no_C"
    ]

    eval_data = {}
    summaries = []
    for rname in all_runs:
        rpath = os.path.join(RUNS_DIR, f"{rname}.txt")
        s = evaluate_run(rpath, qrels)
        eval_data[rname] = s
        summaries.append(s)

    eval_csv = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "MAP", "nDCG@10", "P@10", "n_topics"])
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s[k] for k in ["run", "MAP", "nDCG@10", "P@10", "n_topics"]})

    pq_csv = os.path.join(RESULTS_DIR, "per_query_results.csv")
    with open(pq_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "qid", "AP", "nDCG@10", "P@10"])
        for s in summaries:
            for qid, m in sorted(s["per_query"].items(), key=lambda x: int(x[0])):
                writer.writerow([s["run"], qid, f"{m['AP']:.4f}", f"{m['nDCG@10']:.4f}", f"{m['P@10']:.4f}"])

    full_aps = [eval_data["prf_full"]["per_query"][qid]["AP"] for qid in sorted(eval_data["prf_full"]["per_query"], key=int)]

    stat_results = []
    for rname, s in eval_data.items():
        comp_aps = [s["per_query"][qid]["AP"] for qid in sorted(s["per_query"], key=int)]
        t_stat, p_val = stats.ttest_rel(comp_aps, full_aps)
        sig = "*" if p_val < 0.05 else "ns"
        stat_results.append({
            "system": rname,
            "MAP": f"{s['MAP']:.4f}",
            "nDCG@10": f"{s['nDCG@10']:.4f}",
            "P@10": f"{s['P@10']:.4f}",
            "t_stat": f"{t_stat:.4f}",
            "p_val": f"{p_val:.4f}",
            "significance": sig
        })

    stat_csv = os.path.join(RESULTS_DIR, "statistical_tests.csv")
    with open(stat_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "MAP", "nDCG@10", "P@10", "t_stat", "p_val", "significance"])
        writer.writeheader()
        writer.writerows(stat_results)


if __name__ == "__main__":
    run_pipeline()
