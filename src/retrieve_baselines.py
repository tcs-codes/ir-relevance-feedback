import os
import sys
import json
import math
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import setup, tokenise, tokenise_query, parse_topics, load_corpus

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS_PATH = os.path.join(ROOT, "corpus", "documents.jsonl")
TOPICS_PATH = os.path.join(ROOT, "topics", "topics.txt")
RUNS_DIR = os.path.join(ROOT, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

K1 = 1.5
B = 0.75
N_PR = 10
N_PI = 10
TOP_K = 1000

ALPHA = 1.0
BETA = 0.75
GAMMA = 0.15

RM3_TERMS = 20
RM3_LAMBDA = 0.5


class BM25Index:
    def __init__(self, docs):
        self.docs = docs
        self.N = len(docs)
        self.doc_ids = [d["doc_id"] for d in docs]

        self.tok_docs = []
        self.doc_len = []
        for d in docs:
            toks = tokenise(d["title"] + " " + d["text"])
            self.tok_docs.append(toks)
            self.doc_len.append(len(toks))

        self.avgdl = sum(self.doc_len) / self.N

        self.inv = defaultdict(dict)
        for i, toks in enumerate(self.tok_docs):
            for t, cnt in Counter(toks).items():
                self.inv[t][i] = cnt

        self.idf = {}
        for t, postings in self.inv.items():
            df = len(postings)
            self.idf[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens, doc_idx):
        s = 0.0
        dl = self.doc_len[doc_idx]
        for t in set(query_tokens):
            if t not in self.inv or doc_idx not in self.inv[t]:
                continue
            tf = self.inv[t][doc_idx]
            idf = self.idf.get(t, 0.0)
            numer = tf * (K1 + 1)
            denom = tf + K1 * (1 - B + B * dl / self.avgdl)
            s += idf * numer / denom
        return s

    def retrieve(self, query_tokens, top_n=TOP_K):
        candidates = set()
        for t in query_tokens:
            candidates.update(self.inv.get(t, {}).keys())
        scored = [(i, self.score(query_tokens, i)) for i in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]


def write_run(path, run_id, results):
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(results, key=int):
            for rank, (doc_id, score) in enumerate(results[qid], 1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_id}\n")


def run_bm25(index, topics):
    results = {}
    for t in topics:
        qid = t["num"]
        qtoks = tokenise_query(t["title"] + " " + t["desc"])
        hits = index.retrieve(qtoks)
        results[qid] = [(index.doc_ids[i], s) for i, s in hits]
    return results


def tfidf_vec(tok_doc, index):
    tf_map = Counter(tok_doc)
    n = max(len(tok_doc), 1)
    vec = {}
    for t, tf in tf_map.items():
        vec[t] = (tf / n) * index.idf.get(t, 0.0)
    return vec


def run_rocchio(index, topics, bm25_results):
    results = {}
    for t in topics:
        qid = t["num"]
        qtoks = tokenise_query(t["title"] + " " + t["desc"])
        q_vec = {tok: 1.0 for tok in set(qtoks)}

        hits = bm25_results[qid]
        pr_hits = hits[:N_PR]
        pi_hits = hits[N_PR:N_PR + N_PI]

        pr_centroid = defaultdict(float)
        for doc_id, _ in pr_hits:
            idx = index.doc_ids.index(doc_id)
            for term, w in tfidf_vec(index.tok_docs[idx], index).items():
                pr_centroid[term] += w / N_PR

        pi_centroid = defaultdict(float)
        for doc_id, _ in pi_hits:
            idx = index.doc_ids.index(doc_id)
            for term, w in tfidf_vec(index.tok_docs[idx], index).items():
                pi_centroid[term] += w / N_PI

        new_q = defaultdict(float)
        for term, w in q_vec.items():
            new_q[term] += ALPHA * w
        for term, w in pr_centroid.items():
            new_q[term] += BETA * w
        for term, w in pi_centroid.items():
            new_q[term] -= GAMMA * w

        expanded_toks = []
        for term, w in new_q.items():
            if w > 0:
                expanded_toks.extend([term] * max(1, round(w * 10)))

        hits2 = index.retrieve(expanded_toks)
        results[qid] = [(index.doc_ids[i], s) for i, s in hits2]
    return results


def run_rm3(index, topics, bm25_results):
    results = {}
    for t in topics:
        qid = t["num"]
        qtoks = tokenise_query(t["title"] + " " + t["desc"])

        hits = bm25_results[qid]
        pr_hits = hits[:N_PR]

        scores = [s for _, s in pr_hits]
        min_s, max_s = min(scores), max(scores)
        span = max_s - min_s if max_s > min_s else 1.0
        p_d_q = [(doc_id, (s - min_s) / span + 0.01) for doc_id, s in pr_hits]
        z = sum(w for _, w in p_d_q)
        p_d_q = [(doc_id, w / z) for doc_id, w in p_d_q]

        p_t_r = defaultdict(float)
        for doc_id, p_dq in p_d_q:
            idx = index.doc_ids.index(doc_id)
            toks = index.tok_docs[idx]
            tf_map = Counter(toks)
            dl = max(len(toks), 1)
            for term, tf in tf_map.items():
                p_t_r[term] += (tf / dl) * p_dq

        q_set = set(qtoks)
        rm3_terms = sorted(
            [(t, p) for t, p in p_t_r.items() if t not in q_set],
            key=lambda x: -x[1]
        )[:RM3_TERMS]

        p_t_q = {}
        for term in set(qtoks):
            p_t_q[term] = 1.0 / max(len(set(qtoks)), 1)

        interp = defaultdict(float)
        for term, p in p_t_q.items():
            interp[term] += (1 - RM3_LAMBDA) * p
        for term, p in rm3_terms:
            interp[term] += RM3_LAMBDA * p

        expanded_toks = []
        for term, w in interp.items():
            expanded_toks.extend([term] * max(1, round(w * 100)))

        hits2 = index.retrieve(expanded_toks)
        results[qid] = [(index.doc_ids[i], s) for i, s in hits2]
    return results


def build_pool(all_runs, top_n=20):
    pool = defaultdict(set)
    for run_results in all_runs.values():
        for qid, hits in run_results.items():
            for doc_id, _ in hits[:top_n]:
                pool[qid].add(doc_id)
    return pool


def write_pool(path, pool):
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(pool, key=int):
            for doc_id in sorted(pool[qid]):
                f.write(f"{qid} {doc_id}\n")


if __name__ == "__main__":
    setup()
    docs = load_corpus(DOCS_PATH)
    topics = parse_topics(TOPICS_PATH)

    index = BM25Index(docs)
    bm25_res = run_bm25(index, topics)
    write_run(os.path.join(RUNS_DIR, "bm25.txt"), "BM25", bm25_res)

    rocchio_res = run_rocchio(index, topics, bm25_res)
    write_run(os.path.join(RUNS_DIR, "rocchio.txt"), "Rocchio", rocchio_res)

    rm3_res = run_rm3(index, topics, bm25_res)
    write_run(os.path.join(RUNS_DIR, "rm3.txt"), "RM3", rm3_res)

    pool = build_pool({"bm25": bm25_res, "rocchio": rocchio_res, "rm3": rm3_res})
    write_pool(os.path.join(RUNS_DIR, "pool.txt"), pool)
    print("Baseline retrieval runs generated.")
