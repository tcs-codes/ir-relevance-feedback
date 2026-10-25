import os
import sys
import json
import glob
import csv
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(ROOT, "corpus", "documents.jsonl")
TOPICS_PATH = os.path.join(ROOT, "topics", "topics.txt")
QRELS_PATH = os.path.join(ROOT, "qrels", "qrels.txt")
RUNS_DIR = os.path.join(ROOT, "runs")
RESULTS_DIR = os.path.join(ROOT, "results")


def verify_corpus():
    if not os.path.exists(DOCS_PATH):
        return False, f"Missing {DOCS_PATH}"
    docs = []
    required_keys = {"doc_id", "source", "url", "title", "text", "word_count"}
    sources = Counter()
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                d = json.loads(line)
                missing = required_keys - set(d.keys())
                if missing:
                    return False, f"Doc line {line_no} missing keys"
                docs.append(d)
                sources[d["source"]] += 1
            except Exception as e:
                return False, f"Corpus JSON error at line {line_no}: {e}"

    if len(docs) != 125:
        return False, f"Expected 125 documents, got {len(docs)}"
    return True, "Corpus verified."


def verify_topics():
    if not os.path.exists(TOPICS_PATH):
        return False, f"Missing {TOPICS_PATH}"
    with open(TOPICS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = [b.strip() for b in content.split("<top>") if b.strip()]
    if len(blocks) != 15:
        return False, f"Expected 15 topics, got {len(blocks)}"
    for i, b in enumerate(blocks, 1):
        for tag in ["<num>", "<title>", "<desc>", "<narr>", "</top>"]:
            if tag not in b:
                return False, f"Topic block {i} missing {tag}"
    return True, "Topics verified."


def verify_qrels():
    if not os.path.exists(QRELS_PATH):
        return False, f"Missing {QRELS_PATH}"
    lines = []
    topics_in_qrels = set()
    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 4:
                return False, f"Qrels line {line_no} invalid columns"
            qid, _, _, grade = parts
            g = int(grade)
            if g not in (0, 1, 2):
                return False, f"Invalid grade {g} at line {line_no}"
            topics_in_qrels.add(qid)
            lines.append(parts)

    if len(topics_in_qrels) != 15:
        return False, f"Expected 15 topics in qrels, found {len(topics_in_qrels)}"
    return True, "Qrels verified."


def verify_runs():
    run_files = sorted(glob.glob(os.path.join(RUNS_DIR, "*.txt")))
    run_files = [r for r in run_files if "pool" not in os.path.basename(r)]

    expected_runs = [
        "bm25.txt",
        "rocchio.txt",
        "rm3.txt",
        "prf_frequency.txt",
        "prf_association.txt",
        "prf_fa.txt",
        "prf_positional.txt",
        "prf_discriminative.txt",
        "prf_fapd.txt",
        "prf_contextual.txt",
        "prf_full.txt",
        "ablation_no_F.txt",
        "ablation_no_A.txt",
        "ablation_no_P.txt",
        "ablation_no_D.txt",
        "ablation_no_C.txt",
    ]

    found_names = [os.path.basename(r) for r in run_files]
    for r in expected_runs:
        if r not in found_names:
            return False, f"Missing run file: {r}"

    for rf in run_files:
        name = os.path.basename(rf)
        entries = defaultdict(list)
        with open(rf, "r", encoding="utf-8") as f:
            for l_idx, line in enumerate(f, 1):
                p = line.strip().split()
                if len(p) != 6:
                    return False, f"{name}:{l_idx} format error"
                qid, _, _, rank, score, _ = p
                entries[qid].append((int(rank), float(score)))

        if len(entries) != 15:
            return False, f"{name} covers only {len(entries)} topics"

        for qid, items in entries.items():
            ranks = [it[0] for it in items]
            scores = [it[1] for it in items]
            if ranks != list(range(1, len(ranks) + 1)):
                return False, f"{name} topic {qid} non-consecutive ranks"
            for k in range(len(scores) - 1):
                if scores[k] < scores[k + 1] - 1e-9:
                    return False, f"{name} topic {qid} non-descending score"

    return True, "Run files verified."


def verify_eval_metrics():
    csv_path = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    if not os.path.exists(csv_path):
        return False, f"Missing {csv_path}"
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if len(rows) < 10:
        return False, "Evaluation summary has insufficient rows"
    return True, "Evaluation metrics verified."


if __name__ == "__main__":
    tests = [
        ("Corpus Integrity", verify_corpus),
        ("Topics Integrity", verify_topics),
        ("Frozen Qrels", verify_qrels),
        ("TREC Runs", verify_runs),
        ("Evaluation Summary", verify_eval_metrics),
    ]

    all_passed = True
    for name, test_fn in tests:
        passed, msg = test_fn()
        if not passed:
            print(f"[FAIL] {name}: {msg}")
            all_passed = False
            break

    if all_passed:
        print("All verification checks passed.")
    else:
        sys.exit(1)
