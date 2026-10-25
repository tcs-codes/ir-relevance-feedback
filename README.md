# Learning from Searchers: Relevance Feedback and Query Expansion

This repository contains the complete two-stage pseudo-relevance feedback (PRF) and query expansion pipeline evaluated on a 125-document Indian news collection.

## Structure

```
corpus/
  documents.jsonl          # 125 documents (25 per source across 5 news outlets)
  collection_meta.json     # Metadata and collection counts
topics/
  topics.txt               # 15 TREC-format topics (301-315)
qrels/
  qrels.txt                # Frozen relevance judgments (487 entries)
runs/
  bm25.txt                 # Baseline first-pass retrieval
  rocchio.txt              # Rocchio PRF baseline
  rm3.txt                  # RM3 PRF baseline
  prf_frequency.txt        # Frequency signal (F)
  prf_association.txt      # Query-term association signal (A)
  prf_fa.txt               # Combined F + A
  prf_positional.txt       # Positional proximity signal (P)
  prf_discriminative.txt   # Pseudo-irrelevant discrimination signal (D)
  prf_fapd.txt             # 4 Lexical signals (F + A + P + D)
  prf_contextual.txt       # Contextual passage relevance (C)
  prf_full.txt             # Proposed 5-signal fusion model
  ablation_*.txt           # Leave-one-out ablation runs
results/
  evaluation_summary.csv   # MAP, nDCG@10, and P@10 for all 16 runs
  per_query_results.csv    # Per-topic AP, nDCG, and P@10 breakdown
  statistical_tests.csv    # Paired t-test statistics and p-values
  expansion_terms_analysis.csv # Expansion terms selected per topic
src/
  preprocess.py            # Tokenization, stopword removal, and Porter stemming
  retrieve_baselines.py    # BM25, Rocchio, and RM3 retrieval engines
  auto_judge.py            # Generates ground truth judgments
  feedback_signals.py      # Core functions for F, A, P, D, and C signals
  run_experiments.py       # Full PRF pipeline, ablation runs, and stats
  evaluate.py              # MAP, nDCG@10, and P@10 calculator
  verify_all.py            # Pipeline verification script
```

## Setup

```bash
pip install nltk sentence-transformers numpy scipy
```

## Usage

1. Build baseline retrieval runs (BM25, Rocchio, RM3):
   ```bash
   python src/retrieve_baselines.py
   ```

2. Run the PRF experiments, ablations, and statistical tests:
   ```bash
   python src/run_experiments.py
   ```

3. Evaluate runs against qrels:
   ```bash
   python src/evaluate.py
   ```

4. Verify repository integrity:
   ```bash
   python src/verify_all.py
   ```

## Results

| Model / System | Signals Used | MAP | nDCG@10 | P@10 |
| :--- | :--- | :---: | :---: | :---: |
| **BM25** | Unexpanded Query | 0.9445 | 0.9630 | 0.5600 |
| **Ablation (No F)** | A + P + D + C | 0.9214 | 0.9514 | 0.5400 |
| **Proposed Lexical (FAPD)** | F + A + P + D | 0.9153 | 0.9544 | 0.5467 |
| **Ablation (No D)** | F + A + P + C | 0.9101 | 0.9427 | 0.5400 |
| **PRF Positional** | P alone | 0.9078 | 0.9371 | 0.5333 |
| **PRF Frequency** | F alone | 0.9078 | 0.9455 | 0.5400 |
| **PRF F+A** | F + A | 0.9042 | 0.9449 | 0.5400 |
| **Ablation (No P)** | F + A + D + C | 0.8981 | 0.9476 | 0.5467 |
| **PRF Contextual** | C alone | 0.8878 | 0.9295 | 0.5133 |
| **PRF Association** | A alone | 0.8864 | 0.9491 | 0.5467 |
| **Full Fusion** | 0.25F + 0.20A + 0.20P + 0.20D + 0.15C | 0.8742 | 0.9261 | 0.5267 |
| **RM3** | Top 10 Docs, 20 terms | 0.8737 | 0.9291 | 0.5267 |
| **Ablation (No A)** | F + P + D + C | 0.8715 | 0.9264 | 0.5267 |
| **PRF Discriminative**| D alone | 0.8697 | 0.9335 | 0.5400 |
| **Rocchio** | Top 10 positive, ranks 11-20 negative | 0.5233 | 0.5929 | 0.4733 |
