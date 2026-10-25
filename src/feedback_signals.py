import re
import numpy as np
from sentence_transformers import SentenceTransformer

ASSOCIATION_WINDOW = 20
TOP_EXPANSION_TERMS = 10


def extract_candidate_terms(tok_docs_pr, q_tokens, idf_map):
    candidates = set()
    for toks in tok_docs_pr:
        for t in set(toks):
            if t not in q_tokens and len(t) > 2 and idf_map.get(t, 0.0) > 0.1:
                candidates.add(t)
    return candidates


def compute_frequency_signal(candidates, tok_docs_pr):
    n_docs = len(tok_docs_pr)
    if n_docs == 0:
        return {t: 0.0 for t in candidates}
    f_scores = {}
    for t in candidates:
        doc_count = sum(1 for doc in tok_docs_pr if t in doc)
        f_scores[t] = doc_count / float(n_docs)
    return f_scores


def compute_association_signal(candidates, tok_docs_pr, q_tokens, window=ASSOCIATION_WINDOW):
    a_scores = {}
    for t in candidates:
        docs_with_t = 0
        docs_with_assoc = 0
        for doc in tok_docs_pr:
            if t not in doc:
                continue
            docs_with_t += 1
            t_positions = [i for i, tok in enumerate(doc) if tok == t]
            q_positions = [i for i, tok in enumerate(doc) if tok in q_tokens]

            associated = False
            for tp in t_positions:
                for qp in q_positions:
                    if abs(tp - qp) <= window:
                        associated = True
                        break
                if associated:
                    break
            if associated:
                docs_with_assoc += 1

        a_scores[t] = (docs_with_assoc / float(docs_with_t)) if docs_with_t > 0 else 0.0
    return a_scores


def compute_positional_signal(candidates, tok_docs_pr, q_tokens):
    p_scores = {}
    for t in candidates:
        p_local_vals = []
        for doc in tok_docs_pr:
            t_positions = [i for i, tok in enumerate(doc) if tok == t]
            if not t_positions:
                continue
            q_positions = [i for i, tok in enumerate(doc) if tok in q_tokens]
            if not q_positions:
                for _ in t_positions:
                    p_local_vals.append(0.0)
                continue

            for tp in t_positions:
                min_dist = min(abs(tp - qp) for qp in q_positions)
                p_local = 1.0 / (1.0 + min_dist)
                p_local_vals.append(p_local)

        p_scores[t] = (sum(p_local_vals) / len(p_local_vals)) if p_local_vals else 0.0
    return p_scores


def compute_discriminative_signal(candidates, tok_docs_pr, tok_docs_pi):
    n_pr = float(len(tok_docs_pr)) if tok_docs_pr else 1.0
    n_pi = float(len(tok_docs_pi)) if tok_docs_pi else 1.0

    d_scores = {}
    for t in candidates:
        p_pr = sum(1 for d in tok_docs_pr if t in d) / n_pr
        p_pi = sum(1 for d in tok_docs_pi if t in d) / n_pi
        d_scores[t] = p_pr - p_pi
    return d_scores


def chunk_text_into_passages(text, sentences_per_passage=3):
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
    if not sentences:
        return [text] if text.strip() else []

    passages = []
    step = max(1, sentences_per_passage - 1)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + sentences_per_passage])
        if len(chunk.split()) >= 15:
            passages.append(chunk)

    if not passages:
        passages = [text]
    return passages


def compute_contextual_signal(model, candidates, pr_docs_raw, query_text, tokenise_fn):
    q_emb = model.encode([query_text], normalize_embeddings=True)[0]

    doc_passages = []
    for doc in pr_docs_raw:
        body = (doc.get("title", "") + ". " + doc.get("text", "")).strip()
        chunks = chunk_text_into_passages(body)
        for chunk in chunks:
            toks = set(tokenise_fn(chunk, stem=True))
            doc_passages.append({"text": chunk, "tokens": toks})

    if not doc_passages:
        return {t: 0.0 for t in candidates}

    passage_texts = [p["text"] for p in doc_passages]
    p_embs = model.encode(passage_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    cos_sims = np.dot(p_embs, q_emb)

    for i, p in enumerate(doc_passages):
        p["score"] = float(cos_sims[i])

    c_scores = {}
    for t in candidates:
        relevant_scores = [p["score"] for p in doc_passages if t in p["tokens"]]
        c_scores[t] = max(relevant_scores) if relevant_scores else 0.0

    return c_scores


def min_max_normalize(scores):
    if not scores:
        return {}
    vals = list(scores.values())
    s_min, s_max = min(vals), max(vals)
    span = s_max - s_min
    if span <= 1e-9:
        return {k: 1.0 if s_max > 0 else 0.0 for k in scores}
    return {k: (v - s_min) / span for k, v in scores.items()}


def combine_signals(candidates, norm_signals, weights):
    w_sum = sum(weights.values())
    w_norm = {k: v / w_sum for k, v in weights.items()}
    combined = {}
    for c in candidates:
        s = sum(w_norm[sig] * norm_signals[sig].get(c, 0.0) for sig in weights)
        combined[c] = s
    return combined


def expand_query(qtoks, term_weights):
    weighted_query = list(qtoks)
    for term, w in term_weights:
        count = max(1, round(w * 5))
        weighted_query.extend([term] * count)
    return weighted_query
