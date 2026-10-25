import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = None


def setup():
    for resource in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource
                           else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    global stop_words
    stop_words = set(stopwords.words("english"))


def tokenise(text, stem=True):
    if stop_words is None:
        setup()
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def tokenise_query(query_text, stem=True):
    return tokenise(query_text, stem=stem)


def parse_topics(topics_path):
    topics = []
    with open(topics_path, "r", encoding="utf-8") as f:
        content = f.read()

    for block in content.strip().split("<top>"):
        block = block.strip()
        if not block:
            continue
        topic = {}
        for field, key in [("<num>", "num"), ("<title>", "title"),
                           ("<desc>", "desc"), ("<narr>", "narr")]:
            pattern = rf"{re.escape(field)}\s*(.*?)(?=<(?:num|title|desc|narr|/top)>|$)"
            m = re.search(pattern, block, re.DOTALL)
            if m:
                val = m.group(1).strip()
                if key == "num":
                    val = val.replace("Number:", "").strip()
                topic[key] = val
        if "num" in topic and "title" in topic:
            topics.append(topic)
    return topics


def load_corpus(docs_path):
    import json
    docs = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs
