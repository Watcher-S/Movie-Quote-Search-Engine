# matching.py
# movie quote search engine - tf-idf and keyword hybrid retrieval

import sys
import subprocess

REQUIRED_PACKAGES = [
    "scikit-learn",
    "pandas",
    "numpy",
    "rank-bm25",
    "rapidfuzz",
]

def ensure_dependencies():
    missing = []
    try:
        import sklearn  # noqa: F401
    except Exception:
        missing.append("scikit-learn")
    try:
        import pandas  # noqa: F401
    except Exception:
        missing.append("pandas")
    try:
        import numpy  # noqa: F401
    except Exception:
        missing.append("numpy")
    try:
        import rank_bm25  # noqa: F401
    except Exception:
        missing.append("rank-bm25")
    try:
        import rapidfuzz  # noqa: F401
    except Exception:
        missing.append("rapidfuzz")
    if missing:
        print("The following packages are missing:", ", ".join(missing))
        print("Attempting to install them now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("Install complete. Continuing...")
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "Auto-install failed. On macOS zsh, try:\n"
                "  python3 -m pip install scikit-learn pandas numpy rank-bm25 rapidfuzz\n"
                "Or add them to requirements.txt and install."
            )

ensure_dependencies()

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

# configuration toggles
USE_CUSTOM_TFIDF = False
USE_CUSTOM_BM25 = False

# 1. load preprocessed scripts
print("Loading scripts from imsdb_scripts_sample.csv ...")
scripts_df = pd.read_csv("imsdb_scripts_sample.csv")
scripts_df = scripts_df.dropna(subset=["script"]).reset_index(drop=True)
print(f"Loaded {len(scripts_df)} scripts")

def normalize(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()

def tokenize(text):
    return normalize(text).split()

# 2. build the tf-idf + bm25 models
corpus_tokens = [tokenize(s) for s in scripts_df["script"]]

def build_inverted_index(corpus_tokens):
    N = len(corpus_tokens)
    df = {}
    postings = {}
    doc_lengths = np.zeros(N, dtype=np.int32)

    for doc_id, tokens in enumerate(corpus_tokens):
        doc_lengths[doc_id] = len(tokens)
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for t, tf in counts.items():
            df[t] = df.get(t, 0) + 1
            lst = postings.get(t)
            if lst is None:
                postings[t] = [(doc_id, tf)]
            else:
                lst.append((doc_id, tf))

    # idf for tf-idf (log-scaled)
    idf_tfidf = {}
    for t, dfi in df.items():
        idf_tfidf[t] = np.log((N + 1) / (dfi + 1)) + 1.0

    # idf for BM25 (Okapi)
    idf_bm25 = {}
    for t, dfi in df.items():
        idf_bm25[t] = np.log((N - dfi + 0.5) / (dfi + 0.5) + 1.0)

    # precompute doc norms for tf-idf cosine
    doc_norms = np.zeros(N, dtype=np.float64)
    for doc_id, tokens in enumerate(corpus_tokens):
        # per-doc term frequency map
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        ssum = 0.0
        for t, tf in counts.items():
            idf = idf_tfidf.get(t)
            if idf is None:
                continue
            tf_scaled = 1.0 + np.log(tf)
            w = tf_scaled * idf
            ssum += w * w
        doc_norms[doc_id] = np.sqrt(ssum) if ssum > 0 else 1.0

    avgdl = float(doc_lengths.mean()) if N > 0 else 1.0
    return {
        "N": N,
        "df": df,
        "postings": postings,
        "doc_lengths": doc_lengths,
        "idf_tfidf": idf_tfidf,
        "idf_bm25": idf_bm25,
        "doc_norms": doc_norms,
        "avgdl": avgdl,
    }

print("Building indexes ...")
index_data = build_inverted_index(corpus_tokens)

if not USE_CUSTOM_TFIDF:
    print("Building TF-IDF matrix (sklearn) ...")
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(scripts_df["script"])
    print("TF-IDF matrix shape:", tfidf_matrix.shape)

if not USE_CUSTOM_BM25:
    print("Building BM25 index (library) ...")
    bm25 = BM25Okapi(corpus_tokens)


# 3. keyword search helpers


def keyword_search(query):
    """fuzzy substring search: ignore punctuation/case and count occurrences"""
    results = []
    norm_query = normalize(query)
    for _, row in scripts_df.iterrows():
        norm_script = normalize(row["script"])
        matches = norm_script.count(norm_query)
        if matches > 0:
            results.append({"movie": row["movie"], "matches": matches})
    return sorted(results, key=lambda x: x["matches"], reverse=True)


# 4. tf-idf + hybrid search functions
def tfidf_search(query, top_n=5):
    if USE_CUSTOM_TFIDF:
        tokens = tokenize(query)
        # compute query weights
        q_counts = {}
        for t in tokens:
            q_counts[t] = q_counts.get(t, 0) + 1
        q_weights = {}
        for t, tf in q_counts.items():
            idf = index_data["idf_tfidf"].get(t)
            if idf is None:
                continue
            tf_scaled = 1.0 + np.log(tf)
            q_weights[t] = tf_scaled * idf
        q_norm = np.sqrt(sum(w * w for w in q_weights.values())) or 1.0

        # accumulate dot products over postings for query terms
        dot = np.zeros(index_data["N"], dtype=np.float64)
        for t, qw in q_weights.items():
            postings = index_data["postings"].get(t)
            if not postings:
                continue
            idf = index_data["idf_tfidf"][t]
            for doc_id, tf in postings:
                tf_scaled = 1.0 + np.log(tf)
                dw = tf_scaled * idf
                dot[doc_id] += qw * dw

        # cosine similarity
        sims = dot / (q_norm * index_data["doc_norms"])
        top_indices = np.argsort(sims)[::-1][:top_n]
        return [{"movie": scripts_df.iloc[i]["movie"], "score": round(float(sims[i]), 3)} for i in top_indices]
    else:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_n]
        return [{"movie": scripts_df.iloc[i]["movie"], "score": round(float(similarities[i]), 3)} for i in top_indices]


def bm25_search(query, top_n=5, k1=1.5, b=0.75):
    tokens = tokenize(query)
    if USE_CUSTOM_BM25:
        scores = np.zeros(index_data["N"], dtype=np.float64)
        for t in set(tokens):
            postings = index_data["postings"].get(t)
            if not postings:
                continue
            idf = index_data["idf_bm25"].get(t, 0.0)
            for doc_id, tf in postings:
                dl = index_data["doc_lengths"][doc_id]
                denom = tf + k1 * (1.0 - b + b * dl / index_data["avgdl"])
                score = idf * (tf * (k1 + 1.0)) / (denom if denom > 0 else 1.0)
                scores[doc_id] += score
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [{"movie": scripts_df.iloc[i]["movie"], "score": round(float(scores[i]), 3)} for i in top_indices]
    else:
        scores = np.array(bm25.get_scores(tokens))
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [{"movie": scripts_df.iloc[i]["movie"], "score": round(float(scores[i]), 3)} for i in top_indices]


def combined_search(query, top_n=5, w_tfidf=0.4, w_bm25=0.4, w_fuzzy=0.2):
    # get tf-idf and bm25 via selected implementations
    tfidf_res = tfidf_search(query, top_n=len(scripts_df))
    bm25_res = bm25_search(query, top_n=len(scripts_df))

    # maps from movie to score; we need per index; easier: compute arrays aligned to doc order
    tfidf_scores = np.zeros(len(scripts_df))
    bm25_scores = np.zeros(len(scripts_df))
    # build lookup for movie order
    movie_to_indices = {scripts_df.iloc[i]["movie"]: i for i in range(len(scripts_df))}
    for r in tfidf_res:
        i = movie_to_indices.get(r["movie"])  # unique movies assumed
        if i is not None:
            tfidf_scores[i] = r["score"]
    for r in bm25_res:
        i = movie_to_indices.get(r["movie"])  # unique movies assumed
        if i is not None:
            bm25_scores[i] = r["score"]

    # normalize
    tfidf_norm = tfidf_scores / (tfidf_scores.max() if tfidf_scores.max() > 0 else 1.0)
    bm25_norm = bm25_scores / (bm25_scores.max() if bm25_scores.max() > 0 else 1.0)

    # compute fuzzy only on union of top candidates for efficiency
    top_k = 200 if len(scripts_df) > 200 else len(scripts_df)
    tf_top = np.argsort(tfidf_norm)[::-1][:top_k]
    bm_top = np.argsort(bm25_norm)[::-1][:top_k]
    cand = np.unique(np.concatenate([tf_top, bm_top]))

    fuzzy_norm = np.zeros(len(scripts_df), dtype=float)
    norm_query = normalize(query)
    for i in cand:
        s = normalize(scripts_df.iloc[i]["script"])[:20000]
        score = fuzz.token_set_ratio(norm_query, s) / 100.0
        fuzzy_norm[i] = score

    combined = w_tfidf * tfidf_norm + w_bm25 * bm25_norm + w_fuzzy * fuzzy_norm
    top_indices = np.argsort(combined)[::-1][:top_n]
    return [
        {
            "movie": scripts_df.iloc[i]["movie"],
            "score": round(float(combined[i]), 3)
        }
        for i in top_indices
    ]


def hybrid_search(query, top_n=5):
    kw_results = keyword_search(query)
    if kw_results:
        return [f"Exact match -> {r['movie']} (x{r['matches']} hits)" for r in kw_results[:top_n]]
    results = combined_search(query, top_n)
    return [f"Similar -> {r['movie']} (score={r['score']})" for r in results]


# 5. run modes (file / interactive)
def run_from_file(input_file="input.txt", output_file="output.txt"):
    """read queries from a file and write ranked results to an output file"""
    try:
        with open(input_file, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: file {input_file} not found.")
        return

    print(f"Running {len(queries)} predefined test quotes...")
    with open(output_file, "w") as out:
        for q in queries:
            out.write(f"Query: {q}\n")
            results = hybrid_search(q)
            for r in results:
                out.write(f"   {r}\n")
            out.write("\n")
    print(f"Results saved to {output_file}")


def run_interactive():
    """prompt the user for quotes until 'quit' is entered"""
    print("\nEnter a movie quote to search (type 'quit' to exit):")
    while True:
        query = input("> ").strip()
        if query.lower() == "quit":
            print("Exiting search.")
            break
        if not query:
            continue
        results = hybrid_search(query)
        print("\nResults:")
        for r in results:
            print("  ", r)
        print("-" * 40)


# 6. simple main menu
if __name__ == "__main__":
    print("\nSelect mode:")
    print("  1 -> run predefined quotes from input.txt (save to output.txt)")
    print("  2 -> interactive search mode (type 'quit' to exit)\n")

    choice = input("Enter your choice (1 or 2): ").strip()
    if choice == "1":
        run_from_file()
    elif choice == "2":
        run_interactive()
    else:
        print("Invalid choice. Exiting.")
