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

# we've experimented with custom bm25 and tf-idf implementations,
# but sklearn and rank-bm25 are efficient and well-optimized
# so we use them here.


# -----------------
# configuration
# -----------------
class Config:
    STOP_WORDS = "english"
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_SUBLINEAR = True
    TOP_K_CANDIDATES = 200
    WEIGHTS = {
        "tfidf": 0.4,
        "bm25": 0.4,
        "fuzzy": 0.2,
    }



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
print("Building TF-IDF matrix ...")
vectorizer = TfidfVectorizer(
    stop_words=Config.STOP_WORDS,
    lowercase=True,
    ngram_range=Config.TFIDF_NGRAM_RANGE,
    sublinear_tf=Config.TFIDF_SUBLINEAR,
)
tfidf_matrix = vectorizer.fit_transform(scripts_df["script"])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

print("Building BM25 index ...")
corpus_tokens = [tokenize(s) for s in scripts_df["script"]]
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
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = []
    for i in top_indices:
        results.append({
            "movie": scripts_df.iloc[i]["movie"],
            "score": round(similarities[i], 3)
        })
    return results


def bm25_search(query, top_n=5):
    tokens = tokenize(query)
    scores = np.array(bm25.get_scores(tokens))
    top_indices = np.argsort(scores)[::-1][:top_n]
    results = []
    for i in top_indices:
        results.append({
            "movie": scripts_df.iloc[i]["movie"],
            "score": round(float(scores[i]), 3)
        })
    return results


def combined_search(query, top_n=5, w_tfidf=None, w_bm25=None, w_fuzzy=None):
    # use configured weights if not provided
    w_tfidf = Config.WEIGHTS.get("tfidf") if w_tfidf is None else w_tfidf
    w_bm25 = Config.WEIGHTS.get("bm25") if w_bm25 is None else w_bm25
    w_fuzzy = Config.WEIGHTS.get("fuzzy") if w_fuzzy is None else w_fuzzy
    q_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    tfidf_norm = tfidf_scores / (tfidf_scores.max() if tfidf_scores.max() > 0 else 1.0)

    tokens = tokenize(query)
    bm25_scores = np.array(bm25.get_scores(tokens))
    bm25_norm = bm25_scores / (bm25_scores.max() if bm25_scores.max() > 0 else 1.0)

    # compute fuzzy only on union of top candidates for efficiency
    top_k_default = Config.TOP_K_CANDIDATES
    top_k = top_k_default if len(scripts_df) > top_k_default else len(scripts_df)
    tf_top = np.argsort(tfidf_scores)[::-1][:top_k]
    bm_top = np.argsort(bm25_scores)[::-1][:top_k]
    cand = np.unique(np.concatenate([tf_top, bm_top]))

    fuzzy_norm = np.zeros(len(scripts_df), dtype=float)
    norm_query = normalize(query)
    for i in cand:
        s = normalize(scripts_df.iloc[i]["script"])[:20000]
        # cap length to reduce cost; normalization keeps behavior consistent
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
