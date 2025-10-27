# matching.py
# movie quote search engine - tf-idf and keyword hybrid retrieval

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. load preprocessed scripts
print("Loading scripts from imsdb_scripts_sample.csv ...")
scripts_df = pd.read_csv("imsdb_scripts_sample.csv")
scripts_df = scripts_df.dropna(subset=["script"]).reset_index(drop=True)
print(f"Loaded {len(scripts_df)} scripts")

# 2. build the tf-idf model
print("Building TF-IDF matrix ...")
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
tfidf_matrix = vectorizer.fit_transform(scripts_df["script"])
print("TF-IDF matrix shape:", tfidf_matrix.shape)


# 3. keyword search helpers
def normalize(text):
    """lowercase, remove punctuation, collapse spaces"""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


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


def hybrid_search(query, top_n=5):
    kw_results = keyword_search(query)
    if kw_results:
        return [f"Exact match -> {r['movie']} (x{r['matches']} hits)" for r in kw_results[:top_n]]
    else:
        sim_results = tfidf_search(query, top_n)
        return [f"Similar -> {r['movie']} (score={r['score']})" for r in sim_results]


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
