# Movie Quote Search Engine (CS410 Fall 2025)

This repository contains a simple movie quote search engine built for CS410 (Fall 2025). It includes two main scripts:

- `import_requests.py` — scrapes movie script text from IMSDb and saves scripts to a CSV.
- `matching.py` — loads the CSV and provides a hybrid search (keyword + TF-IDF) to find movies by quote.


## Group

Movie Quote Search Engine

- Carson Lee — carsonl4@illinois.edu — University of Illinois at Urbana-Champaign
- Devon Reynolds — dreyno23@illinois.edu — University of Illinois at Urbana-Champaign
- Noah Carroll — noahlc2@illinois.edu — University of Illinois at Urbana-Champaign
- Szymon Czuwal — sczuwal2@illinois.edu — University of Illinois at Urbana-Champaign


## quick start (venv)

After cloning this repo, enter the virtual environment and activate it:

```bash
source .venv/bin/activate
```

## how `import_requests.py` works

- purpose: crawl IMSDb to collect plain-text movie scripts and save them to `imsdb_scripts_sample.csv`.
- main steps:
  1. fetch the index page and collect links to individual movie pages.
  2. for each movie page, find the "Read Script" link and follow it (skips PDF links).
  3. extract script text from the page (looks for `<pre>` text), perform light cleaning (remove parenthetical stage directions, repeated blank lines, some scene markers), and save results.
  4. existing CSVs are detected and preserved — new scripts are appended; failed links are saved to `imsdb_failed.csv`.

usage:

```bash
# run the importer and follow prompts
python import_requests.py
```

Notes:
- the script is written to be polite (adds small delays between requests).
- it will skip files it cannot parse or that return non-200 HTTP codes; failures are logged to `imsdb_failed.csv`.


## how `matching.py` works

- purpose: provide a robust search interface over the saved scripts to find likely movies for a given quote.
- main idea: hybrid retrieval that prefers exact/fuzzy substring matches, and blends TF‑IDF with BM25 scores for better partial phrase handling.
- main steps:
  1. load `imsdb_scripts_sample.csv` into a pandas DataFrame.
  2. build lexical indices:
    - TF‑IDF matrix using scikit‑learn's `TfidfVectorizer` with bigrams and sublinear TF.
    - BM25 index using `rank_bm25` over tokenized scripts.
  3. search modes:
    - keyword search: normalize text (lowercase, remove punctuation) and count substring occurrences — returns exact match hits first.
    - tf‑idf search: compute cosine similarity between a query and all script vectors; returns top similar movies.
    - bm25 search: score tokenized query with BM25 for strong lexical recall and phrase sensitivity.
  4. hybrid search (`combined_search`):
    - blends normalized TF‑IDF, BM25, and fuzzy token‑set ratio (`rapidfuzz`) with weights.
    - computes fuzzy scores only on a candidate set (union of top TF‑IDF and BM25) for speed.
    - final ranking is a weighted sum: `w_tfidf * tfidf + w_bm25 * bm25 + w_fuzzy * fuzzy`.

### tuning
- adjust weights in `combined_search` (defaults: `w_tfidf=0.4`, `w_bm25=0.4`, `w_fuzzy=0.2`).
- increase `ngram_range` in TF‑IDF for more phrase sensitivity (at memory cost).
- change candidate `top_k` size for fuzzy scoring trade‑off between quality and speed.

usage options:

```bash
# interactive mode (type 'quit' to exit)
python matching.py
# choose option 2 when prompted

# file mode: put queries (one per line) in input.txt and run the program
python matching.py
# choose option 1 when prompted (results will be written to output.txt)
```

### dependencies
The script will attempt to auto‑install missing packages, but you can install manually:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install scikit-learn pandas numpy rank-bm25 rapidfuzz
```

---

This project was prepared for CS410 (Fall 2025) by the group listed above.
