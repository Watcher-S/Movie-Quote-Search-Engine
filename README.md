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

- purpose: provide a simple search interface over the saved scripts to find likely movies for a given quote.
- main idea: hybrid retrieval that prefers exact/fuzzy substring matches but falls back to TF-IDF similarity.
- main steps:
  1. load `imsdb_scripts_sample.csv` into a pandas DataFrame.
  2. build a TF-IDF matrix over the script text using scikit-learn's `TfidfVectorizer`.
  3. provide two search modes:
     - keyword search: normalize text (lowercase, remove punctuation) and count fuzzy substring occurrences — returns exact match hits first.
     - tf-idf search: compute cosine similarity between a query and all script vectors; returns top similar movies.
  4. hybrid search: return keyword matches if present; otherwise return TF-IDF matches.

usage options:

```bash
# interactive mode (type 'quit' to exit)
python matching.py
# choose option 2 when prompted

# file mode: put queries (one per line) in input.txt and run the program
python matching.py
# choose option 1 when prompted (results will be written to output.txt)
```
---

This project was prepared for CS410 (Fall 2025) by the group listed above.
