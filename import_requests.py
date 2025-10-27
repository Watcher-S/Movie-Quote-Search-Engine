# import_requests.py
# movie script importer to csv

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

BASE = "https://imsdb.com"
INDEX_URL = f"{BASE}/all-scripts.html"
CSV_PATH = "imsdb_scripts_sample.csv"
FAILED_PATH = "imsdb_failed.csv"

def get_movie_pages():
    """Step 1: get links to all Movie Script info pages."""
    res = requests.get(INDEX_URL)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    movie_pages = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/Movie Scripts/"):
            movie_pages.append(BASE + href)
    return movie_pages

def get_script_link(movie_page_url):
    """Step 2: visit a Movie Script page and extract its 'Read Script' link."""
    res = requests.get(movie_page_url)
    soup = BeautifulSoup(res.text, "html.parser")
    for a in soup.find_all("a", href=True):
        if a.text.strip().lower().startswith("read"):
            href = a["href"]
            if href.startswith("/scripts/"):
                return BASE + href
    return None

def clean_text(text):
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"INT\.|EXT\.|CUT TO:|FADE.*?:", "", text, flags=re.I)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def fetch_script(script_url):
    """Step 3: download and clean the actual script text (skip pdfs, 404s)."""
    if script_url.endswith(".pdf"):
        print("  ⚠️  Skipping PDF:", script_url)
        return None
    try:
        res = requests.get(script_url)
        if res.status_code != 200:
            print(f"  ⚠️  {res.status_code} error for {script_url}")
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        pre = soup.find("pre")
        if not pre:
            print("  ⚠️  No <pre> text found.")
            return None
        return clean_text(pre.get_text())
    except Exception as e:
        print(f"  ⚠️  Error fetching {script_url}: {e}")
        return None


# load existing dataset if present
if os.path.exists(CSV_PATH):
    existing_df = pd.read_csv(CSV_PATH)
    existing_movies = set(existing_df["movie"].tolist())
    print(f"Found existing dataset with {len(existing_movies)} scripts.")
else:
    existing_df = pd.DataFrame(columns=["movie", "script"])
    existing_movies = set()
    print("No existing dataset found, starting fresh.")

# ask the user how many scripts to add
while True:
    try:
        n_add = int(input("How many new scripts would you like to add (1-100)? ").strip())
        if 1 <= n_add <= 100:
            break
        else:
            print("Please enter a number between 1 and 100.")
    except ValueError:
        print("Invalid input, please enter a valid integer.")

# fetch movie list
movie_pages = get_movie_pages()
print(f"Found {len(movie_pages)} movie info pages on IMSDb")

scripts, failed = [], []
added_count = 0

# iterate and skip already saved items
for i, movie_page in enumerate(movie_pages):
    if added_count >= n_add:
        break

    movie_name = movie_page.split("/")[-1].replace("%20Script.html", "").replace("Movie%20Scripts/", "")
    if movie_name in existing_movies:
        print(f"[{i+1}] Skipping already saved: {movie_name}")
        continue
    print(f"[{i+1}] Movie page: {movie_name}")
    script_link = get_script_link(movie_page)
    if not script_link:
        print("  Warning: no script link found")
        failed.append(movie_page)
        continue
    print("  Script link:", script_link)
    script_text = fetch_script(script_link)
    if script_text:
        scripts.append({"movie": movie_name, "script": script_text})
        existing_movies.add(movie_name)
        added_count += 1
        print(f"  Added {movie_name} ({added_count}/{n_add})")
    else:
        failed.append(script_link)

    # polite delay between requests
    time.sleep(1)

# merge and save the results
if scripts:
    new_df = pd.DataFrame(scripts)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(CSV_PATH, index=False)
    print(f"\nAdded {len(new_df)} new scripts (total now {len(combined_df)}).")
else:
    print("\nNo new scripts added.")

# save failed attempts for later inspection
if failed:
    pd.DataFrame({"failed_links": failed}).to_csv(FAILED_PATH, index=False)
    print(f"Warning: {len(failed)} failed or skipped scripts logged in {FAILED_PATH}")

print("Done.")
