"""
Homework 3 - Text Preprocessing using the Coachella dataset

Steps:
    1. Extract hashtags into a new column "hashtags"
    2. Extract emails into a new column "emails"
    3. Clean the tweet text into a new column "clean_text" using:
         remove_usernames, remove_links, remove_non_ascii_symbols, to_lower,
         remove_stop_words, remove_digits, remove_special_characters

Input:  Coachella-2015-2-DFE.csv
Output: coachella_cleaned.csv (original columns + hashtags, emails, clean_text)
        output.txt (summary and before/after examples)

Uses pandas and Python's built-in re and html modules.
"""

import html
import re
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "Coachella-2015-2-DFE.csv"
OUTPUT_FILE = HERE / "coachella_cleaned.csv"

# English stop words (the NLTK list). Contractions are also listed without the
# apostrophe ("dont", "im") because they may lose it in other cleaning steps.
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn",
    "i'm", "im", "i've", "ive", "i'll", "ill", "i'd", "id", "you're", "youre",
    "you've", "youve", "you'll", "youll", "it's", "that's", "thats",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "can't", "cant",
    "won't", "wont", "isn't", "aren't", "wasn't", "weren't", "rt",
}


# ---------------------------------------------------------------------------
# Step 1: Extract hashtags
# ---------------------------------------------------------------------------
HASHTAG_PATTERN = re.compile(r"#\w+")


def extract_hashtags(text):
    """Return a list of all hashtags in the text, e.g. ['#Coachella2015']."""
    return HASHTAG_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Step 2: Extract emails
# ---------------------------------------------------------------------------
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")


def extract_emails(text):
    """Return a list of all email addresses in the text."""
    return EMAIL_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Step 3: Cleaning functions
# ---------------------------------------------------------------------------
def remove_usernames(text):
    """Remove Twitter usernames such as @coachella or @beyonce.

    (?<![\\w.]) makes sure the "@" is not in the middle of a word, so the
    "@" inside an email address is not treated as a username.
    """
    return re.sub(r"(?<![\w.])@\w+", " ", text)


def remove_links(text):
    """Remove links that start with http:// or https://."""
    return re.sub(r"https?://\S+", " ", text)


def remove_non_ascii_symbols(text):
    """Remove every character outside the ASCII range (emojis, accents, ...)."""
    return re.sub(r"[^\x00-\x7F]+", " ", text)


def to_lower(text):
    """Convert all letters to lower case."""
    return text.lower()


def remove_stop_words(text):
    """Remove common words (the, a, an, and, ...) that add little meaning.

    A word is compared without its surrounding punctuation, so "the," and
    "I'm" are also recognised as stop words.
    """
    kept = []
    for word in text.split():
        if word.strip(".,!?;:\"'()[]{}*-").lower() not in STOP_WORDS:
            kept.append(word)
    return " ".join(kept)


def remove_digits(text):
    """Remove all digits, e.g. 2015, 10, 100."""
    return re.sub(r"\d+", " ", text)


def remove_special_characters(text):
    """Remove punctuation and other special characters, keeping letters and spaces.

    Runs of whitespace left behind by the earlier steps are collapsed to one space.
    """
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def decode_html_entities(text):
    """Turn HTML entities from the Twitter export (&amp; &lt; &gt;) back into characters.

    Without this, "&amp;" would leave the meaningless word "amp" after
    special characters are removed.
    """
    return html.unescape(text)


CLEANING_STEPS = [
    decode_html_entities,
    remove_usernames,
    remove_links,
    remove_non_ascii_symbols,
    to_lower,
    remove_stop_words,
    remove_digits,
    remove_special_characters,
]


def clean_text(text):
    """Run the tweet through every cleaning step in order."""
    for step in CLEANING_STEPS:
        text = step(text)
    return text


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
def load_dataset(path):
    # The file is not UTF-8 (tweets were exported with broken emoji bytes),
    # so it is read as latin-1, which accepts every byte.
    df = pd.read_csv(path, encoding="latin-1")
    df["text"] = df["text"].fillna("")
    return df


def main():
    df = load_dataset(INPUT_FILE)

    df["hashtags"] = df["text"].apply(extract_hashtags)
    df["emails"] = df["text"].apply(extract_emails)
    df["clean_text"] = df["text"].apply(clean_text)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Loaded {len(df)} tweets from {INPUT_FILE.name}")
    print(f"Saved cleaned dataset to {OUTPUT_FILE.name}")
    print()

    # --- Hashtags ---
    all_hashtags = [tag.lower() for tags in df["hashtags"] for tag in tags]
    print("1. Hashtags")
    print(f"Tweets with at least one hashtag: {(df['hashtags'].str.len() > 0).sum()}")
    print(f"Total hashtags: {len(all_hashtags)} ({len(set(all_hashtags))} unique)")
    print("Top 10 hashtags (case-insensitive):")
    for tag, count in Counter(all_hashtags).most_common(10):
        print(f"  {tag:<25} {count}")
    print()

    # --- Emails ---
    print("2. Emails")
    print(f"Tweets with at least one email: {(df['emails'].str.len() > 0).sum()}")
    print("(The dataset contains no email addresses, so the column is empty.)")
    demo = "Write to example@gmail.com or user@yahoo.com, not @coachella."
    print(f"Check on a sample sentence: {demo!r}")
    print(f"  -> {extract_emails(demo)}")
    print()

    # --- Cleaning, step by step on one tweet ---
    print("3. Cleaning steps applied to one tweet")
    sample = df.loc[3, "text"]
    print(f"  {'original':<27} {sample!r}")
    for step in CLEANING_STEPS:
        sample = step(sample)
        print(f"  {step.__name__:<27} {sample!r}")
    print()

    # --- Before / after examples ---
    print("4. Before / after examples")
    for i in [0, 1, 2, 4, 6, 10, 25, 100]:
        print(f"[{i}] text:       {df.loc[i, 'text']!r}")
        print(f"     hashtags:   {df.loc[i, 'hashtags']}")
        print(f"     clean_text: {df.loc[i, 'clean_text']!r}")
        print()

    print("First rows of the new columns:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(df[["hashtags", "emails", "clean_text"]].head(10).to_string())


if __name__ == "__main__":
    main()
