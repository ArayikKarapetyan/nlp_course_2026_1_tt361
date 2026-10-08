"""
Homework 4 - BBC News Article Classification using custom Bag of Words and TF-IDF

Steps:
    1. Data preprocessing: lowercase, remove punctuation, remove stop words, tokenize
    2. Bag of Words (from scratch):
         build_vocabulary  - vocabulary from the training set only
         bow_vector        - document -> vector of word counts
    3. TF-IDF (from scratch), using the formulas from the lecture:
         compute_tf        - tf(t, d)
         compute_idf       - idf(t) = ln( N / N(t) )
         tfidf_vectors     - tfidf(t, d) = tf(t, d) * idf(t)
    4. Classification with the custom vectors (train/test split 80/20)
    5. Analysis:
         top 10 words by average TF-IDF for each category
         words with high TF but low IDF, and words with high IDF but low TF

BoW and TF-IDF are written with plain Python and numpy only. scikit-learn is
used only for the train/test split, the classifiers and the metrics, and to
check that the custom BoW gives the same counts as CountVectorizer.
"""

import math
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "bbc-text.csv"
RANDOM_STATE = 42

# English stop words (the NLTK list).
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
    # contractions written without the apostrophe in the dataset
    "dont", "cant", "doesnt", "didnt", "isnt", "wasnt", "wont", "im", "ive",
}


# ---------------------------------------------------------------------------
# Step 1: Data preprocessing
# ---------------------------------------------------------------------------
PUNCTUATION_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


def clean_text(text):
    """Lowercase the text, replace punctuation with spaces and drop digits."""
    text = text.lower()
    text = text.translate(PUNCTUATION_TABLE)
    text = re.sub(r"\d+", " ", text)
    return text


def tokenize(text):
    """Split the text into words."""
    return text.split()


def remove_stop_words(tokens):
    """Drop stop words and single letters left over from removed punctuation."""
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def preprocess(text):
    """Full preprocessing: clean -> tokenize -> remove stop words."""
    return remove_stop_words(tokenize(clean_text(text)))


# ---------------------------------------------------------------------------
# Step 2: Bag of Words
# ---------------------------------------------------------------------------
def build_vocabulary(tokenized_docs, min_df=2, max_size=None):
    """Build the vocabulary from the training documents.

    min_df:   keep a word only if it appears in at least this many documents
              (drops typos and one-off words).
    max_size: keep only the most frequent words (like max_features in sklearn).

    Returns a dict word -> column index, with words sorted A -> Z (as in the
    lecture), so column i means the same word in every vector.
    """
    doc_freq = Counter()
    total_freq = Counter()
    for tokens in tokenized_docs:
        total_freq.update(tokens)
        doc_freq.update(set(tokens))

    words = [word for word in total_freq if doc_freq[word] >= min_df]
    if max_size is not None:
        words = sorted(words, key=lambda w: -total_freq[w])[:max_size]

    return {word: index for index, word in enumerate(sorted(words))}


def bow_vector(tokens, vocabulary):
    """Convert one tokenized document into a BoW vector of word counts.

    Words that are not in the vocabulary are ignored.
    """
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    for token in tokens:
        index = vocabulary.get(token)
        if index is not None:
            vector[index] += 1
    return vector


def bow_matrix(tokenized_docs, vocabulary):
    """Stack the BoW vectors of all documents into an N x V matrix."""
    return np.vstack([bow_vector(tokens, vocabulary) for tokens in tokenized_docs])


# ---------------------------------------------------------------------------
# Step 3: TF-IDF
# ---------------------------------------------------------------------------
def compute_tf(bow, method="normalized"):
    """Term frequency tf(t, d) for every term in every document.

    bow is the N x V count matrix. Variations from the lecture:
        "raw":        tf = count(t, d)
        "normalized": tf = count(t, d) / total words in d   (default)
        "log":        tf = ln(1 + count(t, d))

    Normalized TF is the default so long articles do not get higher scores
    just for being long.
    """
    if method == "raw":
        return bow.copy()
    if method == "normalized":
        lengths = bow.sum(axis=1, keepdims=True)
        lengths[lengths == 0] = 1  # avoid dividing by zero for empty documents
        return bow / lengths
    if method == "log":
        return np.log1p(bow)
    raise ValueError(f"unknown TF method: {method}")


def compute_idf(bow):
    """Inverse document frequency idf(t) = ln( N / N(t) ).

    N    = number of documents
    N(t) = number of documents that contain term t

    It is computed from the training documents only and then reused for the
    test documents. N(t) is never 0 because the vocabulary comes from the
    same training documents.
    """
    n_docs = bow.shape[0]
    doc_freq = (bow > 0).sum(axis=0)
    return np.log(n_docs / doc_freq).astype(np.float32)


def tfidf_vectors(bow, idf, tf_method="normalized"):
    """tfidf(t, d) = tf(t, d) * idf(t) for every document (N x V matrix)."""
    return compute_tf(bow, tf_method) * idf


# ---------------------------------------------------------------------------
# Step 4: Classification helpers
# ---------------------------------------------------------------------------
def nearest_centroid_predict(train_vectors, train_labels, test_vectors):
    """Classify by cosine similarity to the average vector of each category.

    This is "Way B" from the lecture (closest document), but compared with
    one average document per category instead of every training document.
    """
    classes = sorted(set(train_labels))
    labels = np.array(train_labels)
    centroids = np.vstack([train_vectors[labels == c].mean(axis=0) for c in classes])

    def unit(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    similarity = unit(test_vectors) @ unit(centroids).T
    return [classes[i] for i in similarity.argmax(axis=1)]


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} articles from {DATA_FILE.name}")
    print(df["category"].value_counts().to_string())

    # ---- 1. Preprocessing ----
    print_header("1. Data preprocessing")
    df["tokens"] = df["text"].apply(preprocess)
    example = df.loc[0, "text"]
    print("Original (first 200 chars):")
    print(f"  {example[:200]}...")
    print("Tokens (first 25):")
    print(f"  {df.loc[0, 'tokens'][:25]}")
    print(f"Average tokens per article: {df['tokens'].str.len().mean():.1f}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["category"], random_state=RANDOM_STATE
    )
    train_tokens, test_tokens = list(train_df["tokens"]), list(test_df["tokens"])
    y_train, y_test = list(train_df["category"]), list(test_df["category"])
    print(f"Train / test split (80/20, stratified): {len(train_df)} / {len(test_df)} articles")

    # ---- 2. Bag of Words ----
    print_header("2. Bag of Words")
    vocabulary = build_vocabulary(train_tokens, min_df=2)
    index_to_word = {index: word for word, index in vocabulary.items()}
    all_train_words = {word for tokens in train_tokens for word in tokens}
    print(f"Unique words in training set: {len(all_train_words)}")
    print(f"Vocabulary size V (words in >= 2 training articles): {len(vocabulary)}")

    X_train_bow = bow_matrix(train_tokens, vocabulary)
    X_test_bow = bow_matrix(test_tokens, vocabulary)
    print(f"BoW matrix shapes: train {X_train_bow.shape}, test {X_test_bow.shape}")

    row = X_train_bow[0]
    nonzero = np.argsort(-row)[:10]
    print("Largest counts in the first training article:")
    print("  " + ", ".join(f"{index_to_word[i]}={int(row[i])}" for i in nonzero))

    # Check: sklearn's CountVectorizer with the same vocabulary must give the same counts.
    reference = CountVectorizer(vocabulary=vocabulary, analyzer=lambda tokens: tokens)
    same = np.array_equal(reference.transform(train_tokens).toarray(), X_train_bow)
    print(f"Same counts as sklearn CountVectorizer: {same}")

    # Small example from the lecture slides.
    lecture_docs = [["i", "like", "eggs"], ["i", "hate", "cats"],
                    ["i", "like", "eggs", "and", "i", "like", "cats"]]
    lecture_vocab = build_vocabulary(lecture_docs, min_df=1)
    lecture_bow = bow_matrix(lecture_docs, lecture_vocab)
    print(f"Lecture example, vocabulary {list(lecture_vocab)}:")
    for doc, vec in zip(lecture_docs, lecture_bow):
        print(f"  {' '.join(doc):<28} {vec.astype(int).tolist()}")

    # ---- 3. TF-IDF ----
    print_header("3. TF-IDF")
    idf = compute_idf(X_train_bow)
    X_train_tfidf = tfidf_vectors(X_train_bow, idf)
    X_test_tfidf = tfidf_vectors(X_test_bow, idf)
    print("tf(t, d)  = count(t, d) / number of words in d   (normalized TF)")
    print("idf(t)    = ln( N / N(t) ),  N = training articles")
    print("tfidf     = tf * idf")
    print(f"TF-IDF matrix shapes: train {X_train_tfidf.shape}, test {X_test_tfidf.shape}")
    print(f"IDF range: {idf.min():.3f} (most common word) .. {idf.max():.3f} (rarest words)")

    lecture_idf = compute_idf(lecture_bow)
    lecture_tfidf = tfidf_vectors(lecture_bow, lecture_idf, tf_method="raw")
    print("Lecture example with raw counts (should match the slide):")
    print("  idf: " + ", ".join(f"{w}={lecture_idf[i]:.2f}" for w, i in lecture_vocab.items()))
    for doc, vec in zip(lecture_docs, lecture_tfidf):
        print(f"  {' '.join(doc):<28} [{', '.join(f'{v:.2f}' for v in vec)}]")

    # ---- 4. Classification ----
    print_header("4. Classification (test set)")
    results = []
    models = [
        ("Multinomial Naive Bayes", "BoW", MultinomialNB(), X_train_bow, X_test_bow),
        ("Logistic Regression", "BoW", LogisticRegression(max_iter=2000),
         X_train_bow, X_test_bow),
        ("Logistic Regression", "TF-IDF", LogisticRegression(C=100, max_iter=2000),
         X_train_tfidf, X_test_tfidf),
    ]
    predictions = {}
    for name, features, model, X_tr, X_te in models:
        model.fit(X_tr, y_train)
        pred = model.predict(X_te)
        predictions[(name, features)] = pred
        results.append((name, features, accuracy_score(y_test, pred)))

    for features, X_tr, X_te in [("BoW", X_train_bow, X_test_bow),
                                 ("TF-IDF", X_train_tfidf, X_test_tfidf)]:
        pred = nearest_centroid_predict(X_tr, y_train, X_te)
        predictions[("Nearest centroid (cosine)", features)] = pred
        results.append(("Nearest centroid (cosine)", features, accuracy_score(y_test, pred)))

    print(f"{'Model':<28}{'Features':<10}{'Accuracy':>10}")
    print("-" * 48)
    for name, features, acc in results:
        print(f"{name:<28}{features:<10}{acc:>10.4f}")

    best_name, best_features, best_acc = max(results, key=lambda r: r[2])
    best_pred = predictions[(best_name, best_features)]
    print()
    print(f"Best: {best_name} on {best_features} ({best_acc:.4f})")
    print(classification_report(y_test, best_pred, digits=4))
    classes = sorted(set(y_test))
    matrix = confusion_matrix(y_test, best_pred, labels=classes)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(matrix, index=classes, columns=classes).to_string())

    # ---- 5. Analysis ----
    print_header("5a. Top 10 words by average TF-IDF in each category (training set)")
    train_labels = np.array(y_train)
    for category in classes:
        mean_scores = X_train_tfidf[train_labels == category].mean(axis=0)
        top = np.argsort(-mean_scores)[:10]
        print(f"{category}:")
        print("  " + ", ".join(f"{index_to_word[i]} ({mean_scores[i]:.4f})" for i in top))

    print_header("5b. TF vs IDF")
    avg_tf = compute_tf(X_train_bow).mean(axis=0)
    tf_high, tf_low = np.percentile(avg_tf, 90), np.percentile(avg_tf, 10)
    idf_high, idf_low = np.percentile(idf, 90), np.percentile(idf, 10)
    print("avg TF = average normalized TF of a word over all training articles")
    print(f"High TF >= {tf_high:.7f} (top 10%), low TF <= {tf_low:.7f} (bottom 10%)")
    print(f"High IDF >= {idf_high:.3f} (top 10%), low IDF <= {idf_low:.3f} (bottom 10%)")

    doc_count = (X_train_bow > 0).sum(axis=0)
    n_train = X_train_bow.shape[0]

    def show(indices):
        print(f"  {'word':<16}{'avg TF':>11}{'IDF':>8}{'in docs':>12}")
        for i in indices:
            print(f"  {index_to_word[i]:<16}{avg_tf[i]:>11.7f}{idf[i]:>8.3f}"
                  f"{doc_count[i]:>7}/{n_train}")

    high_tf_low_idf = np.where((avg_tf >= tf_high) & (idf <= idf_low))[0]
    high_tf_low_idf = high_tf_low_idf[np.argsort(-avg_tf[high_tf_low_idf])][:15]
    print()
    print(f"High TF, low IDF ({len(high_tf_low_idf)} shown, sorted by TF):")
    print("  Frequent everywhere -> TF-IDF scales them down; they barely help to")
    print("  tell categories apart (e.g. 'said', 'year', 'also' work like stop words).")
    show(high_tf_low_idf)

    low_tf_high_idf = np.where((avg_tf <= tf_low) & (idf >= idf_high))[0]
    total = len(low_tf_high_idf)
    low_tf_high_idf = low_tf_high_idf[np.argsort(avg_tf[low_tf_high_idf])][:15]
    print()
    print(f"High IDF, low TF ({total} words match; 15 with the lowest TF shown):")
    print("  Rare words mentioned once or twice in long articles -> high IDF, but")
    print("  their small TF keeps the TF-IDF score low.")
    show(low_tf_high_idf)


if __name__ == "__main__":
    main()
