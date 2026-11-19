import os
import pandas as pd
from datasets import load_dataset
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel

from preprocess import preprocess_tokens


def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Stage 1 & 2: Problem Identification & Dataset Collection
    print("Loading zeroshot/twitter-financial-news-topic dataset...")
    ds = load_dataset("zeroshot/twitter-financial-news-topic")
    df_train = ds["train"].to_pandas()
    df_valid = ds["validation"].to_pandas() if "validation" in ds else pd.DataFrame()
    df = pd.concat([df_train, df_valid], ignore_index=True)

    # Save a clean sample CSV for testing the Streamlit app
    sample_df = df[["text"]].dropna().drop_duplicates().sample(n=min(200, len(df)), random_state=42)
    sample_df.to_csv("data/sample_finance_docs.csv", index=False)

    # Stage 3: Data Cleaning
    initial_count = len(df)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Cleaned rows: {len(df)} (dropped {initial_count - len(df)} empty/duplicate rows)")

    # Stage 4: Text Preprocessing
    print("Preprocessing text tokens...")
    df["tokens"] = df["text"].apply(preprocess_tokens)
    df = df[df["tokens"].map(len) >= 3].reset_index(drop=True)

    # Stage 5: Quick EDA Summary
    doc_lengths = df["tokens"].map(len)
    print(
        f"EDA -> Docs: {len(df)} | Avg tokens/doc: {doc_lengths.mean():.1f} | "
        f"Median: {doc_lengths.median():.1f} | Max: {doc_lengths.max()}"
    )

    # Stage 6: Feature Engineering (Dictionary + Bag-of-Words)
    dictionary = corpora.Dictionary(df["tokens"])
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000)
    corpus = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]

    # Stage 7 & 8: Model Development & Coherence Evaluation
    candidate_topics = [6, 8, 10, 12]
    results = []

    for k in candidate_topics:
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=8,
            iterations=100,
            alpha="auto",
            eta="auto",
            random_state=42,
        )
        coherence_cv = CoherenceModel(
            model=model,
            texts=df["tokens"].tolist(),
            dictionary=dictionary,
            coherence="c_v",
        ).get_coherence()
        results.append((k, coherence_cv, model))
        print(f"Topics (K={k:2d}) -> C_v Coherence: {coherence_cv:.4f}")

    best_k, best_score, best_model = max(results, key=lambda item: item[1])
    print(f"\nSelected Best Model: K={best_k} topics (C_v Coherence = {best_score:.4f})")

    # Print top words per topic
    for topic_id in range(best_k):
        top_terms = best_model.show_topic(topic_id, topn=8)
        terms_str = ", ".join([f"{word} ({weight:.2f})" for word, weight in top_terms])
        print(f"  Topic {topic_id}: {terms_str}")

    # Persist artifacts for Streamlit GUI
    best_model.save("models/lda_model.gensim")
    dictionary.save("models/dictionary.gensim")
    print("\nSaved model and dictionary to ./models/")


if __name__ == "__main__":
    main()