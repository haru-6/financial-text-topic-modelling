import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel

from src.preprocess import preprocess_tokens

st.set_page_config(
    page_title="Financial Text Topic Modelling",
    page_icon="📈",
    layout="wide",
)

MODEL_PATH = "models/lda_model.gensim"
DICT_PATH = "models/dictionary.gensim"


@st.cache_resource
def load_saved_artifacts():
    model = LdaModel.load(MODEL_PATH)
    dictionary = corpora.Dictionary.load(DICT_PATH)
    return model, dictionary


def extract_documents(uploaded_csv, uploaded_txt_files, csv_text_col=None) -> list[str]:
    docs: list[str] = []

    if uploaded_csv is not None:
        df_up = pd.read_csv(uploaded_csv)
        if csv_text_col and csv_text_col in df_up.columns:
            docs.extend(df_up[csv_text_col].dropna().astype(str).tolist())

    if uploaded_txt_files:
        for txt_file in uploaded_txt_files:
            content = txt_file.read().decode("utf-8", errors="ignore")
            # Split non-empty lines/paragraphs so multi-line files are analyzed cleanly
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            docs.extend(lines)

    return [d.strip() for d in docs if d and d.strip()]


def assign_topics(model: LdaModel, dictionary: corpora.Dictionary, tokens_list: list[list[str]]) -> pd.DataFrame:
    rows = []
    num_topics = model.num_topics

    # Precompute readable labels from top 4 words of each topic
    topic_labels = {}
    for tid in range(num_topics):
        words = [w for w, _ in model.show_topic(tid, topn=4)]
        topic_labels[tid] = f"Topic {tid}: " + ", ".join(words)

    for idx, tokens in enumerate(tokens_list):
        if not tokens:
            rows.append({
                "Doc #": idx + 1,
                "Assigned Topic": "Unassigned (Too Short / Empty)",
                "Confidence": 0.0,
            })
            continue

        bow = dictionary.doc2bow(tokens)
        topic_probs = model.get_document_topics(bow, minimum_probability=0.0)
        best_tid, best_prob = max(topic_probs, key=lambda x: x[1])

        rows.append({
            "Doc #": idx + 1,
            "Assigned Topic": topic_labels[best_tid],
            "Confidence": round(float(best_prob), 3),
        })

    return pd.DataFrame(rows)


def main() -> None:
    st.title("📈 Financial Text Topic Modelling")
    st.write(
        "Discover latent themes in financial news, analyst notes, or customer feedback using **Latent Dirichlet Allocation (LDA)**."
    )

    with st.sidebar:
        st.header("Configuration")
        mode = st.radio(
            "Model Mode",
            options=["Use Pre-trained LDA Model", "Fit New LDA on Uploaded Docs"],
            index=0,
        )
        num_topics = st.slider("Number of Topics (for custom fit)", min_value=2, max_value=15, value=6)
        top_n_words = st.slider("Top Keywords per Topic", min_value=5, max_value=15, value=8)

    st.subheader("1. Upload Documents or Dataset")
    col1, col2 = st.columns(2)

    csv_text_col = None
    with col1:
        uploaded_csv = st.file_uploader("Upload a CSV dataset", type=["csv"])
        if uploaded_csv is not None:
            preview_df = pd.read_csv(uploaded_csv, nrows=5)
            uploaded_csv.seek(0)
            text_cols = [c for c in preview_df.columns if preview_df[c].dtype == object] or list(preview_df.columns)
            csv_text_col = st.selectbox("Select text column", options=text_cols)

    with col2:
        uploaded_txt_files = st.file_uploader(
            "Or upload one or more TXT files",
            type=["txt"],
            accept_multiple_files=True,
        )

    use_sample = st.checkbox("Use sample financial dataset (`data/sample_finance_docs.csv`) if no file uploaded", value=True)

    run_btn = st.button("🚀 Run Topic Modelling", type="primary")

    if run_btn:
        documents = extract_documents(uploaded_csv, uploaded_txt_files, csv_text_col)
        if not documents and use_sample:
            try:
                sample_df = pd.read_csv("data/sample_finance_docs.csv")
                documents = sample_df["text"].dropna().astype(str).tolist()
            except FileNotFoundError:
                st.error("Sample file `data/sample_finance_docs.csv` not found. Run `python src/train_lda.py` first or upload a file.")
                return

        if not documents:
            st.warning("Please upload a CSV/TXT file or enable the sample dataset.")
            return

        with st.spinner("Cleaning text and running topic modelling..."):
            tokens_list = [preprocess_tokens(doc) for doc in documents]

            if mode == "Use Pre-trained LDA Model":
                try:
                    lda_model, dictionary = load_saved_artifacts()
                except Exception as exc:
                    st.error(f"Could not load pre-trained model from `{MODEL_PATH}`: {exc}")
                    return
            else:
                non_empty_tokens = [t for t in tokens_list if len(t) >= 2]
                if len(non_empty_tokens) < num_topics:
                    st.error("Not enough valid documents to fit the requested number of topics.")
                    return
                dictionary = corpora.Dictionary(non_empty_tokens)
                dictionary.filter_extremes(no_below=1, no_above=0.9, keep_n=5000)
                corpus = [dictionary.doc2bow(t) for t in non_empty_tokens]
                lda_model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    passes=6,
                    iterations=80,
                    alpha="auto",
                    random_state=42,
                )

            assignments_df = assign_topics(lda_model, dictionary, tokens_list)
            assignments_df.insert(1, "Document Text", documents)

            valid_tokens = [t for t in tokens_list if len(t) >= 2]
            coherence_score = None
            if len(valid_tokens) >= 2 and len(dictionary) > 0:
                coherence_score = CoherenceModel(
                    model=lda_model,
                    texts=valid_tokens,
                    dictionary=dictionary,
                    coherence="c_v",
                ).get_coherence()

        st.divider()
        st.subheader("2. Discovered Topics & Important Words")
        if coherence_score is not None:
            st.metric("Topic Coherence Score ($C_v$)", f"{coherence_score:.4f}")

        topic_cards = []
        for tid in range(lda_model.num_topics):
            terms = lda_model.show_topic(tid, topn=top_n_words)
            topic_cards.append({
                "Topic ID": f"Topic {tid}",
                "Top Keywords": ", ".join([w for w, _ in terms]),
                "Top Weights": ", ".join([f"{w} ({weight:.2f})" for w, weight in terms[:5]]),
            })
        st.dataframe(pd.DataFrame(topic_cards), use_container_width=True, hide_index=True)

        st.subheader("3. Document Topic Assignments & Distribution")
        counts = assignments_df["Assigned Topic"].value_counts()

        chart_col, table_col = st.columns([1, 1.4])
        with chart_col:
            fig, ax = plt.subplots(figsize=(6, 4))
            counts.plot(kind="barh", ax=ax)
            ax.invert_yaxis()
            ax.set_xlabel("Document Count")
            ax.set_title("Documents per Topic")
            plt.tight_layout()
            st.pyplot(fig)

        with table_col:
            selected_filter = st.selectbox(
                "Filter documents by topic",
                options=["All Topics"] + sorted(assignments_df["Assigned Topic"].unique().tolist()),
            )
            filtered_df = (
                assignments_df
                if selected_filter == "All Topics"
                else assignments_df[assignments_df["Assigned Topic"] == selected_filter]
            )
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        csv_export = assignments_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Topic Assignments CSV",
            data=csv_export,
            file_name="topic_assignments.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()