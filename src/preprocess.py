import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    """Download required NLTK resources quietly if missing."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk_resources()

LEMMATIZER = WordNetLemmatizer()

# Generic words that appear across almost every financial headline/snippet
FINANCE_NOISE_WORDS = {
    "stock", "stocks", "share", "shares", "market", "markets",
    "company", "companies", "firm", "firms", "say", "says", "said",
    "new", "year", "years", "today", "report", "reports", "update",
    "http", "https", "www", "com", "amp", "rt"
}

STOP_WORDS = set(stopwords.words("english")).union(FINANCE_NOISE_WORDS)


def clean_raw_text(text: str) -> str:
    """Stage 3: Clean raw text (URLs, cashtags/handles, HTML entities, extra spaces)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.replace("&amp;", "&")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_tokens(text: str, min_token_len: int = 2) -> list[str]:
    """Stage 4: Normalize, tokenize, remove stopwords, and lemmatize."""
    cleaned = clean_raw_text(text).lower()
    if not cleaned:
        return []
    tokens = word_tokenize(cleaned)
    processed = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if len(tok) >= min_token_len and tok not in STOP_WORDS
    ]
    return processed