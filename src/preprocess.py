import re
import pandas as pd
import io
import pyarabic.araby as araby


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_hard_data(path: str) -> pd.DataFrame:
    """
    Load a HARD dataset file (UTF-16, literal \\t separators).
    Returns a raw DataFrame with original columns.
    """
    with open(path, "r", encoding="utf-16") as f:
        raw_lines = f.readlines()

    # Fix literal \t strings → real tab characters
    fixed_lines = [line.replace("\\t", "\t") for line in raw_lines]

    cleaned_data = io.StringIO("".join(fixed_lines))
    df = pd.read_csv(cleaned_data, sep="\t")

    return df


def binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 1-5 star ratings to binary sentiment.
    1-2 → 0 (negative), 4-5 → 1 (positive), 3 dropped.
    """
    df = df.copy()
    df = df[df["rating"] != 3].reset_index(drop=True)
    df["label"] = (df["rating"] >= 4).astype(int)
    return df


def keep_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only review text and label."""
    return df[["review", "label"]].copy()


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def remove_noise(text: str) -> str:
    """Remove URLs, HTML, English chars, digits, punctuation."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)           # URLs
    text = re.sub(r"<.*?>", "", text)                     # HTML tags
    text = re.sub(r"[a-zA-Z]", "", text)                  # English chars
    text = re.sub(r"\d+", "", text)                       # Digits
    text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)     # Non-Arabic punctuation
    text = re.sub(r"[،؛؟!\"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]", "", text)
    text = re.sub(r"\s+", " ", text).strip()              # Extra whitespace
    return text


def normalize_arabic(text: str) -> str:
    """Normalize Arabic characters."""
    # Normalize alef variants → ا
    text = re.sub(r"[أإآ]", "ا", text)
    # Normalize ya
    text = re.sub(r"ى", "ي", text)
    # Normalize ta marbuta
    text = re.sub(r"ة", "ه", text)
    return text


def remove_diacritics(text: str) -> str:
    """Remove Arabic tashkeel (diacritics)."""
    return araby.strip_tashkeel(text)


def preprocess_for_tfidf(text: str) -> str:
    """Full preprocessing pipeline for TF-IDF model."""
    text = remove_noise(text)
    text = normalize_arabic(text)
    text = remove_diacritics(text)
    return text.strip()


def preprocess_for_arabert(text: str) -> str:
    """
    Lighter preprocessing for AraBERT.
    AraBERT handles its own normalization internally.
    We only remove noise.
    """
    text = remove_noise(text)
    return text.strip()


# ─────────────────────────────────────────────
# APPLY TO DATAFRAME
# ─────────────────────────────────────────────

def apply_preprocessing(df: pd.DataFrame, mode: str = "tfidf") -> pd.DataFrame:
    """
    Apply preprocessing to a DataFrame's review column.
    mode: 'tfidf' or 'arabert'
    """
    df = df.copy()
    fn = preprocess_for_tfidf if mode == "tfidf" else preprocess_for_arabert
    df[f"review_{mode}"] = df["review"].apply(fn)

    # Drop empty reviews after preprocessing
    df = df[df[f"review_{mode}"].str.strip().str.len() > 0].reset_index(drop=True)

    return df