"""Basic NLP preprocessing pipeline."""

import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def tokenize(sentence: str) -> list[str]:
    """Split a sentence into words and punctuation tokens."""
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\w\s]", sentence)


def stem_word(word: str) -> str:
    """Apply simple suffix stemming to an English word."""
    if not word.isalpha():
        return word

    lowered = word.lower()
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if lowered.endswith(suffix) and len(lowered) - len(suffix) >= 3:
            return lowered[: -len(suffix)]
    return lowered


def preprocess_sentence(sentence: str) -> tuple[list[str], list[str], list[str]]:
    """Tokenize, remove English stopwords, and stem the remaining words."""
    # Keep each stage separate so the intermediate results can be displayed.
    original_tokens = tokenize(sentence)
    tokens_without_stopwords = [
        token
        for token in original_tokens
        if not (token.isalpha() and token.lower() in STOPWORDS)
    ]
    stemmed_words = [stem_word(token) for token in tokens_without_stopwords]
    return original_tokens, tokens_without_stopwords, stemmed_words


if __name__ == "__main__":
    # This is the sentence specified in the assignment.
    sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
    original, without_stopwords, stemmed = preprocess_sentence(sentence)

    print("Original Tokens:", original)
    print("Tokens Without Stopwords:", without_stopwords)
    print("Stemmed Words:", stemmed)
