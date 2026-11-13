"""Named Entity Recognition example using spaCy."""

import spacy


def extract_entities(sentence: str) -> list[tuple[str, str, int, int]]:
    """Return entity text, label, and character offsets."""
    # The small English model provides tokenization and pretrained NER labels.
    nlp = spacy.load("en_core_web_sm")
    document = nlp(sentence)
    return [(entity.text, entity.label_, entity.start_char, entity.end_char)
            for entity in document.ents]


if __name__ == "__main__":
    # This is the sentence specified in the assignment.
    sentence = (
        "LeBron James as the NBA player of the United States in Laker "
        "and won the NBA Final MVP in 2020."
    )

    for text, label, start, end in extract_entities(sentence):
        print(f"Entity: {text} | Label: {label} | Start: {start} | End: {end}")
