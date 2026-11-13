# NLP Programming Assignment

**Student:** Bimal Shrestha

**Student ID:** 700780439

This repository contains solutions for three NLP programming tasks.

## Requirements

- Python 3
- NumPy
- spaCy
- spaCy English model: `en_core_web_sm`

## Setup

Create and activate a virtual environment:

```bash
cd /Users/stha/Documents/assingment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy spacy
python -m spacy download en_core_web_sm
```

## Run the solutions

### Q1: NLP preprocessing

Tokenizes a sentence, removes common stopwords, and applies stemming.

```bash
python nlp_preprocessing.py
```

### Q2: Named Entity Recognition

Uses spaCy to extract entity text, labels, and character positions.

```bash
python ner_spacy.py
```

### Q3: Scaled dot-product attention

Computes attention scores, scales them by the key dimension, applies softmax,
and produces the weighted output matrix.

```bash
python scaled_dot_product_attention.py
```

## Short-answer responses

### Q1

**Stemming** removes word endings using simple rules and may produce a
non-dictionary form. For example, `running` may become `runn`. **Lemmatization**
uses vocabulary and grammar to produce the correct dictionary form, so
`running` becomes `run`.

Removing stopwords can reduce text size and focus analysis on meaningful words.
It can be harmful when a stopword changes meaning; for example, removing `not`
from “This movie is not good” reverses the sentiment.

### Q2

Named Entity Recognition identifies entities such as people, organizations,
locations, and dates. Part-of-speech tagging identifies grammatical categories
such as nouns, verbs, adjectives, and adverbs.

NER is used by search engines to understand names and locations in queries. It
is also used in financial-news systems to extract company names, stock symbols,
and monetary information.

### Q3

Dividing attention scores by the square root of the key dimension prevents
large dot products from making softmax overly sharp and keeps gradients stable.
Self-attention lets each word assign greater weight to relevant words elsewhere
in the sentence, helping the model capture context and relationships.
