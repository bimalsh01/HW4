# NLP Programming Assignment

Bimal Shrestha

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
