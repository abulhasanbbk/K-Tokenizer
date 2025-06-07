ner.py

A script for Named Entity Recognition (NER) using K-Tokenizer.

Usage

To run the ner.py script, execute:

python ner.py --corpus <base|umls|mimic> --seed <your seed>

Arguments

--corpus  : Specify the corpus to use. Must be one of:

base  — the base clinical text dataset

umls  — text augmented with UMLS concepts

mimic — the MIMIC-III clinical dataset

--seed    : Integer seed for random number generators to ensure reproducibility.

Examples

# Run on the base corpus with seed 42
env/bin/python ner.py --corpus base --seed 42

# Run on UMLS-augmented corpus with seed 123
python ner.py --corpus umls --seed 123

Outputs

The script outputs token-level NER predictions in a JSON file named <corpus>_ner_results_seed<seed>.json.

Requirements

Python 3.7+

Dependencies listed in requirements.txt (install via pip install -r requirements.txt)

