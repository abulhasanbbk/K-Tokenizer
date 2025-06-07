# ner.py

A script for clincial concept extraction using K-Tokenizer.

## Pre-processing data
- Data can be found in https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/. If unavailable please contact abul.hasan@phc.ox.ac.uk 
- Run ```preprocess.py``` program to create train and test set in onrder to work with our program.

## Usage

To run the ner.py script, execute:

python ner.py --corpus <base|umls|mimic> --seed <your seed>

## Arguments

--corpus  : Specify the corpus to use. Must be one of:

base  — using ClinicalBERT tokenizer

umls  — using K-Tokenizer built from UMLS.

mimic — using K-Tokenizer built from MIMIC-III dataset.

--seed    : Integer seed for random number generators to ensure reproducibility.

Examples

# Run on the base corpus with seed 42
env/bin/python ner.py --corpus base --seed 48 (Will run simply the base tokenizer)

# Run on UMLS-augmented corpus with seed 123
python ner.py --corpus umls --seed 48 (Will run the UMLS based K-Tokenizer)

Outputs

The script outputs token-level prediction results

Requirements

Python 3.7+

Dependencies listed in requirements.txt (install via pip install -r requirements.txt)

