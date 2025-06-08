# ner.py

A script for clincial phenotype extraction using K-Tokenizer.

## Pre-processing data
- Data can be found in https://www.physionet.org/content/radgraph/1.0.0/. If unavailable please contact abul.hasan@phc.ox.ac.uk 
- Run ```pre_process_flat_to_bio.py``` program to create train and test set in onrder to work with our program.

## Usage

To run the ner.py script, execute:

python ner.py --corpus <base|umls|mimic> --seed <your seed> --data_type<type of test dataset>

## Arguments

--corpus  : Specify the corpus to use. Must be one of:

base  — using ClinicalBERT tokenizer

umls  — using K-Tokenizer built from UMLS.

mimic — using K-Tokenizer built from MIMIC-III dataset.

--seed    : Integer seed for random number generators to ensure reproducibility.

--data_type : xr or chexpert
Examples

# Run on the base tokenizer with seed 48
env/bin/python ner.py --corpus base --seed 48 --data_type cxr(Will run simply the base tokenizer)

# Run on K-Tokenzer with seed 48
python ner.py --corpus umls --seed 48 --data_type cxr (Will run the UMLS based K-Tokenizer)

Outputs

The script outputs token-level prediction results

Requirements

Python 3.7+

Dependencies listed in requirements.txt (install via pip install -r requirements.txt)

