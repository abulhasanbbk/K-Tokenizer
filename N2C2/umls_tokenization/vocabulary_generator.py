#!/usr/bin/env python
# coding=utf-8
"""
Author: Abul Hasan
Date: 11/09/2023
A complete step by step process for creating a K-Tokenizer from UMLS
Assuming that we are given dictionaries of drugs and symptoms. These dictionaries are created using UMLS Metamorphosis tool.
Updated on : 06/12/2023
This program will only create drug and symptom vocabularies given number of max iterations. 
Then it will be used to find a sufficient lambda set to create the K-Tokenizer vocabulary 
"""
from utils import (BPEVocabularyBuilder, create_directory_if_not_exists)
import json
import json
import pickle
import os
from collections import OrderedDict
import argparse
if __name__ == "__main__":
    #print()
    parser = argparse.ArgumentParser()
    # parse arguments
    parser.add_argument("--drug_data_path", type=str, required=True,
                        help="Provide location of a drug dictionary ")
    parser.add_argument("--symptom_data_path", type=str, required=True,
                        help="Provide location of symptom dictionary")
    parser.add_argument("--max_iterations", type=int, required=True,
                        help="Provide number of merge operations to perform ")
    args = parser.parse_args()
    drug_data_path=args.drug_data_path
    drug_global_voc_path="drug_char_n_gram_voc.json"
    symptom_data_path=args.symptom_data_path
    symptom_global_voc_path="symptom_char_n_gram_voc.json"
    max_iterations=args.max_iterations
    drug_bpe_builder = BPEVocabularyBuilder(drug_data_path, drug_global_voc_path)
    drug_bpe_builder.build_bpe_vocabulary(max_iterations)
    print("**********************************************************")
    print("Length of drug bpe vocabulary:", len(drug_bpe_builder.get_bpe_vocabulary()))
    drug_bpe_path="./drug_bpe_from_umls_"+str(max_iterations)+"_.json"
    print("***********************************************************")
    drug_bpe_builder.save_bpe_vocabulary(drug_bpe_path)    
    symptom_bpe_builder = BPEVocabularyBuilder(symptom_data_path, symptom_global_voc_path)
    symptom_bpe_builder.build_bpe_vocabulary(max_iterations)
    print("*************************************************************************")
    print("Length of drug bpe vocabulary:", len(symptom_bpe_builder.get_bpe_vocabulary()))
    print("**************************************************************************")
    symptom_bpe_path="./symptom_bpe_from_umls_"+str(max_iterations)+"_.json"
    symptom_bpe_builder.save_bpe_vocabulary(symptom_bpe_path)
    #saving locations for bert-base-cased