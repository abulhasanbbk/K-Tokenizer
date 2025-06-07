#!/usr/bin/env python
# coding=utf-8
"""
Author: Abul Hasan
Date: 11/09/2023
A complete step by step process for creating a K-Tokenizer from UMLS
Assuming that we are given dictionaries of drugs and symptoms. These dictionaries are created using UMLS Metamorphosis tool.
"""
from utils import (BPEVocabularyBuilder, create_directory_if_not_exists)
import json
import json
import pickle
import os
from collections import OrderedDict
from transformers import AutoTokenizer
import math
import argparse

def convert_to_ordered_dict(input_dict):
    # Sort the dictionary by keys and create an ordered dictionary.
    ordered_dict = OrderedDict(sorted(input_dict.items()))

    return ordered_dict

def most_common_elements(ordered_dict, n):
    # Ensure n is within the valid range.
    if n <= 0:
        return {}

    # Get the first n key-value pairs from the ordered dictionary.
    common_elements = list(ordered_dict.items())[:n]

    # Create a new dictionary from the selected key-value pairs.
    result_dict = dict(common_elements)

    return result_dict


def remove_word_pieces(v, n, retain_all=False):
    o=convert_to_ordered_dict(v)
    if retain_all:
        return o
    r=most_common_elements(o,n)
    return r

def calculate_entropy(probs):
    """Entropy calculation for a set of probabilities
    """
    h=0
    for p_i in probs:
        h+=p_i*math.log(p_i)
    return -h

def calculate_prob(voc_dict):
    """Caclulate the probabilities for given vocabulary with frequency
    """
    probs=[]
    tf=0
    #calculate total probability
    for key in voc_dict:
        tf+=voc_dict[key]
    #populate probability list for the given vocabulary
    for key in voc_dict:
        p_i=voc_dict[key]/tf
        probs.append(p_i)
    return probs

def calculate_sufficient_delta_set(probs, w=False):
    """Calculating sufficient delta set following David McKays book
    """
    p_delta=[]
    H=[]
    l=[]
    mn=min(probs) #Find the minimum probability
    probs.sort(reverse=True)
    for i in range(0,10): #Finding entropies for sets setting thresholds for minimum probability+ 0.1+e06
        mn=mn+0.000001
        delta_set = [p for p in probs if p > mn]
        print(len(delta_set))
        h=calculate_entropy(delta_set)
        p_delta.append(mn)
        H.append(h)
        if w:
            l.append(int(len(delta_set)*0.3))
        else: 
            l.append(len(delta_set))
    return p_delta, H, l

class KTokenizerBuilder:
    def __init__(self, language_model, saving_locations):
        self.vocabulary_file="tokenizer.json"
        self.saving_locations=saving_locations
        self.base_vocab = self._load_original_vocab(saving_locations["bert"], language_model)
        self.drug_vocab = self._load_original_vocab(saving_locations["D"], language_model)
        self.ds_vocab = self._load_original_vocab(saving_locations["S"], language_model)
        self.k_vocab = self._load_original_vocab(saving_locations["K"], language_model)
        self.vocab_dict = {
            "D": self.drug_vocab,
            "S": self.ds_vocab
        }

    def _load_original_vocab(self, saving_location, language_model):
        print("Saving in: ", saving_location)
        self._save_tokenizer(saving_location, language_model)
        vocabulary_path=os.path.join(saving_location, self.vocabulary_file)
        with open(vocabulary_path, 'rb') as f:
            vocab= json.load(f)
        return vocab
    
    def _save_tokenizer(self, saving_location, language_model=None, vocab=None):
        create_directory_if_not_exists(saving_location)
        if language_model:
            tokenizer=AutoTokenizer.from_pretrained(language_model)
            tokenizer.save_pretrained(saving_location)
        elif vocab:
            vocabulary_path=os.path.join(saving_location, self.vocabulary_file)
            with open(vocabulary_path, 'w') as f:
                json.dump(vocab,f)
        else:
             raise Exception("Cannot save tokenizer")
    
    
    def build_semantic_tokenizer(self, semantic_type, bpe_file_loc, min_word_pieces):
        """
            Build a semantic tokenizer for a given semantic type.

            Parameters:
            semantic_type (str): The semantic type, either "D" for drug or "S" for disease.
            bpe_file_loc (str): The file location of the BPE vocabulary to load.
            min_word_pieces (int): The maximum number of word pieces to be included.
            saving_loc (str): The location to save the tokenizer.

            Output:
            None

            This function builds a semantic tokenizer based on the specified semantic type. It loads a BPE vocabulary from the
            provided file location, removes word pieces with counts less than min_word_pieces, updates the vocabulary
            of the selected semantic type, prints the length of the updated vocabulary, and saves the tokenizer to the specified
            saving location.

            Example:
                To build a drug tokenizer with semantic_type="D":
                build_semantic_tokenizer("D", "drug_bpe_vocab.json", 5, "drug_tokenizer.pkl")

                To build a disease tokenizer with semantic_type="S":
                build_semantic_tokenizer("S", "disease_bpe_vocab.json", 3, "disease_tokenizer.pkl")
        """
        # Load the BPE vocabulary from the file
        with open(bpe_file_loc, "rb") as f:
            bpe_vocab = json.load(f)
        # Remove word pieces based on min_word_pieces
        r_dr = remove_word_pieces(bpe_vocab, min_word_pieces, retain_all=False)

        # Get the appropriate vocabulary based on semantic_type
        vocab = self.vocab_dict.get(semantic_type)

        if vocab:
            assert len(self.base_vocab["model"]["vocab"])==len(vocab["model"]["vocab"])
            idx = len(vocab["model"]["vocab"])
            for key in r_dr:
                if key not in vocab["model"]["vocab"]:
                    vocab["model"]["vocab"][key] = idx
                    idx += 1
            print(f"length of the {semantic_type} tokenizer: {len(vocab['model']['vocab'])}")
            self._save_tokenizer(self.saving_locations[semantic_type], language_model=None, vocab=vocab)
        else:
            raise Exception("Invalid semantic_type provided.")

        # Print the lengths of base, drug, and disease tokenizers
        print(f"Length of base tokenizer: {len(self.base_vocab['model']['vocab'])}")
        print(f"Length of drug tokenizer: {len(self.drug_vocab['model']['vocab'])}")
        print(f"Length of disease tokenizer: {len(self.ds_vocab['model']['vocab'])}")
    
    def build_k_tokenizer(self,semantic_type):
        assert len(self.base_vocab["model"]["vocab"])==len(self.k_vocab["model"]["vocab"])
        idx=len(self.k_vocab["model"]["vocab"])
        for type in self.vocab_dict:
            vocab= self.vocab_dict.get(type)
            for key in vocab["model"]["vocab"]:
                if key not in self.k_vocab["model"]["vocab"]:
                    self.k_vocab["model"]["vocab"][key]=idx
                    idx+=1
        print("length of the K tokenizer",len(self.k_vocab["model"]["vocab"]))
        self._save_tokenizer(saving_locations[semantic_type], language_model=None, vocab=self.k_vocab)

class ClinicalBERTFromUMLSFileNames:
   """Class to store the lacations for tokenizers
   We need to 
   """
   def __init__(self):
      self.base_tok="emilyalsentzer/Bio_ClinicalBERT"
      self.loc_base="local-clinicalBert-base-cased-"
      self.loc_drug="drug-umls-clinicalBert-base-cased-"
      self.loc_symptom="disease-umls-clinicalBert-base-cased-"
      self.loc_ktok="ktokenizer-umls-clinicalBert-base-cased-"

class GatartronFromUMLSFileNames:
   """Class to store the lacations for tokenizers
   We need to 
   """
   def __init__(self):
      self.base_tok="UFNLP/gatortron-base"
      self.loc_base="local-gatartron-base-cased-"
      self.loc_drug="drug-umls-gatartron-base-cased-"
      self.loc_symptom="disease-umls-gatartron-base-cased-"
      self.loc_ktok="ktokenizer-umls-gatartron-base-cased-"

class PubmedBERTFromUMLSFileNames:
    """Class to store the lacations for tokenizers
   We need to 
   """
    def __init__(self):
        self.base_tok="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.loc_base="local-pubmed-base-cased-"
        self.loc_drug="drug-umls-pubmed-base-cased-"
        self.loc_symptom="disease-umls-pubmed-base-cased-"
        self.loc_ktok="ktokenizer-umls-pubmed-base-cased-"

class BioformerFromUMLSFileNames:
    """Class to store the lacations for tokenizers
   We need to 
   """
    def __init__(self):
        self.base_tok="bioformers/bioformer-16L"
        self.loc_base="local-bioformer-base-cased-"
        self.loc_drug="drug-bioformer-umls-base-cased-"
        self.loc_symptom="disease-bioformer-umls-base-cased-"
        self.loc_ktok="ktokenizer-bioformer-umls-base-cased-"

class BPEfileNamesfromUMLS:
    """class to store bpe file names
    """
    def __init__(self):
      self.drug_bpe="./drug_bpe_from_umls_"
      self.symptom_bpe="./symptom_bpe_from_umls_"

if __name__ == "__main__":
    #print()
    parser = argparse.ArgumentParser()
    # parse arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Provide model name: c for clinicalBERT, g for Gatartron, p for pubmed ")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Provide corpus name umls, mimic")
    parser.add_argument("--max_iterations", type=int, required=True,
                        help="Provide number of merge operations performed")
    args = parser.parse_args()
    model= args.model
    corpus=args.corpus
    max_iterations=args.max_iterations
    """File names for character n-grams will remain same
    """
    drug_global_voc_path="drug_char_n_gram_voc.json"
    symptom_global_voc_path="symptom_char_n_gram_voc.json"
    global_voc_path="umls_n_gram_vocabulary.json"
    if model=='c' and corpus=='umls':
        tokenizerFileNames=ClinicalBERTFromUMLSFileNames()
        bpeFileNames= BPEfileNamesfromUMLS()
    elif model=='g' and corpus=='umls':
        tokenizerFileNames=GatartronFromUMLSFileNames()
        bpeFileNames= BPEfileNamesfromUMLS()
    elif model=='p' and corpus=='umls':
        tokenizerFileNames=PubmedBERTFromUMLSFileNames()
        bpeFileNames= BPEfileNamesfromUMLS()
    elif model=='b' and corpus=='umls':
        tokenizerFileNames=BioformerFromUMLSFileNames()
        bpeFileNames= BPEfileNamesfromUMLS()
    else:
        pass
    language_model=tokenizerFileNames.base_tok
    #saving locations for bert-base-cased
    drug_bpe_path=bpeFileNames.drug_bpe+str(max_iterations)+"_.json"
    symptom_bpe_path=bpeFileNames.symptom_bpe+str(max_iterations)+"_.json"
    print("*************************************************************")
    print("Calculating minimum number of word peices for drug using entropy")
    print("*************************************************************")
    with open(drug_bpe_path, "rb") as f:
        drug_voc= json.load(f)
    probs=calculate_prob(drug_voc)
    lambda_values, entropies, l_drug=calculate_sufficient_delta_set(probs, w=False)
    print(l_drug)
    print("*************************************************************")
    print("Calculating minimum number of word peices for symptoms using entropy")
    print("*************************************************************")
    with open(symptom_bpe_path, "rb") as f:
        symptom_voc= json.load(f)
    probs=calculate_prob(symptom_voc)
    lambda_values, entropies, l_symptom=calculate_sufficient_delta_set(probs, w=True)
    print(l_symptom)
    print("*************************************************************")
    print("Starting to create K-tokenizers")
    print("*************************************************************")
    l_drug=[10000]
    l_symptom=[3000]
    for idx,l in enumerate(l_drug):
        nd=l #number of drugs
        ns=l_symptom[idx] #number of symptoms
        saving_locations={ 
            "bert":tokenizerFileNames.loc_base+str(max_iterations),
            "D":tokenizerFileNames.loc_drug+str(max_iterations)+"-"+str(idx),
            "S":tokenizerFileNames.loc_symptom+str(max_iterations)+"-"+str(idx),
            "K":tokenizerFileNames.loc_ktok+str(max_iterations)+"-"+str(idx),
        }
        
        builder= KTokenizerBuilder(language_model,saving_locations)
        builder.build_semantic_tokenizer(semantic_type="D", bpe_file_loc=drug_bpe_path, min_word_pieces=nd)
        builder.build_semantic_tokenizer(semantic_type="S", bpe_file_loc=symptom_bpe_path, min_word_pieces=ns)
        builder.build_k_tokenizer(semantic_type="K")
        print("***************************************************************************")
        print("K-tokenizer saved in: ")
        print(saving_locations)
        print("***************************************************************************")
    
    
    """Build UMLS ngram vocabulary json which combines drug and symptom
    """
    
    with open(drug_global_voc_path, 'rb') as f:
            drug_global_voc= json.load(f)

    with open(symptom_global_voc_path, 'rb') as f:
            symptom_global_voc= json.load(f)
    ##Build the total global voc
    global_voc={}
    for key in drug_global_voc:
        global_voc[key]=drug_global_voc[key]
    for key in symptom_global_voc:
        if key in global_voc:
            global_voc[key]+=symptom_global_voc[key]
        else:
            global_voc[key]=symptom_global_voc[key]
    #save gobal_vooc
    with open(global_voc_path, 'w') as f:
                json.dump(global_voc,f)
       