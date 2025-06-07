"""KTokenization class for ICD-9 Coding: 11/10/2023
"""
from transformers import AutoTokenizer
import math
import json
import spacy
import os
if 'en_core_web_sm' not in spacy.util.get_installed_models():
    # If not installed, download and load the model
    spacy.cli.download("en_core_web_sm")
SPACEY_TOKENIZER = spacy.load('en_core_web_sm')
UMLS_N_GRAM_FILE_NAME="umls_n_gram_vocabulary.json"
MIMICIII_N_GRAM_FILE_NAME="mimic_n_gram_vocabulary.json"
#UMLS_N_GRAM_LOCATION="../umls_tokenization/umls_n_gram_vocabulary.json"
#BERNOULI_CHAR_NGRAM_DIST="/working/abul/sentence_classification_exp/bernoulli.pkl"
#UMLS_N_GRAM_LOCATION="/working/abul/tokenization/MIMICIII_BERT_KTokenization/mimic_n_gram_vocabulary.json"
"""Generation of character ngrams
"""
class ClinicalBERTFromUMLSFileNames:
   """Class to store the lacations for tokenizers
   We need to 
   """
   def __init__(self):
      self.base_tok="bert-base-cased"
      self.base_loc_drug="drug-umls-bert-base-cased-10000"
      self.base_loc_disease="disease-umls-bert-base-cased-3000"
      self.base_loc_ktok="ktokenizer-umls-bert-base-cased-13000"

class ClinicalBERTFromMIMICFileNames:
   """Class to store the lacations for tokenizers
   We need to 
   """
   def __init__(self):
      self.base_tok="emilyalsentzer/Bio_ClinicalBERT"
      self.base_loc_drug="drug-mimic-clinicalBert-base-cased-20-9"
      self.base_loc_disease="disease-mimic-clinicalBert-base-cased-20-9"
      self.base_loc_ktok="ktokenizer-mimic-clinicalBert-base-cased-20-9"
class Loc:
   def __init__(self):
      self.base_dir=None
      self.base=None
      self.drug=None
      self.disease=None
      self.ktok=None
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    doc =SPACEY_TOKENIZER (text)
    # Access the tokens
    tokens = [token.text for token in doc]
    return tokens
def relative_fertility(bert_tokens, k_bert_tokens):
    f_b=len(bert_tokens)
    f_k=len(k_bert_tokens)
    f=(f_b-f_k)/f_b
    return f

class KTokenizer:
    def __init__(self, locations, fertility=0):
        self.tok=AutoTokenizer.from_pretrained(locations.base)
        self.drug_tok=AutoTokenizer.from_pretrained(locations.drug)
        self.disease_tok=AutoTokenizer.from_pretrained(locations.disease)
        self.k_tok=AutoTokenizer.from_pretrained(locations.ktok)
        self.tf=0
        for n in self.n_gram_voc:
            self.tf+=self.n_gram_voc[n] 
        self.special="##"
        self.mask_token=self.k_tok.mask_token
        self.mask_token_id=self.k_tok.mask_token_id
        self._pad_token=self.k_tok._pad_token
        self.padding_side=self.k_tok.padding_side
        self.pad_token_id=self.k_tok.pad_token_id
        self.pad=self.k_tok.pad
        self.w_drug_label="D"
        self.roll_back=0
        self.f=fertility
        self.max_length=512
    # Callable method to process a list of texts, encode them, and create input for a model.
    def __call__(self, texts, batch=True):
        if batch == True:
            return_inputs={'input_ids':[], 'attention_mask': [], 'token_type_ids':[]}
            for text in texts:
                text= whitespace_tokenize(text)
                kinputs= self.encode(text)
                for key in kinputs:
                    return_inputs[key].append(kinputs[key])
            return return_inputs
            

           
    def get_special_tokens_mask(self, input_ids, already_has_special_tokens=True):
        return self.k_tok.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    
    def convert_tokens_to_ids(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.k_tok.convert_tokens_to_ids(token)

    def __len__(self):
        return len(self.k_tok)
    def _remove_hashtag(self,string):
        """
        Remove '##' character from the beginning of a string.
        """
        if string.startswith(self.special):
            return string[2:]
        else:
            return string
    def _entropy(self,word_list):
        #calculate frequency from subword list
        f=0
        for l in word_list:
            if l in self.n_gram_voc:
                f+=self.n_gram_voc[l]
            else:
                f+=1
        probability=(f+1)/self.tf
        entropy=-probability * math.log2(probability)
        return entropy
        
    def tokenize(self, w, is_bert_tokenizer=False):
        #checking if the word is uppercase.
        #If word is upper case then we ask BERT to tokenize it
        c=self.tok.tokenize(w)
        if is_bert_tokenizer:
            return c
        elif w.isupper():
            return c
        else:
            a=self.drug_tok.tokenize(w)
            b=self.disease_tok.tokenize(w)
            a1=self._entropy(a)
            b1=self._entropy(b)
            c1=self._entropy(c)
            th=1000/self.tf
            thent=-th*math.log2(th)
            if thent>a1 and thent>b1:
                return c
            if a1 <= b1 and a1 <= c1:
                return a
            elif b1 <= a1 and b1 <= c1:
                return b
            else:
                return c
            
    
    # Method to tokenize and encode a given text using specific tokenization rules.
    def encode(self, text):
        tokens = list()
        attn_msk=list()
        token_type_ids=list()
        #Get K-Tokenizer
        is_bert_tokenizer=False
        for i,w in enumerate(text):
            ws = self.tokenize(w, is_bert_tokenizer)  # Tokenize the word
            # Extend the token and word_ids lists with tokenized words and corresponding indices
            tokens.extend(ws)
            attn_msk.extend([1]*len(ws))
            token_type_ids.extend([0]*len(ws))
    
        #Now check which tokenizer to use for the sentence
        bert_tokens=self.tok.tokenize(text, is_split_into_words=True)
        if self.f<=0.0:
            self.roll_back+=1
            is_bert_tokenizer=True
        elif self.f>=1.00:
            is_bert_tokenizer=False
        elif relative_fertility(bert_tokens, tokens)>self.f:
            is_bert_tokenizer=True
            self.roll_back+=1
        else:
            pass
        if is_bert_tokenizer:
            #then tokenize one by one again
            tokens = list()
            attn_msk=list()
            token_type_ids=list()
            for i,w in enumerate(text):
                ws = self.tokenize(w, is_bert_tokenizer)  # Tokenize the word
                tokens.extend(ws)
                attn_msk.extend([1]*len(ws))
                token_type_ids.extend([0]*len(ws))
        
        ids_k = self.k_tok.convert_tokens_to_ids(tokens)[:self.k_tok.model_max_length - 2]
        ids_k=[self.k_tok.cls_token_id] + ids_k + [self.k_tok.sep_token_id]
        attn_msk=[1]+attn_msk[:self.k_tok.model_max_length - 2]+[1]
        token_type_ids=[0]+token_type_ids[:self.k_tok.model_max_length - 2]+[0]
        assert len(ids_k)==len(attn_msk)
        assert len(ids_k)==len(token_type_ids)
        kinput={'input_ids':ids_k, 'attention_mask': attn_msk, 'token_type_ids':token_type_ids}
        return kinput
    
    def get_vocab(self):
        return self.k_tok.get_vocab()
    
    def save_pretrained(self, filename):
        self.k_tok.save_pretrained(filename)

class KTokenizerFromUMLS(KTokenizer):
    def __init__(self,locations, fertilities=0):
        
        file_path = os.path.join(locations.base_dir, UMLS_N_GRAM_FILE_NAME)
        with open(file_path, 'rb') as f:
            self.n_gram_voc= json.load(f)
        super().__init__(locations, fertilities)

class KTokenizerFromMIMICIII(KTokenizer):
    def __init__(self,locations, fertilities=0):
        
        file_path = os.path.join(locations.base_dir, MIMICIII_N_GRAM_FILE_NAME)
        with open(file_path, 'rb') as f:
            self.n_gram_voc= json.load(f)
        super().__init__(locations, fertilities)