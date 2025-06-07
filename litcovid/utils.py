"""KTokenization class for ICD-9 Coding: 11/10/2023
"""
from transformers import AutoTokenizer
import math
import json
import spacy
import os
import time
if 'en_core_web_sm' not in spacy.util.get_installed_models():
    # If not installed, download and load the model
    spacy.cli.download("en_core_web_sm")
SPACEY_TOKENIZER = spacy.load('en_core_web_sm')

#UMLS_N_GRAM_LOCATION="/working/abul/tokenization/UMLS_BERT_KTokenization/umls_n_gram_vocabulary.json"
#BERNOULI_CHAR_NGRAM_DIST="/working/abul/sentence_classification_exp/bernoulli.pkl"
#UMLS_N_GRAM_LOCATION="/working/abul/tokenization/MIMICIII_BERT_KTokenization/mimic_n_gram_vocabulary.json"
UMLS_N_GRAM_FILE_NAME="umls_n_gram_vocabulary.json"
MIMICIII_N_GRAM_FILE_NAME="mimic_n_gram_vocabulary.json"
PUBMED_N_GRAM_FILE_NAME="pumbed_n_gram_vocabulary.json"
"""Generation of character ngrams
"""
class Loc:
   def __init__(self):
      self.base_dir=None
      self.base=None
      self.drug=None
      self.disease=None
      self.ktok=None

class PubmedBERTFromUMLSFileNames:
    """Class to store the lacations for tokenizers
   We need to 
   """
    def __init__(self):
        self.base_tok="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.base_loc_drug="drug-umls-pubmed-base-cased-"
        self.base_loc_disease="disease-umls-pubmed-base-cased-"
        self.base_loc_ktok="ktokenizer-umls-pubmed-base-cased-"
class PubmedBERTFromPubMedFileNames:
    """Class to store the lacations for tokenizers
   We need to 
   """
    def __init__(self):
        self.base_tok="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.base_loc_drug="drug-pubmed-pubmed-base-cased-"
        self.base_loc_disease="disease-pubmed-pubmed-base-cased-"
        self.base_loc_ktok="ktokenizer-pubmed-pubmed-base-cased-"
class BioformerFromPubMedFileNames:
   """Class to store the lacations for tokenizers
   We need to 
   """
   def __init__(self):
      self.base_tok="bioformers/bioformer-16L"
      self.base_loc_drug="drug-bioformer-pubmed-base-cased-"
      self.base_loc_disease="disease-bioformer-pubmed-base-cased-"
      self.base_loc_ktok="ktokenizer-bioformer-pubmed-base-cased-"

class BioformerFromUMLSFileNames:
   """Class to store the lacations for tokenizers
   We need to 
   """
   def __init__(self):
      self.base_tok="bioformers/bioformer-16L"
      self.base_loc_drug="drug-bioformer-umls-base-cased-"
      self.base_loc_disease="disease-bioformer-umls-base-cased-"
      self.base_loc_ktok="ktokenizer-bioformer-umls-base-cased-"

class ClinicalBERTFromUMLSFileNames:
   """Class to store the lacations for tokenizers
   We need to 
   """
   def __init__(self):
      self.base_tok="emilyalsentzer/Bio_ClinicalBERT"
      self.base_loc_drug="drug-umls-clinicalBert-base-cased-"
      self.base_loc_disease="disease-umls-clinicalBert-base-cased-"
      self.base_loc_ktok="ktokenizer-umls-clinicalBert-base-cased-"

class GatartronFromUMLSFileNames:
   """Class to store the lacations for tokenizers
   We need to 
   """
   def __init__(self):
      self.base_tok="UFNLP/gatortron-base"
      self.base_loc_drug="drug-umls-gatartron-base-cased-"
      self.base_loc_disease="disease-umls-gatartron-base-cased-"
      self.base_loc_ktok="ktokenizer-umls-gatartron-base-cased-"


class ClinicalBERTFromMIMICFileNames:
   """Class to store the lacations for tokenizers
   We need to 
   """
   def __init__(self):
      self.base_tok="emilyalsentzer/Bio_ClinicalBERT"
      self.base_loc_drug="drug-mimic-clinicalBert-base-cased-"
      self.base_loc_disease="disease-mimic-clinicalBert-base-cased-"
      self.base_loc_ktok="ktokenizer-mimic-clinicalBert-base-cased-"

class InfusedClinicalBertFromUMLSFileNames:
    """Class to store Embedding initialized clinical bert model file name
    """
    def __init__(self):
      self.infused_bert_model="infused-clinical-bert-model-umls-"

class InfusedClinicalBertFromMIMICFileNames:
    """Class to store Embedding initialized clinical bert model file name
    """
    def __init__(self):
      self.infused_bert_model="infused-clinical-bert-model-mimic-"

class InfusedGatartronFromUMLSFileNames:
    """Class to store Embedding initialized Gatartron model file name
    """
    def __init__(self):
      self.infused_bert_model="infused-gatartron-model-umls-"

class InfusedPubmedFromUMLSFileNames:
    """Class to store Embedding initialized Gatartron model file name
    """
    def __init__(self):
      self.infused_bert_model="infused-pubmed-model-umls-"

class InfusedPubmedFromPubMedFileNames:
    """Class to store Embedding initialized Gatartron model file name
    """
    def __init__(self):
      self.infused_bert_model="infused-pubmed-model-pubmed-"

class InfusedBioformerFromPubMedFileNames:
    """Class to store Embedding initialized Gatartron model file name
    """
    def __init__(self):
      self.infused_bert_model="infused-bioformer-model-pubmed-"

class InfusedBioformerFromUMLSFileNames:
    """Class to store Embedding initialized Gatartron model file name
    """
    def __init__(self):
      self.infused_bert_model="infused-bioformer-model-umls-"


class BackOffDictionaryOfClinicalBertFromUMLSFileNames:
    """Class to store  backoff dicitonary of clinical bert model from the UMLS
    """
    def __init__(self):
      self.back_off_dictionary="backoff-dicitonary-clinical-bert-model-umls-"

class BackOffDictionaryOfClinicalBertFromMIMICFileNames:
    """Class to store  backoff dicitonary of clinical bert model from the UMLS
    """
    def __init__(self):
      self.back_off_dictionary="backoff-dicitonary-clinical-bert-model-mimic-"

class BackOffDictionaryOfGatartronFromUMLSFileNames:
    """Class to store  backoff dicitonary of Gatartron model from the UMLS
    """
    def __init__(self):
      self.back_off_dictionary="backoff-dicitonary-gatartron-model-umls-"

class BackOffDictionaryOfPubmedFromUMLSFileNames:
    """Class to store  backoff dicitonary of Gatartron model from the UMLS
    """
    def __init__(self):
      self.back_off_dictionary="backoff-dicitonary-pubmed-model-umls-"

class BackOffDictionaryOfPubmedFromPubMedFileNames:
    """Class to store  backoff dicitonary of Gatartron model from the UMLS
    """
    def __init__(self):
      self.back_off_dictionary="backoff-dicitonary-pubmed-model-pubmed-"
class BackOffDictionaryOfBioformerFromPubMedFileNames:
    """Class to store  backoff dicitonary of Gatartron model from the UMLS
    """
    def __init__(self):
      self.back_off_dictionary="backoff-dicitonary-bioformer-model-pubmed-"

class BackOffDictionaryFromUMLSLoc:
    def __init__(self):
        loc="back_off_dict_from_umls"
        check_or_create_folder(loc)
        self.location=loc
class BackOffDictionaryFromPubMedLoc:
    def __init__(self):
        loc="back_off_dict_from_pubmed"
        check_or_create_folder(loc)
        self.location=loc
class BackOffDictionaryFromMIMICLoc:
    def __init__(self):
        loc="back_off_dict_from_mimic"
        check_or_create_folder(loc)
        self.location=loc

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    doc =SPACEY_TOKENIZER (text)
    # Access the tokens
    tokens = [token.text for token in doc]
    return tokens

def check_or_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)

def relative_fertility(bert_tokens, k_bert_tokens):
    f_b=len(bert_tokens)
    f_k=len(k_bert_tokens)
    f=(f_b-f_k)/f_b
    return f
class KInput:
    def __init__(self, input_ids=None, attention_mask=None, word_ids=None, labels=None):
        self.input_ids = input_ids or {}
        self.attention_mask = attention_mask or {}
        self.word_ids = word_ids or {}
        self.labels=labels or {}
    def __getitem__(self, key):
        if key == 'input_ids':
            return self.input_ids
        elif key == 'attention_mask':
            return self.attention_mask
        elif key == 'word_ids':
            return self.word_ids
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if key == 'input_ids':
            self.input_ids = value
        elif key == 'attention_mask':
            self.attention_mask = value
        elif key == 'word_ids':
            self.word_ids = value
        elif key=='labels':
            self.labels=value
        else:
            raise KeyError(key)


class KTokenizer:
    def __init__(self,locations,fertility=0):
        print(locations.base)
        print(locations.drug)
        print(locations.disease)
        print(locations.ktok)
        self.tok=AutoTokenizer.from_pretrained(locations.base)
        self.drug_tok=AutoTokenizer.from_pretrained(locations.drug)
        self.disease_tok=AutoTokenizer.from_pretrained(locations.disease)
        self.k_tok=AutoTokenizer.from_pretrained(locations.ktok)
        #self.tok=AutoTokenizer.from_pretrained(self.model_name)
        #with open(UMLS_N_GRAM_LOCATION, 'rb') as f:
        #    self.n_gram_voc= json.load(f)
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
        self.sep_token_id=self.k_tok.sep_token_id
        self.roll_back=0
        self.f=fertility
        self.max_length=512
    
    def __call__(self, 
                 texts,
                 batch=True):
        if batch==True:
            return_inputs={'input_ids':[], 'attention_mask': [], 'token_type_ids':[]}
            for text in texts:
                splitted_text= whitespace_tokenize(text)
                kinputs= self.encode(splitted_text)
                for key in kinputs:
                    return_inputs[key].append(kinputs[key])
            return return_inputs
    
    def get_special_tokens_mask(self, input_ids, already_has_special_tokens=True):
        return self.k_tok.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    
    def convert_tokens_to_ids(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.k_tok.convert_tokens_to_ids(token)

    #def convert_id_to_token(self, index):
    #    """Converts an index (integer) in a token (str) using the vocab."""
    #    return self.tok.convert_tokens_to_ids(index)
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
        if is_bert_tokenizer:
            return self.tok.tokenize(w.lower())
        else:
            a=self.drug_tok.tokenize(w.lower())
            b=self.disease_tok.tokenize(w.lower())
            c=self.tok.tokenize(w.lower())
            a1=self._entropy(a)
            b1=self._entropy(b)
            c1=self._entropy(c)
            th=1200/self.tf
            thent=-th*math.log2(th)
            if thent>a1 and thent>b1:
                return c
            if a1 <= b1 and a1 <= c1:
                return a
            elif b1 <= a1 and b1 <= c1:
                return b
            else:
                return c
            
    
    def encode(self,text):
        tokens = list()
        attn_msk=list()
        token_type_ids=list()
        is_bert_tokenizer=False
        for i,w in enumerate(text):
            ws=self.tokenize(w, is_bert_tokenizer)
            tokens.extend(ws)
            attn_msk.extend([1]*len(ws))
            token_type_ids.extend([0]*len(ws))
        #Now check which tokenizer to use for the sentence
        bert_tokens=self.tok.tokenize(text, is_split_into_words=True)
        if self.f<=0.0: 
            is_bert_tokenizer=True
        elif self.f>=1.00:
            is_bert_tokenizer=False
        elif relative_fertility(bert_tokens, tokens)>=self.f:
            is_bert_tokenizer=True
        else:
            pass       
        if is_bert_tokenizer:
            self.roll_back+=1
            #then tokenize one by one again
            tokens = list()
            attn_msk=list()
            token_type_ids=list()
            #ws = self.tokenize(w, is_bert_tokenizer)  # Tokenize the word
            tokens.extend(bert_tokens)
            attn_msk.extend([1]*len(bert_tokens))
            token_type_ids.extend([0]*len(bert_tokens))
        ids_k = self.k_tok.convert_tokens_to_ids(tokens)[:self.max_length - 2]
        ids_k=[self.k_tok.cls_token_id] + ids_k + [self.k_tok.sep_token_id]
        attn_msk=[1]+attn_msk[:self.max_length- 2]+[1]
        token_type_ids=[0]+token_type_ids[:self.max_length - 2]+[0]
        #add padding 
        ids_k = ids_k+ [self.pad_token_id] * (self.max_length - len(ids_k))
        attn_msk=attn_msk+[0] * (self.max_length - len(attn_msk))
        token_type_ids=token_type_ids+[0]*(self.max_length - len(token_type_ids))
        assert len(ids_k)==len(attn_msk)
        assert len(ids_k)==len(token_type_ids)
        kinput={'input_ids':ids_k, 'attention_mask': attn_msk, 'token_type_ids':token_type_ids}
        return kinput
    
    def save_pretrained(self, filename):
        self.k_tok.save_pretrained(filename)

class KTokenizerFromUMLS(KTokenizer):
    def __init__(self,locations, fertilities=0):
        
        file_path = os.path.join(locations.base_dir, UMLS_N_GRAM_FILE_NAME)
        with open(file_path, 'rb') as f:
            self.n_gram_voc= json.load(f)
        super().__init__(locations, fertilities)

class KTokenizerFromPubMed(KTokenizer):
    def __init__(self,locations, fertilities=0):
        
        file_path = os.path.join(locations.base_dir,PUBMED_N_GRAM_FILE_NAME)
        with open(file_path, 'rb') as f:
            self.n_gram_voc= json.load(f)
        super().__init__(locations, fertilities)

class KTokenizerFromMIMICIII(KTokenizer):
    def __init__(self,locations, fertilities=0):
        
        file_path = os.path.join(locations.base_dir, MIMICIII_N_GRAM_FILE_NAME)
        with open(file_path, 'rb') as f:
            self.n_gram_voc= json.load(f)
        super().__init__(locations, fertilities)