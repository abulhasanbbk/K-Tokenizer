from transformers import AutoTokenizer
import json
import math
import os

#location of all n-gram statistics
UMLS_N_GRAM_FILE_NAME="umls_n_gram_vocabulary.json"
MIMICIII_N_GRAM_FILE_NAME="mimic_n_gram_vocabulary.json"
#GPT generated function
def check_or_create_folder(folder_name):
    current_directory = os.getcwd()  # Get the current working directory

    folder_path = os.path.join(current_directory, folder_name)

    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder '{folder_name}' created successfully.")
        except OSError as e:
            print(f"Error creating folder: {e}")
    else:
        print(f"Folder '{folder_name}' already exists.")
class Loc:
   def __init__(self):
      self.base_dir=None
      self.base=None
      self.drug=None
      self.disease=None
      self.ktok=None

class ClinicalBERTFromUMLSFileNames:
   """Class to store the lacations for tokenizers
   We need to 
   """
   def __init__(self):
      self.base_tok="bert-base-cased"
      self.base_loc_drug="drug-umls-bert-base-cased-10000"
      self.base_loc_disease="disease-umls-bert-base-cased-3000"
      self.base_loc_ktok="ktokenizer-umls-bert-base-cased-13000"

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
      self.base_loc_drug="drug-mimic-clinicalBert-base-cased-20-0"
      self.base_loc_disease="disease-mimic-clinicalBert-base-cased-20-0"
      self.base_loc_ktok="ktokenizer-mimic-clinicalBert-base-cased-20-0"

class InfusedClinicalBertFromUMLSFileNames:
    """Class to store Embedding initialized clinical bert model file name
    """
    def __init__(self):
      self.infused_bert_model="infused_cinical_bert_model_extended"

class InfusedClinicalBertFromMIMICFileNames:
    """Class to store Embedding initialized clinical bert model file name
    """
    def __init__(self):
      self.infused_bert_model="infused-clinical-bert-model-mimic-20-0"

class InfusedGatartronFromUMLSFileNames:
    """Class to store Embedding initialized Gatartron model file name
    """
    def __init__(self):
      self.infused_bert_model="infused-gatartron-model-umls-"

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

class BackOffDictionaryFromUMLSLoc:
    def __init__(self):
        loc="back_off_dict_from_umls"
        check_or_create_folder(loc)
        self.location=loc
class BackOffDictionaryFromMIMICLoc:
    def __init__(self):
        loc="back_off_dict_from_mimic"
        check_or_create_folder(loc)
        self.location=loc


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
    def __init__(self,locations, fertilities=0):
        self.tok=AutoTokenizer.from_pretrained(locations.base)
        self.drug_tok=AutoTokenizer.from_pretrained(locations.drug)
        self.disease_tok=AutoTokenizer.from_pretrained(locations.disease)
        self.k_tok=AutoTokenizer.from_pretrained(locations.ktok)
        
        #with open(UMLS_N_GRAM_LOCATION, 'rb') as f:
        #    self.n_gram_voc= json.load(f)
        #self.n_gram_voc=None 
        
        self.tf=0
        for n in self.n_gram_voc:
            self.tf+=self.n_gram_voc[n] 
        print(self.tf)
        self.special="##"
        self.mask_token=self.k_tok.mask_token
        self.mask_token_id=self.k_tok.mask_token_id
        self._pad_token=self.k_tok._pad_token
        self.padding_side=self.k_tok.padding_side
        self.pad_token_id=self.k_tok.pad_token_id
        self.pad=self.k_tok.pad
        self.roll_back=0
        self.f=fertilities
    # Callable method to process a list of texts, encode them, and create input for a model.
    def __call__(self, texts, batch=True, truncation=True, is_split_into_words=True):
        if batch == True:
            ids = []          # List to store input_ids for each text
            word_idx = []     # List to store word_ids for each text
            attn_msk = []     # List to store attention masks for each text
        
            # Iterate through each text in the input list
            for text in texts:
                inputs, w_ids = self.encode(text)  # Encode the text using appropriate tokenizer
                assert len(inputs) == len(w_ids)
            
                ids.append(inputs)   # Append input_ids to the ids list
                word_idx.append(w_ids)  # Append word_ids to the word_idx list
            
                att = [1 for i in range(len(inputs))]  # Create attention mask for the text
                assert len(inputs) == len(att)
                attn_msk.append(att)  # Append attention mask to the attn_msk list
        kinput = {'input_ids': ids, 'attention_mask': attn_msk, 'word_ids': word_idx}
        return kinput  # Return the input dictionary for the model
    
           
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
            a=self.drug_tok.tokenize(w.lower())
            b=self.disease_tok.tokenize(w.lower())
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
        
        tokens = list()        # List to store tokenized words
        word_ids = list()      # List to store corresponding word indices
        
        #Get K-Tokenizer
        is_bert_tokenizer=False
        for i,w in enumerate(text):
            ws = self.tokenize(w, is_bert_tokenizer)  # Tokenize the +word
            # Extend the token and word_ids lists with tokenized words and corresponding indices
            tokens.extend(ws)
            word_ids.extend([i] * len(ws))
        #Now check which toke+nizer to use for the sentence
        bert_tokens=self.tok.tokenize(text, is_split_into_words=True)
        if self.f==0:
            is_bert_tokenizer=True #if fertility is 0 then we roll back to bert
        elif self.f>=1:
            is_bert_tokenizer=False #If fertility is greater than 1 then we do not roll back 
        elif relative_fertility(bert_tokens, tokens)>=self.f:
            is_bert_tokenizer=True #if neither of above is true then we calculate relative fertility
        elif relative_fertility(bert_tokens, tokens)<0:
            is_bert_tokenizer=True #if ever relative fertility is negative then we always go back to bert 
        else:
            pass
    
        if is_bert_tokenizer:
            self.roll_back+=1
            #then tokenize one by one again
            tokens=list()
            word_ids=list()
            for i,w in enumerate(text):
                ws = self.tokenize(w, is_bert_tokenizer)  # Tokenize the word
                # Extend the token and word_ids lists with tokenized words and corresponding indices
                tokens.extend(ws)
                word_ids.extend([i] * len(ws))
            
        
        # Convert tokenized words to token IDs, truncating if necessary
        ids_k = self.k_tok.convert_tokens_to_ids(tokens)[:self.k_tok.model_max_length - 2]
    
        # Return the encoded sequence along with corresponding word indices
        return [self.k_tok.cls_token_id] + ids_k + [self.k_tok.sep_token_id], [None] + word_ids + [None]
    
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
    