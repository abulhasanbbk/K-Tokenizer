import pickle
import collections
from collections import OrderedDict
import json
import os
import re
import sys

# Function to save data to a pickle file
def save_to_pickle(data, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f'Data saved to {file_path} successfully.')
    except Exception as e:
        print(f'Error saving data to {file_path}: {str(e)}')

# Function to load data from a pickle file
def load_from_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f'Data loaded from {file_path} successfully.')
        return data
    except Exception as e:
        print(f'Error loading data from {file_path}: {str(e)}')
        return None
#Function to craete a directory if it does not exist
def create_directory_if_not_exists(directory_path):
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f'Directory "{directory_path}" created successfully.')
        else:
            print(f'Directory "{directory_path}" already exists.')
    except Exception as e:
        print(f'Error creating directory "{directory_path}": {str(e)}')

#generate character n-grams from a token ['g', 'e', 'b', 'a'], [ge, eb, ba, a],[geb,eba,ba,a]
def generate_ngram(word, n):
    return [word[i:i+n] for i in range(len(word))]

def merge_dicts_recursive(dict1, dict2):
    """
    Recursively merges two dictionaries, handling nested dictionaries and lists.

    Parameters:
    - dict1 (dict): The first dictionary.
    - dict2 (dict): The second dictionary to merge into the first.

    Returns:
    - dict: A new dictionary containing the merged key-value pairs.
    """

    # Check if either of the dictionaries is not actually a dictionary.
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        # If one of them is not a dictionary, return the second dictionary.
        return dict2

    # Create a copy of the first dictionary to preserve the original.
    merged_dict = dict1.copy()

    # Iterate through the key-value pairs of the second dictionary.
    for key, value in dict2.items():
        # Check if the key exists in the merged dictionary.
        if key in merged_dict:
            # If the existing value is a dictionary and the new value is also a dictionary, merge them recursively.
            if isinstance(merged_dict[key], dict) and isinstance(value, dict):
                merged_dict[key] = merge_dicts_recursive(merged_dict[key], value)
            # If the existing value is a list and the new value is also a list, extend the existing list.
            elif isinstance(merged_dict[key], list) and isinstance(value, list):
                merged_dict[key].extend(value)
            # If the key exists but the values are not dictionaries or lists, overwrite the existing value with the new value.
            else:
                merged_dict[key] = value
        else:
            # If the key doesn't exist in the merged dictionary, add it with the new value.
            merged_dict[key] = value

    # Return the merged dictionary.
    return merged_dict

class GenerateCharNGramVocMain:
    """ This class creates char ngram vocabulary from a concept dictionary (i.e drug/symptom).
        A character n-gram vocabulary is a dictionary which has the following form:
        {"cho": 12, "foo": 3, ....}
        where key is a char nrgam and value is its frequency obtained by processing a corpus/ontlogy. 
    """
    def __init__(self, concept_dict):
        """
        Initialize the CreateCharNGramFromUMLS class with file paths to two pickled dictionaries.

        Parameters:
        - concept_dict_loc (str): File path to the pickled dictionary containing a concept dict (e.g. drug/symptom).
        """
        self.concept_dict = concept_dict

    def _generate_char_n_gram_voc(self):
        """
        Note: This is a private function and should not be called from outside of the class
        The funciton works as follows:
            - Each unique term in the dictionary will be processed to create char n-gram statistics
            - This function will not consider prefix and suffixes while constructing char n-ragm. 
                - For example cho and ##cho are same sub words.
        Returns:
        - OrderedDict: A character n-gram vocabulary as an ordered dictionary with n-grams as keys
                       and their frequencies as values.
        """

        # Load the concept dictionary from the specified file path.
    

        # Initialize an ordered dictionary to store the character n-gram vocabulary.
        n_gram_voc = OrderedDict()

        # Iterate through each unique concept.
        for concept in self.concept_dict:
            #A concept may contain a phrase, so split it into words
            concept=concept.lower()
            phrase_list=concept.split()
            for key in phrase_list:
                # Iterate through all possible character n-grams of the word.
                for i in range(0, len(key)):
                    # Generate character n-grams for the current position 'i'.
                    n_gram_list = generate_ngram(key, i + 1)

                    # Populate the character n-gram vocabulary by counting frequencies.
                    for n in n_gram_list:
                        if n in n_gram_voc:
                            n_gram_voc[n] += 1
                        else:
                            n_gram_voc[n] = 1

        # Return the character n-gram vocabulary as an ordered dictionary.
        return n_gram_voc

    def save_to_json(self, file_path):
        """
        Save the character n-gram vocabulary to a JSON file.

        Parameters:
        - file_path (str): File path where the JSON file will be saved.
        """
        char_ngram_vocabulary = self._generate_char_n_gram_voc()
        try:
            with open(file_path, 'w') as json_file:
                json.dump(char_ngram_vocabulary, json_file)
            print(f'Character n-gram vocabulary saved to {file_path} successfully.')
        except Exception as e:
            print(f'Error saving character n-gram vocabulary to {file_path}: {str(e)}')

class GenerateCharNGramSpecialFromUMLS(GenerateCharNGramVocMain):
    """ This class creates char ngram vocabulary from two dictionaries: (i) drug, (ii) symptom.
        A character n-gram vocabulary is a dictionary which has the following form:
        {"cho": 12, "##cho": 3, ....}
        where key is a char nrgam and value is its frequency obtained by processing a corpus/ontlogy.
        This class creates a char n-gram vocabulary from four UMLS semantic types : 
        (i)Pharmacovigilance substance, (ii)Clinical Drug, (iii)Disease and Syndrome, and (iv) Sign or Simptom.
        Here, (i) and (ii) are in a pickle file located in the drug_loc.
        (iii) and (iv) are located in symptom_loc. 
    """
    def __init__(self, drug_loc, symptom_loc):
        """
        Initialize the CreateCharNGramFromUMLS class with file paths to two pickled dictionaries.

        Parameters:
        - drug_loc (str): File path to the pickled dictionary containing drug data.
        - symptom_loc (str): File path to the pickled dictionary containing symptom data.
        """
        self.drug_loc = drug_loc
        self.symptom_loc = symptom_loc

    def generate_char_n_gram_voc_special(self):
        """
        The funciton works as follows:
            - Each unique term in the merged drug and symptom dictionaries will be processed to create char n-gram statistics
            - This function will consider prefix and suffixes while constructing char n-ragm. 
                - For example cho and ##cho are two different sub words.
        Returns:
        - OrderedDict: A character n-gram vocabulary as an ordered dictionary with n-grams as keys
                       and their frequencies as values.
        """
        special = "##"

        # Load the drug and symptom dictionaries from the specified file paths.
        dict1 = load_from_pickle(self.drug_loc)
        dict2 = load_from_pickle(self.symptom_loc)

        # Merge the two dictionaries into a single dictionary of unique tokens.
        umls_unique_tokens = merge_dicts_recursive(dict1, dict2)

        # Initialize an ordered dictionary to store the character n-gram vocabulary.
        n_gram_voc = OrderedDict()

        # Iterate through each unique token in the merged data.
        for key in umls_unique_tokens:
            # Iterate through all possible character n-grams of the token.
            for i in range(0, len(key)):
                # Generate character n-grams for the current position 'i'.
                n_gram_list = generate_ngram(key, i + 1)

                # Initialize a list to store character n-grams with suffixes.
                n_gram_list_with_suffix = []

                # Determine whether each n-gram is a prefix or has a special suffix.
                for j, n in enumerate(n_gram_list):
                    if j == 0:
                        n_gram_list_with_suffix.append(n)
                    else:
                        n_gram_list_with_suffix.append(special + n)

                # Populate the character n-gram vocabulary by counting frequencies.
                for n in n_gram_list_with_suffix:
                    if n in n_gram_voc:
                        n_gram_voc[n] += 1
                    else:
                        n_gram_voc[n] = 1

        # Return the character n-gram vocabulary as an ordered dictionary.
        return n_gram_voc

def divide_into_syllables(word):
    """
    Divide a word into syllables based on predefined patterns.

    This function takes a prefix/word as input and divides it into syllables using regular expressions
    and predefined patterns that account for different syllable structures in English.

    Parameters:
    - word (str): The input word/prefix to be divided into syllables.

    Returns:
    - list: A list of syllables extracted from the input word/prefix.
    """

    # Define patterns for different syllable structures using regular expressions.
    patterns = [
        # Pattern for words ending in a vowel followed by one or more consonants
        r'[aeiouy]+[^aeiouy]+',
        # Pattern for words ending in a vowel
        r'[aeiouy]+',
        # Pattern for words ending in a consonant followed by a vowel
        r'[^aeiouy]+[aeiouy]+',
        # Pattern for words with a single consonant
        r'[^aeiouy]',
        # Pattern for a consonant followed by a vowel and then a consonant
        r'[^aeiouy][aeiouy][^aeiouy]'
    ]

    # Initialize an empty list to store the extracted syllables.
    syllables = []

    # Convert the input word to lowercase to ensure consistent matching.
    remaining_word = word.lower()

    # Iterate through the input word while extracting syllables.
    while remaining_word:
        matched = False
        for pattern in patterns:
            # Attempt to match the current pattern to the remaining word.
            match = re.match(pattern, remaining_word)
            if match:
                # If a match is found, extract the matched syllable and update the remaining word.
                syllable = match.group(0)
                syllables.append(syllable)
                remaining_word = remaining_word[len(syllable):]
                matched = True
                break

        # If no pattern matches, consider the remaining word as a separate syllable.
        if not matched:
            syllables.append(remaining_word)
            remaining_word = ''

    # Return the list of syllables extracted from the input word.
    return syllables

class KBPE:
    """Knowledge infused BPE class. Here, we consider frequencies of entities collected from a corpus (e.g. mimic iii) or 
        an ontology (e.g. UMLS)

    """
    def __init__(self, kvocab, word):
        self.kvocab=kvocab #knowledge vocabulary to find the stats from a corpus/ontology
        self.word=word 
        self._spaced_kvocab=dict() 
        self._bpe_kvocab=dict() # BPE vocabulary
        self._set_spaced_kvocab()

    
    def __call__(self, num_merges):
        self._get_bpe_kvocab(num_merges)
    
    def _set_spaced_kvocab(self):
        """
        Generate a spaced character n-gram vocabulary for a word.

        This function generates character n-grams from the word and filters them based on whether they
        exist in a provided character n-gram vocabulary. It then creates a spaced character n-gram
        vocabulary with spaced n-grams and their corresponding frequencies.
        Example:
        word: albumin
        {
            'a</w>': 248309, 
            'l</w>': 326408, 
            'b</w>': 73941, 
            'u</w>': 54037, 
            'm</w>': 162255, 
            'i</w>': 188613, 
            'n</w>': 394736, 
            'a l</w>': 36698, 
            'l b</w>': 1663, 
            'b u</w>': 4082, 
            'u m</w>': 64230, 
            'm i</w>': 18709, 
            'i n</w>': 195806, 
            'a l b</w>': 288, 
            'l b u</w>': 315, 
            'b u m</w>': 574, 
            'u m i</w>': 420, 
            'm i n</w>': 13100, 
            'a l b u</w>': 92, 
            'l b u m</w>': 45, 
            'b u m i</w>': 35, 
            'u m i n</w>': 672, 
            'a l b u m</w>': 40, 
            'l b u m i</w>': 20, 
            'b u m i n</w>': 147, 
            'a l b u m i</w>': 20, 
            'l b u m i n</w>': 91, 
            'a l b u m i n</w>': 77
        }

        Note: This is a private method and should not be called directly from outside the class.

        """
        # Get the length of the input word.
        l = len(self.word)

        # Iterate through all possible character n-grams of the word.
        for i in range(0, l):
            # Generate character n-grams for the current position 'i'.
            n_gram_list = list(generate_ngram(self.word, i + 1))

            # Iterate through the generated n-grams.
            for l in range(0, len(n_gram_list)):
                ngram = n_gram_list[l]

                # Join the characters of the n-gram with spaces and add a suffix '</w>'.
                ngram_spaced = " ".join(ngram) + "</w>"

                # Check if the n-gram exists in the provided character n-gram vocabulary.
                if ngram in self.kvocab:
                    # If it exists, add it to the filtered character n-gram vocabulary
                    # with its corresponding frequency.
                    self._spaced_kvocab[ngram_spaced] = self.kvocab[ngram]
                else:
                    # If it doesn't exist, add it to the filtered character n-gram vocabulary
                    # with a frequency of 1.
                    self._spaced_kvocab[ngram_spaced] = 1
    
    def get_spaced_kvocab(self):
        """ Returns spaced character ngram vocabulary for a word
        """
        return self._spaced_kvocab
    
    def _get_stats(self, vocab):
        """
        Calculate statistics on character pairs in the spaced character n-gram vocabulary.

        This function calculates statistics on character pairs within the spaced character n-gram
        vocabulary. It counts the frequency of each character pair (bigram) within the spaced n-grams.
        Returns:
        - defaultdict: A dictionary containing character pair (bigram) frequencies.
        """

        # Initialize a defaultdict to store character pair frequencies.
        pairs = collections.defaultdict(int)

        # Iterate through each spaced n-gram and its frequency in the spaced character n-gram vocabulary.
        for word, freq in vocab.items():
            # Split the spaced n-gram into individual symbols (characters).
            symbols = word.split()
            # Iterate through the symbols to count character pairs (bigrams).
            for i in range(len(symbols) - 1):
                # Increment the frequency of the character pair.
                pairs[symbols[i], symbols[i + 1]] += freq

        # Return the dictionary containing character pair frequencies.
        return pairs

    


    def merge_vocab(self, pair, v_in):
        """
            Merge character n-grams in the spaced character n-gram vocabulary based on a given character pair.

            This function takes a character pair and merges character n-grams in the spaced character n-gram
            vocabulary that match the character pair. It creates a new vocabulary with merged n-grams.

            Parameters:
            - pair (tuple): A tuple containing two characters representing the character pair to merge.

            Returns:
            - dict: A new vocabulary containing merged n-grams and their frequencies.
        """
        # Initialize an empty dictionary to store the merged vocabulary.
        v_out = {}

        # Create a regular expression pattern to match the character pair in spaced n-grams.
        bigram_pattern = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram_pattern + r'(?!\S)')

        # Iterate through each spaced n-gram in the spaced character n-gram vocabulary.
        for word in v_in:
            # Use the regular expression pattern to replace the character pair with the merged pair.
            w_out = p.sub(''.join(pair), word)

            # Add the merged n-gram and its frequency to the new vocabulary.
            v_out[w_out] = v_in[word]

        # Return the merged vocabulary containing merged n-grams and their frequencies.
        return v_out

    
    def _get_bpe_kvocab(self, num_merges):
        
        """Build a Byte-Pair Encoding (BPE) vocabulary using a specified number of merges.
            
            This function iteratively performs BPE merges to build a vocabulary. It merges character pairs
            
            based on their frequencies, and the process continues for the specified number of merges.
            
            Parameters:
            - num_merges (int): The number of BPE merges to perform.
        """
        ##fixing the hyper parameters num merges for lenght of the word on 09/11/2023
        #num_merges=len(self.word)
        # Perform the specified number of BPE merges.
        for i in range(num_merges):
            # Calculate character pair frequencies using the _get_stats method.
            pairs = self._get_stats(self._spaced_kvocab)
            # Try to find the character pair with the highest frequency.
            try:
                best = max(pairs, key=pairs.get)
            except ValueError:
                # If no pair has a frequency > 1, stop the process.
                sys.stderr.write('no pair has frequency > 1. Stopping\n')
                break

            # If the frequency of the best character pair is < 2, stop the process.
            if pairs[best] < 2:
                sys.stderr.write('no pair has frequency > 1. Stopping\n')
                break

            # Merge the vocabulary based on the best character pair.
            self._spaced_kvocab = self.merge_vocab(best, self._spaced_kvocab )

            # Extract the two characters that were merged.
            a, b = best

            # Update the BPE vocabulary with the merged character.
            self._bpe_kvocab[a + b.split("</w>")[0]] = pairs[best]
    
    def _get_prefix_segmentation_with_syllable(self, sw_tokens, mx_ln=4):
        """
        This function takes a list of BPE segmented subword tokens. If a subword token has
        a prefix consisting of at least `mx_ln` letters, it is divided into syllables before segmentation.

        Parameters:
        - sw_tokens (list): A list of subword tokens to be segmented.
        - mx_ln (int): The minimum length of prefix letters to trigger syllable division.

        Returns:
        - list: A list of subword tokens resulting from BPE segmentation with syllable division.
        """

        # Initialize a list to store subword tokens with syllables.
        sw_tokens_w_syll = []

        # Iterate through each subword token and its index.
        for i, sw in enumerate(sw_tokens):
            # Check if it's the first subword token and its length is greater than or equal to `mx_ln`.
            if i == 0 and len(sw) >= mx_ln:
            # If so, divide it into syllables using the `divide_into_syllables` function.
                syllables = divide_into_syllables(sw)

                # Append each syllable to the result list.
                for s in syllables:
                    sw_tokens_w_syll.append(s)
            else:
                # If not, simply append the subword token as is.
                sw_tokens_w_syll.append(sw)

    # Return the list of subword tokens resulting from BPE segmentation with syllable division.
        return sw_tokens_w_syll

    
    def get_bpe_segmentation(self, divide_into_syllables=True):
        """
        Perform Byte-Pair Encoding (BPE) segmentation on a given word.

        This function segments the word into subword tokens using the BPE vocabulary.

        Returns:
        - list: A list of subword tokens resulting from BPE segmentation.
        """

        # Define the maximum length for n-grams (subword tokens).
        ngram_max = 5

        # Initialize the list to store subword tokens.
        sw_tokens = []

        # Initialize the starting and ending indices for subword extraction.
        start_idx = 0
        end_idx = min([len(self.word), ngram_max])

        # Iterate through the word while segmenting it into subword tokens.
        while start_idx < len(self.word):
            # Extract a subword from the word based on the current indices.
            subword = self.word[start_idx:end_idx]

            # Check if the subword exists in the BPE vocabulary.
            if subword in self._bpe_kvocab:
                # If it exists, add it as a subword token and update indices.
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(self.word), start_idx + ngram_max])
            elif len(subword) == 1:
                # If the subword is a single character, add it as a subword token and update indices.
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(self.word), start_idx + ngram_max])
            else:
                # If the subword is not in the BPE vocabulary and not a single character,
                # decrement the ending index to extract a shorter subword.
                end_idx -= 1

        # Return the list of subword tokens resulting from BPE segmentation.
        if divide_into_syllables:
            sw_tokens=self._get_prefix_segmentation_with_syllable(sw_tokens)
        return sw_tokens
    

class BPEVocabularyBuilder:
    def __init__(self, drug_data_path, global_voc_path):
        self.concept_dict = self.load_concept_dict(drug_data_path)
        self.global_voc = self.create_global_vocabulary(global_voc_path)
        self.symbol = "##"
        self.bpe_voc = OrderedDict()

    def load_concept_dict(self, path):
        return load_from_pickle(path)  # Replace with your actual loading logic

    def create_global_vocabulary(self, path):
        #generate ngrams for a concept dictionary
        generator=GenerateCharNGramVocMain(self.concept_dict)
        generator.save_to_json(path)
        with open(path) as fp:
            return json.load(fp)

    def build_bpe_vocabulary(self, max_iterations=20):
        for key in self.concept_dict:
            words = key.split()
            for word in words:
                word = word.lower()
                kbpe = KBPE(self.global_voc, word)
                merges=max_iterations
                kbpe(merges)  # Run KBPE algorithm
                sw_tokens = kbpe.get_bpe_segmentation()
                for i, token in enumerate(sw_tokens):
                    # Check if the first character of the token starts with the first character of the word
                    # If they are not equal, then the token must be a suffix.
                    # Add the special symbol with the suffix.
                    if not token[0] == word[0]:
                        token_to_insert = self.symbol + token
                    else:
                        token_to_insert = token

                    if token not in self.bpe_voc:
                        if token in self.global_voc:
                            self.bpe_voc[token_to_insert] = self.global_voc[token]
                        else:
                            print("Following sub word is not found in the global voc:", token)
                            print("The word is:", word)


    def get_bpe_vocabulary(self):
        return self.bpe_voc
    
    def save_bpe_vocabulary(self, path):
        try:
            with open(path, 'w') as json_file:
                json.dump(self.bpe_voc, json_file)
            print(f'BPE vocabulary saved to {path} successfully.')
        except Exception as e:
            print(f'Error saving  BPE vocabulary to {path}: {str(e)}')
      

