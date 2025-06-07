from transformers import AutoTokenizer
loc="./ktokenizer-umls-clinicalBert-base-cased-20-4"
tokenizer=AutoTokenizer.from_pretrained(loc)
print(len(tokenizer))