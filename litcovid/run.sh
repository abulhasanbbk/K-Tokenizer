export CUDA_VISIBLE_DEVICES=1
python document_classification.py \
 --type p \
 --tokenizer_base_loc umls_tokenization \
 --corpus umls \
 --alpha 20 \
 --delta 0 \
 --partition 2 \
 --seed 48 \
 --model_save_loc Model-3-Runs-umls-50-percent \
 --numrun 1
python document_classification.py \
 --type p \
 --tokenizer_base_loc umls_tokenization \
 --corpus umls \
 --alpha 20 \
 --delta 0 \
 --partition 2 \
 --seed 1024 \
 --model_save_loc Model-3-Runs-umls-50-percent \
 --numrun 2
python document_classification.py \
 --type p \
 --tokenizer_base_loc umls_tokenization \
 --corpus umls \
 --alpha 20 \
 --delta 0 \
 --partition 2 \
 --seed 2048 \
 --model_save_loc Model-3-Runs-umls-50-percent \
 --numrun 3