export CUDA_VISIBLE_DEVICES=1
python document_classification.py \
 --type p \
 --tokenizer_base_loc umls_tokenization \
 --corpus base \
 --alpha 20 \
 --delta 0 \
 --partition 2 \
 --seed 2048 \
 --model_save_loc Model-3-Runs-base-50-percent\
 --numrun 3