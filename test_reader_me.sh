
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

python train_reader.py \
    --prediction_results_file ./data/dr_data/reader/dev_results.json.top5_gold \
    --dev_file ./data/dr_data/reader/dev.json \
    --passages_per_question_predict 10 \
    --eval_top_docs 1 5 10 \
    --model_file ./data/dr_exp/reader_gold/dpr_reader.7.10232 \
    --dev_batch_size 50 \
    --sequence_length 512 \
    --max_answer_length 500 \
    --rank_method span \
    # --test_only 
    # --fp16 \
    # --fp16_opt_level O2 \
