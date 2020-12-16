
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

python train_reader.py \
    --prediction_results_file ./data/dr_data/reader/dev_results.json.top10 \
    --dev_file ./data/dr_data/reader/dev.json \
    --passages_per_question_predict 10 \
    --eval_top_docs 1 5 10 \
    --model_file ./data/dr_exp/reader/dpr_reader.5.11730 \
    --dev_batch_size 50 \
    --sequence_length 512 \
    --max_answer_length 500 \
    # --test_only 
