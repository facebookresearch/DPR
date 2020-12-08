
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0


python -m torch.distributed.launch \
    --nproc_per_node=1 train_reader.py \
    --prediction_results_file ./data/dr_data/reader/dev_results.json.dpr \
    --dev_file ./data/dr_data/reader/dev.json\
    --eval_top_docs 1 \
    --model_file ./data/dr_exp/reader/dpr_reader.0.10000 \
    --dev_batch_size 40 \
    --passages_per_question_predict 1 \
    --sequence_length 512 \
    --max_answer_length 500 \
    # --test_only 
    # --fp16 \
    # --fp16_opt_level O2 \
