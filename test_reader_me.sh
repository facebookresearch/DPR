
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0


python -m torch.distributed.launch \
    --nproc_per_node=1 train_reader.py \
    --prediction_results_file ./data/dr_data/reader/test_results.json.dpr \
    --dev_file ./data/dr_data/reader/test.json \
    --eval_top_docs 10 \
    --model_file ./data/dr_exp/reader/dpr_reader.19.5119 \
    --dev_batch_size 200 \
    --passages_per_question_predict 5 \
    --sequence_length 320  \
    --fp16 \
    --fp16_opt_level O2 \
    --test_only 
