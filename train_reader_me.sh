
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1


python -m torch.distributed.launch \
    --nproc_per_node=2 train_reader.py \
    --seed 42 \
    --passages_per_question 5 \
    --passages_per_question_predict 5 \
    --max_answer_length 100 \
    --eval_top_docs 5 \
    --learning_rate 1e-4 \
    --eval_step 12000 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg ./models/RoBERTa-wwm-ext \
    --do_lower_case \
    --train_file data/dr_data/reader/train.json \
    --dev_file data/dr_data/reader/dev.json \
    --warmup_steps 6000 \
    --sequence_length 320 \
    --num_train_epochs 20 \
    --batch_size 8 \
    --dev_batch_size 8 \
    --output_dir data/dr_exp/reader \
    --fp16 \
    --fp16_opt_level O2 > data/dr_exp/reader2/train.log 
