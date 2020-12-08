
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1


python train_reader.py \
    --seed 42 \
    --max_n_answers 1 \
    --passages_per_question 2 \
    --passages_per_question_predict 5 \
    --eval_top_docs 5 \
    --max_answer_length 500 \
    --learning_rate 1e-5 \
    --eval_step 10000 \
    --warmup_steps 10000 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-chinese \
    --do_lower_case \
    --dev_file data/dr_data/reader/dev.json \
    --train_file data/dr_data/reader/train.json \
    --sequence_length 512 \
    --num_train_epochs 10 \
    --batch_size 16 \
    --dev_batch_size 16 \
    --output_dir data/dr_exp/reader \
    --gradient_accumulation_steps 1 \
    > data/dr_exp/reader/train.log 
    # --fp16 \
    # --fp16_opt_level O2 \
