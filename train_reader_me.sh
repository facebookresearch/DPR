
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1


python -m torch.distributed.launch \
    --nproc_per_node=2 train_reader.py \
    --seed 42 \
    --max_n_answers 1 \
    --passages_per_question 3 \
    --passages_per_question_predict 10 \
    --max_answer_length 200 \
    --eval_top_docs 10 \
    --learning_rate 1e-5 \
    --eval_step 6000 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg ./models/RoBERTa-wwm-ext \
    --do_lower_case \
    --train_file data/dr_data/reader/train.json \
    --dev_file data/dr_data/reader/dev.json \
    --warmup_steps 0 \
    --sequence_length 320 \
    --num_train_epochs 100 \
    --batch_size 8 \
    --dev_batch_size 8 \
    --output_dir data/dr_exp/reader \
    --gradient_accumulation_steps 4 \
    > data/dr_exp/reader/train.log 
    #--fp16 \
    #--fp16_opt_level O2 \
