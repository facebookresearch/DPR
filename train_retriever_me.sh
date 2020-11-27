

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch \
    --nproc_per_node=2 train_dense_encoder.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-chinese \
    --seed 12345 \
    --sequence_length 256 \
    --warmup_steps 1237 \
    --batch_size 16 \
    --do_lower_case \
    --train_file data/dr_data/retriever/train.json \
    --dev_file data/dr_data/retriever/dev.json \
    --output_dir data/dr_exp/retriever \
    --learning_rate 2e-05 \
    --num_train_epochs 40 \
    --dev_batch_size 16 \
    --val_av_rank_start_epoch 20 \
    --fp16 \
    --fp16_opt_level O2 \
