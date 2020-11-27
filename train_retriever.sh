

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch \
    --nproc_per_node=1 train_dense_encoder.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-uncased \
    --seed 12345 \
    --sequence_length 256 \
    --warmup_steps 1237 \
    --batch_size 8 \
    --do_lower_case \
    --train_file data/nq_data/retriever/nq-train.json \
    --dev_file data/nq_data/retriever/nq-dev.json \
    --output_dir data/nq_exp \
    --learning_rate 2e-05 \
    --num_train_epochs 40 \
    --dev_batch_size 16 \
    --val_av_rank_start_epoch 20
