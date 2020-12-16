

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch \
    --nproc_per_node=2 train_dense_encoder.py \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-chinese \
    --seed 12345 \
    --sequence_length 200 \
    --dev_batch_size 16 \
    --dev_file data/dr_data/retriever/dev.json \
    --model_file ./data/dr_exp/retriever/dpr_biencoder.35.12827 \
