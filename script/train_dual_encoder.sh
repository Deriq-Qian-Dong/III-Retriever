#!/bin/bash
dataset=marco
add_decoder=true
# l2_normalize=true
sample_num=16
batch_size=64
echo "batch size ${batch_size}"
max_index=64
FN_threshold=.9
retriever_model_name_or_path=./data/RetroMAE_MSMARCO_finetune
n_head_layers=1
warm_start_from=./data/pretrained_decoder_with_retro-ft_freezed-epoch-8.p
top1000=./data/train_negs.tsv
generated_query=./data/RAKE_generated.tsv
learning_rate=2e-5
# learning_rate=1e-3
### 下面是永远不用改的
dev_batch_size=256
min_index=0
EN_threshold=-100
max_seq_len=160
q_max_seq_len=32
p_max_seq_len=140
dev_query=./data/${dataset}/dev.query.txt
warmup_proportion=0.1
eval_step_proportion=0.01
report_step=100
epoch=200
collection=./collection.splitTitle.tsv
qrels=./data/${dataset}/qrels.mrr43-5.tsv
query=./data/${dataset}/train.query.txt
fp16=true
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
gpu_partial=1
# if [[ ${gpu_partial} =~ "0" ]]; then
#     export CUDA_VISIBLE_DEVICES=0,1,2,3
#     master_port=29500
# fi 
# if [[ ${gpu_partial} =~ "1" ]]; then
#     export CUDA_VISIBLE_DEVICES=4,5,6,7
#     master_port=29501
# fi 
master_port=29500
echo "=================start train ${OMPI_COMM_WORLD_RANK:-0}=================="
python -m torch.distributed.launch \
    --log_dir ${log_dir} \
    --nproc_per_node=8 \
    --master_port=${master_port} \
    src/train_dual_encoder.py \
    --vocab_file=${vocab_file} \
    --retriever_model_name_or_path=${retriever_model_name_or_path} \
    --batch_size=${batch_size} \
    --warmup_proportion=${warmup_proportion} \
    --eval_step_proportion=${eval_step_proportion} \
    --report=${report_step} \
    --qrels=${qrels} \
    --query=${query} \
    --dev_query=${dev_query} \
    --collection=${collection} \
    --top1000=${top1000} \
    --min_index=${min_index} \
    --max_index=${max_index} \
    --epoch=${epoch} \
    --sample_num=${sample_num} \
    --dev_batch_size=${dev_batch_size} \
    --pretrain_input_file=${pretrain_input_file} \
    --max_seq_len=${max_seq_len} \
    --learning_rate=${learning_rate} \
    --q_max_seq_len=${q_max_seq_len} \
    --p_max_seq_len=${p_max_seq_len} \
    --warm_start_from=${warm_start_from} \
    --n_head_layers=${n_head_layers} \
    --FN_threshold=${FN_threshold} \
    --EN_threshold=${EN_threshold} \
    --add_decoder=${add_decoder} \
    --l2_normalize=${l2_normalize} \
    --generated_query=${generated_query} \
    --do_train \
    | tee ${log_dir}/train.log

echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="

