#!/bin/bash
dataset=marco
sample_num=24
batch_size=16
# add_adapter=True
echo "batch size ${batch_size}"
dev_batch_size=256
min_index=0
max_index=240
max_seq_len=160
q_max_seq_len=32
p_max_seq_len=128

retriever_model_name_or_path='../data/tas/'
reranker_model_name_or_path=../data/electra-base-discriminator/
reranker_warm_start_from='../data/checkpoint_pt/teacher_electra_base_435/'
# retriever_warm_start_from='../data/tas/'
# adapter_warm_start_from='../data/checkpoint_pt/teacher_electra_base_435/'
# distiller_warm_start_from='distiller393.p'
# online_distill=True
alpha=1.0
beta=0.0
gemma=0.0
omega=0.0
theta1=0.0
theta2=0.0
Temperature=1.0

top1000=../run.msmarco-passage.train.mrr39-6.tsv
dev_top1000=../dev.395.tsv
dev_query=../data/${dataset}/dev.query.txt
learning_rate=3e-5
### 下面是永远不用改的
pretrain_input_file='../data/pretrain/*'
warmup_proportion=0.1
eval_step_proportion=0.01
report_step=100
epoch=40
collection=../collection.splitTitle.tsv
qrels=../data/${dataset}/qrels.tsv
query=../data/${dataset}/train.query.txt
fp16=true
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
# pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.12.0%2Bcu113-cp37-cp37m-linux_x86_64.whl 
# pip install transformers
echo "=================start train ${OMPI_COMM_WORLD_RANK:-0}=================="
python -m torch.distributed.launch \
    --log_dir ${log_dir} \
    --nproc_per_node=8 \
    src/train_distill.py \
    --batch_size=${batch_size} \
    --warmup_proportion=${warmup_proportion} \
    --eval_step_proportion=${eval_step_proportion} \
    --report=${report_step} \
    --qrels=${qrels} \
    --query=${query} \
    --dev_query=${dev_query} \
    --collection=${collection} \
    --top1000=${top1000} \
    --dev_top1000=${dev_top1000} \
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
    --retriever_model_name_or_path=${retriever_model_name_or_path} \
    --reranker_model_name_or_path=${reranker_model_name_or_path} \
    --reranker_warm_start_from=${reranker_warm_start_from} \
    --retriever_warm_start_from=${retriever_warm_start_from} \
    --adapter_warm_start_from=${adapter_warm_start_from} \
    --alpha=${alpha} \
    --beta=${beta} \
    --gemma=${gemma} \
    --omega=${omega} \
    --theta1=${theta1} \
    --theta2=${theta2} \
    --Temperature=${Temperature} \
    --distiller_warm_start_from=${distiller_warm_start_from} \
    --online_distill=${online_distill} \
    --add_adapter=${add_adapter} \
    | tee ${log_dir}/train.log

echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="
