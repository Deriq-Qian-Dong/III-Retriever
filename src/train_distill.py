import os

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import argparse
import random
import subprocess
import tempfile
import time
from collections import defaultdict

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import distributed
import torch_optimizer as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertModel

import dataset_factory
import utils
from modeling import DenseModel, Distiller
from msmarco_eval import calc_mrr
from utils import add_prefix, build_engine, load_qid, merge, read_embed, search

SEED = 2023
best_mrr_retriever=-1
best_mrr_reranker=-1

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
def define_args():
    parser = argparse.ArgumentParser('BERT-retrieval model')
    parser.add_argument('--pretrain_input_file', type=str, default="/home/dongqian06/hdfs_data/data_train/pretrain/*")
    parser.add_argument('--pretrain_batch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--dev_batch_size', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=160)
    parser.add_argument('--q_max_seq_len', type=int, default=160)
    parser.add_argument('--p_max_seq_len', type=int, default=160)
    parser.add_argument('--model_out_dir', type=str, default="output")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--eval_step_proportion', type=float, default=1.0)
    parser.add_argument('--report', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--qrels', type=str, default="/home/dongqian06/hdfs_data/data_train/qrels.train.remains.tsv")
    parser.add_argument('--top1000', type=str, default="/home/dongqian06/codes/anserini/runs/run.msmarco-passage.train.remains.tsv")
    parser.add_argument('--dev_top1000', type=str, default="/home/dongqian06/codes/anserini/runs/run.msmarco-passage.train.remains.tsv")
    parser.add_argument('--collection', type=str, default="/home/dongqian06/hdfs_data/data_train/marco/collection.remains.tsv")
    parser.add_argument('--query', type=str, default="/home/dongqian06/hdfs_data/data_train/train.query.remains.txt")
    parser.add_argument('--dev_query', type=str, default="/home/dongqian06/hdfs_data/data_train/train.query.remains.txt")
    parser.add_argument('--min_index', type=int, default=0)
    parser.add_argument('--max_index', type=int, default=256)
    parser.add_argument('--sample_num', type=int, default=256)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--gradient_checkpoint', type=bool, default=False)
    parser.add_argument('--negatives_x_device', type=bool, default=True)
    parser.add_argument('--untie_encoder', type=bool, default=True)
    parser.add_argument('--add_pooler', type=bool, default=False)

    parser.add_argument('--online_distill', type=bool, default=False)
    parser.add_argument('--validate_reranker_first', type=bool, default=False)
    parser.add_argument('--validate_retriever_first', type=bool, default=True)
    parser.add_argument('--reranker_model_name_or_path', type=str, default="../data/miniLM/")
    parser.add_argument('--retriever_model_name_or_path', type=str, default="../data/co-condenser-marco-retriever/")
    parser.add_argument('--distiller_warm_start_from', type=str, default="")
    parser.add_argument('--reranker_warm_start_from', type=str, default="")
    parser.add_argument('--retriever_warm_start_from', type=str, default="")
    
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--gemma', type=float, default=0.01)
    parser.add_argument('--omega', type=float, default=0.01)
    parser.add_argument('--Temperature', type=float, default=5.0)

    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args

def merge_reranker(eval_cnts, file_pattern='output/res.step-%d.part-0%d'):
    f_list = []
    total_part = torch.distributed.get_world_size()
    for part in range(total_part):
        f0 = open(file_pattern % (eval_cnts, part))
        f_list+=f0.readlines()
    f_list = [l.strip().split("\t") for l in f_list]
    dedup = defaultdict(dict)
    for qid,pid,score in f_list:
        dedup[int(float(qid))][int(float(pid))] = float(score)
    mp = defaultdict(list)
    for qid in dedup:
        for pid in dedup[qid]:
            mp[qid].append((pid, dedup[qid][pid]))
    for qid in mp:
        mp[qid].sort(key=lambda x:x[1], reverse=True)
    with open(file_pattern.replace('.part-0%d','')%eval_cnts, 'w') as f:
        for qid in mp:
            for idx, (pid, score) in enumerate(mp[qid]):
                f.write(str(qid)+"\t"+str(pid)+'\t'+str(idx+1)+"\t"+str(score)+'\n')
    for part in range(total_part):
        os.remove(file_pattern % (eval_cnts, part))

def run_distill(args, model, optimizer):
    epoch = 0
    local_rank = torch.distributed.get_rank()
    if local_rank==0:
        print(f'Starting training, upto {args.epoch} epochs, LR={args.learning_rate}', flush=True)

    # 加载retriever测试数据集
    query_dataset = dataset_factory.QueryDataset(args)
    query_loader = DataLoader(query_dataset, batch_size=args.dev_batch_size, collate_fn=query_dataset._collate_fn, num_workers=3)
    passage_dataset = dataset_factory.PassageDataset(args)
    passage_loader = DataLoader(passage_dataset, batch_size=args.dev_batch_size, collate_fn=passage_dataset._collate_fn, num_workers=3)
    if args.validate_retriever_first:
        validate_retriever(model, query_loader, passage_loader, epoch, args)

    # 加载reranker测试数据集
    if args.online_distill or args.validate_reranker_first:
        dev_dataset = dataset_factory.CrossEncoderDevDataset(args)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset)
        dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, collate_fn=dev_dataset._collate_fn, sampler=dev_sampler, num_workers=3)
        validate_reranker(model, dev_loader, epoch, args)

    # train_dataset = dataset_factory.RetrievalPAIRPretrainDataset(args)
    # train_dataset = dataset_factory.RetrievalRocketQADataset(args)
    train_dataset = dataset_factory.DualEncoderDistillDataset(args)

    for epoch in range(1, args.epoch+1):
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_sampler.set_epoch(epoch)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset._collate_fn, sampler=train_sampler, num_workers=4)
        loss = train_iteration_multi_gpu(model, optimizer, train_loader, epoch, args)
        del train_loader
        torch.distributed.barrier()
        if epoch%1==0:
            validate_retriever(model, query_loader, passage_loader, epoch, args)
            if args.online_distill:
                args.dev_top1000 = 'output/res.top1000.step%d'%epoch
                dev_dataset = dataset_factory.CrossEncoderDevDataset(args)
                dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset)
                dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, collate_fn=dev_dataset._collate_fn, sampler=dev_sampler, num_workers=3)
                validate_reranker(model, dev_loader, epoch, args)
                del dev_dataset
                del dev_loader
                del dev_sampler
            torch.distributed.barrier()

def validate_retriever(model, query_loader, passage_loader, epoch, args):
    global best_mrr_retriever
    local_start = time.time()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    _output_file_name = 'output/_para.index.part%d'%local_rank
    output_file_name = 'output/para.index.part%d'%local_rank
    top_k = 1000
    q_output_file_name = 'output/query.emb.step%d.npy'%epoch
    if local_rank==0:
        q_embs = []
        with torch.no_grad():
            model.eval()
            for records in query_loader:
                tok, mask = records
                tok = torch.tensor(tok).long().cuda()
                mask = torch.tensor(mask).float().cuda()
                batch = {"input_ids":tok, "attention_mask":mask}
                if args.fp16:
                    with autocast():
                        q_reps = model(query_inputs=batch)
                else:
                    q_reps = model(query_inputs=batch)
                q_embs.append(q_reps.cpu().detach().numpy())
        emb_matrix = np.concatenate(q_embs, axis=0)
        np.save(q_output_file_name, emb_matrix)
        print("predict q_embs cnt: %s" % len(emb_matrix))
    with torch.no_grad():
        model.eval()
        para_embs = []
        for records in tqdm(passage_loader, disable=args.local_rank >=1):
            tok, mask = records
            tok = torch.tensor(tok).long().cuda()
            mask = torch.tensor(mask).float().cuda()
            batch = {"input_ids":tok, "attention_mask":mask}
            if args.fp16:
                with autocast():
                    p_reps = model(passage_inputs=batch)
            else:
                p_reps = model(passage_inputs=batch)
            para_embs.append(p_reps.cpu().detach().numpy())
    para_embs = np.concatenate(para_embs, axis=0)
    print("predict embs cnt: %s" % len(para_embs))
    engine = build_engine(para_embs, 768)
    faiss.write_index(engine, _output_file_name)
    np.save('output/_para.emb.part%d.npy'%local_rank, para_embs)
    print('create index done!')
    qid_list = load_qid(args.dev_query)
    search(engine, q_output_file_name, qid_list, "output/res.top%d.part%d.step%d"%(top_k, local_rank, epoch), top_k=top_k)
    torch.distributed.barrier() 
    if local_rank==0:
        f_list = []
        for part in range(world_size):
            f_list.append('output/res.top%d.part%d.step%d' % (top_k, part, epoch))
        shift = np.load("output/_para.emb.part0.npy").shape[0]
        merge(world_size, shift, top_k, epoch)
        metrics = calc_mrr(args.qrels, 'output/res.top%d.step%d'%(top_k, epoch))
        for run in f_list:
            os.remove(run)
        mrr = metrics['MRR @10']
        if mrr>best_mrr_retriever:
            print("*"*50)
            print("retriever new top")
            print("*"*50)
            best_mrr_retriever = mrr
            for part in range(world_size):
                os.rename('output/_para.index.part%d'%part, 'output/para.index.part%d'%part)
                os.rename('output/_para.emb.part%d.npy'%part, 'output/para.emb.part%d.npy'%part)
            torch.save(model.module.lm_q.state_dict(), "output/retriever_q.p")
            torch.save(model.module.lm_p.state_dict(), "output/retriever_p.p")
        seconds = time.time()-local_start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print("******************eval, mrr@10: %.10f,"%(mrr),"report used time:%02d:%02d:%02d," % (h, m, s))
    torch.distributed.barrier() 

def validate_reranker(model, dev_loader, epoch, args):
    global best_mrr_reranker
    local_start = time.time()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    with torch.no_grad():
        model.eval()
        scores_lst = []
        qids_lst = []
        pids_lst = []
        for record1, record2 in tqdm(dev_loader, disable=args.local_rank >=1):
            with autocast():
                scores = model(reranker_inputs=_prepare_inputs(record1))
            qids = record2['qids']
            pids = record2['pids']
            scores_lst.append(scores.detach().cpu().numpy().copy())
            qids_lst.append(qids.copy())
            pids_lst.append(pids.copy())
        qids_lst = np.concatenate(qids_lst).reshape(-1)
        pids_lst = np.concatenate(pids_lst).reshape(-1)
        scores_lst = np.concatenate(scores_lst).reshape(-1)
        with open("output/res.step-%d.part-0%d"%(epoch, local_rank), 'w') as f:
            for qid,pid,score in zip(qids_lst, pids_lst, scores_lst):
                f.write(str(qid)+'\t'+str(pid)+'\t'+str(score)+'\n')
        torch.distributed.barrier() 
        if local_rank==0:
            merge_reranker(epoch)
            metrics = calc_mrr(args.qrels, 'output/res.step-%d'%epoch)
            mrr = metrics['MRR @10']
            if mrr>best_mrr_reranker:
                print("*"*50)
                print("reranker new top")
                print("*"*50)
                best_mrr_reranker = mrr
                torch.save(model.module.reranker.state_dict(), "output/reranker.p")


       
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()#进程数
    return rt

def _prepare_inputs(record):
    prepared = {}
    local_rank = torch.distributed.get_rank()
    for key in record:
        x = record[key]
        if isinstance(x, torch.Tensor):
            prepared[key] = x.to(local_rank)
        else:
            prepared[key] = _prepare_inputs(x)
    return prepared

def train_iteration_multi_gpu(model, optimizer, data_loader, epoch, args):
    total = 0
    model.train()
    total_loss = 0.
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    start = time.time()
    local_start = time.time()
    all_steps_per_epoch = len(data_loader)
    step = 0
    scaler = GradScaler()
    total_s2t_loss = 0.
    total_t2s_loss = 0.
    total_retriever_loss = 0.
    total_reranker_loss = 0.
    total_teacher_self_loss = 0.
    total_student_cross_loss = 0.
    for record in data_loader:
        record = _prepare_inputs(record)
        if args.fp16:
            with autocast():
                ret = model(query_inputs=record['query'], passage_inputs=record['passage'], reranker_inputs=record['reranker'])
        else:
            ret = model(query_inputs=record['query'], passage_inputs=record['passage'], reranker_inputs=record['reranker'])
        s2t_loss = ret.s2t_loss
        retriever_ce_loss = ret.retriever_ce_loss            
        if args.online_distill:
            reranker_ce_loss = ret.reranker_ce_loss
            t2s_loss = ret.t2s_loss
            loss = s2t_loss+retriever_ce_loss+t2s_loss+reranker_ce_loss
        else:
            loss = s2t_loss+retriever_ce_loss
        torch.distributed.barrier() 
        reduced_loss = reduce_tensor(loss.data)
        total_loss += reduced_loss.item()
        # optimize
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step+=1
        total_s2t_loss += float(s2t_loss.cpu().detach().numpy())
        total_retriever_loss += float(retriever_ce_loss.cpu().detach().numpy())
        if args.online_distill:
            total_t2s_loss += float(t2s_loss.cpu().detach().numpy())
            total_reranker_loss += float(reranker_ce_loss.cpu().detach().numpy())
        if step%args.report==0 and local_rank==0:
            seconds = time.time()-local_start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            local_start = time.time()
            print(f"epoch:{epoch} training step: {step}/{all_steps_per_epoch}, mean loss: {total_loss/step}, s2t loss: {total_s2t_loss/step}, teacher self loss: {total_teacher_self_loss/step}, student cross loss: {total_student_cross_loss/step}, retriever loss: {total_retriever_loss/step}, t2s loss: {total_t2s_loss/step}, reranker loss: {total_reranker_loss/step}, ", "report used time:%02d:%02d:%02d," % (h, m, s), end=' ')
            seconds = time.time()-start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
            print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
    if local_rank==0:
        # model.save(os.path.join(args.model_out_dir, "weights.epoch-%d.p"%(epoch)))
        torch.save(model.module.state_dict(), os.path.join(args.model_out_dir, "weights.epoch-%d.p"%(epoch)))
        seconds = time.time()-start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print(f'train epoch={epoch} loss={total_loss}')
        print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
        print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
    return total_loss

if __name__ == '__main__':
    args = define_args()
    args = vars(args)
    args = utils.HParams(**args)
    # 加载到多卡
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    local_rank = torch.distributed.get_rank()
    if local_rank==0:
        args.print_config()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = Distiller(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    params = {'params': [v for k, v in params]}
    # optimizer = torch.optim.Adam([params], lr=args.learning_rate, weight_decay=0.0)
    optimizer = optim.Lamb([params], lr=args.learning_rate, weight_decay=0.0)
    

    if args.reranker_warm_start_from:
        print('reranker warm start from ', args.reranker_warm_start_from)
        state_dict = torch.load(os.path.join(args.reranker_warm_start_from, 'reranker.p'), map_location=device)
        # reranker_state_dict = {}
        # for key in list(state_dict.keys()):
        #     if 'lm.' in key:
        #         reranker_state_dict[key.replace("module.lm.","").replace("lm.","")] = state_dict.pop(key)
        # model.reranker.load_state_dict(reranker_state_dict)
        model.reranker.load_state_dict(state_dict)
    if args.retriever_warm_start_from:
        print('retriever warm start from ', args.retriever_warm_start_from)
        q_state_dict = torch.load(os.path.join(args.retriever_warm_start_from, 'retriever_q.p'), map_location=device)
        p_state_dict = torch.load(os.path.join(args.retriever_warm_start_from, 'retriever_p.p'), map_location=device)
        model.lm_q.load_state_dict(q_state_dict,strict=False)
        model.lm_p.load_state_dict(p_state_dict,strict=False)
    if args.distiller_warm_start_from:  # 会覆盖掉上面加载的参数，，
        print('distiller warm start from ', args.distiller_warm_start_from)
        state_dict = torch.load(args.distiller_warm_start_from, map_location=device)
        model.load_state_dict(state_dict)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    print("model loaded on GPU%d"%local_rank)
    print(args.model_out_dir)
    os.makedirs(args.model_out_dir, exist_ok=True)
    run_distill(args, model, optimizer)
