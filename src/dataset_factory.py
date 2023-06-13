import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from transformers import AutoTokenizer

class PassageDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t",header=None, quoting=3)
        self.collection.columns=['pid',  'title', 'para']
        self.collection = self.collection.fillna("NA")        
        self.collection.index = self.collection.pid 
        total_cnt = len(self.collection)
        shard_cnt = total_cnt//self.n_procs
        if self.rank!=self.n_procs-1:
            self.collection = self.collection[self.rank*shard_cnt:(self.rank+1)*shard_cnt]
        else:
            self.collection = self.collection[self.rank*shard_cnt:]
        self.num_samples = len(self.collection)
        print('rank:',self.rank,'samples:',self.num_samples)

    def _collate_fn(self, psgs):
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        if self.args.add_decoder:
            generated_query_len = self.args.generated_query_len
            generated_query = " ".join([self.tokenizer.mask_token]*generated_query_len)
            p_de_records = self.tokenizer([generated_query]*len(psgs), padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
            p_records['decoder_inputs'] = p_de_records
        return p_records

    def __getitem__(self, idx):
        cols = self.collection.iloc[idx]
        title = cols.title
        para = cols.para
        psg = title+" [SEP] "+para
        return psg

    def __len__(self):
        return self.num_samples


class QueryDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        self.args = args
        self.collection = pd.read_csv(args.dev_query, sep="\t",header=None, quoting=3)
        self.collection.columns = ['qid','qry']
        self.collection = self.collection.fillna("NA")
        self.num_samples = len(self.collection)
        
    def _collate_fn(self, qrys):
        return self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)

    def __getitem__(self, idx):
        return self.collection.iloc[idx].qry

    def __len__(self):
        return self.num_samples
    
class DualEncoderTrainDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t",header=None, quoting=3)
        self.collection.columns=['pid', 'title','para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args.query,sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = []
        with open(args.top1000,'r') as f:
            lines = f.readlines()
            for line in lines:
                qid,pids = line.strip().split()
                pids = pids.split(',')
                self.top1000.append((int(qid), [int(pid) for pid in pids]))
        self.len = len(self.top1000)
        self.min_index = args.min_index
        self.max_index = args.max_index
        qrels = defaultdict(list)
        with open(args.qrels,'r') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.strip().split()
                qrels[int(qid)].append(int(pid))
        self.generated_query = {}
        with open(args.generated_query) as f:
            lines = f.readlines()
            for line in lines:
                try:
                    pid, query = line.split("\t")
                    if query.strip()=="":
                        self.generated_query[int(pid)] = 'what'
                    else:
                        self.generated_query[int(pid)] = query.strip()
                except:
                    pass
        self.qrels = qrels
        self.sample_num = args.sample_num-1   
        self.epoch = 0
        self.num_samples = len(self.top1000)

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def sample(self, neg_pids, sample_num):
        if sample_num==0:
            return []
        if len(neg_pids)<sample_num:
            pad_num = sample_num - len(neg_pids)
            neg_pids+=[random.randint(0, 8841822) for _ in range(pad_num)]  # 用random neg补充
        interval = len(neg_pids)//sample_num
        offset = self.epoch%interval
        sample_pids = neg_pids[offset::interval][:sample_num]
        return sample_pids

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        generated_queries = []
        # neg_pids = list(cols[1]['pid'])
        # neg_scores = list(cols[1]['score'])
        neg_pids = cols[1]
        pos_id = random.choice(self.qrels.get(qid))
        sample_neg_pids = self.sample(neg_pids, self.sample_num)
        query = self.query.loc[qid]['text']
        data = self.collection.loc[pos_id]
        psgs = [data['title']+" [SEP] "+data['para']]
        generated_queries.append(self.generated_query.get(pos_id, 'what'))
        for neg_pid in sample_neg_pids:
            data = self.collection.loc[neg_pid]
            psgs.append(data['title']+" [SEP] "+data['para'])
            generated_queries.append(self.generated_query.get(neg_pid, 'what'))
        return [query], psgs, generated_queries

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        generated_query_len = self.args.generated_query_len
        generated_query = " ".join([self.tokenizer.mask_token]*generated_query_len)
        generated_queries = []
        for q, p, gen_q in sample_list:
            qrys+=q 
            psgs+=p 
            generated_queries+=gen_q
        q_records = self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        if self.args.add_decoder:
            p_de_records = self.tokenizer([generated_query]*len(psgs), padding=True, truncation=True, return_tensors="pt", max_length=generated_query_len)
            decoder_labels = self.tokenizer(generated_queries, padding='max_length', truncation=True, return_tensors="pt", max_length=generated_query_len, add_special_tokens=False)
            decoder_labels['input_ids'][decoder_labels['input_ids']==self.tokenizer.pad_token_id] = -100
            decoder_labels = decoder_labels['input_ids']
            # decoder_labels = self.ConstantPad(decoder_labels)
            mlm_labels = {"encoder_mlm_labels":None, "decoder_mlm_labels":decoder_labels}
            p_records['decoder_inputs'] = p_de_records
            p_records['mlm_labels'] = mlm_labels
        return {"query_inputs":q_records, "passage_inputs":p_records}

    def __len__(self):
        return self.num_samples

class DualEncoderDistillWithScoresDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t",header=None, quoting=3)
        self.collection.columns=['pid', 'title','para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args.query,sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')

        with open(args.top1000) as f:
            lines = f.readlines()  
            content=[]
            for line in lines:
                content.append(json.loads(line))
        self.top1000 = content
        # self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        # self.top1000.columns=['qid','pid','index','score']
        # self.top1000 = list(self.top1000.groupby("qid"))
        self.len = len(self.top1000)
        self.min_index = args.min_index
        self.max_index = args.max_index
        qrels = defaultdict(list)
        with open(args.qrels,'r') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid,idx,score = line.strip().split()
                qrels[int(qid)].append((int(pid), float(score)))
        self.qrels = qrels
        self.sample_num = args.sample_num-1   
        self.epoch = 0
        self.num_samples = len(self.top1000)
        self.generated_query = {}
        with open(args.generated_query) as f:
            lines = f.readlines()
            for line in lines:
                try:
                    pid, query = line.split("\t")
                    if query.strip()=="":
                        self.generated_query[int(pid)] = 'what'
                    else:
                        self.generated_query[int(pid)] = query.strip()
                except:
                    pass

    def set_epoch(self, epoch):
        self.epoch = epoch
        print(self.epoch)
    
    def sample(self, negatives, sample_num):
        '''
        qid:int
        negs:dataframe qid,pid,index,score
        pos_score: float 正psg的分数
        sample_num:int
        '''
        # pids = [pid for pid in pids if pid not in self.qrels[qid]] 已经过滤了
        # pids = pids[self.args.min_index:self.args.max_index] 不用这个方式过滤了，用ce分数
        if sample_num==0:
            return [], []
        neg_pids = negatives['doc_id']
        neg_pids = [int(pid) for pid in neg_pids]
        neg_scores = negatives['score']
        neg_scores = [float(score) for score in neg_scores]
        if len(neg_pids)<sample_num:
            pad_num = sample_num - len(neg_pids)
            neg_pids+=[random.randint(0, 8841822) for _ in range(pad_num)]  # 用random neg补充
            neg_scores+=[-15 for _ in range(pad_num)]
        interval = len(neg_pids)//sample_num
        offset = self.epoch%interval
        sample_pids = neg_pids[offset::interval][:sample_num]
        sample_scores = neg_scores[offset::interval][:sample_num]
        return sample_pids, sample_scores

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        # qid = cols[0]
        # negs = cols[1]
        positives = cols['positives']
        idxs = list(range(len(positives['doc_id'])))
        idx = random.choice(idxs)
        pos_id, pos_score = int(positives['doc_id'][idx]), float(positives['score'][idx])
        scores = [pos_score]
        sample_neg_pids, sample_neg_scores = self.sample(cols['negatives'], self.sample_num)
        scores+=sample_neg_scores
        query = cols['query']
        data = self.collection.loc[pos_id]
        psgs = [data['title']+" [SEP] "+data['para']]
        generated_queries = []
        generated_queries.append(self.generated_query.get(pos_id, 'what'))
        for neg_pid in sample_neg_pids:
            data = self.collection.loc[neg_pid]
            psgs.append(data['title']+" [SEP] "+data['para'])
            generated_queries.append(self.generated_query.get(neg_pid, 'what'))
        return [query], psgs, scores, generated_queries

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        scores = []
        generated_query_len = self.args.generated_query_len
        generated_query = " ".join([self.tokenizer.mask_token]*generated_query_len)
        generated_queries = []
        for q, p, s, gen_q in sample_list:
            qrys+=q 
            psgs+=p 
            scores+=s
            generated_queries+=gen_q
        scores = torch.from_numpy(np.array(scores))
        q_records = self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        if self.args.add_decoder:
            p_de_records = self.tokenizer([generated_query]*len(psgs), padding=True, truncation=True, return_tensors="pt", max_length=generated_query_len)
            decoder_labels = self.tokenizer(generated_queries, padding='max_length', truncation=True, return_tensors="pt", max_length=generated_query_len, add_special_tokens=False)
            decoder_labels['input_ids'][decoder_labels['input_ids']==self.tokenizer.pad_token_id] = -100
            decoder_labels = decoder_labels['input_ids']
            # decoder_labels = self.ConstantPad(decoder_labels)
            mlm_labels = {"encoder_mlm_labels":None, "decoder_mlm_labels":decoder_labels}
            p_records['decoder_inputs'] = p_de_records
            p_records['mlm_labels'] = mlm_labels
        return {"query_inputs":q_records, "passage_inputs":p_records, "reranker_scores":scores}

    def __len__(self):
        return self.num_samples
