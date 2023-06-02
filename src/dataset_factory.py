import os
import random

import numpy as np
import pandas as pd
from collections import defaultdict
import pytorch_pretrained_bert
import torch
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorForWholeWordMask

class CorpusPretrainDataset(Dataset):
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
        self.num_samples = len(self.collection)
        self.encoder_wwm = DataCollatorForWholeWordMask(self.tokenizer, mlm_probability=.35)
        self.ConstantPad = torch.nn.ConstantPad2d(padding=(1,1,0,0), value=-100)

    def generate_pseudo_query_passage(self, psg):
        '''
        输入psg:str
        输出qry, pos_psg: str, str
        '''
        psg = psg.split()
        if len(psg)==1:
            return " ".join(psg), " ".join(psg)
        qry_len = min(max(min(np.random.poisson(5), len(psg)//2), 1), 10)
        idxs = list(range(len(psg)-qry_len))
        start = random.choice(idxs)
        qry = " ".join(psg[start:start+qry_len])
        psg = " ".join(psg[:start]+psg[start+qry_len:])
        return qry, psg

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        generated_query_len = self.args.generated_query_len
        generated_query = " ".join([self.tokenizer.mask_token]*generated_query_len)
        for q, psg in sample_list:
            qrys.append(q)
            psgs.append(psg)
        q_records = self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        p_records.update(self.encoder_wwm(p_records['input_ids'].numpy(), return_tensors="pt"))
        if self.args.add_decoder:
            p_de_records = self.tokenizer([generated_query]*len(psgs), padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
            decoder_labels = self.tokenizer(qrys, padding='max_length', truncation=True, return_tensors="pt", max_length=generated_query_len, add_special_tokens=False)
            decoder_labels['input_ids'][decoder_labels['input_ids']==self.tokenizer.pad_token_id] = -100
            decoder_labels = decoder_labels['input_ids']
            decoder_labels = self.ConstantPad(decoder_labels)
            mlm_labels = {"encoder_mlm_labels":p_records.pop("labels"), "decoder_mlm_labels":decoder_labels}
            p_records['decoder_inputs'] = p_de_records
            p_records['mlm_labels'] = mlm_labels
        return {"query_inputs":q_records, "passage_inputs":p_records}

    def __getitem__(self, idx):
        cols = self.collection.iloc[idx]
        title = cols.title
        para = cols.para
        psg = title+' '+para
        qry, psg = self.generate_pseudo_query_passage(psg)
        return qry, psg

    def __len__(self):
        return self.num_samples

class RetrievalPAIRDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        self.top1000 = shuffle(self.top1000)
        self.top1000.columns=['query','pos_title','pos_para','neg_title','neg_para','label']
        self.top1000 = self.top1000.fillna("NA") 
        self.num_samples = len(self.top1000)
        self.ConstantPad = torch.nn.ConstantPad2d(padding=(1,1,0,0), value=-100)

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        generated_query_len = self.args.generated_query_len
        generated_query = " ".join([self.tokenizer.mask_token]*generated_query_len)
        for q, pos, neg in sample_list:
            qrys.append(q)
            psgs.append(pos)
            psgs.append(neg)
        q_records = self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        if self.args.add_decoder:
            p_de_records = self.tokenizer([generated_query]*len(psgs), padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
            decoder_labels = self.tokenizer(qrys, padding='max_length', truncation=True, return_tensors="pt", max_length=generated_query_len, add_special_tokens=False)
            decoder_labels['input_ids'][decoder_labels['input_ids']==self.tokenizer.pad_token_id] = -100
            decoder_labels = decoder_labels['input_ids']
            decoder_labels = self.ConstantPad(decoder_labels)
            mlm_labels = {"encoder_mlm_labels":None, "decoder_mlm_labels":decoder_labels}
            p_records['decoder_inputs'] = p_de_records
            p_records['mlm_labels'] = mlm_labels
        return {"query_inputs":q_records, "passage_inputs":p_records}

    def __getitem__(self, idx):
        cols = self.top1000.iloc[idx]
        return cols.query, cols.pos_title+" "+cols.pos_para, cols.neg_title+" "+cols.neg_para

    def __len__(self):
        return self.num_samples


class RetrievalPAIR4BARTDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        self.top1000 = shuffle(self.top1000)
        self.top1000.columns=['query','pos_title','pos_para','neg_title','neg_para','label']
        self.top1000 = self.top1000.fillna("NA") 
        self.num_samples = len(self.top1000)

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        q_gens = []
        prompt = "The generated query is "+self.tokenizer.mask_token*(self.args.q_max_seq_len-6)
        for q, pos, neg in sample_list:
            qrys.append(q)
            psgs.append(pos)
            psgs.append(neg)
            q_gens.append(prompt)
            q_gens.append(prompt)
        q_records = self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        p_de_records = self.tokenizer(psgs, q_gens, padding=True, truncation='only_first', return_tensors="pt", max_length=self.args.max_seq_len)
        p_records['decoder_input_ids'] = p_de_records['input_ids']
        p_records['decoder_attention_mask'] = p_de_records['attention_mask']
        decoder_labels = self.tokenizer(qrys, padding='max_length', truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len-6,add_special_tokens=False)
        return {"query":q_records, "passage":p_records, "decoder_labels":decoder_labels}

    def __getitem__(self, idx):
        cols = self.top1000.iloc[idx]
        return cols.query, cols.pos_title+" "+cols.pos_para, cols.neg_title+" "+cols.neg_para

    def __len__(self):
        return self.num_samples



class TrainDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t",header=None, quoting=3)
        self.collection.columns=['pid','para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args.query,sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        self.top1000.columns=['qid','pid','index']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.len = len(self.top1000)
        self.num_samples = len(self.top1000)
        self.min_index = args.min_index
        self.max_index = args.max_index
        qrels={}
        with open(args.qrels,'r') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels
        self.sample_num = args.sample_num-2        
        
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks
    
    def sample(self, qid, pids, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        sample_pids = []
        pids = pids[self.min_index:self.max_index]
        interval = len(pids)//sample_num
        for i in range(sample_num):
            found = False
            for _ in range(interval):
                neg_id = random.choice(pids[i*interval:(i+1)*interval])
                if neg_id not in self.qrels[qid]:
                    found = True
                    break
            if not found:
                neg_id = 1
            sample_pids.append(neg_id)
        return sample_pids
    
    def _truncate_seq(self, text, max_length):
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:max_length]
        return tokens
    
    def convert_data_to_features(self, slot):
        tokens = self._truncate_seq(slot, self.args.max_seq_len-1)
        slot = ['[CLS]'] + tokens

        input_ids = self.tokenizer.convert_tokens_to_ids(slot)
        input_ids = np.array(input_ids, dtype=np.int64)
        mask_ids = np.array([1]*(1+len(tokens)),dtype=np.int64)
        input_ids = np.pad(input_ids, (0, self.args.max_seq_len-len(input_ids)), 'constant', constant_values=(0,0))
        mask_ids = np.pad(mask_ids, (0, self.args.max_seq_len-len(mask_ids)), 'constant', constant_values=(0,0))
        return [input_ids, mask_ids]
    
    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid,pids,self.sample_num)
        pos_id = random.choice(self.qrels.get(qid))
        query = self.query.loc[qid]['text']
        features = []
        feature = self.convert_data_to_features(query)
        features.append(feature)
        feature = self.convert_data_to_features(self.collection.loc[pos_id]['para'])
        features.append(feature)
        for neg_pid in sample_neg_pids:
            feature = self.convert_data_to_features(self.collection.loc[neg_pid]['para'])
            features.append(feature)
        data = [np.stack(s, axis=0) for s in list(zip(*features))]
        return data
        
    def collate_fn(self, sample_list):
        batch_size = len(sample_list)
        tmp = list(zip(*sample_list))
        batch = [np.concatenate(s, axis=0) for s in tmp]
        input_ids, mask_ids = batch
        list_size = self.args.sample_num-1
        labels = np.array(range(list_size*batch_size*self.rank, list_size*batch_size*(1+self.rank), list_size),dtype='int64')
        labels = torch.tensor(labels).long().cuda()
        input_ids = torch.tensor(input_ids).long().cuda()
        mask_ids = torch.tensor(mask_ids).float().cuda()
        return input_ids, mask_ids, labels
        

    def __len__(self):
        return self.len


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
    

class CrossEncoderTrainDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t",header=None, quoting=3)
        self.collection.columns=['pid', 'title', 'para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args.query,sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        self.top1000.columns=['qid','pid','index']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.len = len(self.top1000)
        self.min_index = args.min_index
        self.max_index = args.max_index
        qrels={}
        with open(args.qrels,'r') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels
        self.sample_num = args.sample_num-1   
        self.epoch = 0
        self.num_samples = len(self.top1000)

    def set_epoch(self, epoch):
        self.epoch = epoch
        print(self.epoch)
    
    def sample(self, qid, pids, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        pids = [pid for pid in pids if pid not in self.qrels[qid]]
        pids = pids[self.args.min_index:self.args.max_index]
        interval = len(pids)//sample_num
        offset = self.epoch%interval
        sample_pids = pids[offset::interval][:sample_num]
        return sample_pids

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid, pids, self.sample_num)
        pos_id = random.choice(self.qrels.get(qid))
        query = self.query.loc[qid]['text']
        data = [(query, self.collection.loc[pos_id]['title']+' '+self.collection.loc[pos_id]['para'])]
        for neg_pid in sample_neg_pids:
            data.append((query, self.collection.loc[neg_pid]['title']+' '+self.collection.loc[neg_pid]['para']))
        return data

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        for qp_pairs in sample_list:
            for q,p in qp_pairs:
                qrys.append(q)
                psgs.append(p)
        features = self.tokenizer(qrys, psgs,  padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
        return features

    def __len__(self):
        return self.num_samples

class CrossEncoderDevDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t",header=None, quoting=3)
        self.collection.columns=['pid', 'title', 'para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args.dev_query,sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(args.dev_top1000, sep="\t",header=None)
        self.num_samples = len(self.top1000)


    def __getitem__(self, idx):
        cols = self.top1000.iloc[idx]
        qid = cols[0]
        pid = cols[1]
        return self.query.loc[qid]['text'], self.collection.loc[pid]['title']+' '+self.collection.loc[pid]['para'], qid, pid

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        qids = []
        pids = []
        for q,p,qid,pid in sample_list:
            qrys.append(q)
            psgs.append(p)
            qids.append(qid)
            pids.append(pid)
        features = self.tokenizer(qrys, psgs,  padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
        return features, {"qids":np.array(qids),"pids":np.array(pids)}
        
    def __len__(self):
        return self.num_samples

class CrossEncoderFilter4NegDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t",header=None, quoting=3)
        self.collection.columns=['pid', 'title', 'para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args.query,sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        self.top1000.columns=['qid','pid','index']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.len = len(self.top1000)
        self.min_index = args.min_index
        self.max_index = args.max_index
        qrels={}
        with open(args.qrels,'r') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels
        self.sample_num = args.sample_num
        self.epoch = 0
        self.num_samples = len(self.top1000)


    def sample(self, qid, pids):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        pids = [pid for pid in pids if pid not in self.qrels[qid]]
        pids = pids[self.args.min_index:self.args.max_index]
        return pids

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid, pids)
        query = self.query.loc[qid]['text']
        data = []
        for neg_pid in sample_neg_pids:
            data.append((query, self.collection.loc[neg_pid]['title']+' '+self.collection.loc[neg_pid]['para']))
        return data, [qid]*len(sample_neg_pids), sample_neg_pids

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        qids_lst = []
        pids_lst = []
        for qp_pairs, qids, pids in sample_list:
            for i in range(len(qp_pairs)):
                q,p = qp_pairs[i]
                qid = qids[i]
                pid = pids[i]
                qrys.append(q)
                psgs.append(p)
                qids_lst.append(qid)
                pids_lst.append(pid)
        features = self.tokenizer(qrys, psgs,  padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
        return features, {"qids":np.array(qids_lst),"pids":np.array(pids_lst)}
        
    def __len__(self):
        return self.num_samples

class CrossEncoderFilter4PosDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t",header=None, quoting=3)
        self.collection.columns=['pid', 'title', 'para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args.query,sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.min_index = args.min_index
        self.max_index = args.max_index
        top1000 = []
        with open(args.qrels,'r') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                top1000.append((qid, pid))
        self.top1000 = top1000
        self.sample_num = args.sample_num
        self.epoch = 0
        self.num_samples = len(self.top1000)

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pid = cols[1]
        query = self.query.loc[qid]['text']
        passage = self.collection.loc[pid]['title']+' '+self.collection.loc[pid]['para']
        return query, passage, qid, pid

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        qids_lst = []
        pids_lst = []
        for q, p, qid, pid in sample_list:
            qrys.append(q)
            psgs.append(p)
            qids_lst.append(qid)
            pids_lst.append(pid)
        features = self.tokenizer(qrys, psgs,  padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
        return features, {"qids":np.array(qids_lst),"pids":np.array(pids_lst)}
        
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
        self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        self.top1000.columns=['qid','pid','index','score']
        self.top1000 = list(self.top1000.groupby("qid"))
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
        self.ConstantPad = torch.nn.ConstantPad2d(padding=(1,1,0,0), value=-100)

    def set_epoch(self, epoch):
        self.epoch = epoch
        print(self.epoch)
    
    def sample(self, qid, negs, pos_score, sample_num):
        '''
        qid:int
        negs:dataframe qid,pid,index,score
        pos_score: float 正psg的分数
        sample_num:int
        '''
        # pids = [pid for pid in pids if pid not in self.qrels[qid]] 已经过滤了
        # pids = pids[self.args.min_index:self.args.max_index] 不用这个方式过滤了，用ce分数
        negs = negs.copy()  # 防止改了原始数据
        min_score = min(negs['score'].min(), pos_score-1e-9)
        pos_score -= min_score
        assert pos_score>0
        negs['score'] -= min_score
        negs['score'] /= pos_score  # 放缩到0-1
        negs = negs[(negs['score']<self.args.FN_threshold)&(negs['score']>self.args.EN_threshold)]
        neg_pids = list(negs['pid'])[self.args.min_index:self.args.max_index]
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
        # neg_pids = list(cols[1]['pid'])
        # neg_scores = list(cols[1]['score'])
        negs = cols[1]
        pos_id, pos_score = random.choice(self.qrels.get(qid))
        sample_neg_pids = self.sample(qid, negs, pos_score, self.sample_num)
        query = self.query.loc[qid]['text']
        data = self.collection.loc[pos_id]
        psgs = [data['title']+" [SEP] "+data['para']]
        for neg_pid in sample_neg_pids:
            data = self.collection.loc[neg_pid]
            psgs.append(data['title']+" [SEP] "+data['para'])
        return [query], psgs

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        generated_query_len = self.args.generated_query_len
        generated_query = " ".join([self.tokenizer.mask_token]*generated_query_len)
        for q, p in sample_list:
            qrys+=q 
            psgs+=p 
        q_records = self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        if self.args.add_decoder:
            p_de_records = self.tokenizer([generated_query]*len(psgs), padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
            decoder_labels = self.tokenizer(qrys, padding='max_length', truncation=True, return_tensors="pt", max_length=generated_query_len, add_special_tokens=False)
            decoder_labels['input_ids'][decoder_labels['input_ids']==self.tokenizer.pad_token_id] = -100
            decoder_labels = decoder_labels['input_ids']
            decoder_labels = self.ConstantPad(decoder_labels)
            mlm_labels = {"encoder_mlm_labels":None, "decoder_mlm_labels":decoder_labels}
            p_records['decoder_inputs'] = p_de_records
            p_records['mlm_labels'] = mlm_labels
        return {"query_inputs":q_records, "passage_inputs":p_records}

    def __len__(self):
        return self.num_samples


class DualEncoderDistillDataset(Dataset):
    def __init__(self, args):
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name_or_path)
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
        self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        self.top1000.columns=['qid','pid','index']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.len = len(self.top1000)
        self.min_index = args.min_index
        self.max_index = args.max_index
        qrels={}
        with open(args.qrels,'r') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels
        self.sample_num = args.sample_num-1   
        self.epoch = 0
        self.num_samples = len(self.top1000)

    def set_epoch(self, epoch):
        self.epoch = epoch
        print(self.epoch)
    
    def sample(self, qid, pids, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        pids = [pid for pid in pids if pid not in self.qrels[qid]]
        pids = pids[self.args.min_index:self.args.max_index]
        interval = len(pids)//sample_num
        offset = self.epoch%interval
        sample_pids = pids[offset::interval][:sample_num]
        return sample_pids

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid, pids, self.sample_num)
        pos_id = random.choice(self.qrels.get(qid))
        query = self.query.loc[qid]['text']
        data = self.collection.loc[pos_id]
        psgs = [data['title']+" [SEP] "+data['para']]
        for neg_pid in sample_neg_pids:
            data = self.collection.loc[neg_pid]
            psgs.append(data['title']+" [SEP] "+data['para'])
        return [query], psgs

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        qrys4reranker = []
        for q, p in sample_list:
            qrys+=q 
            psgs+=p 
            qrys4reranker+=q*self.args.sample_num
        q_records = self.retriever_tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.retriever_tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        reranker_records = self.reranker_tokenizer(qrys4reranker, psgs,  padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
        return {"query":q_records, "passage":p_records, "reranker":reranker_records}

    def __len__(self):
        return self.num_samples


class DualEncoderQueryAwareDistillDataset(Dataset):
    def __init__(self, args):
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name_or_path)
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
        self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        self.top1000.columns=['qid','pid','index']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.len = len(self.top1000)
        self.min_index = args.min_index
        self.max_index = args.max_index
        qrels={}
        with open(args.qrels,'r') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels
        self.sample_num = args.sample_num-1   
        self.epoch = 0
        self.num_samples = len(self.top1000)

    def set_epoch(self, epoch):
        self.epoch = epoch
        print(self.epoch)
    
    def sample(self, qid, pids, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        pids = [pid for pid in pids if pid not in self.qrels[qid]]
        pids = pids[self.args.min_index:self.args.max_index]
        interval = len(pids)//sample_num
        offset = self.epoch%interval
        sample_pids = pids[offset::interval][:sample_num]
        return sample_pids

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid, pids, self.sample_num)
        pos_id = random.choice(self.qrels.get(qid))
        query = self.query.loc[qid]['text']
        data = self.collection.loc[pos_id]
        psgs = [data['title']+" [SEP] "+data['para']]
        for neg_pid in sample_neg_pids:
            data = self.collection.loc[neg_pid]
            psgs.append(data['title']+" [SEP] "+data['para'])
        return [query], psgs

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        qrys4reranker = []
        for q, p in sample_list:
            qrys+=q 
            psgs+=p 
            qrys4reranker+=q*self.args.sample_num
        q_records = self.retriever_tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.retriever_tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        reranker_records = self.reranker_tokenizer(qrys4reranker, psgs,  padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
        return {"query":q_records, "passage":p_records, "reranker":reranker_records}

    def __len__(self):
        return self.num_samples


class RetrievalRocketQA4DistillDataset(Dataset):
    def __init__(self, args):
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        self.top1000.columns=['query','title','para', 'label']
        self.top1000 = self.top1000.fillna("NA") 
        self.num_samples = len(self.top1000)//128

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        qrys4reranker = []
        for q,p in sample_list:
            qrys+=q
            psgs+=p 
            qrys4reranker+=q*len(p)
        q_records = self.retriever_tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.retriever_tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        
        reranker_records = self.reranker_tokenizer(qrys4reranker, psgs,  padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
        return {"query":q_records, "passage":p_records, "reranker":reranker_records}

    def __getitem__(self, idx):
        idx = idx*128
        data = self.top1000[idx:idx+128]
        qrys = data['query'].iloc[0]
        psgs = list(data['title']+" [SEP] "+data['para'])
        return [qrys], psgs


    def __len__(self):
        return self.num_samples
class RetrievalRocketQADataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.top1000 = pd.read_csv(args.top1000, sep="\t",header=None)
        self.top1000.columns=['query','title','para', 'label']
        self.top1000 = self.top1000.fillna("NA") 
        self.num_samples = len(self.top1000)//128

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        for q,p in sample_list:
            qrys+=q
            psgs+=p 
        q_records = self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        return {"query":q_records, "passage":p_records}

    def __getitem__(self, idx):
        idx = idx*128
        data = self.top1000[idx:idx+128]
        qrys = data['query'].iloc[0]
        psgs = list(data['title']+" [SEP] "+data['para'])
        return [qrys], psgs


    def __len__(self):
        return self.num_samples


