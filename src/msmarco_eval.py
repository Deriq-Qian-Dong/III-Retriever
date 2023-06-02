"""
This module computes evaluation metrics for MSMARCO dataset on the ranking task.
Command line:
python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
Creation Date : 06/12/2018
Last Modified : 1/21/2019
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""
import itertools
import sys
from collections import Counter

import numpy as np
import pandas as pd

MaxMRRRank = 10
import paddle


class Mrr(paddle.metric.Metric):
    """doc"""
    def __init__(self, cfg):
        """doc"""
        self.reset()
        self._name = "Mrr"
        qrels={}
        with open(cfg['qrels'],'rb') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels

    def reset(self):
        """doc"""
        self.qid_saver = np.array([], dtype=np.int64)
        self.pid_saver = np.array([], dtype=np.int64)
        self.label_saver = np.array([], dtype=np.int64)
        self.pred_saver = np.array([], dtype=np.float32)

    def update(self, qid, pid, label, pred):
        if isinstance(qid, paddle.Tensor):
            qid = qid.numpy()
        if isinstance(pid, paddle.Tensor):
            pid = pid.numpy()
        if isinstance(label, paddle.Tensor):
            label = label.numpy()
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        if not (qid.shape[0] == label.shape[0] == pred.shape[0]):
            raise ValueError(
                'Mrr dimention not match: qid[%s] label[%s], pred[%s]' %
                (qid.shape, label.shape, pred.shape))
        self.qid_saver = np.concatenate(
            [self.qid_saver, qid.reshape([-1]).astype(np.int64)])
        self.pid_saver = np.concatenate(
            [self.pid_saver, pid.reshape([-1]).astype(np.int64)])
        self.label_saver = np.concatenate(
            [self.label_saver, label.reshape([-1]).astype(np.int64)])
        self.pred_saver = np.concatenate(
            [self.pred_saver, pred.reshape([-1]).astype(np.float32)])
        
    def accumulate(self):
        """doc"""
        def _key_func(tup):
            return tup[0]
        def _calc_func(tup):
            ranks = [1. / (rank + 1.) for rank, (_, l, p) in enumerate(sorted(tup, key=lambda t: t[2], reverse=True)[:MaxMRRRank]) if l != 0]
            if len(ranks):
                return ranks[0]
            else:
                return 0.
        mrr_for_qid = [
            _calc_func(tup)
            for _, tup in itertools.groupby(
                sorted(
                    zip(self.qid_saver, self.label_saver, self.pred_saver),
                    key=_key_func),
                key=_key_func)
        ]
        mrr = np.float32(sum(mrr_for_qid) / len(mrr_for_qid))
        return mrr
    def accumulate_map(self):
        """doc"""
        def _key_func(tup):
            return tup[0]
        def _calc_func(tup):
            tmp = sorted(tup, key=lambda t: t[2], reverse=True)
            qid = tmp[0][0]
            ranks = [rank+1 for rank, (qid, l, p) in enumerate(tmp) if l != 0]
            ranks = [(i+1)/rank for i,rank in enumerate(ranks)]
            if len(ranks):
                return sum(ranks)/len(self.qrels.get(qid, [0]))
            else:
                return 0.
        map_for_qid = [
            _calc_func(tup)
            for _, tup in itertools.groupby(
                sorted(
                    zip(self.qid_saver, self.label_saver, self.pred_saver),
                    key=_key_func),
                key=_key_func)
        ]
        map = np.float32(sum(map_for_qid) / len(map_for_qid))
        return map
    # ndcg
    def get_dcg(self, y_pred, y_true, k):
        #注意y_pred与y_true必须是一一对应的，并且y_pred越大越接近label=1(用相关性的说法就是，与label=1越相关)
        df = pd.DataFrame({"y_pred":y_pred, "y_true":y_true})
        df = df.sort_values(by="y_pred", ascending=False)  # 对y_pred进行降序排列，越排在前面的，越接近label=1
        df = df.iloc[0:k, :]  # 取前K个
        dcg = (2 ** df["y_true"] - 1) / np.log2(np.arange(1, df["y_true"].count()+1) + 1) # 位置从1开始计数
        dcg = np.sum(dcg)
        return dcg
        
    def accumulate_ndcg(self):
        """doc"""
        def _key_func(tup):
            return tup[0]
        def _calc_func(tup):
            ranks = [(l, p) for rank, (_, l, p) in enumerate(sorted(tup, key=lambda t: t[2], reverse=True))]
            dcg = self.get_dcg([r[1] for r in ranks],[r[0] for r in ranks], 10)
            idcg = self.get_dcg([r[0] for r in ranks],[r[0] for r in ranks], 10)
            if idcg==0:
                return 0
            ndcg = dcg/idcg
            return ndcg
        ndcg_for_qid = [
            _calc_func(tup)
            for _, tup in itertools.groupby(
                sorted(
                    zip(self.qid_saver, self.label_saver, self.pred_saver),
                    key=_key_func),
                key=_key_func)
        ]
        ndcg = np.float32(sum(ndcg_for_qid) / len(ndcg_for_qid))
        return ndcg

    def name(self):
        """
        Returns metric name
        """
        return self._name
def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_to_relevant_passageids = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            qid = int(l[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(int(l[1]))
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids


def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    with open(path_to_reference, 'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids


def load_candidate_from_stream(f):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_passages = {}
    for l in f:
        try:
            l = l.strip().split()
            qid = int(l[0])
            pid = int(l[1])
            rank = int(l[2])
            if qid in qid_to_ranked_candidate_passages:
                pass
            else:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                tmp = [0] * 1000
                qid_to_ranked_candidate_passages[qid] = tmp
            qid_to_ranked_candidate_passages[qid][rank - 1] = pid
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qid_to_ranked_candidate_passages


def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """

    with open(path_to_candidate, 'r') as f:
        qid_to_ranked_candidate_passages = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_passages


def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Perform quality checks on the dictionaries
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set(
            [item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids - set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message


def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = 0
    qids_with_relevant_passages = 0
    ranking = []
    recall_q_top1 = set()
    recall_q_top50 = set()
    recall_q_all = set()

    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    MRR += 1.0 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
            for i, pid in enumerate(candidate_pid):
                if pid in target_pid:
                    recall_q_all.add(qid)
                    if i < 50:
                        recall_q_top50.add(qid)
                    if i == 0:
                        recall_q_top1.add(qid)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    MRR = MRR / len(qids_to_ranked_candidate_passages)
    recall_top1 = len(recall_q_top1) * 1.0 / len(qids_to_ranked_candidate_passages)
    recall_top50 = len(recall_q_top50) * 1.0 / len(qids_to_ranked_candidate_passages)
    recall_all = len(recall_q_all) * 1.0 / len(qids_to_ranked_candidate_passages)
    all_scores['MRR @10'] = MRR
    all_scores["recall@1"] = recall_top1
    all_scores["recall@50"] = recall_top50
    all_scores["recall@all"] = recall_all
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores


def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True):
    """Compute MRR metric
    Args:
    p_path_to_reference_file (str): path to reference file.
        Reference file should contain lines in the following format:
            QUERYID\tPASSAGEID
            Where PASSAGEID is a relevant passage for a query. Note QUERYID can repeat on different lines with different PASSAGEIDs
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file sould contain lines in the following format:
            QUERYID\tPASSAGEID1\tRank
            If a user wishes to use the TREC format please run the script with a -t flag at the end. If this flag is used the expected format is
            QUERYID\tITER\tDOCNO\tRANK\tSIM\tRUNID
            Where the values are separated by tabs and ranked in order of relevance
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """

    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)

    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)


def main():
    """Command line:
    python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
    """

    if len(sys.argv) == 3:
        path_to_reference = sys.argv[1]
        path_to_candidate = sys.argv[2]

    else:
        path_to_reference = 'metric/qp_reference.all.tsv'
        path_to_candidate = 'metric/ranking_res'
        #print('Usage: msmarco_eval_ranking.py <reference ranking> <candidate ranking>')
        #exit()

    metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)
    print('#####################')
    for metric in sorted(metrics):
        print('{}: {}'.format(metric, metrics[metric]))
    print('#####################')


def calc_mrr(path_to_reference, path_to_candidate):
    """Command line:
    python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
    """

    metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)
    print('#####################')
    for metric in sorted(metrics):
        print('{}: {}'.format(metric, metrics[metric]))
    print('#####################')
    return metrics


def get_mrr(path_to_reference="/home/dongqian06/codes/NAACL2021-RocketQA/corpus/marco/qrels.dev.tsv", path_to_candidate="output/step_0_pred_dev_scores.txt"):
    all_data = pd.read_csv(path_to_candidate,sep="\t",header=None)
    all_data.columns = ["qid","pid","score"]
    all_data = all_data.groupby("qid").apply(lambda x: x.sort_values('score', ascending=False).reset_index(drop=True))
    all_data.columns = ['query_id',"para_id","score"]
    all_data = all_data.reset_index()
    all_data.pop("qid")
    all_data.columns = ["index","qid","pid","score"]
    all_data = all_data.loc[:,["qid","pid","index","score"]]
    all_data['index']+=1
    path_to_candidate = path_to_candidate.replace("txt","qrels")
    all_data.to_csv(path_to_candidate, header=None,index=False,sep="\t")
    metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)
    return metrics['MRR @10']

if __name__ == '__main__':
    main()
