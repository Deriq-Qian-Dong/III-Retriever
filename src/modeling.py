import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (AutoConfig, AutoModel, AutoTokenizer,BertForMaskedLM,
                          AutoModelForSequenceClassification, BertModel, BertLayer,
                          PreTrainedModel)
from transformers.file_utils import ModelOutput
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from utils import filter_stop_words

logger = logging.getLogger(__name__)

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
        self.kl = nn.KLDivLoss(reduction="mean")

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = self.kl(p_s, p_t) * (self.T**2)
        return loss

def lambda_mrr_loss(y_pred, y_true, eps=1e-10, padded_value_indicator=-1, reduction="mean", sigma=1.):
    """
    y_pred: FloatTensor [bz, topk]
    y_true: FloatTensor [bz, topk]
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    clamp_val = 1e8 if y_pred.dtype==torch.float32 else 1e4

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")
    #assert torch.sum(padded_mask) == 0

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    inv_pos_idxs = 1. / torch.arange(1, y_pred.shape[1] + 1).to(device)
    weights = torch.abs(inv_pos_idxs.view(1,-1,1) - inv_pos_idxs.view(1,1,-1)) # [1, topk, topk]

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-clamp_val, max=clamp_val)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    losses = torch.log(1. + torch.exp(-scores_diffs)) * weights #[bz, topk, topk]

    if reduction == "sum":
        loss = torch.sum(losses[padded_pairs_mask])
    elif reduction == "mean":
        loss = torch.mean(losses[padded_pairs_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss

@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

@dataclass
class DistillOutput(ModelOutput):
    s2t_loss: Optional[Tensor] = None
    t2s_loss: Optional[Tensor] = None
    retriever_ce_loss: Optional[Tensor] = None
    reranker_ce_loss: Optional[Tensor] = None

class Reranker(nn.Module):
    def __init__(self, args):
        super(Reranker, self).__init__()
        self.lm = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=1, output_hidden_states=True)
        if args.gradient_checkpoint:
            self.lm.gradient_checkpointing_enable()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch):
        ret = self.lm(**batch, return_dict=True)
        logits = ret.logits
        if self.training:
            scores = logits.view(-1, self.args.sample_num)  # q_batch_size, sample_num
            target_label = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = self.cross_entropy(scores, target_label)
            return loss
        return logits

class Distiller(nn.Module):
    def __init__(self, args):
        super(Distiller, self).__init__()
        self.reranker = AutoModelForSequenceClassification.from_pretrained(args.reranker_model_name_or_path, num_labels=1, output_hidden_states=True)
        if not args.online_distill:
            for i in self.parameters():
                i.requires_grad = False
        # self.lm_q = BertModel.from_pretrained(args.retriever_model_name_or_path, add_pooling_layer=False)
        # self.lm_p = BertModel.from_pretrained(args.retriever_model_name_or_path, add_pooling_layer=False)
        self.lm_q = AutoModel.from_pretrained(args.retriever_model_name_or_path)
        self.lm_p = AutoModel.from_pretrained(args.retriever_model_name_or_path)
        if args.gradient_checkpoint:
            self.lm_q.gradient_checkpointing_enable()
            self.lm_p.gradient_checkpointing_enable()
            self.reranker.gradient_checkpointing_enable()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_div = DistillKL(args.Temperature)
        if self.args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def forward(self, query_inputs=None, passage_inputs=None, reranker_inputs=None):            
        if self.training:
            ret = self.reranker(**reranker_inputs, return_dict=True)
            qry_out = self.lm_q(**query_inputs, return_dict=True)
            q_hidden = qry_out.last_hidden_state
            q_reps = q_hidden[:, 0]

            psg_out = self.lm_p(**passage_inputs, return_dict=True)
            p_hidden = psg_out.last_hidden_state
            p_reps = p_hidden[:, 0]
            p_reps = p_reps.view(-1, self.args.sample_num, 768)

            student_scores = torch.matmul(q_reps.unsqueeze(1), p_reps.transpose(2,1))
            student_scores = student_scores.squeeze(1)

            reranker_logits = ret.logits
            reranker_scores = reranker_logits.view(-1, self.args.sample_num)

            s2t_loss = self.kl_div(student_scores, reranker_scores.detach())
            p_reps = p_reps.view(-1, 768)
            if self.args.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            retriever_ce_loss = self.cross_entropy(scores, target)

            # loss = self.args.alpha*s2t_loss + self.args.beta*retriever_ce_loss
            if self.args.online_distill:  # online蒸馏
                reranker_label = torch.zeros(reranker_scores.size(0), device=reranker_scores.device, dtype=torch.long)
                reranker_ce_loss = self.cross_entropy(reranker_scores, reranker_label)
                t2s_loss = self.kl_div(reranker_scores, student_scores.detach())
                return DistillOutput(
                        s2t_loss=self.args.alpha*s2t_loss, 
                        retriever_ce_loss=self.args.beta*retriever_ce_loss,
                        t2s_loss=self.args.gemma*t2s_loss, 
                        reranker_ce_loss=self.args.omega*reranker_ce_loss
                    )
            return DistillOutput(
                    s2t_loss=self.args.alpha*s2t_loss, 
                    retriever_ce_loss=self.args.beta*retriever_ce_loss
                )
        else: # inference
            if reranker_inputs is not None:
                ret = self.reranker(**reranker_inputs, return_dict=True)
                reranker_logits = ret.logits
                return reranker_logits
            else:
                if query_inputs is not None:
                    qry_out = self.lm_q(**query_inputs, return_dict=True)
                    q_hidden = qry_out.last_hidden_state
                    q_reps = q_hidden[:, 0]
                    return q_reps
                else:
                    psg_out = self.lm_p(**passage_inputs, return_dict=True)
                    p_hidden = psg_out.last_hidden_state
                    p_reps = p_hidden[:, 0]
                    return p_reps

class BARTDualEncoder(nn.Module):
    def __init__(self, args):
        super(BARTDualEncoder, self).__init__()
        self.lm_q = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False)
        self.lm_p = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False)
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        self.lm_head = nn.Linear(self.lm_p.shared.embedding_dim, self.lm_p.shared.num_embeddings, bias=False)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.lm_p.shared.num_embeddings)))

        if args.gradient_checkpoint:
            self.lm_q.gradient_checkpointing_enable()
            self.lm_p.gradient_checkpointing_enable()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if self.args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def forward(self, query_inputs=None, passage_inputs=None, decoder_mlm_labels=None):            
        if self.training:
            qry_out = self.lm_q(**query_inputs, return_dict=True)
            q_hidden = qry_out.last_hidden_state
            q_input_ids = query_inputs['input_ids']
            eos_mask = q_input_ids.eq(self.tokenizer.eos_token_id)

            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            q_reps = q_hidden[eos_mask, :].view(q_hidden.size(0), -1, q_hidden.size(-1))[:, -1, :]

            psg_out = self.lm_p(**passage_inputs, return_dict=True)
            p_hidden = psg_out.last_hidden_state
            p_input_ids = passage_inputs['decoder_input_ids']
            eos_mask = p_input_ids.eq(self.tokenizer.eos_token_id)
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            p_reps = p_hidden[eos_mask, :].view(p_hidden.size(0), -1, p_hidden.size(-1))[:, -1, :]

            qgen_mask = p_input_ids.eq(self.tokenizer.mask_token_id)
            generated_q_hidden = p_hidden[qgen_mask].view(p_hidden.size(0), -1, p_hidden.size(-1))  # [bz*sample_num, 26, 768]
            # 只对positive计算generation loss
            generated_q_hidden = generated_q_hidden[:,None,:,:].view(q_reps.size(0), self.args.sample_num, -1, p_hidden.size(-1))[:,0,:,:]  # [bz, 26, 768]

            mlm_mask = decoder_mlm_labels['input_ids']!=self.tokenizer.pad_token_id
            lm_logits = self.lm_head(generated_q_hidden[mlm_mask]) + self.final_logits_bias   # [-1, vocab_size]


            generation_loss = self.cross_entropy(lm_logits.view(-1, self.lm_p.shared.num_embeddings), decoder_mlm_labels['input_ids'][mlm_mask].view(-1))

            if self.args.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            retriever_ce_loss = self.cross_entropy(scores, target)
            return retriever_ce_loss, generation_loss
        else: # inference
            if query_inputs is not None:
                qry_out = self.lm_q(**query_inputs, return_dict=True)
                q_hidden = qry_out.last_hidden_state
                q_input_ids = query_inputs['input_ids']
                eos_mask = q_input_ids.eq(self.tokenizer.eos_token_id)

                if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")
                q_reps = q_hidden[eos_mask, :].view(q_hidden.size(0), -1, q_hidden.size(-1))[:, -1, :]
                return q_reps
            else:
                psg_out = self.lm_p(**passage_inputs, return_dict=True)
                p_hidden = psg_out.last_hidden_state
                p_input_ids = passage_inputs['decoder_input_ids']
                eos_mask = p_input_ids.eq(self.tokenizer.eos_token_id)
                if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")
                p_reps = p_hidden[eos_mask, :].view(p_hidden.size(0), -1, p_hidden.size(-1))[:, -1, :]
                return p_reps

class RepsMergeHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class DualEncoder(nn.Module):
    def __init__(self, args):
        super(DualEncoder, self).__init__()
        try:
            self.lm_q = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False,  add_pooling_layer=False, output_hidden_states=True)
            self.lm_p = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False,  add_pooling_layer=False, output_hidden_states=True)
        except:
            self.lm_q = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False, output_hidden_states=True)
            self.lm_p = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False, output_hidden_states=True)
        if args.add_decoder:
            config = AutoConfig.from_pretrained(args.retriever_model_name_or_path, num_labels=1, cache_dir='~/.cache/')
            config.is_decoder = True
            config.add_cross_attention = True
            self.c_head = nn.ModuleList(
                    [BertLayer(config) for _ in range(args.n_head_layers)]
                )
            self.config = config
            self.c_head.apply(self.lm_p._init_weights)
            self.before_merge_layer = RepsMergeHead(config)
            self.cls = BertOnlyMLMHead(config)
            # self.cls = self.lm_p.cls
        

        if args.gradient_checkpoint:
            self.lm_q.gradient_checkpointing_enable()
            self.lm_p.gradient_checkpointing_enable()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if self.args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def encode_query(self, query_inputs):
        qry_out = self.lm_q(**query_inputs, return_dict=True)
        q_hidden = qry_out.hidden_states[-1]
        q_reps = q_hidden[:, 0]
        if self.args.l2_normalize:
            q_reps = F.normalize(q_reps, dim=-1)
        return q_reps

    def encode_passage(self, passage_inputs):
        mlm_labels = None 
        passage_decoder_inputs = None
        if "mlm_labels" in passage_inputs:
            mlm_labels = passage_inputs.pop("mlm_labels")
        if "decoder_inputs" in passage_inputs:
            passage_decoder_inputs = passage_inputs.pop("decoder_inputs")
        psg_out = self.lm_p(**passage_inputs, return_dict=True)
        p_hidden = psg_out.hidden_states[-1]  # 需要输入decoder
        p_reps = p_hidden[:, 0]
        if self.args.l2_normalize:
            p_reps = F.normalize(p_reps, dim=-1)
        mlm_loss = 0.
        if mlm_labels is not None and mlm_labels['encoder_mlm_labels'] is not None:
            mlm_loss = self.mlm_loss(p_hidden, mlm_labels['encoder_mlm_labels'])
        if self.args.add_decoder:
            assert passage_decoder_inputs is not None 
            input_shape = passage_decoder_inputs['input_ids'].size()
            bz_plus_sample_num, _ = input_shape
            bz = bz_plus_sample_num//self.args.sample_num
            decoder_inputs_embeds = self.lm_p.embeddings(input_ids=passage_decoder_inputs['input_ids'])
            encoder_attention_mask = _expand_mask(passage_inputs['attention_mask'], decoder_inputs_embeds.dtype, tgt_len=input_shape[-1])
            decoder_attention_mask = _expand_mask(passage_decoder_inputs['attention_mask'], decoder_inputs_embeds.dtype)
            p_dec_hidden = decoder_inputs_embeds
            for layer in self.c_head:
                layer_out = layer(
                    hidden_states=p_dec_hidden,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=p_hidden,
                    encoder_attention_mask=encoder_attention_mask
                )
                p_dec_hidden = layer_out[0]
            generated_q_hidden = p_dec_hidden
            query_aware_p_reps = generated_q_hidden[:, 0]
            if self.training:
                # 只对positive计算generation loss
                generated_q_hidden = generated_q_hidden[:,None,:,:].view(bz, self.args.sample_num, -1, p_hidden.size(-1))[:,0,:,:]  # [bz, seq_len, 768]
                if mlm_labels is not None and mlm_labels['decoder_mlm_labels'] is not None:
                    mlm_loss += self.mlm_loss(generated_q_hidden, mlm_labels['decoder_mlm_labels'])  # query生成loss
            p_reps = self.before_merge_layer(torch.stack([p_reps, query_aware_p_reps], dim=1))  # [bz*sample_num, 2, 768]
            p_reps = p_reps.max(1).values
            # TODO:用col bert会不会更好点
        return p_reps, mlm_loss

    def decode_passage(self, passage_inputs):
        passage_decoder_inputs = passage_inputs.pop("decoder_inputs")
        psg_out = self.lm_p(**passage_inputs, return_dict=True)
        p_hidden = psg_out.hidden_states[-1]  # 需要输入decoder
        p_reps = p_hidden[:, 0]
        input_shape = passage_decoder_inputs['input_ids'].size()
        bz_plus_sample_num, _ = input_shape
        bz = bz_plus_sample_num//self.args.sample_num
        decoder_inputs_embeds = self.lm_p.embeddings(input_ids=passage_decoder_inputs['input_ids'])
        encoder_attention_mask = _expand_mask(passage_inputs['attention_mask'], decoder_inputs_embeds.dtype, tgt_len=input_shape[-1])
        decoder_attention_mask = _expand_mask(passage_decoder_inputs['attention_mask'], decoder_inputs_embeds.dtype)
        p_dec_hidden = decoder_inputs_embeds
        for layer in self.c_head:
            layer_out = layer(
                hidden_states=p_dec_hidden,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=p_hidden,
                encoder_attention_mask=encoder_attention_mask
            )
            p_dec_hidden = layer_out[0]
        generated_q_hidden = p_dec_hidden
        pred_scores = self.cls(generated_q_hidden)
        ret = pred_scores.view(-1, self.config.vocab_size)
        # return " ".join(filter_stop_words(tokenizer.batch_decode(torch.topk(ret, 20)[1].numpy())))
        return ret

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.cls(hiddens)
        masked_lm_loss = self.cross_entropy(pred_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss
    def forward(self, query_inputs=None, passage_inputs=None):
        if self.training:
            q_reps = self.encode_query(query_inputs)
            p_reps, mlm_loss = self.encode_passage(passage_inputs)

            if self.args.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            if self.args.negatives_in_device:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))
            else:
                p_reps = p_reps.view(-1, self.args.sample_num, 768)
                scores = torch.matmul(q_reps.unsqueeze(1), p_reps.transpose(2,1))
                scores = scores.squeeze(1)
                target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = self.cross_entropy(scores, target)
            return loss, mlm_loss
        else:
            if query_inputs is not None:
                return self.encode_query(query_inputs)
            else:
                p_reps, _ =  self.encode_passage(passage_inputs)
                return p_reps





class EncoderPooler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderPooler, self).__init__()
        self._config = {}

    def forward(self, q_reps, p_reps):
        raise NotImplementedError('EncoderPooler is an abstract class')

    def load(self, pooler_path: str):
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f'Loading Pooler from {pooler_path}')
                state_dict = torch.load(pooler_path, map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = BertModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False,
                 gradient_checkpoint: bool =True
                 ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder
        self.gradient_checkpoint = gradient_checkpoint
        if gradient_checkpoint:
            self.lm_q.gradient_checkpointing_enable()
            if untie_encoder:
                self.lm_p.gradient_checkpointing_enable()
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
            return EncoderOutput(loss=loss)
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            args,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(args.model_name_or_path):
            if args.untie_encoder:
                _qry_model_path = os.path.join(args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = args.model_name_or_path
                    _psg_model_path = args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                print(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                print(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if args.untie_encoder else lm_q

        if args.add_pooler:
            pooler = cls.build_pooler(args)
        else:
            pooler = None
        if args.gradient_checkpoint:
            lm_q.gradient_checkpointing_enable()
            if args.untie_encoder:
                lm_p.gradient_checkpointing_enable() 
        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=args.negatives_x_device,
            untie_encoder=args.untie_encoder
        )
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path,
            **hf_kwargs,
    ):
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
                untie_encoder = False
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(pooler_weights, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=untie_encoder
        )
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)
        if self.pooler:
            self.pooler.save_pooler(output_dir)

class DensePooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True):
        super(DensePooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q[:, 0])
        elif p is not None:
            return self.linear_p(p[:, 0])
        else:
            raise ValueError


class DenseModel(EncoderModel):
    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        return p_reps

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden)
        else:
            q_reps = q_hidden[:, 0]
        return q_reps

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = DensePooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = DensePooler(
            768,
            768,
            tied=not model_args.untie_encoder
        )
        try:
            pooler.load(model_args.model_name_or_path)
        except:
            pass
        return pooler
