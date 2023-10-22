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
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=-1)
        p_t = F.log_softmax(y_t, dim=-1)
        loss = self.kl(p_s, p_t)
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
    mlm_loss: Optional[Tensor] = None
    retriever_ce_loss: Optional[Tensor] = None
    reranker_ce_loss: Optional[Tensor] = None

class DistillerWithoutReranker(nn.Module):
    def __init__(self, args):
        super(DistillerWithoutReranker, self).__init__()
        self.dual_encoder = DualEncoder(args)
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_div = DistillKL(args.Temperature)
        if self.args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query_inputs=None, passage_inputs=None, reranker_scores=None):            
        if self.training:
            q_reps = self.dual_encoder.encode_query(query_inputs)
            p_reps, mlm_loss = self.dual_encoder.encode_passage(passage_inputs)
            p_reps = p_reps.view(-1, self.args.sample_num, 768)
            scores = torch.matmul(q_reps.unsqueeze(1), p_reps.transpose(2,1))
            student_scores = scores.squeeze(1)

            p_reps = p_reps.view(-1, 768)
            student_loss = None
            if self.args.negatives_x_device:
                q_reps = self.dual_encoder._dist_gather_tensor(q_reps)
                p_reps = self.dual_encoder._dist_gather_tensor(p_reps)
            if self.args.negatives_in_device:
                scores = self.dual_encoder.compute_similarity(q_reps, p_reps)
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))
                student_loss = self.cross_entropy(scores, target)

            reranker_scores = reranker_scores.view(-1, self.args.sample_num)
            s2t_loss = self.kl_div(student_scores, reranker_scores)
            return DistillOutput(
                    s2t_loss=s2t_loss,
                    retriever_ce_loss=student_loss,
                    mlm_loss=mlm_loss,
                )
        else: # inference
            if query_inputs is not None:
                return self.dual_encoder.encode_query(query_inputs)
            else:
                p_reps, _ = self.dual_encoder.encode_passage(passage_inputs)
                return p_reps

class DualEncoder(nn.Module):
    def __init__(self, args):
        super(DualEncoder, self).__init__()
        try:
            self.lm_q = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False,  add_pooling_layer=False, output_hidden_states=True)
            self.lm_p = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False,  add_pooling_layer=False, output_hidden_states=True)
        except:
            self.lm_q = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False, output_hidden_states=True)
            self.lm_p = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False, output_hidden_states=True)

        # for i in self.parameters():
            # i.requires_grad = False

        if args.add_decoder:
            config = AutoConfig.from_pretrained(args.retriever_model_name_or_path, num_labels=1, cache_dir='~/.cache/')
            self.d_head = nn.ModuleList(
                    [BertLayer(config) for _ in range(args.n_head_layers)]
                )
            config.is_decoder = True
            config.add_cross_attention = True
            self.c_head = nn.ModuleList(
                    [BertLayer(config) for _ in range(args.n_head_layers)]
                )
            self.config = config
            self.c_head.apply(self.lm_p._init_weights)
            self.d_head.apply(self.lm_p._init_weights)
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
        mlm_loss = 0.
        if mlm_labels is not None and mlm_labels['encoder_mlm_labels'] is not None:
            mlm_loss = self.mlm_loss(p_hidden, mlm_labels['encoder_mlm_labels'])
        if self.args.add_decoder:
            assert passage_decoder_inputs is not None 
            input_shape = passage_decoder_inputs['input_ids'].size()
            bz_plus_sample_num, _ = input_shape
            bz = bz_plus_sample_num//self.args.sample_num
            decoder_inputs_embeds = self.lm_p.embeddings(input_ids=passage_decoder_inputs['input_ids'])
            decoder_attention_mask = _expand_mask(passage_decoder_inputs['attention_mask'], decoder_inputs_embeds.dtype)
            p_dec_hidden = decoder_inputs_embeds
            for layer in self.c_head:
                layer_out = layer(
                    hidden_states=p_dec_hidden,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=p_reps.unsqueeze(1),
                )
                p_dec_hidden = layer_out[0]
            generated_q_hidden = p_dec_hidden
            hiddens = torch.cat([p_reps.unsqueeze(1), generated_q_hidden.mean(dim=1).unsqueeze(1)], dim=1)
            for layer in self.d_head:
                layer_out = layer(hidden_states=hiddens)
                hiddens = layer_out[0]
            p_reps = hiddens[:, 0]
            if self.training:
                # 只对positive计算generation loss
                # generated_q_hidden = generated_q_hidden[:,None,:,:].view(bz, self.args.sample_num, -1, p_hidden.size(-1))[:,0,:,:]  # [bz, seq_len, 768]
                if mlm_labels is not None and mlm_labels['decoder_mlm_labels'] is not None:
                    mlm_loss += self.mlm_loss(generated_q_hidden, mlm_labels['decoder_mlm_labels'])  # query生成loss
            # TODO:用col bert会不会更好点
        if self.args.l2_normalize:
            p_reps = F.normalize(p_reps, dim=-1)
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
            if self.args.ret_loss:
                loss = self.cross_entropy(scores, target)
                return loss, mlm_loss
            else:
                return scores
        else:
            if query_inputs is not None:
                return self.encode_query(query_inputs)
            else:
                p_reps, _ =  self.encode_passage(passage_inputs)
                return p_reps
