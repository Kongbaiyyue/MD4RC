import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import PositionwiseFeedForward, LayerNorm, MultiHeadedAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attn):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, plan, log, metrics, src_mask, plan_mask):
        input_norm = self.layer_norm(inputs)
        context, attn = self.self_attn(input_norm, plan, log, metrics, src_mask, plan_mask)

        out = self.dropout(context) + inputs
        return self.feed_forward(out), attn

class CrossTransformer(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, attn_modules):
        super(CrossTransformer, self).__init__()

        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout, attn_modules[i])
             for i in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)


    def forward(self, sql, plan, log, metrics, sql_mask, plan_mask):
        '''
        :param src: [src_len, batch_size]
        :param bond: [batch_size, src_len, src_len, 7]
        :return:
        '''

        # words = src.transpose(0, 1)
        # w_batch, w_len = words.size()
        # padding_idx = self.embeddings.word_padding_idx
        # mask = words.data.eq(padding_idx).unsqueeze(1) \
        #     .expand(w_batch, w_len, w_len)

        out = sql
        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out, attn = self.transformer[i](out, plan, log, metrics, sql_mask, plan_mask)

        out = self.layer_norm(out)
        # out = out.transpose(0, 1).contiguous()
        # edge_out = self.layer_norm(edge_feature) if edge_feature is not None else None
        return out#, edge_out




class GlobalLocalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attn, attn_l_m=None):
        super(GlobalLocalTransformerEncoderLayer, self).__init__()

        self.self_attn = attn
        self.self_attn_l_m = attn_l_m
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward2 = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, plan, log, metrics, src_mask, plan_mask):
        input_norm = self.layer_norm(inputs)
        context, attn = self.self_attn(input_norm, plan, log, metrics, src_mask, plan_mask)

        out = self.dropout(context) + inputs
        # sql_plan_emb = self.feed_forward(out)

        # input_norm = self.layer_norm2(sql_plan_emb)
        # context, attn = self.self_attn_l_m(input_norm, log, metrics, src_mask, plan_mask)
        # out = self.dropout2(context) + sql_plan_emb

        return self.feed_forward2(out), attn

class GlobalLocalCrossTransformer(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, attn_modules, attn_l_m):
        super(GlobalLocalCrossTransformer, self).__init__()

        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [GlobalLocalTransformerEncoderLayer(d_model, heads, d_ff, dropout, attn_modules[i], attn_l_m[i])
             for i in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)


    def forward(self, sql, plan, log, metrics, sql_mask, plan_mask):
        '''
        :param src: [src_len, batch_size]
        :param bond: [batch_size, src_len, src_len, 7]
        :return:
        '''

        # words = src.transpose(0, 1)
        # w_batch, w_len = words.size()
        # padding_idx = self.embeddings.word_padding_idx
        # mask = words.data.eq(padding_idx).unsqueeze(1) \
        #     .expand(w_batch, w_len, w_len)

        out = sql
        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out, attn = self.transformer[i](out, plan, log, metrics, sql_mask, plan_mask)

        out = self.layer_norm(out)
        # out = out.transpose(0, 1).contiguous()
        # edge_out = self.layer_norm(edge_feature) if edge_feature is not None else None
        return out#, edge_out