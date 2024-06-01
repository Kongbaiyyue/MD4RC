import torch
from torch import nn
from model.modules.QueryFormer.QueryFormer import QueryFormer
from torch.nn import functional as F
from models import LogModel, TimeSeriesModel
import numpy as np


class TopModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        # self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 12)
        # self.pred_label_cross = nn.Linear(emb_dim, 12)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 12)
        # self.pred_opt_cross = nn.Linear(emb_dim, 12)
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_label_cross = nn.Linear(emb_dim, 6)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_opt_cross = nn.Linear(emb_dim, 6)
        self.pred_label_concat = nn.Linear(emb_dim * 2, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 2, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        

        self.cross_mean = cross_mean
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        # with torch.no_grad():
        #     sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        # log_emb = self.log_model(log)
        
        time_emb = time.unsqueeze(1)
        time_emb = self.time_model(time_emb)
        
        # sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        # if self.cross_model is not None:
        #     sql_emb = self.sql_last_emb(sql_emb)
        #     emb = self.cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
        #     # if self.cross_mean:
        #     #     emb = emb.mean(dim=1)
        #     # else:
        #     #     emb = emb[:, 0, :]
        #     emb = emb.mean(dim=1)
        # else:
        # 维度变换为相同大小
        # sql_emb = sql_emb[:, 0, :]
        # sql_emb = self.sql_last_emb(sql_emb)

        plan_emb = plan_emb[:, 0, :]
        time_emb = torch.flatten(time_emb, start_dim=1)
        time_emb = self.time_tran_emb(time_emb)

        # emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        emb = torch.cat([plan_emb, time_emb], dim=1)
        # if self.cross_model is not None:
        #     pred_label = self.pred_label_cross(emb)
        #     pred_opt = self.pred_opt_cross(emb)
        #     # pred_opt = F.softmax(pred_opt, dim=-1)
        #     # pred_opt = F.tanh(pred_opt)
        # else:
        pred_label = self.pred_label_concat(emb)
        pred_opt = self.pred_opt_concat(emb)
            # pred_opt = F.tanh(pred_opt)
            
            # pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
    

# ----------------------------------------- 单模态 -----------------------------------

class SQLOptModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.activation = nn.ReLU()        

        self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        sql_emb = sql_emb[:, 0, :]
        sql_emb = self.sql_last_emb(sql_emb)
        sql_emb = self.activation(sql_emb)
        
        pred_label = self.pred_label_cross(sql_emb)
        pred_opt = self.pred_opt_cross(sql_emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
    

class PlanOptModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)

        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        self.activation = nn.ReLU()        
        
        self.init_params()
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        plan_emb = self.plan_model(plan)

        plan_emb = plan_emb[:, 0, :]
        plan_emb = self.activation(plan_emb)

        pred_label = self.pred_label_cross(plan_emb)
        pred_opt = self.pred_opt_cross(plan_emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt


class LogOptModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        self.activation = nn.ReLU()        
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        log_emb = self.log_model(log)
        log_emb = self.activation(log_emb)
        
        pred_label = self.pred_label_cross(log_emb)
        pred_opt = self.pred_opt_cross(log_emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt


class TimeOptModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.time_model = time_model
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        # self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim, emb_dim)
        self.activation = nn.ReLU()        
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        time_emb = time.unsqueeze(1)
        time_emb = self.time_model(time_emb)
        
        time_emb = torch.flatten(time_emb, start_dim=1)
        time_emb = self.time_tran_emb(time_emb)
        time_emb = self.activation(time_emb)
        
        pred_label = self.pred_label_cross(time_emb)
        pred_opt = self.pred_opt_cross(time_emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
# ----------------------------------------- 单模态 -----------------------------------

# Concat
class ConcatOptModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        # self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim, emb_dim)
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 12)
        # self.pred_label_cross = nn.Linear(emb_dim, 12)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 12)
        # self.pred_opt_cross = nn.Linear(emb_dim, 12)
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_label_cross = nn.Linear(emb_dim, 6)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_opt_cross = nn.Linear(emb_dim, 6)
        self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        

        self.cross_mean = cross_mean
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        sql_emb = sql_emb.last_hidden_state
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        
        time_emb = time.unsqueeze(1)
        time_emb = self.time_model(time_emb)
        
        sql_emb = sql_emb[:, 0, :]
        sql_emb = self.sql_last_emb(sql_emb)

        plan_emb = plan_emb[:, 0, :]
        time_emb = torch.flatten(time_emb, start_dim=1)
        time_emb = self.time_tran_emb(time_emb)

        emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        pred_label = self.pred_label_concat(emb)
        pred_opt = self.pred_opt_concat(emb)
        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt



class TopRealEstModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, plan_args=None, cross_model=None, time_model=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 12)
        # self.pred_label_cross = nn.Linear(emb_dim, 12)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 12)
        # self.pred_opt_cross = nn.Linear(emb_dim, 12)
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_label_cross = nn.Linear(emb_dim, 6)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_opt_cross = nn.Linear(emb_dim, 6)
        self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)

        self.RelEstFuse = nn.Linear(emb_dim * 3, emb_dim)
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        if self.cross_model is not None:
            sql_emb = self.sql_last_emb(sql_emb)
            sql_plan_emb = self.cross_model(sql_emb, plan_emb, None, None, None, None)

            sql_plan_emb = sql_plan_emb.mean(dim=1)
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)
            emb = torch.cat([sql_plan_emb, log_emb, time_emb], dim=1)

            # emb = emb.mean(dim=1)
            emb = self.RelEstFuse(emb)
        else:
            # 维度变换为相同大小
            sql_emb = sql_emb[:, 0, :]
            sql_emb = self.sql_last_emb(sql_emb)

            plan_emb = plan_emb[:, 0, :]
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)

            emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
    
    
class TopConstractModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, plan_args=None, cross_model=None, time_model=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_label_cross = nn.Linear(emb_dim, 6)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_opt_cross = nn.Linear(emb_dim, 6)
        self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)

        self.RelEstFuse = nn.Linear(emb_dim * 3, emb_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(plan_args.dropout)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        if self.cross_model is not None:
            sql_emb = self.sql_last_emb(sql_emb)
            sql_global = sql_emb[:, 0].unsqueeze(1)
            sql_plan_emb = self.cross_model(sql_emb, plan_emb, None, None, None, None)

            sql_plan_emb = sql_plan_emb.mean(dim=1)
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)
            emb = torch.cat([sql_plan_emb, log_emb, time_emb], dim=1)

            # emb = emb.mean(dim=1)
            emb = self.RelEstFuse(emb)
        else:
            # 维度变换为相同大小
            sql_emb = sql_emb[:, 0, :]
            sql_emb = self.sql_last_emb(sql_emb)

            plan_emb = plan_emb[:, 0, :]
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)

            emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
            # pred_opt = F.softmax(pred_opt, dim=-1)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)
            # pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)
        scores = torch.matmul(sql_global, plan_emb.transpose(1, 2))
        attn = torch.softmax(scores, dim=-1)
        drop_attn = self.dropout(attn)
        sql_plan_global_emb = torch.matmul(drop_attn, plan_emb).squeeze(1)

        return pred_label, pred_opt, sql_plan_global_emb, self.logit_scale.exp()
    


class TopMultiScaleModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, plan_args=None, cross_model=None, time_model=None, cross_mod="none") -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        self.pred_label_cross = nn.Linear(emb_dim, 6)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        self.pred_opt_cross = nn.Linear(emb_dim, 6)
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_label_cross = nn.Linear(emb_dim, 5)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_opt_cross = nn.Linear(emb_dim, 5)

        self.cross_mod = cross_mod
        # if self.cross_mod == "l_g":
        #     self.RelEstFuse = nn.Linear(emb_dim * 3, emb_dim)
        # else:
        #     self.RelEstFuse = nn.Linear(emb_dim * 5, emb_dim)
        # self.RelEstFuse = nn.Linear(emb_dim * 5, emb_dim)
        self.RelEstFuse = nn.Linear(emb_dim * 3, emb_dim)
        # self.RelEstFuse = nn.Linear(emb_dim * 4, emb_dim)
        
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(plan_args.dropout)
        
        self.proj_drop = nn.Dropout(plan_args.dropout)
        
        self.init_params()
        self.sql_model = sql_model
        
        
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        if self.cross_model is not None:
            sql_emb = self.sql_last_emb(sql_emb)
            # sql_global = sql_emb[:, 0].unsqueeze(1)
            scores = torch.matmul(plan_emb[:, 0].unsqueeze(1), sql_emb.transpose(1, 2))
            # attn = torch.sigmoid(scores)
            attn = torch.tanh(scores)
            drop_attn = self.dropout(attn)
            sql_global = torch.matmul(drop_attn, sql_emb)
            
            # sql_emb = sql_emb + sql_emb * sql_global.softmax(dim=-1)
            sql_emb = sql_emb + sql_emb * attn.transpose(1, 2)
            
            # sql_emb[:, 0] = sql_global.squeeze(1)
            
            # if self.cross_mod == "l_g":
            #     sql_plan_emb = self.cross_model(torch.cat([sql_global, sql_emb], 1), plan_emb, None, None, None, None)
                
            # else:
            sql_plan_emb = self.cross_model(plan_emb, sql_emb, None, None, None, None)
            # sql_plan_emb = self.cross_model(sql_emb, plan_emb, None, None, None, None)

            sql_plan_emb = sql_plan_emb.mean(dim=1)
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)
            # emb = torch.cat([sql_global.squeeze(1), plan_emb[:, 0], sql_plan_emb, log_emb, time_emb], dim=1)
            emb = torch.cat([sql_plan_emb, log_emb, time_emb], dim=1)
            

            # emb = torch.cat([sql_global.squeeze(1), plan_emb[:, 0], log_emb, time_emb], dim=1)

            # emb = emb.mean(dim=1)
            emb = self.RelEstFuse(emb)
        else:
            # 维度变换为相同大小
            sql_emb = sql_emb[:, 0, :]
            sql_emb = self.sql_last_emb(sql_emb)

            plan_emb = plan_emb[:, 0, :]
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)

            emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)
        scores = torch.matmul(sql_global, plan_emb.transpose(1, 2))
        attn = torch.softmax(scores, dim=-1)
        drop_attn = self.dropout(attn)
        sql_plan_global_emb = torch.matmul(drop_attn, plan_emb).squeeze(1)

        return pred_label, pred_opt, sql_plan_global_emb
    

class TopMultiDiffModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, plan_args=None, cross_model=None, time_model=None, cross_mod="none") -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        self.pred_label_cross = nn.Linear(emb_dim, 6)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        self.pred_opt_cross = nn.Linear(emb_dim, 6)
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_label_cross = nn.Linear(emb_dim, 5)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_opt_cross = nn.Linear(emb_dim, 5)

        self.cross_mod = cross_mod
        # if self.cross_mod == "l_g":
        #     self.RelEstFuse = nn.Linear(emb_dim * 3, emb_dim)
        # else:
        #     self.RelEstFuse = nn.Linear(emb_dim * 5, emb_dim)
        # self.RelEstFuse = nn.Linear(emb_dim * 5, emb_dim)
        self.RelEstFuse = nn.Linear(emb_dim * 3, emb_dim)
        # self.RelEstFuse = nn.Linear(emb_dim * 4, emb_dim)
        
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(plan_args.dropout)
        
        self.proj_drop = nn.Dropout(plan_args.dropout)
        
        self.init_params()
        self.sql_model = sql_model
        
        
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        if self.cross_model is not None:
            sql_emb = self.sql_last_emb(sql_emb)
            # sql_global = sql_emb[:, 0].unsqueeze(1)
            scores = torch.matmul(plan_emb[:, 0].unsqueeze(1), sql_emb.transpose(1, 2))
            # attn = torch.sigmoid(scores)
            attn_sql = torch.tanh(scores)
            drop_attn_sql = self.dropout(attn_sql)
            sql_global = torch.matmul(drop_attn_sql, sql_emb)
            
            scores = torch.matmul(sql_emb[:, 0].unsqueeze(1), plan_emb.transpose(1, 2))
            attn_plan = torch.tanh(scores)
            drop_attn_plan = self.dropout(attn_plan)
            plan_global = torch.matmul(drop_attn_plan, plan_emb)
            
            # sql_emb = sql_emb + sql_emb * sql_global.softmax(dim=-1)
            # sql_emb = sql_emb + sql_emb * attn.transpose(1, 2)
            diff_sql_emb = sql_emb - sql_emb * drop_attn_sql.transpose(1, 2)
            diff_plan_emb = plan_emb - plan_emb * drop_attn_plan.transpose(1, 2)
            common_emb = torch.cat([sql_emb * drop_attn_sql.transpose(1, 2), plan_emb * drop_attn_plan.transpose(1, 2)], dim=1)
            
            # sql_emb[:, 0] = sql_global.squeeze(1)
            
            # if self.cross_mod == "l_g":
            #     sql_plan_emb = self.cross_model(torch.cat([sql_global, sql_emb], 1), plan_emb, None, None, None, None)
                
            # else:
            sql_plan_emb = self.cross_model(common_emb, diff_sql_emb, diff_plan_emb, None, None, None)
            # sql_plan_emb = self.cross_model(sql_emb, plan_emb, None, None, None, None)

            sql_plan_emb = sql_plan_emb.mean(dim=1)
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)
            # emb = torch.cat([sql_global.squeeze(1), plan_emb[:, 0], sql_plan_emb, log_emb, time_emb], dim=1)
            emb = torch.cat([sql_plan_emb, log_emb, time_emb], dim=1)
            

            # emb = torch.cat([sql_global.squeeze(1), plan_emb[:, 0], log_emb, time_emb], dim=1)

            # emb = emb.mean(dim=1)
            emb = self.RelEstFuse(emb)
        else:
            # 维度变换为相同大小
            sql_emb = sql_emb[:, 0, :]
            sql_emb = self.sql_last_emb(sql_emb)

            plan_emb = plan_emb[:, 0, :]
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)

            emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)
        scores = torch.matmul(sql_global, plan_emb.transpose(1, 2))
        attn = torch.softmax(scores, dim=-1)
        drop_attn = self.dropout(attn)
        sql_plan_global_emb = torch.matmul(drop_attn, plan_emb).squeeze(1)

        return pred_label, pred_opt, sql_plan_global_emb
    
    
class TopConstractOriModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, plan_args=None, cross_model=None, time_model=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        self.pred_label_cross = nn.Linear(emb_dim, 6)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        self.pred_opt_cross = nn.Linear(emb_dim, 6)
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_label_cross = nn.Linear(emb_dim, 5)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_opt_cross = nn.Linear(emb_dim, 5)

        self.RelEstFuse = nn.Linear(emb_dim * 4, emb_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(plan_args.dropout)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        if self.cross_model is not None:
            sql_emb = self.sql_last_emb(sql_emb)
            sql_global = sql_emb[:, 0].unsqueeze(1)
            emb = self.cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)

            # sql_plan_emb = sql_plan_emb.mean(dim=1)
            # time_emb = torch.flatten(time_emb, start_dim=1)
            # time_emb = self.time_tran_emb(time_emb)
            # emb = torch.cat([sql_emb[:, 0], plan[:, 0], log_emb, time_emb], dim=1)

            emb = emb.mean(dim=1)
            # emb = self.RelEstFuse(emb)
        else:
            # 维度变换为相同大小
            sql_emb = sql_emb[:, 0, :]
            sql_emb = self.sql_last_emb(sql_emb)
            sql_global = sql_emb.unsqueeze(1) 

            plan_emb_cat = plan_emb[:, 0, :]
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)

            emb = torch.cat([sql_emb, plan_emb_cat, log_emb, time_emb], dim=1)
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)
        scores = torch.matmul(sql_global, plan_emb.transpose(1, 2))
        attn = torch.softmax(scores, dim=-1)
        drop_attn = self.dropout(attn)
        sql_plan_global_emb = torch.matmul(drop_attn, plan_emb).squeeze(1)

        return pred_label, pred_opt, sql_plan_global_emb, self.logit_scale.exp()
    
    
class TopHeriOriModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, plan_args=None, cross_model=None, time_model=None, cross_model_real=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        self.cross_model_real = cross_model_real
        self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        self.pred_label_cross = nn.Linear(emb_dim, 6)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        self.pred_opt_cross = nn.Linear(emb_dim, 6)
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_label_cross = nn.Linear(emb_dim, 5)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_opt_cross = nn.Linear(emb_dim, 5)

        self.RelEstFuse = nn.Linear(emb_dim * 4, emb_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(plan_args.dropout)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_concat_1 = nn.Linear(emb_dim, emb_dim * 2)
        self.cross_concat_2 = nn.Linear(emb_dim * 2, emb_dim)
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        if self.cross_model is not None:
            sql_emb = self.sql_last_emb(sql_emb)
            sql_global = sql_emb[:, 0].unsqueeze(1)
            sql_plan_emb = self.cross_model(sql_emb, plan_emb, None, None, None, None)
            
            emb = self.cross_model_real(sql_plan_emb, log_emb, None, time_emb, None, None)
            emb_cross = emb.mean(dim=1)

            # sql_plan_emb = sql_plan_emb.mean(dim=1)
            # time_emb = torch.flatten(time_emb, start_dim=1)
            # time_emb = self.time_tran_emb(time_emb)
            # emb = torch.cat([sql_emb[:, 0], plan_emb[:, 0], log_emb, time_emb], dim=1)
            # emb_concat = self.RelEstFuse(emb)
            
            # emb = emb_cross + emb_concat
            # emb = self.cross_concat_1(emb_cross)
            # emb = torch.relu(emb)
            # emb = self.cross_concat_2(emb)
            # emb = torch.relu(emb)
            emb = emb_cross
            
        else:
            # 维度变换为相同大小
            sql_emb = sql_emb[:, 0, :]
            sql_emb = self.sql_last_emb(sql_emb)
            sql_global = sql_emb.unsqueeze(1) 

            plan_emb_cat = plan_emb[:, 0, :]
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)

            emb = torch.cat([sql_emb, plan_emb_cat, log_emb, time_emb], dim=1)
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)
        # scores = torch.matmul(sql_global, plan_emb.transpose(1, 2))
        # attn = torch.softmax(scores, dim=-1)
        # drop_attn = self.dropout(attn)
        # sql_plan_global_emb = torch.matmul(drop_attn, plan_emb).squeeze(1)

        # return pred_label, pred_opt, sql_plan_global_emb, self.logit_scale.exp()
        return pred_label, pred_opt



class TopDiffComModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, plan_args=None, cross_model=None, time_model=None, cross_model_real=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        self.cross_model_real = cross_model_real
        self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        self.pred_label_cross = nn.Linear(emb_dim, 6)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        self.pred_opt_cross = nn.Linear(emb_dim, 6)
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_label_cross = nn.Linear(emb_dim, 5)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_opt_cross = nn.Linear(emb_dim, 5)

        self.RelEstFuse = nn.Linear(emb_dim * 4, emb_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(plan_args.dropout)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_concat_1 = nn.Linear(emb_dim, emb_dim * 2)
        self.cross_concat_2 = nn.Linear(emb_dim * 2, emb_dim)
        
        self.init_params()
        self.sql_model = sql_model
        
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        if self.cross_model is not None:
            sql_emb = self.sql_last_emb(sql_emb)
            sql_global = sql_emb[:, 0].unsqueeze(1)
            sql_plan_emb = self.cross_model(sql_emb, plan_emb, None, None, None, None)
            
            emb = self.cross_model_real(sql_plan_emb, log_emb, None, time_emb, None, None)
            emb_cross = emb.mean(dim=1)

            # sql_plan_emb = sql_plan_emb.mean(dim=1)
            # time_emb = torch.flatten(time_emb, start_dim=1)
            # time_emb = self.time_tran_emb(time_emb)
            # emb = torch.cat([sql_emb[:, 0], plan_emb[:, 0], log_emb, time_emb], dim=1)
            # emb_concat = self.RelEstFuse(emb)
            
            # emb = emb_cross + emb_concat
            # emb = self.cross_concat_1(emb_cross)
            # emb = torch.relu(emb)
            # emb = self.cross_concat_2(emb)
            # emb = torch.relu(emb)
            emb = emb_cross
            
        else:
            # 维度变换为相同大小
            sql_emb = sql_emb[:, 0, :]
            sql_emb = self.sql_last_emb(sql_emb)
            sql_global = sql_emb.unsqueeze(1) 

            plan_emb_cat = plan_emb[:, 0, :]
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)

            emb = torch.cat([sql_emb, plan_emb_cat, log_emb, time_emb], dim=1)
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)
            pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)
        # scores = torch.matmul(sql_global, plan_emb.transpose(1, 2))
        # attn = torch.softmax(scores, dim=-1)
        # drop_attn = self.dropout(attn)
        # sql_plan_global_emb = torch.matmul(drop_attn, plan_emb).squeeze(1)

        # return pred_label, pred_opt, sql_plan_global_emb, self.logit_scale.exp()
        return pred_label, pred_opt
    
    
class OnlyPlanModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 12)
        # self.pred_label_cross = nn.Linear(emb_dim, 12)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 12)
        # self.pred_opt_cross = nn.Linear(emb_dim, 12)
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_label_cross = nn.Linear(emb_dim, 6)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_opt_cross = nn.Linear(emb_dim, 6)
        self.pred_label_concat = nn.Linear(emb_dim * 1, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 1, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        

        self.cross_mean = cross_mean
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad:
            sql_emb = self.sql_model(**sql)
        sql_emb = sql_emb.last_hidden_state
        sql_emb = self.sql_last_emb(sql_emb)
        sql_emb = sql_emb.mean(dim=1)
        
        # plan_emb = self.plan_model(plan)
        # time_emb = self.time_model(time)
                
        # plan_emb = plan_emb.mean(dim=1)
        # time_emb = torch.flatten(time_emb, start_dim=1)
        # time_emb = self.time_tran_emb(time_emb)
        
        # emb = torch.cat([plan_emb, time_emb], dim=1)
        # emb = plan_emb
        emb = sql_emb
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
            # pred_opt = F.softmax(pred_opt, dim=-1)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)
            # pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
    
    
class CommonSpecialModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        # self.sql_last_emb = nn.Linear(768, emb_dim)
        # self.plan_last_emb = nn.Linear(emb_dim, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_label_cross = nn.Linear(emb_dim, 6)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_opt_cross = nn.Linear(emb_dim, 6)
        self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=emb_dim))
        self.project_t.add_module('project_t_activation', nn.LeakyReLU())
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(emb_dim))
        
        self.project_p = nn.Sequential()
        self.project_p.add_module('project_p', nn.Linear(in_features=emb_dim, out_features=emb_dim))
        self.project_p.add_module('project_p_activation', nn.LeakyReLU())
        self.project_p.add_module('project_p_layer_norm', nn.LayerNorm(emb_dim))
        
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=emb_dim, out_features=emb_dim))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())
        
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=emb_dim, out_features=emb_dim))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_p = nn.Sequential()
        self.private_p.add_module('private_v_1', nn.Linear(in_features=emb_dim, out_features=emb_dim))
        self.private_p.add_module('private_v_activation_1', nn.Sigmoid())
        

        self.cross_mean = cross_mean
        self.init_params()
        
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        if self.cross_model is not None:
            sql_emb = self.project_t(sql_emb)
            # sql_global = sql_emb[:, 0].unsqueeze(1)
            plan_emb = self.project_p(plan_emb)
            
            share_sql_emb = self.shared(sql_emb)
            share_plan_emb = self.shared(plan_emb)
            
            private_sql_emb = self.private_t(sql_emb)
            private_plan_emb = self.private_p(plan_emb)
            
            sql_emb_new = share_sql_emb + private_sql_emb
            plan_emb_new = share_plan_emb + private_plan_emb
            emb = self.cross_model(sql_emb_new, plan_emb_new, log_emb, time_emb, None, None)

            emb = emb.mean(dim=1)
            # sql_plan_emb = sql_plan_emb.mean(dim=1)
            # time_emb = torch.flatten(time_emb, start_dim=1)
            # time_emb = self.time_tran_emb(time_emb)
            # emb = torch.cat([sql_emb_new.mean(dim=1), plan_emb_new.mean(dim=1), log_emb, time_emb], dim=1)
            
        else:
            # 维度变换为相同大小
            sql_emb = sql_emb[:, 0, :]
            sql_emb = self.sql_last_emb(sql_emb)

            plan_emb = plan_emb[:, 0, :]
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)

            emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
            # pred_opt = F.softmax(pred_opt, dim=-1)
            # pred_opt = F.tanh(pred_opt)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)
            # pred_opt = F.tanh(pred_opt)
            # pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt, share_sql_emb, share_plan_emb, private_sql_emb, private_plan_emb
    
class PlanMainModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_label_cross = nn.Linear(emb_dim, 6)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_opt_cross = nn.Linear(emb_dim, 6)
        self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        

        self.cross_mean = cross_mean
        self.init_params()
        self.sql_model = sql_model
        
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        if self.cross_model is not None:
            sql_emb = self.sql_last_emb(sql_emb)
            emb = self.cross_model(plan_emb, sql_emb, log_emb, time_emb, None, None)
            # if self.cross_mean:
            #     emb = emb.mean(dim=1)
            # else:
            #     emb = emb[:, 0, :]
            emb = emb.mean(dim=1)
        else:
            # 维度变换为相同大小
            sql_emb = sql_emb[:, 0, :]
            sql_emb = self.sql_last_emb(sql_emb)

            plan_emb = plan_emb[:, 0, :]
            time_emb = torch.flatten(time_emb, start_dim=1)
            time_emb = self.time_tran_emb(time_emb)

            emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
            # pred_opt = F.softmax(pred_opt, dim=-1)
            # pred_opt = F.tanh(pred_opt)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)
            # pred_opt = F.tanh(pred_opt)
            
            # pred_opt = F.softmax(pred_opt, dim=-1)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt

    

class gateModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_label_cross = nn.Linear(emb_dim, 6)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_opt_cross = nn.Linear(emb_dim, 6)
        self.pred_label_concat = nn.Linear(emb_dim * 4, 1)
        # self.pred_label_cross = nn.Linear(emb_dim, 1)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 1)
        # self.pred_opt_cross = nn.Linear(emb_dim, 1)
        self.pred_label_cross_list = nn.ModuleList()
        self.pred_opt_cross_list = nn.ModuleList()
        for i in range(5):
            self.pred_label_cross_list.append(nn.Linear(emb_dim, 1))
            self.pred_opt_cross_list.append(nn.Linear(emb_dim, 1))
        
        self.cross_mean = cross_mean
        
        self.gate_sql = nn.ModuleList()
        self.gate_sql_activate = nn.ModuleList()
        self.gate_plan = nn.ModuleList()
        self.gate_plan_activate = nn.ModuleList()
        self.gate_log = nn.ModuleList()
        self.gate_log_activate = nn.ModuleList()
        self.gate_metrics = nn.ModuleList()
        self.gate_metrics_activate = nn.ModuleList()
        self.gate_metrics_norm = nn.ModuleList()
        self.gate_out_dim = 1
        for i in range(5):
            gate_sql_0 = nn.Sequential()
            gate_sql_0.add_module('gate_sql', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_sql_0.add_module('gate_sql_activation', nn.LeakyReLU())
            # gate_sql_0.add_module('gate_sql_layer_norm', nn.LayerNorm(emb_dim))
            
            self.gate_sql.append(gate_sql_0)
            self.gate_sql_activate.append(nn.Sigmoid())
        
            gate_plan_0 = nn.Sequential()
            gate_plan_0.add_module('gate_plan', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_plan_0.add_module('gate_plan_activation', nn.LeakyReLU())
            # gate_plan_0.add_module('gate_plan_layer_norm', nn.LayerNorm(emb_dim))
            self.gate_plan.append(gate_plan_0)
            self.gate_plan_activate.append(nn.Sigmoid())
        
            gate_log_0 = nn.Sequential()
            gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=emb_dim))
            # gate_log_0.add_module('gate_log_activation', nn.LeakyReLU())
            # gate_log_0.add_module('gate_log_layer_norm', nn.BatchNorm1d(emb_dim))
            self.gate_log.append(gate_log_0)
            self.gate_log_activate.append(nn.Sigmoid())
            
            gate_metrics_0 = nn.Sequential()
            gate_metrics_0.add_module('gate_metrics', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_metrics_0.add_module('gate_metrics_activation', nn.LeakyReLU())
            # gate_metrics_0.add_module('gate_metrics_layer_norm', nn.BatchNorm1d(7))
            self.gate_metrics.append(gate_metrics_0)
            self.gate_metrics_activate.append(nn.Sigmoid())
            # self.gate_metrics_norm.append(nn.BatchNorm1d(emb_dim))
        
        self.init_params()
        
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        sql_emb = self.sql_last_emb(sql_emb)
        
        for i in range(5):
        # sql_emb = F.relu(sql_emb)
            sql_emb_tmp = self.gate_sql[i](sql_emb)
            sql_emb_tmp = self.gate_sql_activate[i](sql_emb_tmp) * sql_emb
            
            plan_emb_tmp = self.gate_plan[i](plan_emb)
            plan_emb_tmp = self.gate_plan_activate[i](plan_emb_tmp) * plan_emb
            
            log_emb_tmp = self.gate_log[i](log_emb)
            log_emb_tmp = self.gate_log_activate[i](log_emb_tmp) * log_emb
            
            # batch_size, _, emb_size = sql_emb.shape
            time_emb_tmp = self.gate_metrics[i](time_emb)
            # time_emb_tmp = time_emb_tmp.view(-1, emb_size)
            # time_emb_tmp = self.gate_metrics_norm[i](time_emb_tmp)
            # time_emb_tmp = time_emb_tmp.view(batch_size, -1, emb_size)
            time_emb_tmp = self.gate_metrics_activate[i](time_emb_tmp) * time_emb

            if self.cross_model is not None:
                # sql_emb = torch.relu(sql_emb)
                
                emb = self.cross_model(sql_emb_tmp, plan_emb_tmp, log_emb_tmp, time_emb_tmp, None, None)
                # if self.cross_mean:
                #     emb = emb.mean(dim=1)
                # else:
                #     emb = emb[:, 0, :]
                emb = emb.mean(dim=1)
            else:
                # 维度变换为相同大小
                # sql_emb = sql_emb[:, 0, :]

                # plan_emb = plan_emb[:, 0, :]
                sql_emb_tmp = torch.mean(sql_emb_tmp, dim=1)
                plan_emb_tmp = torch.mean(plan_emb_tmp, dim=1)
                time_emb_tmp = torch.flatten(time_emb_tmp, start_dim=1)
                time_emb_tmp = self.time_tran_emb(time_emb_tmp)

                emb = torch.cat([sql_emb_tmp, plan_emb_tmp, log_emb_tmp, time_emb_tmp], dim=1)
            if self.cross_model is not None:
                pred_label = self.pred_label_cross_list[i](emb)
                pred_opt = self.pred_opt_cross_list[i](emb)
                # pred_label = self.pred_label_cross(emb)
                # pred_opt = self.pred_opt_cross(emb)
                # pred_opt = F.softmax(pred_opt, dim=-1)
                # pred_opt = F.tanh(pred_opt)
            else:
                pred_label = self.pred_label_concat(emb)
                pred_opt = self.pred_opt_concat(emb)
                # pred_opt = F.tanh(pred_opt)
                
                # pred_opt = F.softmax(pred_opt, dim=-1)

            pred_label = torch.sigmoid(pred_label)
            if i == 0:
                pred_opt_output = pred_opt
                pred_label_output = pred_label
            else:
                pred_opt_output = torch.cat([pred_opt_output, pred_opt], dim=-1)
                pred_label_output = torch.cat([pred_label_output, pred_label], dim=-1)

        return pred_label_output, pred_opt_output
        # return pred_label, pred_opt
        
        
class gateHierarchicalModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, plan_args=None, cross_model=None, time_model=None, cross_mean=True, cross_model_real=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = plan_args.use_sample, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)

        self.sql_ln = nn.LayerNorm(emb_dim)
        self.log_bn = nn.BatchNorm1d(emb_dim)
        self.metrics_bn = nn.BatchNorm1d(7)
        # 融合三个模态

        self.cross_model = cross_model
        self.cross_model_real = cross_model_real
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_label_cross = nn.Linear(emb_dim, 6)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 6)
        # self.pred_opt_cross = nn.Linear(emb_dim, 6)
        self.pred_label_concat = nn.Linear(emb_dim * 4, 1)
        # self.pred_label_cross = nn.Linear(emb_dim, 1)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 1)
        # self.pred_opt_cross = nn.Linear(emb_dim, 1)
        self.pred_label_cross_list = nn.ModuleList()
        self.pred_opt_cross_list = nn.ModuleList()
        for i in range(5):
            self.pred_label_cross_list.append(nn.Linear(emb_dim, 1))
            self.pred_opt_cross_list.append(nn.Linear(emb_dim, 1))
        
        self.cross_mean = cross_mean
        
        self.gate_sql = nn.ModuleList()
        self.gate_sql_activate = nn.ModuleList()
        self.gate_plan = nn.ModuleList()
        self.gate_plan_activate = nn.ModuleList()
        self.gate_log = nn.ModuleList()
        self.gate_log_activate = nn.ModuleList()
        self.gate_metrics = nn.ModuleList()
        self.gate_metrics_activate = nn.ModuleList()
        self.gate_metrics_norm = nn.ModuleList()
        self.gate_out_dim = 1
        
        for i in range(5):
            gate_sql_0 = nn.Sequential()
            gate_sql_0.add_module('gate_sql', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_sql_0.add_module('gate_sql_activation', nn.LeakyReLU())
            # gate_sql_0.add_module('gate_sql_layer_norm', nn.LayerNorm(emb_dim))
            
            self.gate_sql.append(gate_sql_0)
            self.gate_sql_activate.append(nn.Sigmoid())
        
            gate_plan_0 = nn.Sequential()
            gate_plan_0.add_module('gate_plan', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_plan_0.add_module('gate_plan_activation', nn.LeakyReLU())
            # gate_plan_0.add_module('gate_plan_layer_norm', nn.LayerNorm(emb_dim))
            self.gate_plan.append(gate_plan_0)
            self.gate_plan_activate.append(nn.Sigmoid())
        
            gate_log_0 = nn.Sequential()
            gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=emb_dim))
            # gate_log_0.add_module('gate_log_activation', nn.LeakyReLU())
            # gate_log_0.add_module('gate_log_layer_norm', nn.BatchNorm1d(emb_dim))
            self.gate_log.append(gate_log_0)
            self.gate_log_activate.append(nn.Sigmoid())
            
            gate_metrics_0 = nn.Sequential()
            gate_metrics_0.add_module('gate_metrics', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_metrics_0.add_module('gate_metrics_activation', nn.LeakyReLU())
            # gate_metrics_0.add_module('gate_metrics_layer_norm', nn.BatchNorm1d(7))
            self.gate_metrics.append(gate_metrics_0)
            self.gate_metrics_activate.append(nn.Sigmoid())
            # self.gate_metrics_norm.append(nn.BatchNorm1d(emb_dim))
        
        self.init_params()
        
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        sql_emb = self.sql_last_emb(sql_emb)
        sql_emb = self.sql_ln(sql_emb)
        log_emb = self.log_bn(log_emb)
        time_emb = self.metrics_bn(time_emb)
        
        for i in range(5):
        # sql_emb = F.relu(sql_emb)
            sql_emb_tmp = self.gate_sql[i](sql_emb)
            sql_emb_tmp = self.gate_sql_activate[i](sql_emb_tmp) * sql_emb
            
            plan_emb_tmp = self.gate_plan[i](plan_emb)
            plan_emb_tmp = self.gate_plan_activate[i](plan_emb_tmp) * plan_emb
            
            log_emb_tmp = self.gate_log[i](log_emb)
            log_emb_tmp = self.gate_log_activate[i](log_emb_tmp) * log_emb
            
            # batch_size, _, emb_size = sql_emb.shape
            time_emb_tmp = self.gate_metrics[i](time_emb)
            # time_emb_tmp = time_emb_tmp.view(batch_size, -1)
            # time_emb_tmp = self.gate_metrics_norm[i](time_emb_tmp)
            # time_emb_tmp = time_emb_tmp.view(batch_size, -1, emb_size)
            time_emb_tmp = self.gate_metrics_activate[i](time_emb_tmp) * time_emb

            if self.cross_model is not None:
                # sql_emb = torch.relu(sql_emb)
                
                sql_plan_emb = self.cross_model(sql_emb_tmp, plan_emb_tmp, None, None, None, None)
                # emb = self.cross_model_real(sql_plan_emb, log_emb_tmp, None, time_emb_tmp, None, None)
                emb = self.cross_model_real(sql_plan_emb, time_emb_tmp, log_emb_tmp, None, None, None)
                # emb_cross = emb.mean(dim=1)
                # if self.cross_mean:
                #     emb = emb.mean(dim=1)
                # else:
                #     emb = emb[:, 0, :]
                emb = emb.mean(dim=1)
            else:
                # 维度变换为相同大小
                # sql_emb = sql_emb[:, 0, :]

                # plan_emb = plan_emb[:, 0, :]
                sql_emb_tmp = torch.mean(sql_emb_tmp, dim=1)
                plan_emb_tmp = torch.mean(plan_emb_tmp, dim=1)
                time_emb_tmp = torch.flatten(time_emb_tmp, start_dim=1)
                time_emb_tmp = self.time_tran_emb(time_emb_tmp)

                emb = torch.cat([sql_emb_tmp, plan_emb_tmp, log_emb_tmp, time_emb_tmp], dim=1)
            if self.cross_model is not None:
                # pred_label = self.pred_label_cross(emb)
                # pred_opt = self.pred_opt_cross(emb)
                pred_label = self.pred_label_cross_list[i](emb)
                pred_opt = self.pred_opt_cross_list[i](emb)
                
                # pred_opt = F.softmax(pred_opt, dim=-1)
                # pred_opt = F.tanh(pred_opt)
            else:
                pred_label = self.pred_label_concat(emb)
                pred_opt = self.pred_opt_concat(emb)
                print("use concat")
                # pred_opt = F.tanh(pred_opt)
                
                # pred_opt = F.softmax(pred_opt, dim=-1)

            pred_label = torch.sigmoid(pred_label)
            if i == 0:
                pred_opt_output = pred_opt
                pred_label_output = pred_label
            else:
                pred_opt_output = torch.cat([pred_opt_output, pred_opt], dim=-1)
                pred_label_output = torch.cat([pred_label_output, pred_label], dim=-1)

        return pred_label_output, pred_opt_output
        # return pred_label, pred_opt


class CrossSQLPlanModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True, cross_model_real=None, cross_model_CrossSQLPlan=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = True, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        self.cross_model_real = cross_model_real
        self.cross_model_CrossSQLPlan = cross_model_CrossSQLPlan
        self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        
        self.log_bn = nn.BatchNorm1d(emb_dim)
        self.metrics_bn1 = nn.BatchNorm1d(7)
        self.metrics_bn2 = nn.BatchNorm1d(emb_dim)
        self.cross_mean = cross_mean
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)
        
        log_emb = self.log_model(log)
        log_emb = torch.relu(log_emb)
        log_emb = self.log_bn(log_emb)
        
        time_emb = self.time_model(time)
        time_emb = torch.relu(time_emb)
        time_emb = self.metrics_bn1(time_emb)
        # time_emb = torch.flatten(time_emb, start_dim=1)
        # time_emb = self.time_tran_emb(time_emb)
        # time_emb = torch.relu(time_emb)
        # time_emb = self.metrics_bn2(time_emb)

        sql_emb = self.sql_last_emb(sql_emb)
        sql_plan_emb = self.cross_model(sql_emb, plan_emb, None, None, None, None)
        plan_sql_emb = self.cross_model_real(plan_emb, sql_emb, None, None, None, None)
        
        # sql_plan_concat_emb = torch.cat([sql_plan_emb, plan_sql_emb], dim=1)
        # emb = self.cross_model_CrossSQLPlan(sql_plan_concat_emb, log_emb, None, time_emb, None, None)
        emb = self.cross_model_CrossSQLPlan(sql_plan_emb, plan_sql_emb, log_emb, time_emb, None, None)
        
        # sql_plan_emb = sql_plan_emb.mean(dim=1)
        # plan_sql_emb = plan_sql_emb.mean(dim=1)

        # emb = torch.cat([sql_plan_emb, plan_sql_emb, log_emb, time_emb], dim=1)
        # if self.cross_mean:
        #     emb = emb.mean(dim=1)
        # else:
        #     emb = emb[:, 0, :]
        emb = emb.mean(dim=1)
            
        # pred_label = self.pred_label_concat(emb)
        pred_label = self.pred_label_cross(emb)
        # pred_opt = self.pred_opt_concat(emb)
        pred_opt = self.pred_opt_cross(emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
    
    
class GateAttnModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True, cross_model_real=None, cross_model_CrossSQLPlan=None, rootcause_cross_model=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = True, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.common_cross_model = cross_model
        self.rootcause_cross_model = rootcause_cross_model
        self.pred_label_cross_list = nn.ModuleList()
        self.pred_opt_cross_list = nn.ModuleList()
        for i in range(5):
            self.pred_label_cross_list.append(nn.Linear(emb_dim, 1))
            self.pred_opt_cross_list.append(nn.Linear(emb_dim, 1))
        
        self.log_bn = nn.BatchNorm1d(emb_dim)
        self.metrics_bn1 = nn.BatchNorm1d(7)
        self.metrics_bn2 = nn.BatchNorm1d(emb_dim)
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)
        
        log_emb = self.log_model(log)
        log_emb = torch.relu(log_emb)
        log_emb = self.log_bn(log_emb)
        
        time_emb = self.time_model(time)
        time_emb = torch.relu(time_emb)
        time_emb = self.metrics_bn1(time_emb)

        sql_emb = self.sql_last_emb(sql_emb)
        
        common_emb = self.common_cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
        for i in range(len(self.rootcause_cross_model)):
            emb = self.rootcause_cross_model[i](sql_emb, plan_emb, log_emb, time_emb, None, None)
            emb = common_emb + emb
            emb = emb.mean(dim=1)
            
            pred_label = self.pred_label_cross_list[i](emb)
            pred_opt = self.pred_opt_cross_list[i](emb)

            pred_label = torch.sigmoid(pred_label)
            if i == 0:
                pred_opt_output = pred_opt
                pred_label_output = pred_label
            else:
                pred_opt_output = torch.cat([pred_opt_output, pred_opt], dim=-1)
                pred_label_output = torch.cat([pred_label_output, pred_label], dim=-1)
        return pred_label_output, pred_opt_output
    

class GateCommonDiffAttnModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True, cross_model_real=None, cross_model_CrossSQLPlan=None, rootcause_cross_model=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = True, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.common_cross_model = cross_model
        self.rootcause_cross_model = rootcause_cross_model
        self.pred_label_cross_list = nn.ModuleList()
        self.pred_opt_cross_list = nn.ModuleList()
        for _ in range(5):
            self.pred_label_cross_list.append(nn.Linear(emb_dim, 1))
            self.pred_opt_cross_list.append(nn.Linear(emb_dim, 1))
        
        self.log_bn = nn.BatchNorm1d(emb_dim)
        self.metrics_bn1 = nn.BatchNorm1d(7)
        self.metrics_bn2 = nn.BatchNorm1d(emb_dim)
        
        self.gate_sql = nn.ModuleList()
        self.gate_sql_activate = nn.ModuleList()
        self.gate_plan = nn.ModuleList()
        self.gate_plan_activate = nn.ModuleList()
        self.gate_log = nn.ModuleList()
        self.gate_log_activate = nn.ModuleList()
        self.gate_metrics = nn.ModuleList()
        self.gate_metrics_activate = nn.ModuleList()
        self.gate_metrics_norm = nn.ModuleList()
        self.gate_out_dim = 1
        for i in range(5):
            gate_sql_0 = nn.Sequential()
            gate_sql_0.add_module('gate_sql', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_sql_0.add_module('gate_sql_activation', nn.LeakyReLU())
            # gate_sql_0.add_module('gate_sql_layer_norm', nn.LayerNorm(emb_dim))
            
            self.gate_sql.append(gate_sql_0)
            self.gate_sql_activate.append(nn.Sigmoid())
        
            gate_plan_0 = nn.Sequential()
            gate_plan_0.add_module('gate_plan', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_plan_0.add_module('gate_plan_activation', nn.LeakyReLU())
            # gate_plan_0.add_module('gate_plan_layer_norm', nn.LayerNorm(emb_dim))
            self.gate_plan.append(gate_plan_0)
            self.gate_plan_activate.append(nn.Sigmoid())
        
            gate_log_0 = nn.Sequential()
            gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=emb_dim))
            # gate_log_0.add_module('gate_log_activation', nn.LeakyReLU())
            # gate_log_0.add_module('gate_log_layer_norm', nn.BatchNorm1d(emb_dim))
            self.gate_log.append(gate_log_0)
            self.gate_log_activate.append(nn.Sigmoid())
            
            gate_metrics_0 = nn.Sequential()
            gate_metrics_0.add_module('gate_metrics', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_metrics_0.add_module('gate_metrics_activation', nn.LeakyReLU())
            # gate_metrics_0.add_module('gate_metrics_layer_norm', nn.BatchNorm1d(7))
            self.gate_metrics.append(gate_metrics_0)
            self.gate_metrics_activate.append(nn.Sigmoid())
            # self.gate_metrics_norm.append(nn.BatchNorm1d(emb_dim))
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)
        
        log_emb = self.log_model(log)
        log_emb = torch.relu(log_emb)
        log_emb = self.log_bn(log_emb)
        
        time_emb = self.time_model(time)
        time_emb = torch.relu(time_emb)
        time_emb = self.metrics_bn1(time_emb)

        sql_emb = self.sql_last_emb(sql_emb)
        
        common_emb = self.common_cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
        for i in range(len(self.rootcause_cross_model)):
            sql_emb_tmp = self.gate_sql[i](sql_emb)
            sql_emb_tmp = self.gate_sql_activate[i](sql_emb_tmp) * sql_emb
            
            plan_emb_tmp = self.gate_plan[i](plan_emb)
            plan_emb_tmp = self.gate_plan_activate[i](plan_emb_tmp) * plan_emb
            
            log_emb_tmp = self.gate_log[i](log_emb)
            log_emb_tmp = self.gate_log_activate[i](log_emb_tmp) * log_emb
            
            time_emb_tmp = self.gate_metrics[i](time_emb)
            time_emb_tmp = self.gate_metrics_activate[i](time_emb_tmp) * time_emb
            
            emb = self.rootcause_cross_model[i](sql_emb_tmp, plan_emb_tmp, log_emb_tmp, time_emb_tmp, None, None)
            
            emb = common_emb + emb
            emb = emb.mean(dim=1)
            
            pred_label = self.pred_label_cross_list[i](emb)
            pred_opt = self.pred_opt_cross_list[i](emb)

            pred_label = torch.sigmoid(pred_label)
            if i == 0:
                pred_opt_output = pred_opt
                pred_label_output = pred_label
            else:
                pred_opt_output = torch.cat([pred_opt_output, pred_opt], dim=-1)
                pred_label_output = torch.cat([pred_label_output, pred_label], dim=-1)
        return pred_label_output, pred_opt_output
    


class GateCommonAttnModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True, cross_model_real=None, cross_model_CrossSQLPlan=None, rootcause_cross_model=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = True, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.common_cross_model = cross_model
        self.pred_label_cross_list = nn.ModuleList()
        self.pred_opt_cross_list = nn.ModuleList()
        for _ in range(5):
            self.pred_label_cross_list.append(nn.Linear(emb_dim, 1))
            self.pred_opt_cross_list.append(nn.Linear(emb_dim, 1))
        
        self.log_bn = nn.BatchNorm1d(emb_dim)
        self.metrics_bn1 = nn.BatchNorm1d(7)
        self.metrics_bn2 = nn.BatchNorm1d(emb_dim)
        
        self.gate_sql = nn.ModuleList()
        self.gate_sql_activate = nn.ModuleList()
        self.gate_plan = nn.ModuleList()
        self.gate_plan_activate = nn.ModuleList()
        self.gate_log = nn.ModuleList()
        self.gate_log_activate = nn.ModuleList()
        self.gate_metrics = nn.ModuleList()
        self.gate_metrics_activate = nn.ModuleList()
        self.gate_metrics_norm = nn.ModuleList()
        self.gate_out_dim = 1
        for i in range(5):
            gate_sql_0 = nn.Sequential()
            gate_sql_0.add_module('gate_sql', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_sql_0.add_module('gate_sql_activation', nn.LeakyReLU())
            # gate_sql_0.add_module('gate_sql_layer_norm', nn.LayerNorm(emb_dim))
            
            self.gate_sql.append(gate_sql_0)
            self.gate_sql_activate.append(nn.Sigmoid())
        
            gate_plan_0 = nn.Sequential()
            gate_plan_0.add_module('gate_plan', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_plan_0.add_module('gate_plan_activation', nn.LeakyReLU())
            # gate_plan_0.add_module('gate_plan_layer_norm', nn.LayerNorm(emb_dim))
            self.gate_plan.append(gate_plan_0)
            self.gate_plan_activate.append(nn.Sigmoid())
        
            gate_log_0 = nn.Sequential()
            gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=emb_dim))
            # gate_log_0.add_module('gate_log_activation', nn.LeakyReLU())
            # gate_log_0.add_module('gate_log_layer_norm', nn.BatchNorm1d(emb_dim))
            self.gate_log.append(gate_log_0)
            self.gate_log_activate.append(nn.Sigmoid())
            
            gate_metrics_0 = nn.Sequential()
            gate_metrics_0.add_module('gate_metrics', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_metrics_0.add_module('gate_metrics_activation', nn.LeakyReLU())
            # gate_metrics_0.add_module('gate_metrics_layer_norm', nn.BatchNorm1d(7))
            self.gate_metrics.append(gate_metrics_0)
            self.gate_metrics_activate.append(nn.Sigmoid())
            # self.gate_metrics_norm.append(nn.BatchNorm1d(emb_dim))
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)
        
        log_emb = self.log_model(log)
        log_emb = torch.relu(log_emb)
        log_emb = self.log_bn(log_emb)
        
        time_emb = self.time_model(time)
        time_emb = torch.relu(time_emb)
        time_emb = self.metrics_bn1(time_emb)

        sql_emb = self.sql_last_emb(sql_emb)
        
        common_emb = self.common_cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
        for i in range(5):
            sql_emb_tmp = self.gate_sql[i](sql_emb)
            sql_emb_tmp = self.gate_sql_activate[i](sql_emb_tmp) * sql_emb
            
            plan_emb_tmp = self.gate_plan[i](plan_emb)
            plan_emb_tmp = self.gate_plan_activate[i](plan_emb_tmp) * plan_emb
            
            log_emb_tmp = self.gate_log[i](log_emb)
            log_emb_tmp = self.gate_log_activate[i](log_emb_tmp) * log_emb
            
            time_emb_tmp = self.gate_metrics[i](time_emb)
            time_emb_tmp = self.gate_metrics_activate[i](time_emb_tmp) * time_emb
            
            emb = self.common_cross_model(sql_emb_tmp, plan_emb_tmp, log_emb_tmp, time_emb_tmp, None, None)
        
            emb = common_emb + emb
            emb = emb.mean(dim=1)
            
            pred_label = self.pred_label_cross_list[i](emb)
            pred_opt = self.pred_opt_cross_list[i](emb)

            pred_label = torch.sigmoid(pred_label)
            if i == 0:
                pred_opt_output = pred_opt
                pred_label_output = pred_label
            else:
                pred_opt_output = torch.cat([pred_opt_output, pred_opt], dim=-1)
                pred_label_output = torch.cat([pred_label_output, pred_label], dim=-1)
        return pred_label_output, pred_opt_output
    

class GateComDiff1AttnModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True, cross_model_real=None, cross_model_CrossSQLPlan=None, rootcause_cross_model=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = True, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.common_cross_model = cross_model
        self.rootcause_cross_model = rootcause_cross_model
        self.pred_label_cross_list = nn.ModuleList()
        self.pred_opt_cross_list = nn.ModuleList()
        for _ in range(5):
            self.pred_label_cross_list.append(nn.Linear(emb_dim, 1))
            self.pred_opt_cross_list.append(nn.Linear(emb_dim, 1))
        
        self.log_bn = nn.BatchNorm1d(emb_dim)
        self.metrics_bn1 = nn.BatchNorm1d(7)
        self.metrics_bn2 = nn.BatchNorm1d(emb_dim)
        # self.last_layer = nn.Linear(emb_dim, emb_dim)
        # self.last_activate = nn.LeakyReLU()
        # self.last_bn = nn.BatchNorm1d(emb_dim)
        
        
        self.gate_sql = nn.ModuleList()
        self.gate_sql_activate = nn.ModuleList()
        self.gate_plan = nn.ModuleList()
        self.gate_plan_activate = nn.ModuleList()
        self.gate_log = nn.ModuleList()
        self.gate_log_activate = nn.ModuleList()
        self.gate_metrics = nn.ModuleList()
        self.gate_metrics_activate = nn.ModuleList()
        self.gate_metrics_norm = nn.ModuleList()
        self.gate_out_dim = 1
        for i in range(5):
            gate_sql_0 = nn.Sequential()
            gate_sql_0.add_module('gate_sql', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_sql_0.add_module('gate_sql_activation', nn.LeakyReLU())
            # gate_sql_0.add_module('gate_sql_layer_norm', nn.LayerNorm(emb_dim))
            
            self.gate_sql.append(gate_sql_0)
            self.gate_sql_activate.append(nn.Sigmoid())
        
            gate_plan_0 = nn.Sequential()
            gate_plan_0.add_module('gate_plan', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_plan_0.add_module('gate_plan_activation', nn.LeakyReLU())
            # gate_plan_0.add_module('gate_plan_layer_norm', nn.LayerNorm(emb_dim))
            self.gate_plan.append(gate_plan_0)
            self.gate_plan_activate.append(nn.Sigmoid())
        
            gate_log_0 = nn.Sequential()
            gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=emb_dim))
            # gate_log_0.add_module('gate_log_activation', nn.LeakyReLU())
            # gate_log_0.add_module('gate_log_layer_norm', nn.BatchNorm1d(emb_dim))
            self.gate_log.append(gate_log_0)
            self.gate_log_activate.append(nn.Sigmoid())
            
            gate_metrics_0 = nn.Sequential()
            gate_metrics_0.add_module('gate_metrics', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_metrics_0.add_module('gate_metrics_activation', nn.LeakyReLU())
            # gate_metrics_0.add_module('gate_metrics_layer_norm', nn.BatchNorm1d(7))
            self.gate_metrics.append(gate_metrics_0)
            self.gate_metrics_activate.append(nn.Sigmoid())
            # self.gate_metrics_norm.append(nn.BatchNorm1d(emb_dim))
        
        self.init_params()
        self.sql_model = sql_model
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)
        
        log_emb = self.log_model(log)
        log_emb = torch.relu(log_emb)
        log_emb = self.log_bn(log_emb)
        
        time_emb = self.time_model(time)
        time_emb = torch.relu(time_emb)
        time_emb = self.metrics_bn1(time_emb)

        sql_emb = self.sql_last_emb(sql_emb)
        
        common_emb = self.common_cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
        for i in range(5):
            sql_emb_tmp = self.gate_sql[i](sql_emb)
            sql_emb_tmp = self.gate_sql_activate[i](sql_emb_tmp) * sql_emb
            
            plan_emb_tmp = self.gate_plan[i](plan_emb)
            plan_emb_tmp = self.gate_plan_activate[i](plan_emb_tmp) * plan_emb
            
            log_emb_tmp = self.gate_log[i](log_emb)
            log_emb_tmp = self.gate_log_activate[i](log_emb_tmp) * log_emb
            
            time_emb_tmp = self.gate_metrics[i](time_emb)
            time_emb_tmp = self.gate_metrics_activate[i](time_emb_tmp) * time_emb
            
            emb = self.rootcause_cross_model(sql_emb_tmp, plan_emb_tmp, log_emb_tmp, time_emb_tmp, None, None)
        
            emb = common_emb + emb
            emb = emb.mean(dim=1)
            
            # emb = self.last_layer(emb)
            # emb = self.last_activate(emb)
            # emb = self.last_bn(emb)
            
            pred_label = self.pred_label_cross_list[i](emb)
            pred_opt = self.pred_opt_cross_list[i](emb)

            pred_label = torch.sigmoid(pred_label)
            if i == 0:
                pred_opt_output = pred_opt
                pred_label_output = pred_label
            else:
                pred_opt_output = torch.cat([pred_opt_output, pred_opt], dim=-1)
                pred_label_output = torch.cat([pred_label_output, pred_label], dim=-1)
        return pred_label_output, pred_opt_output


class GateContrastCommonAttnModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True, cross_model_real=None, cross_model_CrossSQLPlan=None, rootcause_cross_model=None) -> None:
        super().__init__()

        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = True, use_hist = False, \
                 pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.common_cross_model = cross_model
        self.pred_label_cross_list = nn.ModuleList()
        self.pred_opt_cross_list = nn.ModuleList()
        for _ in range(5):
            self.pred_label_cross_list.append(nn.Linear(emb_dim, 1))
            self.pred_opt_cross_list.append(nn.Linear(emb_dim, 1))
        
        self.log_bn = nn.BatchNorm1d(emb_dim)
        self.metrics_bn1 = nn.BatchNorm1d(7)
        self.metrics_bn2 = nn.BatchNorm1d(emb_dim)
        
        self.gate_sql = nn.ModuleList()
        self.gate_sql_activate = nn.ModuleList()
        self.gate_plan = nn.ModuleList()
        self.gate_plan_activate = nn.ModuleList()
        self.gate_log = nn.ModuleList()
        self.gate_log_activate = nn.ModuleList()
        self.gate_metrics = nn.ModuleList()
        self.gate_metrics_activate = nn.ModuleList()
        self.gate_metrics_norm = nn.ModuleList()
        self.gate_out_dim = 1
        for i in range(5):
            gate_sql_0 = nn.Sequential()
            gate_sql_0.add_module('gate_sql', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_sql_0.add_module('gate_sql_activation', nn.LeakyReLU())
            # gate_sql_0.add_module('gate_sql_layer_norm', nn.LayerNorm(emb_dim))
            
            self.gate_sql.append(gate_sql_0)
            self.gate_sql_activate.append(nn.Sigmoid())
        
            gate_plan_0 = nn.Sequential()
            gate_plan_0.add_module('gate_plan', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_plan_0.add_module('gate_plan_activation', nn.LeakyReLU())
            # gate_plan_0.add_module('gate_plan_layer_norm', nn.LayerNorm(emb_dim))
            self.gate_plan.append(gate_plan_0)
            self.gate_plan_activate.append(nn.Sigmoid())
        
            gate_log_0 = nn.Sequential()
            gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=emb_dim))
            # gate_log_0.add_module('gate_log_activation', nn.LeakyReLU())
            # gate_log_0.add_module('gate_log_layer_norm', nn.BatchNorm1d(emb_dim))
            self.gate_log.append(gate_log_0)
            self.gate_log_activate.append(nn.Sigmoid())
            
            gate_metrics_0 = nn.Sequential()
            gate_metrics_0.add_module('gate_metrics', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_metrics_0.add_module('gate_metrics_activation', nn.LeakyReLU())
            # gate_metrics_0.add_module('gate_metrics_layer_norm', nn.BatchNorm1d(7))
            self.gate_metrics.append(gate_metrics_0)
            self.gate_metrics_activate.append(nn.Sigmoid())
            # self.gate_metrics_norm.append(nn.BatchNorm1d(emb_dim))
        
        self.init_params()
        self.sql_model = sql_model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = self.plan_model(plan)
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)
        
        log_emb = self.log_model(log)
        log_emb = torch.relu(log_emb)
        log_emb = self.log_bn(log_emb)
        
        time_emb = self.time_model(time)
        time_emb = torch.relu(time_emb)
        time_emb = self.metrics_bn1(time_emb)

        sql_emb = self.sql_last_emb(sql_emb)
        
        common_emb = self.common_cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
        for i in range(5):
            sql_emb_tmp = self.gate_sql[i](sql_emb)
            sql_emb_tmp = self.gate_sql_activate[i](sql_emb_tmp) * sql_emb
            
            plan_emb_tmp = self.gate_plan[i](plan_emb)
            plan_emb_tmp = self.gate_plan_activate[i](plan_emb_tmp) * plan_emb
            
            log_emb_tmp = self.gate_log[i](log_emb)
            log_emb_tmp = self.gate_log_activate[i](log_emb_tmp) * log_emb
            
            time_emb_tmp = self.gate_metrics[i](time_emb)
            time_emb_tmp = self.gate_metrics_activate[i](time_emb_tmp) * time_emb
            
            emb = self.common_cross_model(sql_emb_tmp, plan_emb_tmp, log_emb_tmp, time_emb_tmp, None, None)
        
            emb = common_emb + emb
            emb = emb.mean(dim=1)
            
            pred_label = self.pred_label_cross_list[i](emb)
            pred_opt = self.pred_opt_cross_list[i](emb)

            pred_label = torch.sigmoid(pred_label)
            if i == 0:
                pred_opt_output = pred_opt
                pred_label_output = pred_label
            else:
                pred_opt_output = torch.cat([pred_opt_output, pred_opt], dim=-1)
                pred_label_output = torch.cat([pred_label_output, pred_label], dim=-1)
        return pred_label_output, pred_opt_output, common_emb.mean(dim=1), self.logit_scale