import torch
from torch import nn
from model.modules.QueryFormer.QueryFormer import QueryFormer
from torch.nn import functional as F


class LogModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        
        self.ll_1 = nn.Linear(input_dim, hidden_dim)
        self.ll_2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, output_dim)

        self.init_params()
            
    def forward(self, input_ids):
        # output = self.bert(input_ids)
        output = self.ll_1(input_ids)
        output = F.relu(output)
        output = self.ll_2(output)
        output = F.relu(output)
        output = self.cls(output)

        return output
    
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        
        self.ll_1 = nn.Linear(input_dim, hidden_dim)
        self.ll_2 = nn.Linear(hidden_dim, hidden_dim)
        # self.cls = nn.Linear(hidden_dim * 7, output_dim)
        self.cls = nn.Linear(hidden_dim, output_dim)

        self.init_params()
            
    def forward(self, input_ids):
        # output = self.bert(input_ids)
        output = self.ll_1(input_ids)
        output = F.relu(output)
        output = self.ll_2(output)
        # output = torch.flatten(output, start_dim=1)
        output = F.relu(output)
        output = self.cls(output)
        output = F.relu(output)

        return output
    
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class Model(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, freeze_model=False, plan_args=None, cross_model=None) -> None:
        super().__init__()

        self.sql_model = sql_model
        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = True, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 12)
        # self.pred_label_cross = nn.Linear(emb_dim, 12)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 12)
        # self.pred_opt_cross = nn.Linear(emb_dim, 12)
        self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_label_cross = nn.Linear(emb_dim, 5)
        self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
        self.pred_opt_cross = nn.Linear(emb_dim, 5)
        
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
            emb = self.cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
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
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
        
class ThreeMulModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, freeze_model=False, plan_args=None, cross_model=None) -> None:
        super().__init__()

        self.sql_model = sql_model
        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = True, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        self.pred_label_concat = nn.Linear(emb_dim * 3, 12)
        self.pred_label_cross = nn.Linear(emb_dim, 12)
        self.pred_opt_concat = nn.Linear(emb_dim * 3, 12)
        self.pred_opt_cross = nn.Linear(emb_dim, 12)
        

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
            sql_emb = self.sql_model(**sql)
        plan_emb = None
        if plan is not None:
            plan_emb = self.plan_model(plan)
        log_emb = self.log_model(log)
        time_emb = self.time_model(time)
        
        sql_emb = sql_emb.last_hidden_state
        # sql_emb = F.relu(sql_emb)

        if self.cross_model is not None:
            sql_emb = self.sql_last_emb(sql_emb)
            emb = self.cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
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
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
    
class PlanModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None, freeze_model=False, plan_args=None, cross_model=None) -> None:
        super().__init__()

        self.sql_model = sql_model
        self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
                 dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
                 use_sample = True, use_hist = False, \
                 pred_hid = emb_dim)
        
        self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)

        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        self.pred_label_concat = nn.Linear(emb_dim * 2, 12)
        self.pred_label_cross = nn.Linear(emb_dim, 12)
        self.pred_opt_concat = nn.Linear(emb_dim * 2, 12)
        self.pred_opt_cross = nn.Linear(emb_dim, 12)
        

    def forward(self, sql, plan, time, log):
        plan_emb = self.plan_model(plan)
        time_emb = self.time_model(time)

        plan_emb = plan_emb[:, 0, :]
        time_emb = torch.flatten(time_emb, start_dim=1)
        time_emb = self.time_tran_emb(time_emb)

        emb = torch.cat([plan_emb, time_emb], dim=1)
        if self.cross_model is not None:
            pred_label = self.pred_label_cross(emb)
            pred_opt = self.pred_opt_cross(emb)
        else:
            pred_label = self.pred_label_concat(emb)
            pred_opt = self.pred_opt_concat(emb)

        pred_label = torch.sigmoid(pred_label)

        return pred_label, pred_opt
    

