import torch.nn as nn
import torch

from model.emb import AlignmentModel
from model.pretrain.pretrain import Alignment
from models import LogModel
from model.modules.QueryFormer.QueryFormer import QueryFormer, QueryFormerBert
from transformers import BertTokenizer, BertModel

class Predict(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, ff_dim, dropout=0.1):
        super(Predict, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.input_fc = nn.Linear(input_dim, model_dim)  # 调整输入维度到模型维度

    def forward(self, x):
        x = self.input_fc(x)  # 调整输入维度
        x = self.transformer_encoder(x)
        return x


class GateComDiffPretrainModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True, cross_model_real=None, cross_model_CrossSQLPlan=None, rootcause_cross_model=None) -> None:
        super().__init__()
        self.time_model = time_model
        self.log_model = LogModel(1, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.plan_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)

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
            # self.gate_sql_activate.append(nn.Softmax(dim=-1))
        
            gate_plan_0 = nn.Sequential()
            gate_plan_0.add_module('gate_plan', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_plan_0.add_module('gate_plan_activation', nn.LeakyReLU())
            # gate_plan_0.add_module('gate_plan_layer_norm', nn.LayerNorm(emb_dim))
            self.gate_plan.append(gate_plan_0)
            self.gate_plan_activate.append(nn.Sigmoid())
            # self.gate_plan_activate.append(nn.Softmax(dim=-1))
        
            gate_log_0 = nn.Sequential()
            gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=emb_dim))
            # gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_log_0.add_module('gate_log_activation', nn.LeakyReLU())
            # gate_log_0.add_module('gate_log_layer_norm', nn.BatchNorm1d(emb_dim))
            self.gate_log.append(gate_log_0)
            self.gate_log_activate.append(nn.Sigmoid())
            # self.gate_log_activate.append(nn.Softmax(dim=-1))
            
            gate_metrics_0 = nn.Sequential()
            gate_metrics_0.add_module('gate_metrics', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_metrics_0.add_module('gate_metrics_activation', nn.LeakyReLU())
            # gate_metrics_0.add_module('gate_metrics_layer_norm', nn.BatchNorm1d(7))
            self.gate_metrics.append(gate_metrics_0)
            self.gate_metrics_activate.append(nn.Sigmoid())
            # self.gate_metrics_activate.append(nn.Softmax(dim=-1))
            # self.gate_metrics_norm.append(nn.BatchNorm1d(emb_dim))
        
        self.init_params()
        self.device = device
        # self.alignmentModel = AlignmentModel(device=device)
        self.alignmentModel = Alignment(device=device)
        # self.alignmentModel.load_state_dict(torch.load('./pretrain/2500_save.pth')['model_state_dict'])
        # self.alignmentModel.load_state_dict(torch.load('./pretrain/model_500.pth'))
        # self.alignmentModel.load_state_dict(torch.load('./pretrain/best_model.pth'))
        
        # self.plan_model = self.alignmentModel.plan_encoder
        # self.sql_model = self.alignmentModel.sql_encoder
        self.sql_model = sql_model
        # self.sql_model = self.alignmentModel.sql_model
        self.log_model = self.alignmentModel.log_model
        self.plan_model = self.alignmentModel.plan_model
        # self.tokenizer = self.alignmentModel.tokenizer        
        
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
        # sql = self.tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            sql_emb = self.sql_model(**sql)
            # sql_emb = self.sql_model(sql)
            
        plan_emb = self.plan_model(plan)
        sql_emb = sql_emb.last_hidden_state
                
        # sql_emb = F.relu(sql_emb)
        
        # log = log.unsqueeze(2)
        log_emb = self.log_model(log)
        log_emb = torch.relu(log_emb)
        log_emb = self.log_bn(log_emb)
        
        time_emb = time.unsqueeze(1)
        time_emb = self.time_model(time_emb).squeeze(3).squeeze(2)
        # time_emb = time
        # time_emb = self.time_model(time_emb)
        time_emb = torch.relu(time_emb)
        # time_emb = self.metrics_bn1(time_emb)
        time_emb = self.metrics_bn2(time_emb)

        sql_emb = self.sql_last_emb(sql_emb)
        
        center_sql = sql_emb.mean(1)
        center_plan = plan_emb.mean(1)
        center_log = log_emb
        center_time = time_emb
        
        
        # sql_emb = sql_emb[:, 0, :]
        # sql_emb = self.sql_last_emb(sql_emb)

        # plan_emb = self.plan_last_emb(plan_emb)
        # plan_emb = plan_emb[:, 0, :]
        # time_emb = torch.flatten(time_emb, start_dim=1)
        # time_emb = self.time_tran_emb(time_emb)
        # emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        # pred_label = self.pred_label_concat(emb)
        # pred_opt = self.pred_opt_concat(emb)
        
        # return pred_label, pred_opt
        # common_emb = self.common_cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
        all_emb = None
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
        
            # emb = common_emb + emb
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
                all_emb = emb
            else:
                pred_opt_output = torch.cat([pred_opt_output, pred_opt], dim=-1)
                pred_label_output = torch.cat([pred_label_output, pred_label], dim=-1)
                all_emb = torch.cat([all_emb, emb], dim=-1)
        # center_sql, center_plan, center_log, center_time = center_sql.to("cpu"), center_plan.to("cpu"), center_log.to("cpu"), center_time.to("cpu")
        # return pred_label_output, pred_opt_output, center_sql.to("cpu"), center_plan.to("cpu"), center_log.to("cpu"), center_time.to("cpu")
        return all_emb.to("cpu")
    
    

class GatePretrainModel(nn.Module):
    def __init__(self, t_input_dim, l_input_dim, l_hidden_dim, t_hidden_him, emb_dim, sql_model=None, device=None,
                  plan_args=None, cross_model=None, time_model=None, cross_mean=True, cross_model_real=None, cross_model_CrossSQLPlan=None, rootcause_cross_model=None) -> None:
        super().__init__()
        # self.plan_model = QueryFormerBert(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
        #          dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
        #          use_sample = True, use_hist = False, \
        #          pred_hid = emb_dim)
        # self.plan_model = QueryFormer(emb_size = plan_args.embed_size ,ffn_dim = plan_args.ffn_dim, head_size = plan_args.head_size, \
        #          dropout = plan_args.dropout, n_layers = plan_args.n_layers, \
        #          use_sample = True, use_hist = False, \
        #          pred_hid = emb_dim)
        
        # self.time_model = TimeSeriesModel(t_input_dim, t_hidden_him, emb_dim)
        self.time_model = time_model
        # self.log_model = LogModel(l_input_dim, l_hidden_dim, emb_dim)
        # self.log_model = LogModel(1, l_hidden_dim, emb_dim)

        self.sql_last_emb = nn.Linear(768, emb_dim)
        self.plan_last_emb = nn.Linear(768, emb_dim)
        self.time_tran_emb = nn.Linear(emb_dim * 7, emb_dim)
        # 融合三个模态

        # self.common_cross_model = cross_model
        self.rootcause_cross_model = rootcause_cross_model
        self.pred_label_cross_list = nn.ModuleList()
        self.pred_opt_cross_list = nn.ModuleList()
        # for _ in range(5):
        for _ in range(6):
            self.pred_label_cross_list.append(nn.Linear(emb_dim, 1))
            self.pred_opt_cross_list.append(nn.Linear(emb_dim, 1))
        
        self.log_bn = nn.BatchNorm1d(emb_dim)
        # self.log_bn = nn.BatchNorm1d(13)
        self.metrics_bn1 = nn.BatchNorm1d(7)
        self.metrics_bn2 = nn.BatchNorm1d(emb_dim)
        # self.last_layer = nn.Linear(emb_dim, emb_dim)
        # self.last_activate = nn.LeakyReLU()
        # self.last_bn = nn.BatchNorm1d(emb_dim)
        
        # self.pred_label_concat = nn.Linear(emb_dim * 4, 5)
        # self.pred_opt_concat = nn.Linear(emb_dim * 4, 5)
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
        # for i in range(6):
            gate_sql_0 = nn.Sequential()
            gate_sql_0.add_module('gate_sql', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_sql_0.add_module('gate_sql_activation', nn.LeakyReLU())
            # gate_sql_0.add_module('gate_sql_layer_norm', nn.LayerNorm(emb_dim))
            
            self.gate_sql.append(gate_sql_0)
            self.gate_sql_activate.append(nn.Sigmoid())
            # self.gate_sql_activate.append(nn.Softmax(dim=-1))
        
            gate_plan_0 = nn.Sequential()
            gate_plan_0.add_module('gate_plan', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_plan_0.add_module('gate_plan_activation', nn.LeakyReLU())
            # gate_plan_0.add_module('gate_plan_layer_norm', nn.LayerNorm(emb_dim))
            self.gate_plan.append(gate_plan_0)
            self.gate_plan_activate.append(nn.Sigmoid())
            # self.gate_plan_activate.append(nn.Softmax(dim=-1))
        
            gate_log_0 = nn.Sequential()
            gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=emb_dim))
            # gate_log_0.add_module('gate_log', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_log_0.add_module('gate_log_activation', nn.LeakyReLU())
            # gate_log_0.add_module('gate_log_layer_norm', nn.BatchNorm1d(emb_dim))
            self.gate_log.append(gate_log_0)
            self.gate_log_activate.append(nn.Sigmoid())
            # self.gate_log_activate.append(nn.Softmax(dim=-1))
            
            gate_metrics_0 = nn.Sequential()
            gate_metrics_0.add_module('gate_metrics', nn.Linear(in_features=emb_dim, out_features=self.gate_out_dim))
            # gate_metrics_0.add_module('gate_metrics_activation', nn.LeakyReLU())
            # gate_metrics_0.add_module('gate_metrics_layer_norm', nn.BatchNorm1d(7))
            self.gate_metrics.append(gate_metrics_0)
            self.gate_metrics_activate.append(nn.Sigmoid())
            # self.gate_metrics_activate.append(nn.Softmax(dim=-1))
            # self.gate_metrics_norm.append(nn.BatchNorm1d(emb_dim))
        
        self.init_params()
        self.device = device
        # self.alignmentModel = AlignmentModel(device=device)
        self.alignmentModel = Alignment(device=device)
        # self.alignmentModel.load_state_dict(torch.load('./pretrain/2500_save.pth')['model_state_dict'])
        # self.alignmentModel.load_state_dict(torch.load('pretrain/alignment_new/model30.pth')) # now use
        # self.alignmentModel.load_state_dict(torch.load('./pretrain/model_500.pth'))
        # self.alignmentModel.load_state_dict(torch.load('./pretrain/best_model.pth'))
        
        # self.plan_model = self.alignmentModel.plan_encoder
        self.sql_model = self.alignmentModel.sql_model
        # self.sql_model = sql_model
        # self.sql_model = self.alignmentModel.sql_model
        self.log_model = self.alignmentModel.log_model
        self.plan_model = self.alignmentModel.plan_model
        # self.tokenizer = self.alignmentModel.tokenizer        
        
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sql, plan, time, log):
        with torch.no_grad():
        # sql = self.tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            sql_emb = self.sql_model(**sql)
            # sql_emb = self.sql_model(sql)
            
        plan_emb = self.plan_model(plan)
        sql_emb = sql_emb.last_hidden_state
                
        # sql_emb = F.relu(sql_emb)
        
        # log = log.unsqueeze(2)
        log_emb = self.log_model(log)
        log_emb = torch.relu(log_emb)
        log_emb = self.log_bn(log_emb)
        
        # time_emb = time.unsqueeze(1)
        # time_emb = self.time_model(time_emb).squeeze(3).squeeze(2)
        # # time_emb = time
        # # time_emb = self.time_model(time_emb)
        # time_emb = torch.relu(time_emb)
        # # time_emb = self.metrics_bn1(time_emb)
        # time_emb = self.metrics_bn2(time_emb)

        sql_emb = self.sql_last_emb(sql_emb)
    
        # sql_emb = sql_emb[:, 0, :]
        # sql_emb = self.sql_last_emb(sql_emb)

        # plan_emb = self.plan_last_emb(plan_emb)
        # plan_emb = plan_emb[:, 0, :]
        # time_emb = torch.flatten(time_emb, start_dim=1)
        # time_emb = self.time_tran_emb(time_emb)
        # emb = torch.cat([sql_emb, plan_emb, log_emb, time_emb], dim=1)
        # pred_label = self.pred_label_concat(emb)
        # pred_opt = self.pred_opt_concat(emb)
        
        # return pred_label, pred_opt
        # common_emb = self.common_cross_model(sql_emb, plan_emb, log_emb, time_emb, None, None)
        for i in range(5):
        # for i in range(6):
            sql_emb_tmp = self.gate_sql[i](sql_emb)
            sql_emb_tmp = self.gate_sql_activate[i](sql_emb_tmp) * sql_emb
            
            plan_emb_tmp = self.gate_plan[i](plan_emb)
            plan_emb_tmp = self.gate_plan_activate[i](plan_emb_tmp) * plan_emb
            
            log_emb_tmp = self.gate_log[i](log_emb)
            log_emb_tmp = self.gate_log_activate[i](log_emb_tmp) * log_emb
            
            # time_emb_tmp = self.gate_metrics[i](time_emb)
            # time_emb_tmp = self.gate_metrics_activate[i](time_emb_tmp) * time_emb
            
            # emb = self.rootcause_cross_model(sql_emb_tmp, plan_emb_tmp, log_emb_tmp, time_emb_tmp, None, None)
            emb = self.rootcause_cross_model(sql_emb_tmp, plan_emb_tmp, log_emb_tmp, None, None, None)
            # emb = self.rootcause_cross_model(plan_emb_tmp, sql_emb_tmp, log_emb_tmp, None, None, None)
        
            # emb = common_emb + emb
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