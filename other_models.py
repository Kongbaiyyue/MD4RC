import torch
from torch import nn
from model.modules.QueryFormer.QueryFormer import QueryFormer
from torch.nn import functional as F

class GaussModel(nn.Module):
    def __init__(self, plan_input, metrics_input, emb_dim, sql_model=None, device=None, freeze_model=False, plan_args=None, cross_model=None, batch_size=None) -> None:
        super().__init__()

        self.sql_model = sql_model
        self.plan_model = nn.Linear(plan_input, emb_dim)
        self.time_tran_emb = nn.Linear(metrics_input, emb_dim)
        # 融合三个模态

        self.cross_model = cross_model
        self.pred_label_concat = nn.Linear(emb_dim * (500 + 7), 12)
        self.pred_opt_concat = nn.Linear(emb_dim * (500 + 7), 12)
        

    def forward(self, sql, plan, time, log):
        plan_emb = self.plan_model(plan['x'])
        time_emb = self.time_tran_emb(time)
        emb = torch.cat([plan_emb, time_emb], dim=1)

        emb = torch.flatten(emb, start_dim=1)

        pred_label = self.pred_label_concat(emb)
        pred_opt = self.pred_opt_concat(emb)

        pred_label = F.sigmoid(pred_label)

        return pred_label, pred_opt