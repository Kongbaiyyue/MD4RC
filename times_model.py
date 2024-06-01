import torch
from torch import nn
from model.modules.QueryFormer.QueryFormer import QueryFormer
from torch.nn import functional as F


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


class TimeSoftmaxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=None, t=1, use_softmax=False) -> None:
        super().__init__()
        
        self.ll_1 = nn.Linear(input_dim, hidden_dim)
        self.ll_2 = nn.Linear(hidden_dim, hidden_dim)
        # self.cls = nn.Linear(hidden_dim * 7, output_dim)
        self.cls = nn.Linear(hidden_dim, output_dim)

        self.t = t
        self.use_softmax = use_softmax

        self.init_params()
            
    def forward(self, input_ids):
        # output = self.bert(input_ids)
        output = input_ids
        if self.use_softmax:
            output = F.softmax(output / self.t, dim=-1)

        output = self.ll_1(output)
        output = F.relu(output)
        output = self.ll_2(output)
        # output = torch.flatten(output, start_dim=1)
        output = F.relu(output)
        output = self.cls(output)

        return output
    
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)