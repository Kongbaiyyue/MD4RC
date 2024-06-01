import json
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import re
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 请在这里粘贴完整的JSON数据

class PlanEncoder(nn.Module):
    def __init__(self,device):
        super(PlanEncoder, self).__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.node_merge = BertModel.from_pretrained("bert-base-uncased").to(self.device)

    def padding(self, max_len):
        print("max_len", max_len)
        for i in range(len(self.embs_bz)):
            padding = torch.zeros(max_len - self.embs_bz[i].shape[0], 768).to(self.device)
            self.embs_bz[i] = torch.cat([self.embs_bz[i], padding], dim=0)
        self.embs_bz = torch.stack(self.embs_bz)
        
    def preorder_traversal(self,plan, depth=0):
        
        if isinstance(plan, dict):
            node=''
            for key in plan.keys():
                if key != 'Plans':
                    node = node+f'\'{key}\':\'{plan[key]}\','
            tokens = self.tokenizer(node, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            node_emb = self.node_merge(**tokens).pooler_output
        
            self.embs=torch.cat([self.embs,node_emb],dim=0)
            if 'Plans' in plan:
                for subplan in plan['Plans']:
                    self.preorder_traversal(subplan, depth + 1)

    def process_plan_raw(self,plan):
        plan = plan[plan.find('{'):plan.rfind('}') + 1]
        plan = plan.replace('False','\"False\"').replace('\': \'','\": \"').replace('\'}','\"}').replace('{\'','{\"').replace('\', \'','\", \"').replace(', \'',', \"').replace('\':','\":').replace('[\'','[\"').replace('\']','\"]').replace('\"::','\'::').replace('\"0\'','\'0\'').replace('\"(\"','\"(\'').replace(') = \"',') = \'').replace('numeric \"','numeric \'').replace('\'numeric\"','\'numeric\'').replace('\\','').replace('{[\"','{[\'')
        return plan
    
    def forward(self, json_data):
        self.embs_bz = []
        max_len = 2
        error = 0
        if type(json_data) == list:
            for i in range(len(json_data)):
                self.embs=torch.zeros(1,768).to(self.device)
                json_data_tmp = json_data[i]
                # print(i)
                if type(json_data_tmp)==str:
                    try:
                        # json_data_tmp=json.loads(self.process_plan_raw(json_data_tmp))
                        json_data_tmp=json.loads(json_data_tmp)
                    except:
                        error += 1
                        self.embs=torch.zeros(2,768).to(self.device)
                        self.embs_bz.append(self.embs)
                        continue
                try:
                    self.preorder_traversal(json_data_tmp['Plan'])
                except:
                    # print(json_data_tmp)
                    self.embs=torch.zeros(2,768).to(self.device)
                self.embs_bz.append(self.embs)
                if max_len < self.embs.shape[0]: max_len = self.embs.shape[0]
                
            self.padding(max_len)
            print("plan error: ", error)
            return self.embs_bz[:, 1:]
                
        # self.embs=torch.zeros(1,768).to(self.device)
        # # if type(json_data)==str:
        # try:
        #     json_data=json.loads(self.process_plan_raw(json_data))
        #     self.preorder_traversal(json_data['Plan'])
        # except:
        #     self.embs=torch.zeros(1,768).to(self.device)
        #     return self.embs
                
        # return self.embs[1:]
                
        if type(json_data)==str:
            # json_data=json.loads(self.process_plan_raw(json_data))
            json_data=json.loads(json_data)
        self.embs=torch.zeros(1,768).to(self.device)
        self.preorder_traversal(json_data['Plan'])
        return self.embs[1:].unsqueeze(0)

class SQLEncoder(nn.Module):
    def __init__(self,device):
        super(SQLEncoder, self).__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
     
    
    def forward(self, sql):
        tokens = self.tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        # emb = self.model(**tokens).pooler_output
        emb = self.model(**tokens).last_hidden_state
        
        return emb

class AlignmentModel(nn.Module):
    def __init__(self,device):
        super(AlignmentModel, self).__init__()
        self.device = device
        self.plan_encoder = PlanEncoder(device)
        self.sql_encoder = SQLEncoder(device)
       
        self.pridict_model = nn.Transformer(d_model=768).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, dropout=0.1),
            num_layers=3
        ).to(device)

    def forward(self, sql,plan):
        sql_emb = self.sql_encoder(sql)
        plan_emb = self.plan_encoder(plan)
        x = torch.cat((sql_emb, plan_emb)).unsqueeze(0)
        x = self.transformer_encoder(x)

        return x.mean(dim=1)
    
if __name__ == "__main__":
    alignmentModel = AlignmentModel('cuda:0')
    checkpoint = torch.load('./pretrain/2500_save.pth')
    alignmentModel.load_state_dict(checkpoint['model_state_dict'])
    # df = pd.read_csv("dataset/SQL/data.csv")
    # print(df['sql'][0])
    # print(df['plan_json'][0])

    # sql = 'SELECT * FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk WHERE d.d_year = 2020;'
    # plan = '''0    [{'Plan': {'Node Type': 'Gather', 'Parallel Aware': False, 'Startup Cost': 0.0, 'Total Cost': 265.5, 'Plan Rows': 234272, 'Plan Width': 381, 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Join Type': 'Inner', 'Startup Cost': 0.0, 'Total Cost': 48.69, 'Plan Rows': 234272, 'Plan Width': 381, 'Inner Unique': False, 'Hash Cond': '(store_sales.ss_customer_sk = customer.c_customer_sk)', 'Plans': [{'Node Type': 'Redistribution', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Startup Cost': 0.0, 'Total Cost': 27.46, 'Plan Rows': 234272, 'Plan Width': 206, 'Hash Key': 'store_sales.ss_customer_sk', 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Join Type': 'Inner', 'Startup Cost': 0.0, 'Total Cost': 19.91, 'Plan Rows': 234272, 'Plan Width': 206, 'Inner Unique': False, 'Hash Cond': '(store_sales.ss_sold_date_sk = date_dim.d_date_sk)', 'Plans': [{'Node Type': 'Local Gather', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Startup Cost': 0.0, 'Total Cost': 5.93, 'Plan Rows': 1111100, 'Plan Width': 99, 'Plans': [{'Node Type': 'Decode', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Startup Cost': 0.0, 'Total Cost': 5.55, 'Plan Rows': 1111100, 'Plan Width': 99, 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Relation Name': 'store_sales', 'Alias': 'store_sales', 'Startup Cost': 0.0, 'Total Cost': 5.45, 'Plan Rows': 1111100, 'Plan Width': 99}]}]}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Startup Cost': 5.4, 'Total Cost': 5.4, 'Plan Rows': 7266, 'Plan Width': 107, 'Plans': [{'Node Type': 'Broadcast', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Startup Cost': 0.0, 'Total Cost': 5.4, 'Plan Rows': 7266, 'Plan Width': 107, 'Plans': [{'Node Type': 'Local Gather', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Startup Cost': 0.0, 'Total Cost': 5.25, 'Plan Rows': 363, 'Plan Width': 107, 'Plans': [{'Node Type': 'Decode', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Startup Cost': 0.0, 'Total Cost': 5.25, 'Plan Rows': 363, 'Plan Width': 107, 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Relation Name': 'date_dim', 'Alias': 'date_dim', 'Startup Cost': 0.0, 'Total Cost': 5.15, 'Plan Rows': 363, 'Plan Width': 107, 'Filter': '(d_year = 2020)'}]}]}]}]}]}]}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Startup Cost': 5.19, 'Total Cost': 5.19, 'Plan Rows': 76200, 'Plan Width': 175, 'Plans': [{'Node Type': 'Local Gather', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Startup Cost': 0.0, 'Total Cost': 5.19, 'Plan Rows': 76200, 'Plan Width': 175, 'Plans': [{'Node Type': 'Decode', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Startup Cost': 0.0, 'Total Cost': 5.15, 'Plan Rows': 76200, 'Plan Width': 175, 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Relation Name': 'customer', 'Alias': 'customer', 'Startup Cost': 0.0, 'Total Cost': 5.05, 'Plan Rows': 76200, 'Plan Width': 175}]}]}]}]}]}, 'Optimizer': 'HQO version 1.3.0'}]
    # Name: QUERY PLAN, dtype: object'''
    # df = pd.read_csv("data/all_data/add_2023_12_30_2-5-sqlsmith_resplit_lg_500_cost_64_2_random_0.05rate_4_valid.csv")

    # df.loc[:, "json_plan_tensor"] = torch.tensor(0)
    # for i in range(df.shape[0]):
    #     print(i)
    #     print(alignmentModel.plan_encoder(df["plan_json"].iloc[i]))
    #     df["json_plan_tensor"].iloc[i] = alignmentModel.plan_encoder(df["plan_json"].iloc[i]).to("cpu").detach()
    # df.to_pickle("data/all_data/add_2023_12_30_2-5-sqlsmith_resplit_lg_500_cost_64_2_random_0.05rate_4_pretrain_valid.pickle")

    # for plan in df["plan_json"].tolist():
        
    # sql_emb = alignmentModel.sql_encoder(sql)  # 1,768
    # plan_emb = alignmentModel.plan_encoder(plan) # node_nums,768
    # print(sql_emb.shape)
    # print(plan_emb.shape)