import time
from torch.utils import data

import json
from collections import deque
import sys
import pickle
sys.path.append(".")

import numpy as np
import torch

from model.modules.QueryFormer.utils import formatFilter, formatJoin, TreeNode, filterDict2Hist
from model.modules.QueryFormer.utils import *

import torch
from torch.utils import data
import numpy as np

import json
import time

from collections import deque

from model.modules.QueryFormer.utils import formatFilter, formatJoin, TreeNode, filterDict2Hist
from model.modules.QueryFormer.utils import *

class Two_modal_dataset(data.Dataset):
    '''
        sample: [feature, label]
    '''
    def __init__(self, df, device, train=True, encoding=None, tokenizer=None, train_dataset=None):
        super().__init__()
        self.encoding = encoding
        self.treeNodes = []

        samples_list = df.values.tolist()
        samples_data = []

        if train_dataset is None:
            # 训练集归一化操作
            logs = []
            timeseries = []
            opt_labels = []
            querys = []
            for i, samples in enumerate(samples_list):
                querys.append(samples[0])
                logs.append(torch.tensor(samples[2]))
                timeseries.append(torch.tensor(samples[3]))
                opt_labels.append(torch.tensor(samples[5]))
            
            # tokens = tokenizer(querys, return_tensors="pt", padding=True, truncation=True, max_length=512)

            logs = torch.stack(logs, dim=0)
            self.logs_train_mean = logs.mean(dim=0)
            self.logs_train_std = logs.std(dim=0)

            timeseries = torch.stack(timeseries, dim=0)
            self.timeseries_train_mean = timeseries.mean(dim=[0, 2])
            self.timeseries_train_std = timeseries.std(dim=[0, 2])

            opt_labels = torch.stack(opt_labels, dim=0)
            self.opt_labels_train_mean = opt_labels.mean(dim=0)
            self.opt_labels_train_std = opt_labels.std(dim=0)

        else:
            querys = df["query"].values.tolist()
            # tokens = tokenizer(querys, return_tensors="pt", padding=True, truncation=True, max_length=512)

            self.logs_train_mean = train_dataset.logs_train_mean
            self.logs_train_std = train_dataset.logs_train_std
            self.timeseries_train_mean = train_dataset.timeseries_train_mean
            self.timeseries_train_std = train_dataset.timeseries_train_std
            self.opt_labels_train_mean = train_dataset.opt_labels_train_mean
            self.opt_labels_train_std = train_dataset.opt_labels_train_std

        for i, samples in enumerate(samples_list):
            sam = {
                    "query": samples[0], 
                    # "query": {"input_ids": tokens["input_ids"][i], "token_type_ids": tokens["token_type_ids"][i], "attention_mask": tokens["attention_mask"][i]},
                   "plan": samples[1], 
                   "log": (torch.tensor(samples[2]) - self.logs_train_mean) / (self.logs_train_std + 1e-6),
                    "timeseries": (torch.tensor(samples[3]) - self.timeseries_train_mean.unsqueeze(1)) / (self.timeseries_train_std.unsqueeze(1) + 1e-6), 
                    "multilabel": torch.tensor(samples[4]), 
                    "opt_label": (torch.tensor(samples[5])  - self.opt_labels_train_mean) / (self.opt_labels_train_std + 1e-6),
                    "duration": samples[6],
                    "ori_opt_label": torch.tensor(samples[5]),
            }
            samples_data.append(sam)

        self.samples = samples_data
        self.device = device
        self.train = train

            # sam = Sample(samples[0], samples[1], torch.tensor
    def __getitem__(self,index):
        # label = self.samples[index][-1]
        # feature = self.samples[index][:-1]
        # print(self.samples[index])
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

class Multimodal_dataset(data.Dataset):
    '''
        sample: [feature, label]
    '''
    def __init__(self, df, device, train=True, encoding=None, tokenizer=None, train_dataset=None):
        super().__init__()
        self.encoding = encoding
        self.treeNodes = []

        samples_list = df.values.tolist()
        samples_data = []

        if train_dataset is None:
            # 训练集归一化操作
            logs = []
            timeseries = []
            opt_labels = []
            querys = []
            for i, samples in enumerate(samples_list):
                querys.append(samples[0])
                logs.append(torch.tensor(samples[2]))
                timeseries.append(torch.tensor(samples[3]))
                opt_labels.append(torch.tensor(samples[5]))
            
            # tokens = tokenizer(querys, return_tensors="pt", padding=True, truncation=True, max_length=512)

            logs = torch.stack(logs, dim=0)
            self.logs_train_mean = logs.mean(dim=0)
            self.logs_train_std = logs.std(dim=0)

            timeseries = torch.stack(timeseries, dim=0)
            self.timeseries_train_mean = timeseries.mean(dim=[0, 2])
            self.timeseries_train_std = timeseries.std(dim=[0, 2])

            opt_labels = torch.stack(opt_labels, dim=0)
            self.opt_labels_train_mean = opt_labels.mean(dim=0)
            self.opt_labels_train_std = opt_labels.std(dim=0)

        else:
            querys = df["query"].values.tolist()
            # tokens = tokenizer(querys, return_tensors="pt", padding=True, truncation=True, max_length=512)

            self.logs_train_mean = train_dataset.logs_train_mean
            self.logs_train_std = train_dataset.logs_train_std
            self.timeseries_train_mean = train_dataset.timeseries_train_mean
            self.timeseries_train_std = train_dataset.timeseries_train_std
            self.opt_labels_train_mean = train_dataset.opt_labels_train_mean
            self.opt_labels_train_std = train_dataset.opt_labels_train_std

        for i, samples in enumerate(samples_list):
            print("dataset", i, time.time())

            node = json.loads(samples[1])['Plan']
            samples[1] = self.js_node2dict(i, node)
            print("node shape", samples[1]["x"].shape, i)

            # sam = Sample(samples[0], samples[1], torch.tensor(samples[2]), torch.tensor(samples[3]), torch.tensor(samples[4]), torch.tensor(samples[5]))
            # sam = Sample(samples[1], torch.tensor(samples[2]), torch.tensor(samples[3]), torch.tensor(samples[4]), torch.tensor(samples[5]))
            sam = {
                    "query": samples[0], 
                    # "query": {"input_ids": tokens["input_ids"][i], "token_type_ids": tokens["token_type_ids"][i], "attention_mask": tokens["attention_mask"][i]},
                   "plan": samples[1], 
                   "log": (torch.tensor(samples[2]) - self.logs_train_mean) / (self.logs_train_std + 1e-6),
                    "timeseries": (torch.tensor(samples[3]) - self.timeseries_train_mean.unsqueeze(1)) / (self.timeseries_train_std.unsqueeze(1) + 1e-6), 
                    "multilabel": torch.tensor(samples[4]), 
                    "opt_label": (torch.tensor(samples[5])  - self.opt_labels_train_mean) / (self.opt_labels_train_std + 1e-6),
                    "duration": samples[6],
                    "ori_opt_label": torch.tensor(samples[5]),
            }
            samples_data.append(sam)
            

        
        self.samples = samples_data
        self.device = device
        self.train = train


    def __getitem__(self,index):
        # label = self.samples[index][-1]
        # feature = self.samples[index][:-1]
        # print(self.samples[index])
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
    
    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        
        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict
      
    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node = 500, rel_pos_max = 20):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N,N], dtype=torch.bool)
            adj[edge_index[0,:], edge_index[1,:]] = True
            
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy((shortest_path_result)).long()

        
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        
        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }


    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(features),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
#        nodes = []
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
#            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    def traversePlan(self, plan, idx, encoding): # bfs accumulate plan

        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None #plan['Actual Rows']
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.feature = node2feature(root, encoding, None, None)
        #    print(root)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 

def node2feature(node, encoding, hist_file, table_sample):
    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((2,20-num_filter))
    filts = np.array(list(node.filterDict.values())) #cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten() 
    # mask = np.zeros(3)
    mask = np.zeros(20)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    
    # hists = filterDict2Hist(hist_file, node.filterDict, encoding)


    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    sample = np.zeros(1000)
    # if node.table_id == 0:
    #     sample = np.zeros(1000)
    # else:
    #     sample = table_sample[node.query_id][node.table]
    
    #return np.concatenate((type_join,filts,mask))
    return np.concatenate((type_join, filts, mask, table, sample))


class Sample():
    # def __init__(self, query, plan, log, timeseries, multilabel, opt_label) -> None:
    def __init__(self, plan, log, timeseries, multilabel, opt_label) -> None:

        # self.query = query
        self.plan = plan
        self.log = log
        self.timeseries = timeseries
        self.multilabel = multilabel
        self.opt_label = opt_label


class Top_dataset(data.Dataset):
    '''
        sample: [feature, label]
    '''
    def __init__(self, df, device, train=True, encoding=None, tokenizer=None, train_dataset=None):
        super().__init__()
        self.encoding = encoding
        self.treeNodes = []

        samples_list = df.values.tolist()
        samples_data = []

        if train_dataset is None:
            # 训练集归一化操作
            logs = []
            timeseries = []
            opt_labels = []
            querys = []
            for i, samples in enumerate(samples_list):
                querys.append(samples[0])
                logs.append(torch.tensor(samples[2]))
                timeseries.append(torch.tensor(samples[3]))
                opt_labels.append(torch.tensor(samples[5]))
            
            # tokens = tokenizer(querys, return_tensors="pt", padding=True, truncation=True, max_length=512)

            logs = torch.stack(logs, dim=0)
            self.logs_train_mean = logs.mean(dim=0)
            self.logs_train_std = logs.std(dim=0)

            timeseries = torch.stack(timeseries, dim=0)
            self.timeseries_train_mean = timeseries.mean(dim=[0, 2])
            self.timeseries_train_std = timeseries.std(dim=[0, 2])

            # opt_labels = torch.stack(opt_labels, dim=0)
            # self.opt_labels_train_mean = opt_labels.mean(dim=0)
            # self.opt_labels_train_std = opt_labels.std(dim=0)

        else:
            querys = df["query"].values.tolist()
            # tokens = tokenizer(querys, return_tensors="pt", padding=True, truncation=True, max_length=512)

            self.logs_train_mean = train_dataset.logs_train_mean
            self.logs_train_std = train_dataset.logs_train_std
            self.timeseries_train_mean = train_dataset.timeseries_train_mean
            self.timeseries_train_std = train_dataset.timeseries_train_std
            # self.opt_labels_train_mean = train_dataset.opt_labels_train_mean
            # self.opt_labels_train_std = train_dataset.opt_labels_train_std

        for i, samples in enumerate(samples_list):
            print("dataset", i, time.time())

            node = json.loads(samples[1])['Plan']
            samples[1] = self.js_node2dict(i, node)
            print("node shape", samples[1]["x"].shape, i)

            # sam = Sample(samples[0], samples[1], torch.tensor(samples[2]), torch.tensor(samples[3]), torch.tensor(samples[4]), torch.tensor(samples[5]))
            # sam = Sample(samples[1], torch.tensor(samples[2]), torch.tensor(samples[3]), torch.tensor(samples[4]), torch.tensor(samples[5]))
            sam = {
                    "query": samples[0], 
                    # "query": {"input_ids": tokens["input_ids"][i], "token_type_ids": tokens["token_type_ids"][i], "attention_mask": tokens["attention_mask"][i]},
                   "plan": samples[1], 
                   "log": (torch.tensor(samples[2]) - self.logs_train_mean) / (self.logs_train_std + 1e-6),
                    "timeseries": (torch.tensor(samples[3]) - self.timeseries_train_mean.unsqueeze(1)) / (self.timeseries_train_std.unsqueeze(1) + 1e-6), 
                    "multilabel": torch.tensor(samples[4]), 
                    # "opt_label": (torch.tensor(samples[5])  - self.opt_labels_train_mean) / (self.opt_labels_train_std + 1e-6),
                    "top_label": torch.tensor(samples[5]),
                    "duration": samples[6],
                    "ori_opt_label": torch.tensor(samples[5]),
            }
            samples_data.append(sam)
            

        
        self.samples = samples_data
        self.device = device
        self.train = train


    def __getitem__(self,index):
        # label = self.samples[index][-1]
        # feature = self.samples[index][:-1]
        # print(self.samples[index])
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
    
    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        
        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict
      
    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node = 500, rel_pos_max = 20):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N,N], dtype=torch.bool)
            adj[edge_index[0,:], edge_index[1,:]] = True
            
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy((shortest_path_result)).long()

        
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        
        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }


    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(features),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
#        nodes = []
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
#            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    def traversePlan(self, plan, idx, encoding): # bfs accumulate plan

        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None #plan['Actual Rows']
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.feature = node2feature(root, encoding, None, None)
        #    print(root)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 

def node2feature(node, encoding, hist_file, table_sample):
    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((2,30-num_filter))
    filts = np.array(list(node.filterDict.values())) #cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten() 
    # mask = np.zeros(3)
    mask = np.zeros(30)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    
    # hists = filterDict2Hist(hist_file, node.filterDict, encoding)


    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    sample = np.zeros(1000)
    # if node.table_id == 0:
    #     sample = np.zeros(1000)
    # else:
    #     sample = table_sample[node.query_id][node.table]
    global_plan_cost = np.array([float(node.start_up_cost), float(node.total_cost), float(node.plan_rows), float(node.plan_width)])
    local_plan_cost = np.array([float(node.start_up_cost), float(node.total_cost), float(node.plan_rows), float(node.plan_width)])
    
    
    return np.concatenate((type_join, filts, mask, table, sample, global_plan_cost, local_plan_cost), dtype=np.float64)


class PlanEncoder():
    '''
        sample: [feature, label]
    '''
    def __init__(self,data, train=True, encoding=None, tokenizer=None, train_dataset=None):
        super().__init__()
        self.encoding = Encoding(None, {'NA': 0})
        self.treeNodes = []
        self.data = data
        

    def get(self):
        from tqdm import tqdm
        embs=[]
        for i,x in tqdm(enumerate(self.data)):
            t = re.search(r'\[(.*)\]', x.replace('[\'','[\"').replace('\']','\"]').replace(', \'',', \"').replace('\'}','\"}').replace('{\'','{\"').replace('\': ','\": ').replace(': \'',': \"').replace('\', ','\", ').replace('False','false').replace('True','true').replace('\"0\'','\'0\'').replace('\", 0','\', 0'), re.DOTALL).group(0) 
            node = json.loads(t)[0]['Plan']
            # print(i+1)
            a = self.js_node2dict(i+1, node)
            embs.append(a)
        return embs

    
    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        
        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict
      
    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node = 500, rel_pos_max = 20):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N,N], dtype=torch.bool)
            adj[edge_index[0,:], edge_index[1,:]] = True
            
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy((shortest_path_result)).long()

        
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        
        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }


    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            # 'features' : torch.FloatTensor(features),
            'features' : torch.tensor(features, dtype=torch.float64),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
#        nodes = []
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
#            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    def traversePlan(self, plan, idx, encoding): # bfs accumulate plan

        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None #plan['Actual Rows']
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded, plan["Startup Cost"], plan["Total Cost"], plan["Plan Rows"], plan["Plan Width"])
        
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.feature = node2feature(root, encoding, None, None)
        #    print(root)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 
    

class Tensor_Opt_modal_dataset(data.Dataset):
    '''
        sample: [feature, label]
    '''
    def __init__(self, path='data/t2.pickle', train=True, encoding=None, tokenizer=None, train_dataset=None):
        super().__init__()
        self.encoding = encoding
        self.treeNodes = []
        file = open(path,'rb')
        sql,plan_json,time,log,opt_label,multilabel = pickle.load(file)
        encoder = PlanEncoder(plan_json)
        plan = encoder.get()
        
        self.sql = sql
        self.plan =plan
        self.time = time
        self.log = log
        self.opt_label = opt_label
        self.multilabel = multilabel
        self.opt_labels_train_mean = torch.mean(opt_label)
        self.opt_labels_train_std = torch.std(opt_label)
        # self.device = device
        # self.train = train

    def __getitem__(self,index):
        return self.sql[index],self.plan[index],self.time[index],self.log[index],self.opt_label[index],self.multilabel[index]

    def __len__(self):
        return len(self.sql)
    

# if __name__ == "__main__":
#     with open('de/data/t2.pickle','rb') as f:
#         sql,plan_json,time,log,opt_label,multilabel = pickle.load(f)

#     encoder = PlanEncoder(plan_json)
#     plan = encoder.get()
#     with open('./t2.pickle','wb') as f:
#             pickle.dump(plan,f)