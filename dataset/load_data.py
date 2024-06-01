import torch
import torch.nn as nn
from torch import optim

import numpy as np
import pandas as pd
import json
from transformers import BertTokenizer

from dataset.data_tensor import Tensor_Opt_modal_dataset, Tensor_Opt_modal_pretrain_dataset
from model.modules.QueryFormer.utils import Encoding


def load_dataset(data_path, batch_size = 8, device="cpu"):
    # query tokenizer
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")

    path = data_path
    print("data path", path)
    # df = pd.read_csv(path)
    
    df = pd.read_pickle(path)
    print(df.columns)
    print(df.shape)
    # dataset = df[["query", "", "plan_json", "opt_label"]]
    def get_timeseries(x):
        labels = ['CPU_percent', 'IO_read_standard', 'IO_write_standard', 'IO_read',
        'IO_write', 'Memory_percent', 'Memory_used']
        timeseries = []
        for a in labels:
            if pd.isna(x[a]):
                print("error")
            else:
                timeseries.append(json.loads(x[a]))
        return timeseries
    df["timeseries"] = df.apply(get_timeseries, axis=1)
    # print(df["timeseries"].iloc[0])

    def get_log(x):
        feature_k = ["duration", "result_rows", "result_bytes", "read_bytes", "read_rows", "optimization_cost", "start_query_cost", "affected_rows", "affected_bytes", "memory_bytes", "shuffle_bytes", "cpu_time_ms", "physical_reads"]
        x[feature_k] = x[feature_k].fillna(0.0)
        for k in feature_k:
            if x[k] == "\\N":
                x[k] = 0.0
            x[k] = float(x[k])
        return x[feature_k].values.tolist()
    df["log_all"] = df.apply(get_log, axis=1)

    def set_m_lab(x):
        labels = ['JoinOrder', 'update table', 'distributionKey', 'index', 'dictionary']
        m_l = [1 if ((x["duration"] - x[lab]) > 0.05 * x["duration"]) else 0 for lab in labels]
        return m_l
    # df["multilabel"] = df["multilabel"].apply(lambda x: json.loads(x))
    df["multilabel"] = df.apply(set_m_lab, axis=1)
    df["opt_label"] = df["opt_label"].apply(lambda x: json.loads(x))

    # print(df["log_all"].iloc[0])
    # print("multilabel", df["multilabel"])

    encoding = Encoding(None, {'NA': 0})

    # df = df[:10]
    # 获取训练数据、测试数据
    df_train = df[df["dataset_cls"] == "train"]
    df_test = df[df["dataset_cls"] == "test"]
    # df_test = df[df["dataset_cls"] == "train"]
    # df_train = df[df["dataset_cls"] == "test"]
    print(df_test.shape)

    # 创建数据集
    # train_dataset = Tensor_Opt_modal_dataset(df_train[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label", "duration", "mask_opt_label"]], device=device, encoding=encoding, tokenizer=tokenizer)
    train_dataset = Tensor_Opt_modal_dataset(df_train[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer)
    # test_dataset = Tensor_Opt_modal_dataset(df_test[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label", "duration", "mask_opt_label"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)
    test_dataset = Tensor_Opt_modal_dataset(df_test[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)

    print("load dataset over")
    # 数据加载器 
    # todo 分训练集和测试集
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader, len(train_dataset), len(test_dataset), train_dataset 


def load_dataset_valid(data_path, batch_size = 8, device="cpu"):
    # query tokenizer
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")

    path = data_path
    print("data path", path)
    # df = pd.read_csv(path)
    
    df = pd.read_pickle(path)
    # df = df[(df["dataset_cls"] == "train") | (df["dataset_cls"] == "valid")]
    # df2 = pd.read_pickle("data/yewu/hgprecn-cn-tl32b9p7800i_test_norm.pickle")
    # df2.loc[:, "opt_label_rate"] = json.dumps([0.0] * 6)
    # df2["opt_label_rate"] = df2["opt_label_rate"].map(lambda x: json.loads(x))
    
    # df = pd.concat([df, df2])
    
    # print(df.columns)
    # print(df.shape)
    # dataset = df[["query", "", "plan_json", "opt_label"]]
    def get_timeseries(x):
        labels = ['CPU_percent', 'IO_read_standard', 'IO_write_standard', 'IO_read',
        'IO_write', 'Memory_percent', 'Memory_used']
        timeseries = []
        for a in labels:
            if pd.isna(x[a]):
                print("error")
            else:
                timeseries.append(json.loads(x[a]))
        return timeseries
    df["timeseries"] = df.apply(get_timeseries, axis=1)
    # print(df["timeseries"].iloc[0])

    def get_log(x):
        feature_k = ["duration", "result_rows", "result_bytes", "read_bytes", "read_rows", "optimization_cost", "start_query_cost", "affected_rows", "affected_bytes", "memory_bytes", "shuffle_bytes", "cpu_time_ms", "physical_reads"]
        x[feature_k] = x[feature_k].fillna(0.0)
        for k in feature_k:
            if x[k] == "\\N":
                x[k] = 0.0
            x[k] = float(x[k])
        return x[feature_k].values.tolist()
    df["log_all"] = df.apply(get_log, axis=1)

    def set_m_lab(x):
        labels = ['JoinOrder', 'update table', 'distributionKey', 'index', 'dictionary']
        m_l = [1 if ((x["duration"] - x[lab]) > 0.05 * x["duration"]) else 0 for lab in labels]
        return m_l
    # df["multilabel"] = df["multilabel"].apply(lambda x: json.loads(x))
    df["multilabel"] = df.apply(set_m_lab, axis=1)
    # df["opt_label"] = df["opt_label"].apply(lambda x: json.loads(x))

    # print(df["log_all"].iloc[0])
    # print("multilabel", df["multilabel"])

    encoding = Encoding(None, {'NA': 0})

    # df = df[:10]
    # 获取训练数据、测试数据
    df_train = df[df["dataset_cls"] == "train"]
    df_test = df[df["dataset_cls"] == "test"]
    df_test = df_test[df_test["query"].str.contains("oy")]
    # df_test = df_test[df_test["dataset_cls"]=="test"]
    df_valid = df[df["dataset_cls"] == "valid"]
    # df_test = df[df["dataset_cls"] == "train"]
    # df_train = df[df["dataset_cls"] == "test"]
    print(df_test.shape)

    # 创建数据集
    # train_dataset = Tensor_Opt_modal_dataset(df_train[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label", "duration", "mask_opt_label"]], device=device, encoding=encoding, tokenizer=tokenizer)
    train_dataset = Tensor_Opt_modal_dataset(df_train[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer)
    # test_dataset = Tensor_Opt_modal_dataset(df_test[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label", "duration", "mask_opt_label"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)
    test_dataset = Tensor_Opt_modal_dataset(df_test[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)
    
    valid_dataset = Tensor_Opt_modal_dataset(df_valid[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)

    print("load dataset over")
    # 数据加载器 
    # todo 分训练集和测试集
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader, valid_dataloader, len(train_dataset), len(test_dataset), len(valid_dataset), train_dataset 



def padding_plan(plan, max_len):
    padding = torch.zeros(1, max_len - plan.shape[1], 768)
    return torch.cat([plan, padding], dim=1)

def collate_fn(batch):
    querys, plans, logs, timeseries, multilabels, opt_labels, durations, ori_opt_labels = [], {}, [], [], [], [], [], []
    plans_x, plans_attn_bias, plans_rel_pos, plans_heights = [], [], [], []
    max_len_plan = 0
    for sample in batch:
        querys.append(sample["query"])
        p = sample["plan"]
        if max_len_plan < p["x"].shape[1]: max_len_plan = p["x"].shape[1]
        plans_x.append(p["x"])
        plans_attn_bias.append(p["attn_bias"])
        plans_rel_pos.append(p["rel_pos"])
        plans_heights.append(p["heights"])
        
        logs.append(sample["log"])
        timeseries.append(sample["timeseries"])
        multilabels.append(sample["multilabel"])
        opt_labels.append(sample["opt_label"])
        durations.append(sample["duration"])
        ori_opt_labels.append(sample["ori_opt_label"])
    
    max_len_plan = 500
    for i in range(len(plans_x)):
        plans_x[i] = padding_plan(plans_x[i], max_len_plan)
    plans["x"] = torch.stack(plans_x)
    plans["attn_bias"] = torch.stack(plans_attn_bias)
    plans["rel_pos"] = torch.stack(plans_rel_pos)
    plans["heights"] = torch.stack(plans_heights)
    return {
        "query": querys,
        "plan": plans,
        # "plan": torch.stack(plans),
        "log": torch.stack(logs),
        "timeseries": torch.stack(timeseries),
        "multilabel": torch.stack(multilabels),
        "opt_label": torch.stack(opt_labels),
        "duration": torch.tensor(durations),
        "ori_opt_label": torch.stack(ori_opt_labels)
    }


def load_dataset_pretrain_valid(data_path, batch_size = 8, device="cpu"):
    # query tokenizer
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")

    path = data_path
    print("data path", path)
    df = pd.read_csv(path)
    
    # df = pd.read_pickle(path)
    print(df.columns)
    print(df.shape)
    def get_timeseries(x):
        labels = ['CPU_percent', 'IO_read_standard', 'IO_write_standard', 'IO_read',
        'IO_write', 'Memory_percent', 'Memory_used']
        timeseries = []
        for a in labels:
            if pd.isna(x[a]):
                print("error")
            else:
                timeseries.append(json.loads(x[a]))
        return timeseries
    df["timeseries"] = df.apply(get_timeseries, axis=1)
    # print(df["timeseries"].iloc[0])

    def get_log(x):
        feature_k = ["duration", "result_rows", "result_bytes", "read_bytes", "read_rows", "optimization_cost", "start_query_cost", "affected_rows", "affected_bytes", "memory_bytes", "shuffle_bytes", "cpu_time_ms", "physical_reads"]
        x[feature_k] = x[feature_k].fillna(0.0)
        for k in feature_k:
            if x[k] == "\\N":
                x[k] = 0.0
            x[k] = float(x[k])
        return x[feature_k].values.tolist()
    df["log_all"] = df.apply(get_log, axis=1)

    def set_m_lab(x):
        labels = ['JoinOrder', 'update table', 'distributionKey', 'index', 'dictionary']
        m_l = [1 if ((x["duration"] - x[lab]) > 0.05 * x["duration"]) else 0 for lab in labels]
        return m_l
    # df["multilabel"] = df["multilabel"].apply(lambda x: json.loads(x))
    df["multilabel"] = df.apply(set_m_lab, axis=1)
    df["opt_label_rate"] = df["opt_label_rate"].apply(lambda x: json.loads(x))

    # print(df["log_all"].iloc[0])
    # print("multilabel", df["multilabel"])

    encoding = Encoding(None, {'NA': 0})

    # df = df[:10]
    # 获取训练数据、测试数据
    df_train = df[df["dataset_cls"] == "train"]
    df_test = df[df["dataset_cls"] == "test"]
    df_valid = df[df["dataset_cls"] == "valid"]
    # df_test = df[df["dataset_cls"] == "train"]
    # df_train = df[df["dataset_cls"] == "test"]
    print(df_test.shape)

    # 创建数据集
    train_dataset = Tensor_Opt_modal_dataset(df_train[["query", "plan_json", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer)
    test_dataset = Tensor_Opt_modal_dataset(df_test[["query", "plan_json", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)
    valid_dataset = Tensor_Opt_modal_dataset(df_valid[["query", "plan_json", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)

    print("load dataset over")
    # 数据加载器 
    # todo 分训练集和测试集
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader, valid_dataloader, len(train_dataset), len(test_dataset), len(valid_dataset), train_dataset



def load_dataset_pretrain_valid2(data_path, batch_size = 8, device="cpu"):
    # query tokenizer
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")

    path = data_path
    print("data path", path)
    # df = pd.read_csv(path)
    
    df = pd.read_pickle(path)
    print(df.columns)
    print(df.shape)
    def get_timeseries(x):
        labels = ['CPU_percent', 'IO_read_standard', 'IO_write_standard', 'IO_read',
        'IO_write', 'Memory_percent', 'Memory_used']
        timeseries = []
        for a in labels:
            if pd.isna(x[a]):
                print("error")
            else:
                timeseries.append(json.loads(x[a]))
        return timeseries
    df["timeseries"] = df.apply(get_timeseries, axis=1)
    # print(df["timeseries"].iloc[0])

    def get_log(x):
        feature_k = ["duration", "result_rows", "result_bytes", "read_bytes", "read_rows", "optimization_cost", "start_query_cost", "affected_rows", "affected_bytes", "memory_bytes", "shuffle_bytes", "cpu_time_ms", "physical_reads"]
        x[feature_k] = x[feature_k].fillna(0.0)
        for k in feature_k:
            if x[k] == "\\N":
                x[k] = 0.0
            x[k] = float(x[k])
        return x[feature_k].values.tolist()
    df["log_all"] = df.apply(get_log, axis=1)

    def set_m_lab(x):
        labels = ['JoinOrder', 'update table', 'distributionKey', 'index', 'dictionary']
        m_l = [1 if ((x["duration"] - x[lab]) > 0.05 * x["duration"]) else 0 for lab in labels]
        return m_l
    # df["multilabel"] = df["multilabel"].apply(lambda x: json.loads(x))
    df["multilabel"] = df.apply(set_m_lab, axis=1)
    # df["opt_label_rate"] = df["opt_label_rate"].apply(lambda x: json.loads(x))

    # print(df["log_all"].iloc[0])
    # print("multilabel", df["multilabel"])

    encoding = Encoding(None, {'NA': 0})

    # df = df[:10]
    # 获取训练数据、测试数据
    df_train = df[df["dataset_cls"] == "train"]
    df_test = df[df["dataset_cls"] == "test"]
    df_valid = df[df["dataset_cls"] == "valid"]
    # df_test = df[df["dataset_cls"] == "train"]
    # df_train = df[df["dataset_cls"] == "test"]
    print(df_test.shape)

    # 创建数据集
    train_dataset = Tensor_Opt_modal_dataset(df_train[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer)
    test_dataset = Tensor_Opt_modal_dataset(df_test[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)
    valid_dataset = Tensor_Opt_modal_dataset(df_valid[["query", "json_plan_tensor", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)
    
    # train_dataset = Tensor_Opt_modal_dataset(df_train[["query", "plan_json", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer)
    # test_dataset = Tensor_Opt_modal_dataset(df_test[["query", "plan_json", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)
    # valid_dataset = Tensor_Opt_modal_dataset(df_valid[["query", "plan_json", "log_all", "timeseries", "multilabel", "opt_label_rate", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)

    print("load dataset over")
    # 数据加载器 
    # todo 分训练集和测试集
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader, valid_dataloader, len(train_dataset), len(test_dataset), len(valid_dataset), train_dataset 