import torch
import torch.nn as nn
from torch import optim

import numpy as np
import pandas as pd
import json
import scipy.stats as stats
from transformers import BertTokenizer, BertModel

from data.data import Multimodal_dataset
from model.modules.QueryFormer.utils import Encoding, collator
from models import Model
from model.modules.FuseModel.CrossTransformer import CrossTransformer
from model.modules.FuseModel.module import MultiHeadedAttention


# from llama_cpp import Llama

# llm = Llama(model_path="./model.pt")

# 训练参数
# epoch = 100
batch_size = 8
epoch = 50
# hidden_dim = 128
l_input_dim = 13
t_input_dim = 9
l_hidden_dim = 64
t_hidden_dim = 64
input_dim = 12
emb_dim = 32

fuse_num_layers = 3
fuse_ffn_dim = 128
fuse_head_size = 4
dropout = 0.1
use_fuse_model = True
# use_fuse_model = False

opt_threshold = 0.1
# layer_nums = 8
lr = 0.0003
betas = (0.9, 0.999)
mul_label_loss_fn = nn.BCELoss(reduction="mean")
opt_label_loss_fn = nn.MSELoss(reduction="mean")


class Args:
    bs = 1024
    lr = 0.001
    epochs = 200
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'mps'
    newpath = './results/full/cost/'
    to_predict = 'cost'
plan_args = Args()
device = plan_args.device

def ndcg_2(label, pred):
    pred_opt_order_index = torch.argsort(pred, dim=-1, descending=True)
    opt_label_order_index = torch.argsort(label, dim=-1, descending=True)
    print("pred_opt_order_index", pred_opt_order_index)

    pred_opt_ndcg = torch.gather(label, dim=-1, index=pred_opt_order_index)
    label_opt_ndcg = torch.gather(label, dim=-1, index=opt_label_order_index)
    
    log2_table = torch.log2(torch.arange(2, 102))
    def dcg_at_n(rel):
        dcg = torch.sum(torch.divide(torch.pow(2, rel) - 1, log2_table[:rel.shape[1]].unsqueeze(0)), dim=-1)
        return dcg
    label_s = dcg_at_n(label_opt_ndcg) + 1e-10
    pred_s = dcg_at_n(pred_opt_ndcg)
    print((pred_s / label_s).shape)
    return torch.mean(pred_s / label_s)

def top1_margin(label, pred):
    pred_opt_order, pred_index = torch.sort(pred, dim=-1, descending=True)
    opt_label_order, label_index = torch.sort(label, dim=-1, descending=True)
    # print("pred_opt_order_index", pred_opt_order_index)
    all_label = opt_label_order.shape[0]
    top1_cor = 0
    for i in range(opt_label_order.shape[0]):
        if opt_label_order[i][0] < 0.05 and pred_opt_order[i][0] < 0.05:
            top1_cor += 1
        elif pred_index[i][0] == label_index[i][0]:
            top1_cor += 1
        elif opt_label_order[i][0] - label[i][pred_index[i][0]] < 0.01:
            top1_cor += 1 
    
    return top1_cor / (1.0 * all_label)

def mrr(pred, label):
    mrr_res = 0
    
    for i, lab in enumerate(label):
        mrr_res += (1 / (pred[i].index(lab[0])+1))
    return mrr_res / len(label)

def extended_tau_2(list_a, list_b, all_label):
    """ Calculate the extended Kendall tau from two lists. """
    # fill_len = len(list_b) if len(list_b) < len(list_a) else len(list_a)
    if len(list_a) < len(list_b):
        for i in range(len(list_b) - len(list_a)):
            list_a.append((set(all_label) - set(list_a) - set(list_b)).pop())
    if len(list_a) == 0 and len(list_a) == len(list_b):
        return 1.0
    if len(list_b) == 0:
        return 0.0
    ranks = join_ranks(create_rank(list_a), create_rank(list_b)).fillna(12)
    dummy_df = pd.DataFrame([{'rank_a': 12, 'rank_b': 12} for i in range(2*len(list_a)-len(ranks))])
    total_df = pd.concat([ranks, dummy_df])
    return scale_tau(len(list_a), stats.kendalltau(total_df['rank_a'], total_df['rank_b'])[0])

def scale_tau(length, value):
    """ Scale an extended tau correlation such that it falls in [-1, +1]. """
    n_0 = 2*length*(2*length-1)
    n_a = length*(length-1)
    n_d = n_0 - n_a
    min_tau = (2.*n_a - n_0) / (n_d)
    return 2*(value-min_tau)/(1-min_tau) - 1

def create_rank(a):
    """ Convert an ordered list to a DataFrame with ranks. """
    return pd.DataFrame(
                zip(a, range(len(a))),
                columns=['key', 'rank'])\
            .set_index('key')

def join_ranks(rank_a, rank_b):
    """ Join two rank DataFrames. """
    return rank_a.join(rank_b, lsuffix='_a', rsuffix='_b', how='outer')

def evaluate_tau(label_list, pred_list):
    tau_res = []
    all_label = list(range(12))
    for i in range(len(label_list)):
        tau_res.append(extended_tau_2(label_list[i], pred_list[i], all_label))
    return torch.tensor(tau_res).mean().item()

# query tokenizer
def evaluate():

    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")

    # 读取数据，处理输出
    # path = "RootcauseSQL/data/rootcause_train_test.csv"
    # path = "RootcauseSQL/data/rootcause_train_test_threshold_10.csv"
    # path = "RootcauseSQL/data/rootcause_all_raw_log_threshold_10_train_test_3.csv"
    path = "RootcauseSQL/data/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500.csv"
    df = pd.read_csv(path)
    print(df.columns)
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
        return x[feature_k].values.tolist()
    df["log_all"] = df.apply(get_log, axis=1)

    df["multilabel"] = df["multilabel"].apply(lambda x: json.loads(x))
    df["opt_label"] = df["opt_label"].apply(lambda x: json.loads(x))

    print(df["log_all"].iloc[0])

    encoding = Encoding(None, {'NA': 0})

    # df = df[:10]
    # 获取训练数据、测试数据
    df_train = df[df["dataset_cls"] == "train"]
    df_test = df[df["dataset_cls"] == "test"]
    print(df_test.shape)

    # 创建数据集
    train_dataset = Multimodal_dataset(df_train[["query", "plan_json", "log_all", "timeseries", "multilabel", "opt_label", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer)
    test_dataset = Multimodal_dataset(df_test[["query", "plan_json", "log_all", "timeseries", "multilabel", "opt_label", "duration"]], device=device, encoding=encoding, tokenizer=tokenizer, train_dataset=train_dataset)

    print("load dataset over")
    # 数据加载器 
    # todo 分训练集和测试集
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("start train")

    sql_model = BertModel.from_pretrained("./bert-base-uncased")

    fuse_model = None
    if use_fuse_model:
        multihead_attn_modules_cross_attn = nn.ModuleList(
                [MultiHeadedAttention(fuse_head_size, emb_dim, dropout=dropout)
                for _ in range(fuse_num_layers)])
        fuse_model = CrossTransformer(num_layers=fuse_num_layers, d_model=emb_dim, heads=fuse_head_size, d_ff=fuse_ffn_dim, dropout=dropout, attn_modules=multihead_attn_modules_cross_attn)
    model = Model(t_input_dim, l_input_dim, l_hidden_dim, t_hidden_dim, emb_dim, device=device, plan_args=plan_args, sql_model=sql_model, cross_model=fuse_model)

    # model.load_state_dict(torch.load("saved_models/{}.pt".format("rootcause_all_raw_log_threshold_10_fuse_model_cross_attn")))
    # model.load_state_dict(torch.load("saved_models/{}.pt".format("threshold_10_fuse_model")))
    # model.load_state_dict(torch.load("saved_models/{}.pt".format("rootcause_old_threshold_10_cross_attn_2")))
    # model.load_state_dict(torch.load("saved_models/{}.pt".format("rootcause_all_raw_log_threshold_10_new")))
    model.load_state_dict(torch.load("saved_models/cross_attn_simple/{}.pt".format("rootcause_new_threshold_10_cross_attn_sigmoid_log_attn_5")))

    model.to(device)

    # todo test方案，以及给出优化方法

    sum_correct = 0

    label_list = []
    pred_list = []
    label_rows = 0
    test_idx = 0
    test_len = len(test_dataset)
    MSE_loss = 0
    # pred_rows = 0

    test_pred_opt = None
    model.eval()

    for index, input1 in enumerate(test_dataloader):
        sql, plan, time, log, multilabel, opt_label = input1["query"], input1["plan"], input1["timeseries"], input1["log"], input1["multilabel"], input1["opt_label"]
        sql = tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # 去除plan 多出的一维
        plan['x'] = plan['x'].squeeze(1)
        plan["attn_bias"] = plan["attn_bias"].squeeze(1)
        plan["rel_pos"] = plan["rel_pos"].squeeze(1)
        plan["heights"] = plan["heights"].squeeze(1)

        plan['x'] = plan['x'].squeeze(1).to(device)
        plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
        plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
        plan["heights"] = plan["heights"].squeeze(1).to(device)
        sql.to(device)
        time = time.to(device)
        log = log.to(device)
        multilabel = multilabel.to(device)
        opt_label = opt_label.to(device)
        
        pred_label, pred_opt = model(sql, plan, time, log)

        # a = pred_opt.gt(0) & multilabel

        # 输出根因标签，优化时间。
        pred_opt = (pred_opt * train_dataset.opt_labels_train_std.to(device) + train_dataset.opt_labels_train_mean.to(device))
        duration = input1["duration"].to(device)
        # pred_multilabel = pred_opt.gt(0).nonzero()
        opt_min_duration = (duration * opt_threshold).unsqueeze(1)
        pred_multilabel = pred_opt.gt(opt_min_duration).nonzero()
        pred_opt_time = pred_opt[pred_multilabel[:, 0], pred_multilabel[:, 1]]
        sorted_time_index = torch.argsort(pred_opt, dim=1, descending=True)

        # 设置阈值并排序优化时间
        opt_label_m = (opt_label * train_dataset.opt_labels_train_std.to(device) + train_dataset.opt_labels_train_mean.to(device))
        opt_label = input1["ori_opt_label"].to(device)
        label_multilabel = opt_label.gt(opt_min_duration).nonzero()
        label_sorted_time_index = torch.argsort(opt_label, dim=1, descending=True)


        # pred_opt_time = (pred_opt[pred_multilabel[:, 0], pred_multilabel[:, 1]] * train_dataset.opt_labels_train_std + train_dataset.opt_labels_train_mean)

        # pred_opt.gt(0)

        if test_pred_opt is None:
            test_pred_opt = pred_opt
        else:
            test_pred_opt = torch.cat((test_pred_opt, pred_opt), dim=0)

        # 计算mAP@k
        def patk(actual, pred, k):
            #we return 0 if k is 0 because 
            #   we can't divide the no of common values by 0 
            if k == 0:
                return 0
            #taking only the top k predictions in a class 
            k_pred = pred[:k]

            # taking the set of the actual values 
            actual_set = set(actual)
            # print(list(actual_set))
            # taking the set of the predicted values 
            pred_set = set(k_pred)
            # print(list(pred_set))
            
            # 求预测值与真实值得交集
            common_values = actual_set.intersection(pred_set)
            # print(common_values)

            if len(pred[:k]) == 0:
                return 0
            return len(common_values)/len(pred[:k])
        
        def apatk(acutal, pred, k):
            #creating a list for storing the values of precision for each k 
            precision_ = []
            for i in range(1, k+1):
                #calculating the precision at different values of k 
                #      and appending them to the list 
                precision_.append(patk(acutal, pred, i))

            #return 0 if there are no values in the list
            if len(precision_) == 0:
                return 0 

            #returning the average of all the precision values
            return np.mean(precision_)
        
        def mapk(acutal, pred, k):

            #creating a list for storing the Average Precision Values
            average_precision = []
            #interating through the whole data and calculating the apk for each 
            for i in range(len(acutal)):
                ap = apatk(acutal[i], pred[i], k)
                # print(f"AP@k: {ap}")
                average_precision.append(ap)

            #returning the mean of all the data
            return np.mean(average_precision)

        start_row = 0
        label_list.append([])
        # 多标签无顺序 label
        # for row, col in multilabel.nonzero():
        #     if row == start_row:
        #         label_list[label_rows+row].append(col.item())
        #     else:
        #         start_row += 1
        #         while row != start_row:
        #             start_row += 1
        #             label_list.append([])
        #         label_list.append([col.item()])
        # label 多标签优化时间排序
        kk_i = 0
        for row, col in label_multilabel:
            if row == start_row:
                label_list[batch_size * test_idx + row].append(label_sorted_time_index[row][kk_i].item())
                kk_i += 1
            else:
                # start_row += 1
                kk_i = 0
                while row != start_row:
                    start_row += 1
                    label_list.append([])
                # label_list.append([label_sorted_time_index[row][kk_i].item()])
                label_list[batch_size * test_idx + row].append(label_sorted_time_index[row][kk_i].item())
                kk_i += 1

        # print(label_list)
        len_data = (batch_size * (test_idx+1) if batch_size * (test_idx+1) < test_len else test_len)
        label_len = len(label_list)
        if label_len < len_data:
            print("label_len: ", label_len)
            for i in range(len_data - label_len):
                label_list.append([])

        start_row = 0
        pred_list.append([])

        kk_i = 0
        
        for row, col in pred_multilabel:
            if row == start_row:
                # pred_list[pred_rows+row].append(col.item())
                pred_list[batch_size * test_idx + row].append(sorted_time_index[row][kk_i].item())
                kk_i += 1
            else:
                # start_row += 1
                kk_i = 0
                while row != start_row:
                    start_row += 1
                    pred_list.append([])
                # pred_list.append([col.item()])
                # pred_list.append([sorted_time_index[row][kk_i].item()])
                pred_list[batch_size * test_idx + row].append(sorted_time_index[row][kk_i].item())
                kk_i += 1
        
        pred_len = len(pred_list)
        if pred_len < len_data:
            print("pred_len: ", pred_len)
            for i in range(len_data - pred_len):
                pred_list.append([])
        
        # pred_rows = len(pred_list)
        test_idx += 1

        # 计算MSE
        MSE_loss += torch.pow(pred_opt - opt_label_m, 2).mean(-1).sum(0).item()
        # MSE_loss += opt_label_loss_fn(pred_opt, opt_label_m).item()

        # print(pred_list)

        # for i in range(len(pred_list)):
        #     patk_ = patk(label_list[i], pred_list[i], 1)
        #     print(patk_)

    print("mapk@3: ", mapk(label_list, pred_list, 3))
    print("mapk@5: ", mapk(label_list, pred_list, 5))

    all_right_cnt = 0
    for i in range(len(label_list)):
        # print(label_list[i] == pred_list)
        if label_list[i] == pred_list[i]:
            all_right_cnt += 1
    print("all_right_cnt rate: ", all_right_cnt / df_test.shape[0])

    label_dict = {}
    cls_cor = {}
    sig_cor = {}
    sig_label = {}
    sig_pred = {}
    for i, v in enumerate(label_list):
        if str(label_list[i]) == str(pred_list[i]):
            cls_cor[str(v)] = cls_cor.get(str(v), 0) + 1
        label_dict[str(v)] = label_dict.get(str(v), 0) + 1

        label_set = set(label_list[i])
        pred_set = set(pred_list[i])
        cor_set = label_set & pred_set
        for kk in cor_set:
            sig_cor[str(kk)] = sig_cor.get(str(kk), 0) + 1

        for kk in label_set:
            sig_label[str(kk)] = sig_label.get(str(kk), 0) + 1

        for kk in pred_set:
            sig_pred[str(kk)] = sig_pred.get(str(kk), 0) + 1

    pred_dict = {}
    for i, v in enumerate(pred_list):
        pred_dict[str(v)] = pred_dict.get(str(v), 0) + 1

    print("label_dict: ", label_dict)
    print("pred_dict: ", pred_dict)
    print("cls_cor", cls_cor)
    print("sig_cor: ", sig_cor)
    print("sig_label: ", sig_label)
    print("sig_pred: ", sig_pred)

    print("MSE_loss: ", MSE_loss / len(test_dataset))
    print("tau: ", evaluate_tau(label_list, pred_list))

if __name__ == "__main__":
    # evaluate()
    # print(evaluate_tau([[0, 1, 2, 3, 4]], [[4, 3, 2, 1, 0]]))
    # print(evaluate_tau([[0, 1, 2]], [[0, 1, 2, 3]]))
    print(evaluate_tau([[0, 1, 2]], [[0, 2, 1]]))
    print(evaluate_tau([[0, 1, 2]], [[0, 1, 3]]))