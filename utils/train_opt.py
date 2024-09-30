import torch
import torch.nn as nn
from torch import optim

import numpy as np
import pandas as pd
import json
from transformers import BertTokenizer, BertModel

from dataset.data_tensor import Tensor_Opt_modal_dataset
from model.modules.QueryFormer.utils import Encoding, collator
from models import Model, ThreeMulModel
from other_models import GaussModel
from model.modules.FuseModel.CrossTransformer import CrossTransformer, GlobalLocalCrossTransformer
from model.modules.FuseModel.module import MultiHeadedAttention, ThreeMultiHeadedAttention, LocalGlobalMultiHeadedAttention, LogTimesMultiHeadedAttention
from top_model import TopModel, TopRealEstModel, OnlyPlanModel
from times_model import TimeSeriesModel, TimeSoftmaxModel

from evaluate import evaluate_tau


# from llama_cpp import Llama

# llm = Llama(model_path="./model.pt")

# 训练参数
# epoch = 100
# batch_size = 8
# epoch = 50
# # hidden_dim = 128
# l_input_dim = 13
# t_input_dim = 9
# l_hidden_dim = 64
# t_hidden_dim = 64
# input_dim = 12
# emb_dim = 32

# fuse_num_layers = 3
# fuse_ffn_dim = 128
# fuse_head_size = 4
# dropout = 0.1
# use_fuse_model = True
# # use_fuse_model = False
# # 两种模型选择[有plan，无plan] 区别在于是否将plan解析成树，解析成树非常耗时
# # ["cross_attn_no_plan", "cross_attn", "open_gauss"]
# select_model = "cross_attn"
# # ['no_plan', 'plan']
# dataset_select = "plan"

# lr = 0.0003
# # lr = 0.001
# betas = (0.9, 0.999)

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
    device = 'cuda:1'
    newpath = './results/full/cost/'
    to_predict = 'cost'
    input_emb = 1063
    

def load_dataset(data_path, batch_size = 8):
    plan_args = Args()
    device = plan_args.device
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


def train(data_path, select_model, dataset_select, use_fuse_model, train_dataloader, test_dataloader, train_len, test_len, train_dataset, betas = (0.9, 0.999), lr = 0.0003, batch_size = 8, epoch = 50, l_input_dim = 13,
        t_input_dim = 9,l_hidden_dim = 64, t_hidden_dim = 64, input_dim = 12, emb_dim = 32, fuse_num_layers = 3,
        fuse_ffn_dim = 128, fuse_head_size = 4, dropout = 0.1, opt_threshold = 0.1, time_t=1., model_path=None, res_path=None, use_metrics=True, use_log=True, use_softmax=True, multi_head="all_cross"):
    plan_args = Args()
    device = plan_args.device

    # query tokenizer
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    mul_label_loss_fn = nn.BCELoss(reduction="mean")
    opt_label_loss_fn = nn.MSELoss(reduction="mean")
    # opt_label_loss_fn = nn.CrossEntropyLoss(reduction="mean")

    print("start train")

    sql_model = BertModel.from_pretrained("./bert-base-uncased")
    time_model = TimeSoftmaxModel(t_input_dim, t_hidden_dim, emb_dim, device=device, t=time_t, use_softmax=use_softmax)

    fuse_model = None
    # use_metrics = False
    # use_log = False
    if use_fuse_model:
        if multi_head == "all_cross":
            multihead_attn_modules_cross_attn = nn.ModuleList(
                    [MultiHeadedAttention(fuse_head_size, emb_dim, dropout=dropout, use_metrics=use_metrics, use_log=use_log)
                    for _ in range(fuse_num_layers)])
            fuse_model = CrossTransformer(num_layers=fuse_num_layers, d_model=emb_dim, heads=fuse_head_size, d_ff=fuse_ffn_dim, dropout=dropout, attn_modules=multihead_attn_modules_cross_attn)
            
        elif multi_head == "local_global_cross":
            multihead_attn_modules_cross_attn = nn.ModuleList(
                    [LocalGlobalMultiHeadedAttention(fuse_head_size, emb_dim, dropout=dropout, use_metrics=False, use_log=False)
                    for _ in range(fuse_num_layers)])
            attn_l_m_modules = nn.ModuleList(
                    [LogTimesMultiHeadedAttention(fuse_head_size, emb_dim, dropout=dropout, use_metrics=use_metrics, use_log=use_log)
                    for _ in range(fuse_num_layers)])
            fuse_model = GlobalLocalCrossTransformer(num_layers=fuse_num_layers, d_model=emb_dim, heads=fuse_head_size, d_ff=fuse_ffn_dim, dropout=dropout, attn_modules=multihead_attn_modules_cross_attn, attn_l_m=attn_l_m_modules)
    
    if select_model == "TopRealEstModel":
        model = TopRealEstModel(t_input_dim, l_input_dim, l_hidden_dim, t_hidden_dim, emb_dim, device=device, plan_args=plan_args, sql_model=sql_model, cross_model=fuse_model, time_model=time_model)
    elif select_model == "all_fuse":
        # model = TopModel(t_input_dim, l_input_dim, l_hidden_dim, t_hidden_dim, emb_dim, device=device, plan_args=plan_args, sql_model=sql_model, cross_model=fuse_model, time_model=time_model)
        model = OnlyPlanModel(t_input_dim, l_input_dim, l_hidden_dim, t_hidden_dim, emb_dim, device=device, plan_args=plan_args, sql_model=sql_model, cross_model=fuse_model, time_model=time_model)

    # model.load_state_dict(torch.load(model_path))
    model.to(device)
    opt = optim.Adam(model.parameters(), lr, betas)

    for i in range(epoch):
        epoch_loss = 0
        for index, input1 in enumerate(train_dataloader):
            opt.zero_grad()
            sql, plan, time, log, multilabel, opt_label = input1["query"], input1["plan"], input1["timeseries"], input1["log"], input1["multilabel"], input1["opt_label"]
            sql = tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # 去除plan 多出的一维
            
            if select_model == "cross_attn_no_plan":
                plan = None
            else:
                plan['x'] = plan['x'].squeeze(1).to(device).to(torch.float32)
                plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
                plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
                plan["heights"] = plan["heights"].squeeze(1).to(device)
            sql.to(device)
            time = time.to(device)
            log = log.to(device)
            # multilabel = multilabel.to(device)
            opt_label = opt_label.to(device)

            pred_label, pred_opt = model(sql, plan, time, log)
            # output1 = output1.squeeze(1)
            # pred_opt
            pred_opt = pred_opt.view(-1, 1)
            opt_label = opt_label.view(-1, 1)
            # label = label.to(torch.long)
            # mul_label_loss = mul_label_loss_fn(pred_label, multilabel.to(torch.float32))
            opt_label_loss = opt_label_loss_fn(pred_opt, opt_label)

            # loss = mul_label_loss + opt_label_loss
            loss = opt_label_loss
            epoch_loss += loss.item()
            loss.backward()
            opt.step()
        print(i, epoch_loss / train_len)
        # break

    # torch.save(model.state_dict(), "saved_models/cross_attn_2023_11_30_dictionary/{}.pt".format("cross_attn_all"))
    torch.save(model.state_dict(), model_path)


    # todo test方案，以及给出优化方法

    sum_correct = 0

    label_list = []
    pred_list = []
    label_rows = 0
    test_idx = 0
    test_len = test_len
    MSE_loss = 0
    # pred_rows = 0

    test_pred_opt = None
    # model.load_state_dict(torch.load(model_path))
    model.eval()

    for index, input1 in enumerate(test_dataloader):
        sql, plan, time, log, multilabel, opt_label = input1["query"], input1["plan"], input1["timeseries"], input1["log"], input1["multilabel"], input1["opt_label"]
        sql = tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # 去除plan 多出的一维
        # plan['x'] = plan['x'].squeeze(1)
        # plan["attn_bias"] = plan["attn_bias"].squeeze(1)
        # plan["rel_pos"] = plan["rel_pos"].squeeze(1)
        # plan["heights"] = plan["heights"].squeeze(1)

    
        sql.to(device)
        time = time.to(device)
        log = log.to(device)
        multilabel = multilabel.to(device)
        # opt_label = opt_label.to(device)
        if select_model == "cross_attn_no_plan":
            plan = None
        else:
            plan['x'] = plan['x'].squeeze(1).to(device).to(torch.float32)
            plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
            plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
            plan["heights"] = plan["heights"].squeeze(1).to(device)
        
        pred_label, pred_opt_raw = model(sql, plan, time, log)
        pred_opt = pred_opt_raw.detach()
        pred_opt_raw = pred_opt_raw.detach()

        # a = pred_opt.gt(0) & multilabel

        # 输出根因标签，优化时间。
        # batch_size = pred_opt.shape[0]
        pred_opt_raw = pred_opt_raw.to("cpu")
        pred_opt = pred_opt.to("cpu")
        multilabel = multilabel.to("cpu")
        pred_label = pred_label.to("cpu")
        pred_opt = (pred_opt * train_dataset.opt_labels_train_std + train_dataset.opt_labels_train_mean)
        duration = input1["duration"]
        # pred_multilabel = pred_opt.gt(0).nonzero()
        opt_min_duration = (duration * opt_threshold).unsqueeze(1)
        # pred_multilabel = pred_opt.gt(opt_min_duration).nonzero()
        pred_multilabel = pred_opt.gt(opt_threshold).nonzero()
        pred_opt_time = pred_opt[pred_multilabel[:, 0], pred_multilabel[:, 1]]
        sorted_time_index = torch.argsort(pred_opt, dim=1, descending=True)

        # 设置阈值并排序优化时间
        # opt_label_m = (opt_label * train_dataset.opt_labels_train_std.to(device) + train_dataset.opt_labels_train_mean.to(device))
        opt_label_m = opt_label
        opt_label = input1["ori_opt_label"]
        # opt_label = input1["opt_label"].to(device)
        # label_multilabel = opt_label.gt(opt_min_duration).nonzero()
        label_multilabel = opt_label.gt(opt_threshold).nonzero()
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

            if len(pred[:k]) == 0 and len(actual[:k]) == 0:
                return 1
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
        # MSE_loss += torch.pow(pred_opt - opt_label_m, 2).sum().item()
        # MSE_loss += opt_label_loss_fn(pred_opt, opt_label_m).item()
        # MSE_loss += torch.pow(pred_opt - opt_label_m, 2).mean(-1).sum(0).item()
        MSE_loss += torch.pow(pred_opt_raw - opt_label_m, 2).mean(-1).sum().item()
        print("MSE_loss", torch.pow(pred_opt_raw - opt_label_m, 2).mean(-1))
        torch.cuda.empty_cache()
        # print(pred_list)

        # for i in range(len(pred_list)):
        #     patk_ = patk(label_list[i], pred_list[i], 1)
        #     print(patk_)

    with open(res_path, "w") as f:
        print("label_list len", len(label_list))
        print("pred_list len", len(pred_list))
        print("mapk@3: ", mapk(label_list, pred_list, 3), file=f)
        print("mapk@5: ", mapk(label_list, pred_list, 5), file=f)

        all_right_cnt = 0
        for i in range(len(label_list)):
            # print(label_list[i] == pred_list)
            if label_list[i] == pred_list[i]:
                all_right_cnt += 1
        print("all_right_cnt rate: ", all_right_cnt / float(test_len), file=f)

        label_dict = {}
        cls_cor = {}
        sig_cor = {}
        sig_label = {}
        sig_pred = {}
        top_1_cor = 0
        top_1_label = {}
        top_1_pred = {}
        top_1_cor_l = {}
        for i, v in enumerate(label_list):
            if len(label_list[i]) == 0:
                top_1_label[0] = top_1_label.get(0, 0) + 1
            else:
                top_1_label[label_list[i][0]+1] = top_1_label.get(label_list[i][0]+1, 0) + 1
                
            if len(pred_list[i]) == 0:
                top_1_pred[0] = top_1_pred.get(0, 0) + 1
            else:
                top_1_pred[pred_list[i][0]] = top_1_pred.get(pred_list[i][0], 0) + 1
                
            if len(label_list[i]) == 0 and len(pred_list[i]) == 0: 
                top_1_cor += 1
                top_1_cor_l[0] = top_1_cor_l.get(0, 0) + 1
            
            elif  len(label_list[i]) != 0 and len(pred_list[i]) != 0 and label_list[i][0] == pred_list[i][0]: 
                top_1_cor += 1
                top_1_cor_l[label_list[i][0]+1] = top_1_cor_l.get(label_list[i][0]+1, 0) + 1
                
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

        print("label_dict: ", label_dict, file=f)
        print("pred_dict: ", pred_dict, file=f)
        print("cls_cor", cls_cor, file=f)
        print("sig_cor: ", sig_cor, file=f)
        print("sig_label: ", sig_label, file=f)
        print("sig_pred: ", sig_pred, file=f)
        print("top 1 cor: ", top_1_cor, file=f)
        print("top 1 label: ", top_1_label, file=f)
        print("top 1 pred: ", top_1_pred, file=f)
        print("top 1 cor lable: ", top_1_cor_l, file=f)

        print("MSE_loss: ", MSE_loss / float(test_len), file=f)
        print("Kendall's tau: ", evaluate_tau(label_list, pred_list), file=f)
            
            # def map(similarity, topk=1, max_order=True):
            #     """
            #     MAP@k = mean(1/r_{i}), where i = {1, ..., k}.
            #     in 1-to-1 retrieval task, only 1 candidate is related.
            #     """
            #     if not max_order:
            #         similarity = -similarity
            #     _, topk_ids = similarity.topk(topk, dim=-1)
            #     gt = similarity.new(similarity.size(0)).copy_(
            #         torch.arange(0, similarity.size(0))
            #     ).unsqueeze(dim=-1)  # [[0], [1], ..., [B-1]]
            #     rank = similarity.new(similarity.size(0), topk).copy_(
            #         (1. / torch.range(1, topk)).expand(similarity.size(0), topk)
            #     )
            #     map_k = rank.masked_fill(topk_ids != gt, 0.).mean(dim=-1)
            #     return map_k


