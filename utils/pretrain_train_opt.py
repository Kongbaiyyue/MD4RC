import copy
import os
import torch
import torch.nn as nn
from torch import optim

import numpy as np
import pandas as pd
import json
from transformers import BertTokenizer, BertModel
import wandb
from sklearn.metrics import ndcg_score

from dataset.data_tensor import Tensor_Opt_modal_dataset
from model.modules.QueryFormer.utils import Encoding, collator
from utils.models import Model, ThreeMulModel
from utils.other_models import GaussModel
from model.modules.FuseModel.CrossTransformer import CrossTransformer, GlobalLocalCrossTransformer
from model.modules.FuseModel.module import MultiHeadedAttention, ThreeMultiHeadedAttention, LocalGlobalMultiHeadedAttention, LogTimesMultiHeadedAttention
from utils.top_model import TopModel, ConcatOptModel, TopRealEstModel, OnlyPlanModel, CommonSpecialModel, PlanMainModel, gateModel, TopConstractModel, gateHierarchicalModel, CrossSQLPlanModel, GateAttnModel, SQLOptModel, PlanOptModel, LogOptModel, TimeOptModel, GateCommonDiffAttnModel, GateComDiff1AttnModel, GateCommonAttnModel, GateContrastCommonAttnModel
from model.pretrain_opt_model import GateComDiffPretrainModel, GatePretrainModel
from utils.times_model import TimeSeriesModel, TimeSoftmaxModel
from pretrain import CustomConvAutoencoder

from model.loss.loss import CMD, DiffLoss, ThresholdLoss 
from model.loss.loss import MarginLoss, ListnetLoss, ListMleLoss, MSEThresholdLoss

from utils.evaluate import evaluate_tau, ndcg_2, top1_margin


cross_model = {"CrossTransformer": CrossTransformer, "GlobalLocalCrossTransformer": GlobalLocalCrossTransformer}

attn_model = {"MultiHeadedAttention": MultiHeadedAttention}
model_dict = {"TopModel": TopModel, "ConcatOptModel":ConcatOptModel, "OnlyPlanModel": OnlyPlanModel, "CommonSpecialModel": CommonSpecialModel, "PlanMainModel": PlanMainModel, "gateModel": gateModel, "TopConstractModel": TopConstractModel, "gateHierarchicalModel": gateHierarchicalModel, "CrossSQLPlanModel": CrossSQLPlanModel, "GateAttnModel": GateAttnModel,
"SQLModel": SQLOptModel, "PlanOptModel": PlanOptModel, "LogOptModel": LogOptModel, "TimeOptModel": TimeOptModel,
"GateCommonDiffAttnModel": GateCommonDiffAttnModel, "GateComDiff1AttnModel": GateComDiff1AttnModel, "GateCommonAttnModel": GateCommonAttnModel, "GateContrastCommonAttnModel": GateContrastCommonAttnModel, "GateComDiffPretrainModel": GateComDiffPretrainModel, "GatePretrainModel": GatePretrainModel}

margin_loss_types = {"ListnetLoss": ListnetLoss, "MarginLoss": MarginLoss, "ListMleLoss": ListMleLoss}

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
    device = 'cuda:0'
    newpath = './results/full/cost/'
    to_predict = 'cost'
    input_emb = 1063
    
class ArgsPara:
    diff_weight = 0.05
    share_weight = 0.05
    # margin_weight = 1.0
    margin_weight = 1.0
    mul_label_weight = 1.0
    ts_weight = 100
    cons_weigtht = 1.0
    

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

    # df["multilabel"] = df["multilabel"].apply(lambda x: json.loads(x))
    df["multilabel"] = df.apply(lambda x: [], axis=1)
    df["opt_label"] = df["opt_label"].apply(lambda x: json.loads(x))

    print(df["log_all"].iloc[0])
    print("multilabel", df["multilabel"])

    encoding = Encoding(None, {'NA': 0})

    # df = df[:10]
    # 获取训练数据、测试数据
    df_train = df[df["dataset_cls"] == "train"]
    df_test = df[df["dataset_cls"] == "test"]
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

def test(model, test_dataloader, tokenizer, device, wdb, test_len, epoch, model_name, train_dataset, opt_threshold, select_model, batch_size, para_args, best_me_num, model_path, best_model_path):
    label_list = []
    pred_list = []
    label_rows = 0
    test_idx = 0
    test_len = test_len
    MSE_loss = 0
    right_label_all = 0
    # pred_rows = 0
    top1_valid_sum = 0
    top1_valid_num = 0

    test_pred_opt = None
    # model.load_state_dict(torch.load(model_path))
    model.eval()

    for index, input1 in enumerate(test_dataloader):
        sql, plan, time, log, multilabel, opt_label = input1["query"], input1["plan"], input1["timeseries"], input1["log"], input1["multilabel"], input1["opt_label"]
        sql = tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        if select_model == "cross_attn_no_plan":
            plan = None
        else:
            plan['x'] = plan['x'].squeeze(1).to(device).to(torch.float32)
            plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
            plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
            plan["heights"] = plan["heights"].squeeze(1).to(device)
        
        time = time.to(device)
        log = log.to(device)
        multilabel = multilabel.to(device)
        # opt_label = opt_label.to(device)
        # if select_model == "cross_attn_no_plan":
        #     plan = None
        # else:
        #     plan['x'] = plan['x'].squeeze(1).to(device).to(torch.float32)
        #     plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
        #     plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
        #     plan["heights"] = plan["heights"].squeeze(1).to(device)
        
        if model_name == "CommonSpecialModel":
        
            pred_label, pred_opt_raw, share_sql_emb, share_plan_emb, private_sql_emb, private_plan_emb = model(sql, plan, time, log)
        elif model_name == "TopConstractModel" or model_name == "GateContrastCommonAttnModel":
            pred_label, pred_opt_raw, sql_plan_global_emb, logit_scale = model(sql, plan, time, log)
        else:
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
        pred_opt = (pred_opt * (train_dataset.opt_labels_train_std + 1e-6) + train_dataset.opt_labels_train_mean)
        
        duration = input1["duration"]
        # pred_multilabel = pred_opt.gt(0).nonzero()
        opt_min_duration = (duration * opt_threshold).unsqueeze(1)
        # pred_multilabel = pred_opt.gt(opt_min_duration).nonzero()
        pred_multilabel = pred_opt.gt(opt_threshold).nonzero()
        # pred_opt_time = pred_opt[pred_multilabel[:, 0], pred_multilabel[:, 1]]
        sorted_time_index = torch.argsort(pred_opt, dim=1, descending=True)

        # 设置阈值并排序优化时间
        # opt_label_m = (opt_label * train_dataset.opt_labels_train_std.to(device) + train_dataset.opt_labels_train_mean.to(device))
        opt_label = input1["ori_opt_label"]
        opt_label = (opt_label * (train_dataset.opt_labels_train_std+ 1e-6) + train_dataset.opt_labels_train_mean)
        opt_label_m = opt_label
        # opt_label = input1["opt_label"].to(device)
        # label_multilabel = opt_label.gt(opt_min_duration).nonzero()
        label_multilabel = opt_label.gt(opt_threshold).nonzero()
        label_sorted_time_index = torch.argsort(opt_label, dim=1, descending=True)
        
        if para_args.pred_type == "multilabel":
            
            multilabel_true = torch.where(opt_label > opt_threshold, 1, 0)
            multilabel_pred = torch.where(pred_label > 0.5, 1, 0)
            right_label = torch.where(multilabel_true == multilabel_pred, 1, 0).sum()
            right_label_all += right_label
        else:
            multilabel_true = torch.where(opt_label > opt_threshold, 1, 0)
            multilabel_pred = torch.where(pred_opt > opt_threshold, 1, 0)
            right_label = torch.where(multilabel_true == multilabel_pred, 1, 0).sum()
            right_label_all += right_label

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
            # print("label_len: ", label_len)
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
            # print("pred_len: ", pred_len)
            for i in range(len_data - pred_len):
                pred_list.append([])
        
        # pred_rows = len(pred_list)
        test_idx += 1
        
        # 计算top1 有效率
        for i in range(sorted_time_index.shape[0]):
            if pred_opt[i][sorted_time_index[i][0]] > opt_threshold:
                top1_valid_num += 1
                if opt_label[i][sorted_time_index[i][0]] > 0:
                    top1_valid_sum += opt_label[i][sorted_time_index[i][0]]

        # 计算MSE
        # MSE_loss += torch.pow(pred_opt - opt_label_m, 2).sum().item()
        # MSE_loss += opt_label_loss_fn(pred_opt, opt_label_m).item()
        # MSE_loss += torch.pow(pred_opt - opt_label_m, 2).mean(-1).sum(0).item()
        MSE_loss += torch.pow(pred_opt_raw - opt_label_m, 2).mean(-1).sum().item()
        # print("MSE_loss", torch.pow(pred_opt_raw - opt_label_m, 2).mean(-1))
        torch.cuda.empty_cache()
        # print(pred_list)

        # for i in range(len(pred_list)):
        #     patk_ = patk(label_list[i], pred_list[i], 1)
        #     print(patk_)

    # with open(res_path, "w") as f:

    all_right_cnt = 0
    for i in range(len(label_list)):
        # print(label_list[i] == pred_list)
        if label_list[i] == pred_list[i]:
            all_right_cnt += 1
            
    label_dict = {}
    cls_cor = {}
    sig_cor = {}
    sig_label = {}
    sig_pred = {}
    top_1_cor = 0
    lab_cor = 0 
    lt_label = 0
    gt_label = 0
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
            top_1_pred[pred_list[i][0]+1] = top_1_pred.get(pred_list[i][0]+1, 0) + 1
            
        if len(label_list[i]) == 0 and len(pred_list[i]) == 0: 
            top_1_cor += 1
            top_1_cor_l[0] = top_1_cor_l.get(0, 0) + 1
            lab_cor += 1
        
        elif  len(label_list[i]) != 0 and len(pred_list[i]) != 0: 
            if label_list[i][0] == pred_list[i][0]:
                top_1_cor += 1
                top_1_cor_l[label_list[i][0]+1] = top_1_cor_l.get(label_list[i][0]+1, 0) + 1
            if len(label_list[i]) == len(pred_list[i]) and len(set(label_list[i]) & set(pred_list[i])) == len(label_list[i]):
                lab_cor += 1
        if len(label_list[i]) < len(pred_list[i]):
            gt_label += 1
        elif len(label_list[i]) > len(pred_list[i]):
            lt_label += 1
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
    
    map5 = mapk(label_list, pred_list, 5)
    tau = evaluate_tau(label_list, pred_list)
    me_num = (top_1_cor / float(test_len)) + tau
    if me_num > best_me_num:
        best_model_path = model_path
        torch.save(model.state_dict(), "/".join(model_path.split("/")[:-1]) + "/best_model.pt")
    torch.save(model.state_dict(), model_path)
    # wdb.log({"label_list_len": len(label_list), "pred_list len": len(pred_list), "mapk@3": mapk(label_list, pred_list, 3), "mapk@5": map5, "all_right_cnt_rate: ": all_right_cnt / float(test_len),  "top-1": top_1_cor / float(test_len), "lab_cor": lab_cor, "MSE_loss": MSE_loss / float(test_len), "Kendall's tau": tau, "gt_label": gt_label, "lt_label": lt_label, "right_label_all": right_label_all, "top1_opt_rate": top1_valid_sum / float(top1_valid_num)})
    if me_num > best_me_num:
        return me_num, best_model_path
    return best_me_num, best_model_path


def train(data_path, select_model, dataset_select, use_fuse_model, train_dataloader, test_dataloader, valid_dataloader, train_len, test_len, valid_len, train_dataset, betas = (0.9, 0.999), lr = 0.0003, batch_size = 8, epoch = 50, l_input_dim = 13,
        t_input_dim = 9,l_hidden_dim = 64, t_hidden_dim = 64, input_dim = 12, emb_dim = 32, fuse_num_layers = 3,
        fuse_ffn_dim = 128, fuse_head_size = 4, dropout = 0.1, opt_threshold = 0.1, time_t=1., model_path_dir=None,  res_path=None, model_name=None,use_metrics=True, use_log=True, use_softmax=True, use_margin_loss=False, use_label_loss=False, use_weight_loss=False, use_threshold_loss=False, margin_loss_type="MarginLoss", multi_head="all_cross", name=None, plan_args=None, para_args=None, attn_model_name="MultiHeadedAttention", cross_model_name="CrossTransformer"):
    # 超参数监控
    if valid_dataloader is None:
        use_valid_dataset = False
    else:
        use_valid_dataset = True
    # start a new wandb run to track this script
    if not os.path.exists(model_path_dir):
        os.mkdir(model_path_dir)
        
    # wdb = wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="oy",
    #     name=name,
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": lr,
    #     "model_name": model_name,
    #     "dataset": para_args.dataset, 
    #     "use_valid_dataset": use_valid_dataset,
    #     "batch_size": batch_size, 
        
    #     # "cross_model_name": cross_model_name,
    #     # "attn_model_name": attn_model_name,
    #     "use_cross_attn": use_fuse_model,
    #     "use_metrics": use_metrics,
    #     "use_log": use_log,
    #     "use_margin_loss": use_margin_loss,
    #     "use_label_loss": use_label_loss,
    #     "use_weight_loss": use_weight_loss,
    #     "use_threshold_loss": use_threshold_loss,
    #     "res_path": res_path,
    #     "pred_type": para_args.pred_type,
    #     # "contrast_learning": contrast_learning,
    #     "dropout": dropout,
    #     "epochs": epoch,
    #     "margin_loss_type": margin_loss_type,
    #     "contrast experiment": model_name
    #     }
    # )
    
    device = plan_args.device

    # query tokenizer
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    mul_label_loss_fn = nn.BCELoss(reduction="mean")
    opt_label_loss_fn = nn.MSELoss(reduction="mean")
    # opt_label_loss_fn = MSEThresholdLoss(threshold=para_args.std_threshold, factor=para_args.threshold_factor)
    loss_diff = DiffLoss()
    loss_cmd = CMD()
    loss_margin = margin_loss_types[margin_loss_type]()
    loss_ts = ThresholdLoss(threshold=para_args.std_threshold)
    constract_loss_fn = nn.CrossEntropyLoss(reduction="mean")
    
    # opt_label_loss_fn = nn.CrossEntropyLoss(reduction="mean")

    print("start train")

    sql_model = BertModel.from_pretrained("./bert-base-uncased")
    # time_model = TimeSoftmaxModel(t_input_dim, t_hidden_dim, emb_dim, device=device, t=time_t, use_softmax=use_softmax)
    time_model = CustomConvAutoencoder()

    fuse_model = None
    # use_metrics = False
    # use_log = False
    if use_fuse_model:
        if multi_head == "all_cross":
            multihead_attn_modules_cross_attn = nn.ModuleList(
                    [MultiHeadedAttention(fuse_head_size, emb_dim, dropout=dropout, use_metrics=use_metrics, use_log=use_log)
                    for _ in range(fuse_num_layers)])
            fuse_model = CrossTransformer(num_layers=fuse_num_layers, d_model=emb_dim, heads=fuse_head_size, d_ff=fuse_ffn_dim, dropout=dropout, attn_modules=multihead_attn_modules_cross_attn)
    
    r_attn_model = nn.ModuleList(
        [attn_model[attn_model_name](fuse_head_size, emb_dim, dropout=dropout, use_metrics=False, use_log=True)
        for _ in range(int(fuse_num_layers))])
    rootcause_cross_model = cross_model[cross_model_name](num_layers=int(fuse_num_layers), d_model=emb_dim, heads=fuse_head_size, d_ff=fuse_ffn_dim, dropout=dropout, attn_modules=r_attn_model)
    # model = model_dict[model_name](t_input_dim, l_input_dim, l_hidden_dim, t_hidden_dim, emb_dim, device=device, plan_args=plan_args, sql_model=sql_model, cross_model=fuse_model, time_model=time_model, rootcause_cross_model=rootcause_cross_model)
    model = model_dict[model_name](t_input_dim, l_input_dim, l_hidden_dim, t_hidden_dim, emb_dim, device=device, plan_args=plan_args, sql_model=sql_model, cross_model=fuse_model, time_model=time_model)

    # model.time_model.load_state_dict(torch.load('pretrain/time/best_model.pth'))
    # model.time_model.load_state_dict(torch.load('pretrain/time/0.0038.pth'))
    model.to(device)
    opt = optim.Adam(model.parameters(), lr, betas)
    # accum_iter = 4
    best_me_num = 0
    best_model_path = ""
    # # weights = torch.ones((batch_size, 5))
    # for i in range(epoch):
    #     epoch_loss = 0
    #     opt.zero_grad()
    #     import time as timeutil
    #     start_time = timeutil.time()
    #     for index, input1 in enumerate(train_dataloader):
    #         opt.zero_grad()
    #         sql, plan, time, log, multilabel, opt_label = input1["query"], input1["plan"], input1["timeseries"], input1["log"], input1["multilabel"], input1["opt_label"]

    #         sql = tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    #         if select_model == "cross_attn_no_plan":
    #             plan = None
    #         else:
    #             plan['x'] = plan['x'].squeeze(1).to(device).to(torch.float32)
    #             plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
    #             plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
    #             plan["heights"] = plan["heights"].squeeze(1).to(device)
            
    #         time = time.to(device)
    #         log = log.to(device)
    #         multilabel = multilabel.to(device)
    #         opt_label = opt_label.to(device)

    #         if model_name == "CommonSpecialModel":
    #             pred_label, pred_opt, share_sql_emb, share_plan_emb, private_sql_emb, private_plan_emb = model(sql, plan, time, log)
    #         elif model_name == "TopConstractModel" or model_name == "GateContrastCommonAttnModel":
    #             pred_label, pred_opt, sql_plan_global_emb, logit_scale = model(sql, plan, time, log)
    #             batch_size_con = opt_label.shape[0]

    #             cons_label = torch.eye(batch_size_con).to(device)
    #             # cons_pred = torch.matmul(sql_plan_global_emb, sql_plan_global_emb.transpose(0, 1)) / t
    #             cons_pred = logit_scale * sql_plan_global_emb @ sql_plan_global_emb.t()
    #             cons_loss = constract_loss_fn(cons_pred, cons_label)
    #         else:
    #             pred_label, pred_opt = model(sql, plan, time, log)
    #         # output1 = output1.squeeze(1)
    #         # pred_opt
            
    #         if use_margin_loss:
    #             margin_loss = loss_margin(pred_opt, opt_label)

    #         # pred_opt = pred_opt.view(-1, 1)
    #         # opt_label = opt_label.view(-1, 1)
            
    #         # label = label.to(torch.long)
    #         if para_args.pred_type == "multilabel":
    #             opt_label_true = (opt_label * (train_dataset.opt_labels_train_std.to(opt_label.device) + 1e-6) + train_dataset.opt_labels_train_mean.to(opt_label.device))
    #             opt_label_l = torch.where(opt_label_true > opt_threshold, 1.0, 0.0)
    #             opt_label_loss = mul_label_loss_fn(pred_label, opt_label_l)
    #         else:
    #             opt_label_loss = opt_label_loss_fn(pred_opt, opt_label)
    #         # opt_label_loss = 0
    #         wdb.log({"opt_label_loss": opt_label_loss})
    #         if model_name == "CommonSpecialModel":
            
    #             diff_loss = loss_diff(private_sql_emb, share_sql_emb)
    #             diff_loss += loss_diff(private_plan_emb, share_plan_emb)
    #             diff_loss += loss_diff(private_sql_emb, private_plan_emb)
    #             diff_loss = diff_loss / 3.0
                
    #             share_loss = loss_cmd(share_sql_emb[:, 0], share_plan_emb[:, 0], 5)

    #             # loss = mul_label_loss + opt_label_loss
    #             loss = opt_label_loss + diff_loss*para_args.diff_weight + share_loss*para_args.share_weight
                
    #         elif model_name == "TopConstractModel" or model_name == "GateContrastCommonAttnModel":
    #             loss = opt_label_loss + cons_loss*para_args.cons_weigtht
    #             wdb.log({"cons_loss": cons_loss*para_args.cons_weigtht})
    #         else:
    #             loss = opt_label_loss
            
    #         if use_margin_loss:
    #             loss += (margin_loss*para_args.margin_weight)
    #             wdb.log({"margin_loss": margin_loss*para_args.margin_weight})
    #         if use_threshold_loss:
    #             ts_loss = loss_ts(pred_opt, opt_label)
    #             loss += (ts_loss*para_args.ts_weight)
    #             wdb.log({"ts_loss": ts_loss*para_args.ts_weight})
    #         if use_label_loss:
    #             mul_label_loss = mul_label_loss_fn(pred_label, multilabel.to(torch.float32))
    #             loss += (mul_label_loss*para_args.mul_label_weight)
    #             wdb.log({"mul_label_loss": mul_label_loss*para_args.mul_label_weight})
                

    #         #重新分配样本权重
    #         if use_weight_loss:
    #             errors = torch.abs(pred_opt - opt_label)
    #             weights = 1 / (1 + errors)
    #             weights = weights.detach()
    #             loss = weights * loss
            
    #         epoch_loss += loss.item()
    #         # loss = loss / accum_iter 
    #         # if epoch_loss == torch.nan:
    #         #     print(epoch_loss)
    #         loss.backward()
    #         # if ((index + 1) % accum_iter == 0) or (index + 1 == len(train_dataloader)):
    #         #     opt.step()
    #         #     opt.zero_grad()
    #         opt.step()
    #     print(i, epoch_loss / train_len)
    #     end_time = timeutil.time()
    #     print("============diag during time==========: ", end_time - start_time)
    #     if model_name == "CommonSpecialModel":
    #         wdb.log({"loss": epoch_loss / train_len, "diff_loss": diff_loss, "share_loss": share_loss, "epoch": i+1, "lr": opt.state_dict()['param_groups'][0]['lr']})
    #     else:
    #         wdb.log({"loss": epoch_loss / train_len, "epoch": i+1, "lr": opt.state_dict()['param_groups'][0]['lr']})
        
        
    #     model_path = model_path_dir + f"/{i}.pt"
    #     if (i+1) % 5 == 0 or i == 0 or i == epoch-1:
    #         if valid_dataloader is not None:
    #             best_me_num, best_model_path = test(model, valid_dataloader, tokenizer, device, wdb, valid_len, epoch, model_name, train_dataset, opt_threshold, select_model, batch_size, para_args, best_me_num, model_path, best_model_path)
    #         else:
    #             test(model, test_dataloader, tokenizer, device, wdb, test_len, epoch, model_name, train_dataset, opt_threshold, select_model, batch_size, para_args, best_me_num, model_path)
    
    # torch.save(model.state_dict(), model_path)
    
    # model.load_state_dict(torch.load(best_model_path))
    # model.load_state_dict(torch.load(model_path_dir + "/best_model.pt"))
    model.load_state_dict(torch.load(model_path_dir))

     
    # todo test方案，以及给出优化方法

    sum_correct = 0

    label_list = []
    pred_list = []
    pred_opt_list_ndcg = []
    label_opt_list_ndcg = []
    label_rows = 0
    test_idx = 0
    test_len = test_len
    MSE_loss = 0
    right_label_all = 0
    pred_no_in_label_list = []
    label_no_pred_list = []
    # pred_rows = 0

    test_pred_opt = None
    # model.load_state_dict(torch.load(model_path))
    model.eval()
    top1_valid_sum = 0
    top1_valid_num = 0

    center_t = {}
    center_sql_t = {}
    center_plan_t = {}
    center_log_t = {}
    center_time_t = {}
    sqlss = {"sql":[], "predrootcause": []}
    for index, input1 in enumerate(test_dataloader):
        sql, plan, time, log, multilabel, opt_label = input1["query"], input1["plan"], input1["timeseries"], input1["log"], input1["multilabel"], input1["opt_label"]
        
        sqlss["sql"].extend(sql)
        sql = tokenizer(sql, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        if select_model == "cross_attn_no_plan":
            plan = None
        else:
            plan['x'] = plan['x'].squeeze(1).to(device).to(torch.float32)
            plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
            plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
            plan["heights"] = plan["heights"].squeeze(1).to(device)
            
        time = time.to(device)
        log = log.to(device)
        multilabel = multilabel.to(device)
        # opt_label = opt_label.to(device)
        # if select_model == "cross_attn_no_plan":
        #     plan = None
        # else:
        #     plan['x'] = plan['x'].squeeze(1).to(device).to(torch.float32)
        #     plan["attn_bias"] = plan["attn_bias"].squeeze(1).to(device)
        #     plan["rel_pos"] = plan["rel_pos"].squeeze(1).to(device)
        #     plan["heights"] = plan["heights"].squeeze(1).to(device)
        
        if model_name == "CommonSpecialModel":
        
            pred_label, pred_opt_raw, share_sql_emb, share_plan_emb, private_sql_emb, private_plan_emb = model(sql, plan, time, log)
        elif model_name == "TopConstractModel" or model_name == "GateContrastCommonAttnModel":
            pred_label, pred_opt_raw, sql_plan_global_emb, logit_scale = model(sql, plan, time, log)
        else:
            # pred_label, pred_opt_raw, center_sql, center_plan, center_log, center_time = model(sql, plan, time, log)
            pred_label, pred_opt_raw = model(sql, plan, time, log)
            
            # all_emb = model(sql, plan, time, log)
            
            
            opt_label = input1["ori_opt_label"]
            opt_label = (opt_label * (train_dataset.opt_labels_train_std+ 1e-6) + train_dataset.opt_labels_train_mean)
            label_opt, label_sorted_time_index = torch.sort(opt_label, dim=1, descending=True)
            # print(label_opt)
            # print(label_sorted_time_index)
            # top1_rc = torch.where(label_opt[:, 0] > opt_threshold, label_sorted_time_index[:, 0] + 1, 0)
            # print(top1_rc)
            # if len(center_t) == 0:
            #     center_t["all_emb"] = [all_emb.to("cpu").detach()]
            #     center_t["top_rc"] = [top1_rc.to("cpu").detach()]
            #     print(0)
            # else:
            #     center_t["all_emb"][0] = torch.cat([center_t["all_emb"][0], all_emb.to("cpu").detach()], dim=0)
            #     center_t["top_rc"][0] = torch.cat([center_t["top_rc"][0], top1_rc.to("cpu").detach()], dim=0)
                
            #     print(1)
                # print(center_time.device)
                
        # torch.cuda.empty_cache()
        
        pred_opt = pred_opt_raw.detach()
        pred_opt_raw = pred_opt_raw.detach()

        # a = pred_opt.gt(0) & multilabel

        # 输出根因标签，优化时间。
        # batch_size = pred_opt.shape[0]
        pred_opt_raw = pred_opt_raw.to("cpu")
        pred_opt = pred_opt.to("cpu")
        multilabel = multilabel.to("cpu")
        pred_label = pred_label.to("cpu")
        pred_opt = (pred_opt * (train_dataset.opt_labels_train_std + 1e-6) + train_dataset.opt_labels_train_mean)
                
        duration = input1["duration"]
        # pred_multilabel = pred_opt.gt(0).nonzero()
        opt_min_duration = (duration * opt_threshold).unsqueeze(1)
        # pred_multilabel = pred_opt.gt(opt_min_duration).nonzero()
        pred_multilabel = pred_opt.gt(opt_threshold).nonzero()
        # pred_opt_time = pred_opt[pred_multilabel[:, 0], pred_multilabel[:, 1]]
        sorted_time_index = torch.argsort(pred_opt, dim=1, descending=True)

        # 设置阈值并排序优化时间
        opt_label = input1["ori_opt_label"]
        opt_label = (opt_label * (train_dataset.opt_labels_train_std + 1e-6) + train_dataset.opt_labels_train_mean)
        opt_label_m = opt_label
        # opt_label = input1["opt_label"].to(device)
        # label_multilabel = opt_label.gt(opt_min_duration).nonzero()
        label_multilabel = opt_label.gt(opt_threshold).nonzero()
        label_sorted_time_index = torch.argsort(opt_label, dim=1, descending=True)
        
        if para_args.pred_type == "multilabel":
            
            multilabel_true = torch.where(opt_label > opt_threshold, 1, 0)
            multilabel_pred = torch.where(pred_label > 0.5, 1, 0)
            right_label = torch.where(multilabel_true == multilabel_pred, 1, 0).sum()
            right_label_all += right_label
        else:
            multilabel_true = torch.where(opt_label > opt_threshold, 1, 0)
            multilabel_pred = torch.where(pred_opt > opt_threshold, 1, 0)
            right_label = torch.where(multilabel_true == multilabel_pred, 1, 0).sum()
            right_label_all += right_label
        # pred_opt_time = (pred_opt[pred_multilabel[:, 0], pred_multilabel[:, 1]] * train_dataset.opt_labels_train_std + train_dataset.opt_labels_train_mean)

        # pred_opt.gt(0)
        pred_opt_list_ndcg.extend(pred_opt.tolist())
        label_opt_list_ndcg.extend(opt_label.tolist())

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
                if kk_i < 3:
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
                if kk_i < 3:
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
                
        # for l_deal_i in range(batch_size * test_idx, len_data):
        #     temp_list = copy.deepcopy(pred_list[l_deal_i])
        #     del_i = 0
        #     for index_pred, j_pred in enumerate(temp_list):
        #         if temp_list[index_pred] not in label_list[l_deal_i]:
        #             pred_no_in_label_list.append(float(opt_label[l_deal_i - batch_size * test_idx][temp_list[index_pred]]))
        #     for index_pred, j_pred in enumerate(label_list[l_deal_i]):
        #         if label_list[l_deal_i][index_pred] not in pred_list[l_deal_i]:
        #             label_no_pred_list.append(float(opt_label[l_deal_i - batch_size * test_idx][label_list[l_deal_i][index_pred]]))
            
            # for index_pred, j_pred in enumerate(temp_list):
            #     if temp_list[index_pred] not in label_list[l_deal_i] and pred_opt[l_deal_i - batch_size * test_idx][temp_list[index_pred]] < 0.06:
            #         pred_list[l_deal_i].remove(pred_list[l_deal_i][index_pred - del_i])
            #         del_i += 1
            # for index_pred, j_pred in enumerate(label_list[l_deal_i]):
            #     if label_list[l_deal_i][index_pred] not in pred_list[l_deal_i] and opt_label[l_deal_i - batch_size * test_idx][label_list[l_deal_i][index_pred]] > 0.04:
            #         pred_list[l_deal_i].append(label_list[l_deal_i][index_pred])
        
        test_idx += 1
        
        # 计算MSE
        # MSE_loss += torch.pow(pred_opt - opt_label_m, 2).sum().item()
        # MSE_loss += opt_label_loss_fn(pred_opt, opt_label_m).item()
        # MSE_loss += torch.pow(pred_opt - opt_label_m, 2).mean(-1).sum(0).item()
        MSE_loss += torch.pow(pred_opt_raw - opt_label_m, 2).mean(-1).sum().item()
        # MSE_loss += torch.pow(pred_opt - opt_label_m, 2).mean(-1).sum().item()
        print("MSE_loss", torch.pow(pred_opt_raw - opt_label_m, 2).mean(-1))
        torch.cuda.empty_cache()
        
        # 计算top1 有效率
        # for i in range(sorted_time_index.shape[0]):
        #     if pred_opt[i][sorted_time_index[i][0]] > opt_threshold:
        #         top1_valid_num += 1
        #         if opt_label[i][sorted_time_index[i][0]] > 0:
        #             top1_valid_sum += opt_label[i][sorted_time_index[i][0]]
        
        for i in range(label_sorted_time_index.shape[0]):
            if opt_label[i][label_sorted_time_index[i][0]] > opt_threshold:
                top1_valid_num += 1
                if opt_label[i][sorted_time_index[i][0]] > 0:
                    top1_valid_sum += opt_label[i][sorted_time_index[i][0]]
        
        # print(pred_list)

        # for i in range(len(pred_list)):
        #     patk_ = patk(label_list[i], pred_list[i], 1)
        #     print(patk_)
    # with open(res_path.replace(".txt", "_pred.json"), "w") as f2:
    #     json.dump(pred_no_in_label_list, f2)
    # with open(res_path.replace(".txt", ".json"), "w") as f2:
    #     json.dump(label_no_pred_list, f2)
    
    # df = pd.DataFrame(center_t)
    # df.to_pickle("plt/data/fuse_modal_emb.pickle")
    
    # res_path = model_path_dir + "/res_holo.txt"
    # res_path = model_path_dir + "/res_TH.txt"
    # res_path = model_path_dir + "/res.txt"
    pred_error_sample = 0
    res_path = model_path_dir.replace(".pt", "_TH_res.txt")
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
        lab_cor = 0 
        lt_label = 0
        gt_label = 0
        top_1_label = {}
        top_1_pred = {}
        top_1_cor_l = {}
        for i, v in enumerate(label_list):
            sqlss["predrootcause"].append(pred_list[i])
            
            if len(label_list[i]) == 0:
                top_1_label[0] = top_1_label.get(0, 0) + 1
            else:
                top_1_label[label_list[i][0]+1] = top_1_label.get(label_list[i][0]+1, 0) + 1
                
            if len(pred_list[i]) == 0:
                top_1_pred[0] = top_1_pred.get(0, 0) + 1
            else:
                top_1_pred[pred_list[i][0]+1] = top_1_pred.get(pred_list[i][0]+1, 0) + 1
                
            if len(label_list[i]) == 0 and len(pred_list[i]) == 0: 
                top_1_cor += 1
                top_1_cor_l[0] = top_1_cor_l.get(0, 0) + 1
                lab_cor += 1
            
            elif  len(label_list[i]) != 0 and len(pred_list[i]) != 0: 
                if label_list[i][0] == pred_list[i][0]:
                    top_1_cor += 1
                    top_1_cor_l[label_list[i][0]+1] = top_1_cor_l.get(label_list[i][0]+1, 0) + 1
                if len(label_list[i]) == len(pred_list[i]) and len(set(label_list[i]) & set(pred_list[i])) == len(label_list[i]):
                    lab_cor += 1
            if len(label_list[i]) < len(pred_list[i]):
                gt_label += 1
            elif len(label_list[i]) > len(pred_list[i]):
                lt_label += 1
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
            if len(pred_set - label_set) > 0 :
                pred_error_sample += 1

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
        print("lab_cor :", lab_cor, file=f)
        print("gt_label :", gt_label, file=f)
        print("lt_label :", lt_label, file=f)

        print("MSE_loss: ", MSE_loss / float(test_len), file=f)
        print("Kendall's tau: ", evaluate_tau(label_list, pred_list), file=f)
        print("right_label_all: ", right_label_all / test_len / 5, right_label_all, file=f)
        print("top1 提升率: ", top1_valid_sum / float(top1_valid_num), file=f)
        print("pred_error_sample", pred_error_sample / test_len, file=f)
        
        label_opt_list_ndcg = torch.tensor(label_opt_list_ndcg)
        label_opt_list_ndcg = torch.where(label_opt_list_ndcg < 0, 0, label_opt_list_ndcg)
        pred_opt_list_ndcg = torch.tensor(pred_opt_list_ndcg)
        pred_opt_list_ndcg = torch.where(pred_opt_list_ndcg < 0, 0, pred_opt_list_ndcg)
        print("top1 margin acc: ", top1_margin(label_opt_list_ndcg, pred_opt_list_ndcg), file=f)
        
        print("ndgc: ", ndcg_2(label_opt_list_ndcg, pred_opt_list_ndcg), file=f)
        # print("mrr:", mrr(pred_list, label_list), file=f)
    
    # wdb.finish()


