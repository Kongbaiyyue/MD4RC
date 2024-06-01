import pandas as pd
from numpy import random
import json
from matplotlib import pyplot as plt

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.08, 1.02*height, '%s' % int(height), size=10, family="Times new roman")

def get_dataset():
    path = "data/hgprecn-cn-zvp2qdxio002-clone/data/sql_log_dataset_2023-11-05 15:20:00_2023-11-05 15:40:00_rootcause_json_plan_series_cor.csv"
    df = pd.read_csv(path)

    print(df.columns)

    labels = ['JoinOrder', 'update table',
        'PQE rewrite', 'likeSQL', 'UNIQ rewrite', 'rewrite', 'update2insert',
        'distributionKey', 'groupBy', 'Join', 'index', 'dictionary']
    def get_label(rootcause):
        # print(rootcause)
        label = [0] * 12
        if pd.isna(rootcause):
            return label
        rootcause = rootcause.split(",")[:-1]
        for i in range(len(rootcause)):
            label[int(rootcause[i])] = 1
        return label

    def get_opt(row, labels):
        opt_time = []
        for a in labels:
            if pd.isna(row[a]):
                opt_time.append(0)
            else:
                opt_time.append(row["duration"] - row[a])
        return opt_time
        
        
    df["multilabel"] = df["rootcauselabel"].apply(get_label)
    df["opt_label"] = df.apply(get_opt, labels=labels, axis=1)
    # print(df["multilabel"].iloc[:15])
    print(df["opt_label"].iloc[:15])

    df.to_csv("RootcauseSQL/data/rootcause.csv")

def get_dataset_2():
    # df = pd.read_csv("data/hgprecn-cn-zvp2qdxio002-clone/data_all_raw_log/create_sql/sql_log_copilot_dataset_2023-11-15 02:41:00_2023-11-15 06:41:00_rootcause.csv")
    df = pd.read_csv("data/hgprecn-cn-zvp2qdxio002-clone/data_all_raw_log/create_sql/sql_log_copilot_dataset_2023-11-15 15:06:00_2023-11-15 17:52:00_rootcause_json_plan_series.csv")
    print(df.columns)
    def set_threshold(x):
        label = [0] * 12
        # opt_label = json.loads(x["opt_label"])
        # opt_label = sorted(enumerate(opt_label), key=lambda x:x[1], reverse=True)
        label_root_cause = {0: 'JoinOrder', 1: 'update table', 2: 'PQE rewrite', 3: 'likeSQL', 4: 'UNIQ rewrite', 5: 'rewrite', 6: 'update2insert', 7: 'distributionKey', 8: 'groupBy', 9: 'Join', 10: 'index', 11: 'dictionary'}

        k = 0
        # for opt in opt_label:
        for k, v in label_root_cause.items():
            if float(x["duration"] - x[v]) > x["duration"] * 0.1:
            # if float(opt) > 0:
                # label.append(1)
                label[k] = 1
            else:
                # label.append(0)
                label[k] = 0
            
            # k += 1
            # if k >= 3: break
        return json.dumps(label)
    def get_opt(row):
        opt_time = []
        labels = ['JoinOrder', 'update table',
        'PQE rewrite', 'likeSQL', 'UNIQ rewrite', 'rewrite', 'update2insert',
        'distributionKey', 'groupBy', 'Join', 'index', 'dictionary']
        for a in labels:
            if pd.isna(row[a]):
                opt_time.append(0)
            else:
                opt_time.append(float(row["duration"] - row[a]))
        return opt_time
    # df["threshold_20_pre_label"] = df.apply(set_threshold, axis=1)
    df["multilabel"] = df.apply(set_threshold, axis=1)
    df["opt_label"] = df.apply(get_opt, axis=1)

    # df.to_csv("RootcauseSQL/data/rootcause_all_raw_log.csv")


def split_train_test():
    # path = "RootcauseSQL/data/rootcause.csv"
    # path = "RootcauseSQL/data/rootcause_all_raw_log_threshold_10.csv"
    path = "RootcauseSQL/data/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500_2023_12_04.csv"
    df = pd.read_csv(path)
    df.loc[:, "dataset_cls"] = "train"
    # print(df.columns)
    groups = df.groupby("multilabel")
    flag = 0
    for name, group in groups:
        # if group.shape[0] == 1 and random.random() < 0.5:
        if group.shape[0] == 1:
            if flag == 0:
                df.loc[group.index, "dataset_cls"] = "test"
            else:
                flag = (flag+1)%5
        else:
            indices = list(range(group.shape[0]))
            random.shuffle(indices)
            test_indices = indices[:int(group.shape[0] * 0.1)] if group.shape[0] > 10 else [indices[0]]
            for idx in test_indices:
                df["dataset_cls"][group.index[idx]] = "test"
        
        print(name, group.shape)
    print(len(groups))
    # df.to_csv("RootcauseSQL/data/rootcause_train_test.csv", index=False)
    df.to_csv("RootcauseSQL/data/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500_2023_12_04.csv", index=False)


def threshold_split_train_test():
    path = "RootcauseSQL/data/rootcause_threshold_10_pre_label.csv"
    df = pd.read_csv(path)
    # print(df.columns)
    df.loc[:, "dataset_cls"] = "train"
    groups = df.groupby("threshold_10_pre_label")
    flag = 0
    for name, group in groups:
        # if group.shape[0] == 1 and random.random() < 0.5:
        if group.shape[0] == 1:
            if flag == 0:
                df.loc[group.index, "dataset_cls"] = "test"
            else:
                flag = (flag+1)%5
        else:
            indices = list(range(group.shape[0]))
            random.shuffle(indices)
            test_indices = indices[:int(group.shape[0] * 0.1)] if group.shape[0] > 10 else [indices[0]]
            for idx in test_indices:
                df["dataset_cls"][group.index[idx]] = "test"
        
        print(name, group.shape)
    print(len(groups))
    # df.to_csv("RootcauseSQL/data/rootcause_train_test_threshold_10_new.csv", index=False)

def threshold_split_train_test_top():
    path = "RootcauseSQL/data/2023_12_03/2023_12_03_5_labels_top_dataset.csv"
    df = pd.read_csv(path)
    # print(df.columns)
    df.loc[:, "dataset_cls"] = "train"
    groups = df.groupby("top_label")
    flag = 0
    for name, group in groups:
        # if group.shape[0] == 1 and random.random() < 0.5:
        if group.shape[0] == 1:
            if flag == 0:
                df.loc[group.index, "dataset_cls"] = "test"
            else:
                flag = (flag+1)%5
        else:
            indices = list(range(group.shape[0]))
            random.shuffle(indices)
            test_indices = indices[:int(group.shape[0] * 0.2)] if group.shape[0] > 10 else [indices[0]]
            for idx in test_indices:
                df["dataset_cls"][group.index[idx]] = "test"
        
        # print(name, group.shape)
    print(len(groups))
    df.to_csv("RootcauseSQL/data/2023_12_03/2023_12_03_5_labels_top_dataset_0.2test.csv", index=False)
    
    
def split_test_valid():
    path = "data/all_data/add_2023_12_30_2-5-sqlsmith_resplit_lg_500_cost_64_2_random_0.05rate_4.pickle"
    df = pd.read_pickle(path)
    # print(df.columns)
    # df.loc[:, "dataset_cls"] = "train"
    df_test = df[df["dataset_cls"] == "test"]
    groups = df_test.groupby("top_label")
    flag = 0
    for name, group in groups:
        # if group.shape[0] == 1 and random.random() < 0.5:
        if group.shape[0] == 1:
            if flag == 0:
                df.loc[group.index, "dataset_cls"] = "valid"
            else:
                flag = (flag+1)%5
        else:
            indices = list(range(group.shape[0]))
            random.shuffle(indices)
            test_indices = indices[:int(group.shape[0] * 0.3)] if group.shape[0] > 10 else [indices[0]]
            for idx in test_indices:
                df["dataset_cls"][group.index[idx]] = "valid"
        
        # print(name, group.shape)
    print(len(groups))
    # df.to_csv("RootcauseSQL/data/2023_12_03/2023_12_03_5_labels_top_dataset_0.2test.csv", index=False)
    print(df[df["dataset_cls"] == "test"].shape)
    print(df[df["dataset_cls"] == "valid"].shape)
    print(df[df["dataset_cls"] == "train"].shape)
    
    df.to_pickle("data/all_data/add_2023_12_30_2-5-sqlsmith_resplit_lg_500_cost_64_2_random_0.05rate_4_valid.pickle")
    

def threshold():
    df = pd.read_csv("RootcauseSQL/data/rootcause_train_test.csv")

    def set_threshold(x):
        label = [0] * 12
        opt_label = json.loads(x["opt_label"])
        opt_label = sorted(enumerate(opt_label), key=lambda x:x[1], reverse=True)

        k = 0
        # for opt in opt_label:
        for index, opt in opt_label:
            if float(opt) > x["duration"] * 0.2:
            # if float(opt) > 0:
                # label.append(1)
                label[index] = 1
            else:
                # label.append(0)
                label[index] = 0
            
            k += 1
            # if k >= 3: break
        return json.dumps(label)
    df["threshold_20_pre_label"] = df.apply(set_threshold, axis=1)
    print(df["threshold_20_pre_label"])

    groups = df.groupby("threshold_20_pre_label")
    # print(df["multilabel"][856])
    # print(df)
    x_list = list(range(len(groups)))
    y_list = []
    x_label = []
    for name, group in groups:
        # print(group.index, len(group))
        print(name, len(group))
        y_list.append(len(group))
        x_label.append(name)
        # print(group["threshold_10_pre_label"])
    
    # plt.xticks(x_list, x_label, fontproperties='Times New Roman', size = 10, rotation=90)
    # plt.yticks(fontproperties='Times New Roman', size = 10)

    # cm = plt.bar(x_list, y_list)
    # # cm = plt.barh(y_list, x_list)
    # autolabel(cm)
    # plt.tight_layout()   #xlable坐标轴显示不全
    # plt.show()
    check_threshold_data(df)
    # df.to_csv("RootcauseSQL/data/rootcause_threshold_10_pre_label.csv")

def threshold_2():
    # df = pd.read_csv("data/hgprecn-cn-zvp2qdxio002-clone/data_all_raw_log/create_sql/sql_log_copilot_dataset_2023-11-15 02:41:00_2023-11-15 06:41:00_rootcause.csv")
    # df = pd.read_csv("data/hgprecn-cn-zvp2qdxio002-clone/data_all_raw_log/create_sql/sql_log_copilot_dataset_2023-11-15 15:06:00_2023-11-15 17:52:00_update_label_rootcause.csv")
    # df = pd.read_csv("RootcauseSQL/data/rootcause_train_test_threshold_10.csv")
    # df = pd.read_csv("RootcauseSQL/data/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500_2023_11_30_dictionary.csv")
    df = pd.read_csv("RootcauseSQL/data/2023_12_03/2023_12_03_5_labels_top_dataset_0.2test.csv")

    def set_threshold(x):
        label = [0] * 12
        # opt_label = json.loads(x["opt_label"])
        # opt_label = sorted(enumerate(opt_label), key=lambda x:x[1], reverse=True)
        label_root_cause = {0: 'JoinOrder', 1: 'update table', 2: 'PQE rewrite', 3: 'likeSQL', 4: 'UNIQ rewrite', 5: 'rewrite', 6: 'update2insert', 7: 'distributionKey', 8: 'groupBy', 9: 'Join', 10: 'index', 11: 'dictionary'}

        k = 0
        # for opt in opt_label:
        for k, v in label_root_cause.items():
            # if float(x["duration"] - x[v]) > x["duration"] * 0.1:
            if eval(x["opt_label"])[k] > x["duration"] * 0.1:
            # if float(opt) > 0:
                # label.append(1)
                label[k] = 1
            else:
                # label.append(0)
                label[k] = 0
            
            # k += 1
            # if k >= 3: break
        return json.dumps(label)
    df["threshold_20_pre_label"] = df.apply(set_threshold, axis=1)

    # df_old = pd.read_csv("RootcauseSQL/data/rootcause_train_test.csv")

    # def set_threshold_2(x):
    #     label = [0] * 12
    #     opt_label = json.loads(x["opt_label"])
    #     opt_label = sorted(enumerate(opt_label), key=lambda x:x[1], reverse=True)

    #     k = 0
    #     # for opt in opt_label:
    #     for index, opt in opt_label:
    #         if float(opt) > x["duration"] * 0.1:
    #         # if float(opt) > 0:
    #             # label.append(1)
    #             label[index] = 1
    #         else:
    #             # label.append(0)
    #             label[index] = 0
            
    #         k += 1
    #         # if k >= 3: break
    #     return json.dumps(label)
    # df_old["threshold_20_pre_label"] = df_old.apply(set_threshold_2, axis=1)
    # df = pd.concat([df, df_old])
    # print(df["threshold_20_pre_label"])

    # groups = df.groupby("threshold_20_pre_label")
    # # print(df["multilabel"][856])
    # # print(df)
    # x_list = list(range(len(groups)))
    # y_list = []
    # x_label = []
    # for name, group in groups:
    #     # print(group.index, len(group))
    #     print(name, len(group))
    #     y_list.append(len(group))
    #     x_label.append(name)
    #     # print(group["threshold_10_pre_label"])
    
    # plt.xticks(x_list, x_label, fontproperties='Times New Roman', size = 10, rotation=90)
    # plt.yticks(fontproperties='Times New Roman', size = 10)

    # cm = plt.bar(x_list, y_list)
    # # cm = plt.barh(y_list, x_list)
    # autolabel(cm)
    # plt.tight_layout()   #xlable坐标轴显示不全
    # plt.show()
    check_threshold_data(df)

def threshold_3():
    # df = pd.read_csv("data/hgprecn-cn-zvp2qdxio002-clone/data_all_raw_log/create_sql/sql_log_copilot_dataset_2023-11-15 02:41:00_2023-11-15 06:41:00_rootcause.csv")
    df = pd.read_csv("data/hgprecn-cn-zvp2qdxio002-clone/data_all_raw_log/create_sql/sql_log_copilot_dataset_2023-11-15 15:06:00_2023-11-15 17:52:00_update_label_rootcause.csv")

    def set_threshold(x):
        label = [0] * 12
        # opt_label = json.loads(x["opt_label"])
        # opt_label = sorted(enumerate(opt_label), key=lambda x:x[1], reverse=True)
        label_root_cause = {0: 'JoinOrder', 1: 'update table', 2: 'PQE rewrite', 3: 'likeSQL', 4: 'UNIQ rewrite', 5: 'rewrite', 6: 'update2insert', 7: 'distributionKey', 8: 'groupBy', 9: 'Join', 10: 'index', 11: 'dictionary'}

        k = 0
        # for opt in opt_label:
        for k, v in label_root_cause.items():
            if float(x["duration"] - x[v]) > x["duration"] * 0.1:
            # if float(opt) > 0:
                # label.append(1)
                label[k] = 1
            else:
                # label.append(0)
                label[k] = 0
            
            # k += 1
            # if k >= 3: break
        return json.dumps(label)
    df["threshold_20_pre_label"] = df.apply(set_threshold, axis=1)

    # groups = df.groupby("threshold_20_pre_label")
    # # print(df["multilabel"][856])
    # # print(df)
    # x_list = list(range(len(groups)))
    # y_list = []
    # x_label = []
    # for name, group in groups:
    #     # print(group.index, len(group))
    #     print(name, len(group))
    #     y_list.append(len(group))
    #     x_label.append(name)
    #     # print(group["threshold_10_pre_label"])
    
    # plt.xticks(x_list, x_label, fontproperties='Times New Roman', size = 10, rotation=90)
    # plt.yticks(fontproperties='Times New Roman', size = 10)

    # cm = plt.bar(x_list, y_list)
    # # cm = plt.barh(y_list, x_list)
    # autolabel(cm)
    # plt.tight_layout()   #xlable坐标轴显示不全
    # plt.show()
    check_threshold_data_2(df)

def check_test_data():
    df = pd.read_csv("RootcauseSQL/data/rootcause_train_test.csv")
    df2 = df[df["dataset_cls"] == "test"]
    groups = df2.groupby("multilabel")
    # print(df["multilabel"][856])
    # print(df)
    for name, group in groups:
        print(group.index, len(group))

    groups2 = df.groupby("multilabel")
    for name, group in groups2:
        print(group.index, len(group))
        # print(group["multilabel"])
        print(group["opt_label"])
    

def check_threshold_data(df):
    # path = "RootcauseSQL/data/rootcause_threshold_10_pre_label.csv"
    # df = pd.read_csv(path)
    global cnt
    cnt = {}
    for i in range(12):
        cnt[i] = 0
    def check(x):
        global cnt
        label_list = json.loads(x)
        for i, label in enumerate(label_list):
            if label == 1:
                cnt[i] += 1
        return None
    df["threshold_20_pre_label"].apply(check)
    x_list = []
    y_list = []
    x_label = ['JoinOrder', 'update table',
        'PQE rewrite', 'likeSQL', 'UNIQ rewrite', 'rewrite', 'update2insert',
        'distributionKey', 'groupBy', 'Join', 'index', 'dictionary']
    for k, v in cnt.items():
        x_list.append(k)
        y_list.append(v)

    pop_list = []
    for i in range(len(x_list)):
        if y_list[i] == 0:
            pop_list.append(i)
    print(pop_list)
    pop_i = 0
    for i in pop_list:
        # x_list.pop(i-pop_i)
        y_list.pop(i-pop_i)
        x_label.pop(i-pop_i)
        pop_i += 1
    x_list = list(range(len(x_label)))
    print(len(x_label))
    print(len(y_list))
    print(x_list)

    # color=['red','black','peru','orchid','deepskyblue', 'gold', 'green', 'blue', 'pink', 'purple', 'yellow', 'gray']
    plt.xticks(x_list, x_label, fontproperties='Times New Roman', size = 10, rotation=60)
    plt.yticks(fontproperties='Times New Roman', size = 10)
    # cm = plt.bar(x_list, y_list, color=color)
    cm = plt.bar(x_list, y_list)
    
    autolabel(cm)
    plt.tight_layout()
    plt.show()

def check_threshold_data_2(df):
    # path = "RootcauseSQL/data/rootcause_threshold_10_pre_label.csv"
    # df = pd.read_csv(path)
    global cnt
    cnt = {}
    for i in range(12):
        cnt[i] = 0
    def check(x):
        global cnt
        label_list = json.loads(x)
        for i, label in enumerate(label_list):
            if label == 1:
                cnt[i] += 1
        return None
    df["threshold_20_pre_label"].apply(check)
    x_list = []
    y_list = []
    x_label = ['JoinOrder', 'update table',
        'PQE rewrite', 'likeSQL', 'UNIQ rewrite', 'rewrite', 'update2insert',
        'distributionKey', 'groupBy', 'Join', 'index', 'dictionary']
    
    for k, v in cnt.items():
        x_list.append(k)
        y_list.append(v)
    
    color=['red','black','peru','orchid','deepskyblue', 'gold', 'green', 'blue', 'pink', 'purple', 'yellow', 'gray']
    plt.xticks(x_list, x_label, fontproperties='Times New Roman', size = 10, rotation=60)
    plt.yticks(fontproperties='Times New Roman', size = 10)
    cm = plt.bar(x_list, y_list, color=color)
    autolabel(cm)
    plt.show()


if __name__ == "__main__":
    split_test_valid()
# threshold()
# get_dataset_2()
# threshold_2()
# df = pd.read_csv("RootcauseSQL/data/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500_2023_11_30_diskey.csv")
# df = df[df["duration"] - df["JoinOrder"] > df["duration"] * 0.1]
# df = df[df["duration"] - df["update table"] > df["duration"] * 0.1]
# # label_root_cause = {0: 'JoinOrder', 1: 'update table', 2: 'PQE rewrite', 3: 'likeSQL', 4: 'UNIQ rewrite', 5: 'rewrite', 6: 'update2insert', 7: 'distributionKey', 8: 'groupBy', 9: 'Join', 10: 'index', 11: 'dictionary'}
# # keys = label_root_cause.keys()
# # for k in keys:
# #     df = df[df["duration"] - df[label_root_cause[k]] < df["duration"] * 0.1]
# print(df)

# df = pd.read_csv("RootcauseSQL/data/rootcause_train_test_threshold_10.csv")
# print(df[df["dataset_cls"].str.contains("test")])
# df_new = df[df["dataset_cls"].str.contains("test")]
# label_dict = {}
# for i in range(df_new.shape[0]):
#     label = df_new["threshold_10_pre_label"].iloc[i]
#     label_dict[label] = label_dict.get(label, 0) + 1

# print(label_dict)
# print(len(label_dict))
# split_train_test()
# check_threshold_data()
# threshold_split_train_test()

# df = pd.read_csv("RootcauseSQL/data/rootcause_train_test_threshold_10.csv")
# print(df[df["dataset_cls"] == "test"])
# df = pd.read_csv("RootcauseSQL/data/rootcause_train_test.csv")
# print(df[df["duration"] > 800])
# print(df.shape)
# print(df["query"][0])

# df = pd.read_csv("RootcauseSQL/data/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500_2023_12_04.csv")
# # print(df[df["dataset_cls"] == "train"])
# def f(x):
#     label = json.loads(x)
#     labels = [0] * 12
#     for l in label:
#         labels[l] = 1
#     return json.dumps(labels)
# df["multilabel"] = df["multilabel"].apply(f)
# print(df['multilabel'])

# df.to_csv("RootcauseSQL/data/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500/rootcause_all_raw_log_threshold_10_train_test_3_node_lt_500_2023_12_04.csv", index=False)

# threshold_split_train_test_top()