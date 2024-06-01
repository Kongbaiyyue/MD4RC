import pandas as pd
import numpy as np
import torch


from datasetQF import PlanTreeDataset
from utils import Encoding, collator
from QueryFormer import QueryFormer


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

path = "data/hgprecn-cn-zvp2qdxio002-clone/data/sql_log_dataset_2023-11-05 15:20:00_2023-11-05 15:40:00_rootcause_json_plan_series_cor.csv"
df = pd.read_csv(path)["plan_json"]

df = df[:20]
# print(df.iloc[0])

# encoding_ckpt = torch.load('RootcauseSQL/dataset/QF/checkpoint/encoding.pt')
# encoding = encoding_ckpt['encoding']
encoding = Encoding(None, {'NA': 0})

# 考虑后续合适的特征： Parallel Aware， Startup Cost，total Cost， Plan Rows, Plan Width, Parent Relationship, Scan Direction, Index Name, Shard Prune, 是否有Bitmap Filter
train_ds = PlanTreeDataset(df, None, encoding, None, None, None, None, None)
# print(dataset.collated_dicts[0])
# model = QueryFormer(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
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
args = Args()
device = args.device
model = QueryFormer(emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
                 dropout = args.dropout, n_layers = args.n_layers, \
                 use_sample = True, use_hist = True, \
                 pred_hid = args.pred_hid
                )

rng = np.random.default_rng()
train_idxs = rng.permutation(len(train_ds))
model.to(device=device)
for idxs in chunks(train_idxs, 10):

    batch = collator([train_ds[j] for j in idxs])
    batch = batch.to(device)

    cost_preds = model(batch)
    print(cost_preds)