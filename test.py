import pandas as pd
import torch
import pickle
# multilabel = torch.randint(0, 2, [12196,5])
# opt_label = torch.rand([12196,5])
# log = torch.rand([12196,13])
# time = torch.rand([12196,7,9])
# df = pd.read_csv('1/data/query_plan.csv')
# with open('./t2.pickle','wb') as f:
#     pickle.dump((df['sql'].values,df['plan_json'].values,time,log,opt_label,multilabel),f)
from .de.model.plan_encoding import PlanEncoder
with open('./t2.pickle','rb') as f:
    sql,plan_json,time,log,opt_label,multilabel = pickle.load(f)

encoder = PlanEncoder(plan_json)
plan = encoder.get()
with open('./t2.pickle','wb') as f:
        pickle.dump(plan,f)

