import pandas as pd
import json

if __name__ == "__main__":
    # df = pd.read_csv("data/2023_12_30_2-5-sqlsmith/2024_01_01_01_join_order_label.csv")
    # pd.set_option("max_colwidth", 1000000)
    # print(df["plan_json"].iloc[0])
    # def f_plan(x):
    #     x = x.replace("0    ", "", 1).replace("Name: QUERY PLAN, dtype: object", "", 1)
    #     try:
    #         print(x)
    #         x = json.dumps(eval(x)[0])
            
    #     except Exception as e:
    #         # print(e)
    #         return "0"
    #     return x
    # df["plan_json"] = df["plan_json"].apply(f_plan)
    # # print(df["plan_json"])
    # df = df[df["plan_json"] != "0"]
    # print(df.shape)
    # df.to_csv("data/2023_12_30_2-5-sqlsmith/2024_01_01_01_join_order_label_2.csv", index=False)
    df = pd.read_pickle("data/all_data/top_dataset_plan_tensor_resplit_lg_500_cost_64_log_rate_label.pickle")
    print(df["opt_label"])
    print(df["opt_label_rate"])
    print(df.columns)
    
