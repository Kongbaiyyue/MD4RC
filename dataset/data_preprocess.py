import pandas as pd

def neg2zero():
    df = pd.read_pickle("data/all_data/add_2023_12_30_2-5-sqlsmith_resplit_lg_500_cost_64_2_random_0.05rate_4.pickle")
    
    def set_opt_rate(x):
        labels = ['JoinOrder', 'update table', 'distributionKey', 'index', 'dictionary']
        opt = []
        opt = [(x["duration"] - x[lab]) / x["duration"] if (x["duration"] - x[lab]) > 0.0  else 0.0 for lab in labels ]
        return opt
        
    df["opt_label_rate"] = df.apply(set_opt_rate, axis=1)
    # print(df[df[""]])
    # df.to_pickle("data/all_data/add_2023_12_30_2-5-sqlsmith_resplit_lg_500_cost_64_neg2zero_rate_4.pickle")
    
    # df.drop(columns=["json_plan_tensor"])
    # df.to_csv("data/all_data/add_2023_12_30_2-5-sqlsmith_resplit_lg_500_cost_64_neg2zero_rate_4.csv", index=False)

def set_multilabel():
    df = pd.read_pickle("data/all_data/add_2023_12_30_2-5-sqlsmith_resplit_lg_500_cost_64_2_random_0.05rate_4.pickle")
    print(df["multilabel"])
    
    def set_m_lab(x):
        labels = ['JoinOrder', 'update table', 'distributionKey', 'index', 'dictionary']
        m_l = [1 if ((x["duration"] - x[lab]) > 0.05 * x["duration"]) else 0 for lab in labels]
        return m_l
    df["multilabel"] = df.apply(axis=1)
    df.to_pickle("data/all_data/add_2023_12_30_2-5-sqlsmith_resplit_lg_500_cost_64_2_random_0.05rate_4_multi_0.05.pickle")

if __name__ == "__main__":
    # neg2zero()
    set_multilabel()