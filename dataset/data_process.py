import pandas as pd
import random

def set_zero_opt():
    df = pd.read_pickle("data/2023_12_15_all_sql/2023_12_21_08_labels_top_dataset_plan_tensor_resplit_lg_500_cost_64_log.pickle")
    
    print(df.shape)
    print(df.columns)
    
    
    
    
if __name__ == "__main__":
    set_zero_opt()