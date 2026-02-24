import pandas as pd
import numpy as np
import itertools

if __name__ == "__main__":
    df = pd.read_excel("data/负载型未中毒催化剂数据库V2.xlsx")
    features = df.iloc[:, 4:12]
    feature_col_name = features.columns
    print(feature_col_name)
