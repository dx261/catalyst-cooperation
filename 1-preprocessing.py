import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_excel("data/负载型未中毒催化剂数据库.xlsx")
    print(df.shape)
    feature1 = df.iloc[:, 4:18]
    feature2 = df.iloc[:, 18:24]
    print(feature1, feature2)