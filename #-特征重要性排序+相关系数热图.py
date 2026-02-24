import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
import pickle
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# 显示所有行
pd.set_option('display.max_rows', None)
# 显示所有列
pd.set_option('display.max_columns', None)
# 设置列宽不自动省略
pd.set_option('display.max_colwidth', None)
# 设置输出宽度（字符总宽度）
pd.set_option('display.width', None)


if __name__ == '__main__':
    df = pd.read_excel("data/负载型未中毒催化剂数据库.xlsx")
    target_name = ["NOx_Conv_200°C", "N2_Selc_200°C", "NOx_Conv_300°C", "N2_Selc_300°C", "T50", "T90"]
    i = 0
    df2 = df[df[target_name[i]].notna()]
    target = df2[target_name[i]]

    grouped = df2.groupby(df2.iloc[:, 4])
    feature1 = df2.iloc[:, 4:17]
    feature2 = df2.iloc[:, 18:24]

    # 遍历每个组并处理对应的 feature1、feature2
    # for group_name, group_df in grouped:
    #     print(f"组名：{group_name}")
    #
    #     # 获取该组的 feature1 和 feature2
    #     feature1_group = group_df.iloc[:, 5:18]
    #     feature2_group = group_df.iloc[:, 18:24]
    #
    #     # 示例操作：输出形状
    #     print("feature1 shape:", feature1_group.shape)
    #     print("feature2 shape:", feature2_group.shape)
    #
    #     # 可继续处理，例如保存、聚合、建模等

    X = feature1
    Y = target
    std = StandardScaler()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train_std = std.fit_transform(X_train)
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)
    X_test_std = std.transform(X_test)
    X_test_std = pd.DataFrame(X_test_std, columns=X_train.columns)

    model = pickle.load(open(f"models/RandomForestRegressor_{target_name[i]}_.pkl", "rb"))
    # model = pickle.load(open(f"models/XGBRegressor_{target_name[i]}_.pkl", "rb"))
    # explainer = shap.Explainer(model, X_train_std)
    explainer = shap.TreeExplainer(model)  # 用于树模型
    shap_values = explainer(X_train_std)
    #
    # 可视化特征重要性（排序）
    plt.figure(figsize=(15, 6))
    shap.summary_plot(shap_values, X_train_std, show=False, color_bar=True)
    plt.show()
    feature_importance = shap.plots.bar(shap_values, max_display=10)  # 显示前10个重要特征

    # # 皮尔逊相关系数热图
    # corr_matrix = feature1.corr(method='pearson')
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True,
    #             cbar_kws={'shrink': .5}, linewidths=0.5)
    # plt.title('Pearson Correlation Coefficient Heatmap')
    # plt.tight_layout()
    # plt.show()
