import pickle

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import lightgbm as lgb
pd.set_option('display.max_columns', None)

def split_by_distribution(df, column, test_size=0.2, bins=10, random_state=None, plot=True):
    print("Splitting by distribution...")
    df = df.copy()

    # df['_bin'] = pd.qcut(df[column], q=bins, duplicates='drop')  # 等频分箱
    print(df)
    df['_bin'] = pd.cut(df[column], bins=bins)  # 等宽分箱

    # 2. 每个分箱中按比例采样作为测试集
    test_df = df.groupby('_bin', group_keys=False).apply(
        lambda x: x.sample(frac=test_size, random_state=random_state)
    )

    # ✅ 修复点：先提取 test_df 中的 `_bin` 列副本
    test_bin_counts = test_df['_bin'].value_counts().sort_index()

    # 3. 构造训练集、测试集并去除 `_bin` 列
    train_df = df.drop(index=test_df.index).drop(columns=['_bin'])
    test_df = test_df.drop(columns=['_bin'])

    # ✅ 再提取原始总分布
    if plot:
        original_counts = df['_bin'].value_counts().sort_index()

        # 画图
        plt.figure(figsize=(10, 6))
        width = 0.35
        indices = range(len(original_counts))

        plt.bar(indices, original_counts.values, width=width, label='Original', alpha=0.7)
        plt.bar([i + width for i in indices], test_bin_counts.values, width=width, label='Test Sample', alpha=0.7)

        plt.xticks([i + width / 2 for i in indices], [str(c) for c in original_counts.index], rotation=45)
        plt.xlabel(f'{column} Bin Interval')
        plt.ylabel('Sample Count')
        plt.title(f'Original vs Test Sample Distribution of {column}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/第二类变量_分布_{column}.png")
        plt.clf()

    return train_df, test_df

if __name__ == '__main__':
    df = pd.read_excel("data/负载型催化剂-第二类变量-建模用.xlsx")
    target_name = ["NOx_Conv_200°C", "N2_Selc_200°C", "NOx_Conv_300°C", "N2_Selc_300°C", "T50", "T100"]

    models = [
        LinearRegression(),
        Ridge(random_state=1),
        Lasso(random_state=1),
        ElasticNet(random_state=1),
        MLPRegressor(random_state=1),
        SVR(),
        RandomForestRegressor(random_state=1),
        AdaBoostRegressor(random_state=1),
        ExtraTreesRegressor(random_state=1),
        GradientBoostingRegressor(random_state=1),
        XGBRegressor(random_state=1),
        lgb.LGBMRegressor(random_state=1),
    ]

    result_dict = {}

    for target in target_name:
        df2 = df[df[target].notna()]  # 去掉目标列中有缺失值的行

        feature = df2.iloc[:, 1:6]
        tar = df2[target]

        std = StandardScaler()
        X = feature
        Y = tar

        df3 = pd.concat([X, Y], axis=1)

        # 根据数据的分布划分数据集与标准化
        train, test = split_by_distribution(df3, target, test_size=0.2, bins=10, random_state=1, plot=True)
        X_train, Y_train = train[feature.columns.tolist()], train[target]
        X_test, Y_test = test[feature.columns.tolist()], test[target]
        print(Y_train, Y_test)

        # 普通方法
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        X_train_std = std.fit_transform(X_train)
        X_test_std = std.transform(X_test)
        joblib.dump(std, f"std/第二类变量_standard_scaler_{target}.pkl")

        for model in models:
            model.fit(X_train_std, Y_train)
            Y_pred = model.predict(X_test_std)
            Y_pred_train = model.predict(X_train_std)
            cv_predict = cross_val_predict(model, X_train_std, Y_train, cv=10)
            df_pearson_cv = pd.DataFrame({'x': Y_train, 'y': cv_predict})
            r_cv = df_pearson_cv.corr(method='pearson').loc['x', 'y']
            df_pearson_test = pd.DataFrame({'x': Y_test, 'y': Y_pred})
            r_test = df_pearson_test.corr(method='pearson').loc['x', 'y']
            result_dict[type(model).__name__] = [r2_score(Y_train, Y_pred_train), r2_score(Y_test, Y_pred), r_test,
                                                 r2_score(Y_train, cv_predict), r_cv]
            with open(f"models/{type(model).__name__}_{target}_.pkl", "wb") as f:
                pickle.dump(model, f)

            plt.scatter(Y_train, cv_predict, color="red", s=5)
            for i in range(len(Y_train)):
                plt.text(list(Y_train)[i], list(cv_predict)[i], str(i), fontsize=8, ha='center', va='bottom', color='black')
            plt.plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)],
                     color='red', linestyle='--', label='y = x')
            plt.xlabel("True Value")
            plt.ylabel("Predict Value")
            plt.savefig(f"figures/第二类变量_cv10_{target}_{type(model).__name__}.png")
            plt.clf()

            # 可视化特征重要性（排序）
            try:
                explainer = shap.TreeExplainer(model)  # 用于树模型
            except:
                explainer = shap.PartitionExplainer(model.predict, X_train_std)
            shap_values = explainer(X_train_std)
            # shap.summary_plot(shap_values, X_train_std, show=False, color_bar=True)
            plt.subplots_adjust(
                top=0.948,
                bottom=0.197,
                left=0.328,
                right=0.93,
                hspace=0.2,
                wspace=0.2
            )
            shap_values.feature_names = feature.columns.tolist()
            shap.plots.bar(shap_values, max_display=10, show=False)
            plt.savefig(f"figures/第二类变量_特征重要性排序_{target}_{type(model).__name__}.png")
            plt.clf()

        result_dict = pd.DataFrame(result_dict, index=["train_r2", "test_r2", "test_r", "cv10_r2", "cv10_r"])
        result_dict.to_excel(f"data/第二类变量_建模结果_{target}.xlsx")
        print("建模结果：", result_dict)
