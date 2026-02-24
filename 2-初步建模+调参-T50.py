import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pickle
import joblib

# 显示所有行
pd.set_option('display.max_rows', None)
# 显示所有列
pd.set_option('display.max_columns', None)
# 设置列宽不自动省略
pd.set_option('display.max_colwidth', None)
# 设置输出宽度（字符总宽度）
pd.set_option('display.width', None)

if __name__ == '__main__':
    df = pd.read_excel("data/负载型未中毒催化剂数据库V2.xlsx")
    target_name = ["NOx_Conv_200°C", "N2_Selc_200°C", "NOx_Conv_300°C", "N2_Selc_300°C", "T50", "T90"]
    i = 4
    df2 = df[df[target_name[i]].notna()]  # 去掉目标列中有缺失值的行
    feature1 = df2.iloc[:, 4:12]
    # feature2 = df2.iloc[:, 18:24]
    target = df2[target_name[i]]

    std = StandardScaler()
    X = feature1
    Y = target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)
    joblib.dump(std, f"std/standard_scaler_{target_name[i]}.pkl")

    # models = [
    #     LinearRegression(),
    #     Ridge(alpha=100, solver='lsqr'),
    #     Lasso(alpha=4.641588833612772, selection='cyclic'),
    #     ElasticNet(alpha=2.782559402207126, l1_ratio=0.7),
    #     MLPRegressor(activation='relu', solver='adam', hidden_layer_sizes=(100,), max_iter=5000, alpha=0.001,
    #                  learning_rate='constant'),
    #     SVR(C=104, kernel='rbf', gamma=0.1, epsilon=0.2),
    #     RandomForestRegressor(random_state=89, n_estimators=100, max_depth=None, max_features='sqrt',
    #                           min_samples_leaf=2, min_samples_split=5),
    #     AdaBoostRegressor(learning_rate=0.3, random_state=1, n_estimators=40),
    #     ExtraTreesRegressor(max_depth=None, n_estimators=200, max_features='sqrt', min_samples_leaf=2,
    #                         min_samples_split=2),
    #     GradientBoostingRegressor(learning_rate=0.01, max_depth=5, min_samples_leaf=3, min_samples_split=2,
    #                               n_estimators=100, random_state=2),
    #     XGBRegressor(learning_rate=0.01, n_estimators=150, max_depth=4, random_state=0, gamma=0.1,
    #                  reg_alpha=0.1, reg_lambda=0.1, colsample_bytree=1.0, min_child_weight=3),
    #     lgb.LGBMRegressor(random_state=42, learning_rate=0.01, n_estimators=300, max_depth=3, min_child_samples=10,
    #                       num_leaves=15, subsample=0.6, reg_alpha=1, reg_lambda=1, verbose=-1),
    # ]  # 针对T50的调参结果
    #
    # models = [
    #     LinearRegression(),
    #     Ridge(),
    #     Lasso(),
    #     ElasticNet(),
    #     MLPRegressor(),
    #     SVR(),
    #     RandomForestRegressor(),
    #     AdaBoostRegressor(),
    #     ExtraTreesRegressor(),
    #     GradientBoostingRegressor(),
    #     XGBRegressor(),
    #     lgb.LGBMRegressor(),
    # ]

    models_7features = [
        # LinearRegression(),
        # Ridge(),
        # Lasso(),
        # ElasticNet(),
        # MLPRegressor(),
        # SVR(),
        # RandomForestRegressor(),
        # AdaBoostRegressor(learning_rate=0.4, n_estimators=45, loss='square'),
        ExtraTreesRegressor(max_depth=20, n_estimators=100, max_features='sqrt', min_samples_leaf=1, min_samples_split=5),
        GradientBoostingRegressor(learning_rate=0.01, max_depth=7, min_samples_leaf=3, min_samples_split=2, n_estimators=185),
        XGBRegressor(learning_rate=0.075, max_depth=3, n_estimators=50, colsample_bytree=1,
                     gamma=0.5, min_child_weight=1, subsample=1, reg_alpha=0.01, reg_lambda=1.0, random_state=0),
        # lgb.LGBMRegressor(),
    ]

    result_dict = {}

    for model in models_7features:
        model.fit(X_train_std, Y_train)
        Y_pred = model.predict(X_test_std)
        Y_pred_train = model.predict(X_train_std)
        cv_predict = cross_val_predict(model, X_train_std, Y_train, cv=10)
        df_pearson_cv = pd.DataFrame({'x': Y_train, 'y': cv_predict})
        r_cv = df_pearson_cv.corr(method='pearson').loc['x', 'y']
        df_pearson_test = pd.DataFrame({'x': Y_test, 'y': Y_pred})
        r_test = df_pearson_test.corr(method='pearson').loc['x', 'y']
        result_dict[type(model).__name__] = [r2_score(Y_train, Y_pred_train), r2_score(Y_test, Y_pred), r_test, r2_score(Y_train, cv_predict), r_cv]
        with open(f"models/{type(model).__name__}_{target_name[i]}_.pkl", "wb") as f:
            pickle.dump(model, f)

    result_dict = pd.DataFrame(result_dict, index=["train_r2", "test_r2", "test_r", "cv10_r2", "cv10_r"])
    print("调参后：", result_dict)

    """
    # 分组建模
    grouped = df2.groupby(df2.iloc[:, 4])
    feature1 = df2.iloc[:, 5:18]
    feature2 = df2.iloc[:, 18:24]

    # 遍历每个组并处理对应的 feature1、feature2
    for group_name, group_df in grouped:
        result_dict = {}
        print(f"组名：{group_name}")
        # 获取该组的 feature1 和 feature2
        feature1_group = group_df.iloc[:, 5:18]
        feature2_group = group_df.iloc[:, 18:24]
        target_group = group_df[target_name[i]]

        std = StandardScaler()
        X = feature2_group
        Y = target_group

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=153)
        X_train_std = std.fit_transform(X_train)
        X_test_std = std.transform(X_test)
        

        try:
            for model in models:
                model.fit(X_train_std, Y_train)
                Y_pred = model.predict(X_test_std)
                Y_pred_train = model.predict(X_train_std)
                cv_predict = cross_val_predict(model, X_train_std, Y_train, cv=10)
                df_pearson_cv = pd.DataFrame({'x': Y_train, 'y': cv_predict})
                r_cv = df_pearson_cv.corr(method='pearson').loc['x', 'y']
                df_pearson_test = pd.DataFrame({'x': Y_test, 'y': Y_pred})
                r_test = df_pearson_test.corr(method='pearson').loc['x', 'y']
                result_dict[type(model).__name__ + f"group{group_name}"] = [r2_score(Y_train, Y_pred_train),
                                                                            r2_score(Y_test, Y_pred), r_test,
                                                                            r2_score(Y_train, cv_predict), r_cv]
                with open(f"models/{type(model).__name__}_{target_name[i]}_group{group_name}.pkl", "wb") as f:
                    pickle.dump(model, f)
        except Exception as e:
            print(e)
            continue

        result_dict = pd.DataFrame(result_dict, index=["train_r2", "test_r2", "test_r", "cv10_r2", "cv10_r"])
        result_dict.to_csv(f"组别_特征2_{group_name}.csv")
    """
    # 调参
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    # # 针对SVR：构建 pipeline：标准化 + 回归
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('svr', SVR())
    # ])
    #
    # # 参数网格（只有两个能调）
    # param_grid = {
    #     'svr__kernel': ['rbf'],  # 你也可以试 'linear', 'poly'
    #     'svr__C': [i for i in range(100, 105)],
    #     'svr__epsilon': [0.2],
    #     'svr__gamma': [i/10 for i in range (0, 2)]
    # }

    # 网格搜索
    # grid = GridSearchCV(pipe, param_grid, cv=10, scoring='r2')
    # grid.fit(X_train, Y_train)

    # print("最优参数：", grid.best_params_)
    # print("最优 R² 分数：", grid.best_score_)
    # print("调参前：", result_dict)

    # # 针对adaboost：基础模型（可选）
    # base_model = DecisionTreeRegressor(max_depth=3)
    # # 构建 AdaBoost 模型
    # abr = AdaBoostRegressor(estimator=base_model, random_state=0)
    # # 参数网格
    # param_grid = {
    #     'n_estimators': [i for i in range(0, 50, 5)],
    #     'learning_rate': [i/10 for i in range(1, 10)],
    #     'loss': ['linear', 'square']
    # }
    #
    # # 网格搜索 + 5折交叉验证
    # grid = GridSearchCV(abr, param_grid, cv=10, scoring='r2', n_jobs=-1)  # GridSearchCV和之前计算cross_val_predict的评估方式不同
    #
    # # 拟合
    # grid.fit(X_train, Y_train)
    # print("最优参数：", grid.best_params_)
    # print("最优R²：", grid.best_score_)

    # # Ridge回归的pipeline：标准化 + 回归
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('ridge', Ridge())
    # ])
    #
    # # 网格参数
    # param_grid = {
    #     'ridge__alpha': [0.01, 0.1, 1, 10, 100],
    #     'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr']
    # }

    # # 创建 lasso的 pipeline（标准化 + Lasso）
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('lasso', Lasso(max_iter=10000))
    # ])
    #
    # # 设置参数网格：alpha 是正则化强度（越大越稀疏）
    # param_grid = {
    #     'lasso__alpha': np.logspace(-4, 2, 10),  # 从 0.0001 到 10 的对数间隔
    #     'lasso__selection': ['cyclic', 'random']
    # }

    # # 构建 pipeline（标准化 + ElasticNet）
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('enet', ElasticNet(max_iter=10000))
    # ])
    #
    # # 参数网格
    # param_grid = {
    #     'enet__alpha': np.logspace(-4, 1, 10),  # 正则化强度
    #     'enet__l1_ratio': np.linspace(0.1, 0.9, 9),  # L1 和 L2 的权重比（0=L2, 1=L1）
    # }

    # # 构建 Pipeline：标准化 + MLP
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('mlp', MLPRegressor(max_iter=5000, random_state=42))
    # ])
    #
    # # 参数网格
    # param_grid = {
    #     'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 50)],
    #     'mlp__activation': ['relu', 'tanh'],
    #     'mlp__solver': ['adam'],  # 可选：['adam', 'lbfgs', 'sgd']
    #     'mlp__alpha': [1e-5, 1e-4, 1e-3],  # L2 正则项
    #     'mlp__learning_rate': ['constant', 'adaptive']
    # }

    # # 构建 pipeline（注意：标准化对树模型不是必须的，但便于统一处理）
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('rf', RandomForestRegressor())
    # ])
    #
    # # 参数网格
    # param_grid = {
    #     'rf__n_estimators': [100, 200, 300],
    #     'rf__max_depth': [None, 5, 10, 20],
    #     'rf__min_samples_split': [2, 5, 10],
    #     'rf__min_samples_leaf': [1, 2, 4],
    #     'rf__max_features': ['auto', 'sqrt', 'log2']
    #     'rf__random_state': [i for i in range(0, 100, 1)]
    # }

    # # 极端随机树
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('et', ExtraTreesRegressor(random_state=42))
    # ])
    #
    # # 参数网格
    # param_grid = {
    #     'et__n_estimators': [100, 200, 300],
    #     'et__max_depth': [None, 10, 20],
    #     'et__min_samples_split': [2, 5],
    #     'et__min_samples_leaf': [1, 2],
    #     'et__max_features': ['auto', 'sqrt', 'log2']
    # }

    # # 构建 gbdt pipeline
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('gbr', GradientBoostingRegressor(random_state=42))
    # ])
    #
    # # 参数网格
    # param_grid = {
    #     'gbr__n_estimators': [i for i in range(50, 200, 5)],
    #     'gbr__learning_rate': [0.01, 0.1, 0.2],
    #     'gbr__max_depth': [3, 4, 5, 6, 7],
    #     'gbr__min_samples_split': [2, 3, 4, 5],
    #     'gbr__min_samples_leaf': [1, 2, 3]
    # }

    # # 构建 XGB pipeline
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),  # 可选，对树模型不是必须
    #     ('xgb', XGBRegressor(learning_rate=0.075, max_depth=3, n_estimators=50, colsample_bytree=1,
    #                          gamma=0.5, min_child_weight=1, subsample=1, reg_alpha=0.01, reg_lambda=1.0, random_state=0))
    # ])

    # 设置参数网格
    # param_grid = {
    # 'xgb__n_estimators': [i for i in range(50, 200, 5)],
    # 'xgb__learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.2],
    # 'xgb__max_depth': [2, 3, 4, 5, 6, 7],
    # 'xgb__min_child_weight': [1, 2, 3, 4, 5, 6],
    # 'xgb__gamma': [0, 0.05, 0.1, 0.2, 0.3, 0.5],
    # 'xgb__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    # 'xgb__colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # 'xgb__reg_alpha': [0, 0.001, 0.01, 0.1, 0.5, 1.0],
    # 'xgb__reg_lambda': [0.1, 0.5, 1.0, 1.5, 2.0, 3.0],
    # 'xgb__random_state': [i for i in range(0, 150, 1)],
    # }

    # 构建 lightgbm pipeline
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('lgb', lgb.LGBMRegressor(random_state=42))
    # ])
    #
    # param_grid = {
    # 'lgb__n_estimators': [100, 200, 300, 400, 500],
    # 'lgb__learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
    # 'lgb__max_depth': [3, 4, 5, 6, 7, 8, 10],
    # 'lgb__num_leaves': [15, 31, 63, 127],
    # 'lgb__min_child_samples': [5, 10, 20, 30, 50],
    # 'lgb__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    # 'lgb__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    # 'lgb__reg_alpha': [0.0, 0.01, 0.1, 1.0],
    # 'lgb__reg_lambda': [0.0, 0.01, 0.1, 1.0]
    # }

    # # 网格搜索 + 10折交叉验证
    # grid = GridSearchCV(pipe, param_grid, cv=10, scoring='r2', n_jobs=-1)
    #
    # # 拟合
    # grid.fit(X_train_std, Y_train)
    #
    # # 输出最佳参数和得分
    # print("最优参数：", grid.best_params_)
    # print("最优 R² 分数（CV）：", grid.best_score_)
