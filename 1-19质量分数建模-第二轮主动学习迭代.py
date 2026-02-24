import pandas as pd
import numpy as np
import scipy
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score

# 线性模型
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor, RANSACRegressor, SGDRegressor
)

# SVM
from sklearn.svm import SVR

# 邻域方法
from sklearn.neighbors import KNeighborsRegressor

# 集成学习
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor
)

# 神经网络
from sklearn.neural_network import MLPRegressor

# XGBoost
from xgboost import XGBRegressor

# 特征筛选
from sklearn.feature_selection import SelectKBest, f_regression

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "BayesianRidge": BayesianRidge(),

    "SVR_RBF": SVR(kernel='rbf', C=10, gamma='scale'),

    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=300, random_state=42),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

if __name__ == '__main__':
    target = ["NOx_Conv_200°C", "N2_Selc_200°C", "NOx_Conv_300°C", "N2_Selc_300°C", "T50", "T90"]
    i = 5
    df = pd.read_excel("data/1-19质量分数直接建模.xlsx")
    df.dropna(subset=target[i], inplace=True)

    df_target = df.iloc[:, 8:]
    df_ratio = df.iloc[:, :8]

    y = df_target[target[i]]
    X = df_ratio


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0  # T90是0，其他用42
    )

    results = []

    for name, model in models.items():
        print(f"训练模型：{name}")
        model.fit(X_train, y_train)

        # CV R²
        cv_pred = cross_val_predict(model, X_train, y_train, cv=10)
        cv_r2 = r2_score(y_train, cv_pred)
        cv_r = scipy.stats.pearsonr(y_train, cv_pred)[0]

        # 测试集 R²
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_r = scipy.stats.pearsonr(y_test, y_pred)[0]

        # 综合评分
        score = 0.5 * cv_r2 + 0.5 * test_r2
        print(f"{name} 模型评分：{score:.4f}\n")
        print(f"{name} 模型 testR²：{test_r2}")
        print(f"{name} 模型 cvR²：{cv_r2}")

        results.append([name, cv_r2, test_r2, score, cv_r, test_r])

    results_df = pd.DataFrame(results, columns=["model", "cv_r2", "test_r2", "score", "cv_r", "test_r"])
    results_df = results_df.sort_values("score", ascending=False)
    print("\n所有模型评分：")
    print(results_df)
    results_df.to_csv(f"data/{target[i]}_model_score_1-19质量分数直接建模.csv", index=False)
