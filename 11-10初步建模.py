import joblib
import numpy as np
import pandas as pd
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
    "SGDRegressor": SGDRegressor(max_iter=2000, tol=1e-3),

    "SVR_RBF": SVR(kernel='rbf', C=10, gamma='scale'),
    "SVR_RBF_HV":SVR(C=1642, epsilon=13.6, gamma=0.44),

    # "KNN": KNeighborsRegressor(n_neighbors=5),

    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=300, random_state=42),

    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(64, 64),
                                 max_iter=2000, random_state=42),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

# 目标
target = ["NOx_Conv_200°C", "N2_Selc_200°C", "NOx_Conv_300°C", "N2_Selc_300°C", "T50", "T90"]
i = 0

df_magpie = pd.read_excel("data/12-1提取matminer特征.xlsx").dropna()
df_target = df_magpie.iloc[:, 2:8]
col = pd.read_excel("data/人工筛选后特征.xlsx")
new_col = list(col['features'])

df_magpie = df_magpie[new_col]

y = df_target[target[i]]
X = df_magpie[df_magpie.columns.difference([target[i]])]


# ------------------------------------------
# ★ 新增：SelectKBest 特征筛选（非常基础）
# ------------------------------------------
k = len(new_col)
selector = SelectKBest(score_func=f_regression, k=k)  # 可改 k=20,40...
X_selected = selector.fit_transform(X, y)

selected_feature_names = X.columns[selector.get_support()]
print("入选特征：")
print(selected_feature_names.tolist())


# 训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

results = []

print("开始批量建模...\n")

for name, model in models.items():
    print(f"训练模型：{name}")
    model.fit(X_train, y_train)

    # CV R²
    cv_pred = cross_val_predict(model, X_train, y_train, cv=10)
    cv_r2 = r2_score(y_train, cv_pred)

    # 测试集 R²
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)

    # 综合评分
    score = 0.5 * cv_r2 + 0.5 * test_r2
    print(f"{name} 模型评分：{score:.4f}\n")
    print(f"{name} 模型 testR²：{test_r2}")
    print(f"{name} 模型 cvR²：{cv_r2}")

    results.append([name, cv_r2, test_r2, score])


# 输出结果
results_df = pd.DataFrame(results, columns=["model", "cv_r2", "test_r2", "score"])

print("\n所有模型评分：")
print(results_df)

best = results_df.iloc[results_df["score"].idxmax()]
print("\n==============================")
print(f"{target[i]}最佳模型")
print("==============================")
print(best)

best.to_excel(f"data/{target[i]}_model_score.xlsx")

import matplotlib.pyplot as plt

best_model_name = best["model"]
print(f"\n为最佳模型绘制散点图：{best_model_name}")

# 取出最佳模型对象
best_model = models[best_model_name]

# 用训练集重新训练最佳模型（保证一致性）
best_model.fit(X_train, y_train)
joblib.dump(best_model, f"models/12月元素外推模型_{target[i]}_best_model_{best_model_name}.pkl")

# 十折交叉验证预测（这一部分是关键）
cv_pred_best = cross_val_predict(best_model, X_train, y_train, cv=10)

# 创建散点图
plt.figure(figsize=(6, 6))
plt.scatter(y_train, cv_pred_best, alpha=0.7, edgecolors='k', s=50)

# y = x 理想线
min_val = min(y_train.min(), cv_pred_best.min())
max_val = max(y_train.max(), cv_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.xlabel("True Values (Train Set)")
plt.ylabel("Predicted Values (10-Fold CV)")
plt.title(f"{best_model_name} - 10-Fold CV True vs Predicted\nTarget: {target[i]}")
plt.grid(True)
plt.tight_layout()
plt.show()


'''

'''