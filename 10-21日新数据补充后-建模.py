import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor, RANSACRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

# ========= 用户配置部分 ==========
target_name = ["NOx_Conv_200°C", "N2_Selc_200°C", "NOx_Conv_300°C",
               "N2_Selc_300°C", "T50", "T90"]
i = 5
TARGET_COL = target_name[i]
DATA_PATH = r"C:\Users\12832\Desktop\氧化脱硝\data\负载型未中毒催化剂数据库-10-19.xlsx"
MODEL_SAVE_PATH = f"10-21-new_models/best_model_{TARGET_COL}.pkl"
SCALER_SAVE_PATH = f"10-21-new_models/scaler_{TARGET_COL}.pkl"

# ========= 读取数据 ==========
df = pd.read_excel(DATA_PATH)
df2 = df[df[TARGET_COL].notna()]
X = df2[["Support_ Categorized", "V2O5", "CeO2", "WO3", "CuO", "Fe2O3", "MnO2", "Co2O3"]]
y = df2[TARGET_COL]

# ========= 划分训练/测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ========= 标准化 ==========
scaler = StandardScaler()
# scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========= 定义模型集合 ==========
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "BayesianRidge": BayesianRidge(),
    "HuberRegressor": HuberRegressor(),
    "RANSACRegressor": RANSACRegressor(random_state=42),

    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    # "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42),
    # "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42, n_estimators=200),

    # 给T50和T90用
    "ExtraTrees": ExtraTreesRegressor(max_depth=20, n_estimators=100, max_features='sqrt', min_samples_leaf=1, min_samples_split=5),
    "GradientBoosting": GradientBoostingRegressor(learning_rate=0.01, max_depth=7, min_samples_leaf=3, min_samples_split=2, n_estimators=185),
    "XGBoost": XGBRegressor(learning_rate=0.075, max_depth=3, n_estimators=50, colsample_bytree=1,
                 gamma=0.5, min_child_weight=1, subsample=1, reg_alpha=0.01, reg_lambda=1.0, random_state=0),

    "SVR": SVR(kernel="rbf"),
    "KNN": KNeighborsRegressor(),
    # "XGBoost": XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.05),
    # "LightGBM": LGBMRegressor(random_state=42, n_estimators=300, learning_rate=0.05),
}

# ========= 十折交叉验证 ==========
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = {}

print("开始模型评估...\n")
for name, model in models.items():
    # cross_val_predict 会自动划分折叠集，所以要传入标准化后的数据
    y_pred_cv = cross_val_predict(model, X_train_scaled, y_train, cv=kf, n_jobs=-1)
    r2_cv = r2_score(y_train, y_pred_cv)

    # 再用整个训练集拟合模型，并在测试集上预测
    model.fit(X_train_scaled, y_train)
    y_pred_test = model.predict(X_test_scaled)
    r2_test = r2_score(y_test, y_pred_test)

    results[name] = {"CV_R2": r2_cv, "Test_R2": r2_test}
    print(f"{name:<20} CV_R² = {r2_cv:.4f} | Test_R² = {r2_test:.4f}")

# ========= 找出最佳模型 ==========
best_model_name = max(results, key=lambda k: results[k]["Test_R2"])
best_model = models[best_model_name]
best_cv_r2 = results[best_model_name]["CV_R2"]
best_test_r2 = results[best_model_name]["Test_R2"]

# ===== 绘制真实值 vs 预测值图 =====
plt.figure(figsize=(5, 5))
y_pred_test = best_model.predict(X_test_scaled)
r2_test = r2_score(y_test, y_pred_test)
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.7, label="Predicted vs True")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label="Ideal: y = x")
plt.title(f"{best_model_name}\nR² = {r2_test:.3f}", fontsize=11)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.tight_layout()
plt.show()

print("\n===============================")
print(f"最佳模型: {best_model_name}")
print(f"CV_R²: {best_cv_r2:.4f}")
print(f"Test_R²: {best_test_r2:.4f}")
print("===============================\n")

# ========= 保存模型与标准化器 ==========
best_model.fit(np.vstack([X_train_scaled, X_test_scaled]), np.hstack([y_train, y_test]))
joblib.dump(best_model, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)

print(f"✅ 模型已保存到: {MODEL_SAVE_PATH}")
print(f"✅ 归一化器已保存到: {SCALER_SAVE_PATH}")

'''
T90最佳模型（加调参的模型）: XGBoost
CV_R²: 0.3318
Test_R²: 0.6871

T50最佳模型（加调参的模型）: GradientBoosting
CV_R²: 0.4283
Test_R²: 0.5579

conv200最佳模型: GradientBoosting
CV_R²: 0.4827
Test_R²: 0.6114
'''