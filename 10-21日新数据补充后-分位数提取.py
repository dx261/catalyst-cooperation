import pandas as pd
import numpy as np

# ========= 用户配置部分 ==========
target_name = ["NOx_Conv_200°C", "N2_Selc_200°C", "NOx_Conv_300°C", "N2_Selc_300°C", "T50", "T90"]
i = 0
TARGET_COL = target_name[i]
DATA_PATH = r"C:\Users\12832\Desktop\氧化脱硝\data\负载型未中毒催化剂数据库-10-19.xlsx"

# ========= 读取数据 ==========
df = pd.read_excel(DATA_PATH)
df2 = df[df[TARGET_COL].notna()]  # 只保留目标不为空的样本
X = df2[["V2O5", "CeO2", "WO3", "CuO", "Fe2O3", "MnO2", "Co2O3"]]
y = df2[TARGET_COL]

# ========= 统计每列非零值个数 ==========
nonzero_counts = (X != 0).sum()
print("各特征非零值个数：")
print(list(nonzero_counts), "\n")

# ========= 计算每种氧化物的分位数（仅对非零值） ==========
quantile_dict = {}

for col in X.columns:
    nonzero_values = X[col][X[col] != 0]
    if len(nonzero_values) == 0:
        quantile_dict[col] = [np.nan] * 5
    else:
        q = np.percentile(nonzero_values, [0, 25, 50, 75, 100])
        quantile_dict[col] = q

# ========= 结果汇总为 DataFrame ==========
quantile_df = pd.DataFrame(
    quantile_dict,
    index=["min", "25%", "50%", "75%", "max"]
).T  # 转置让每行代表一个氧化物

# ========= 输出结果 ==========
print("各氧化物非零含量的分位数：")
print(quantile_df.round(4))

# ========= 可选：保存为 Excel ==========
save_path = rf"C:\Users\12832\Desktop\氧化脱硝\10-21-new_models\oxide_fraction_quantiles_{target_name[i]}.xlsx"
quantile_df.to_excel(save_path)
print(f"\n分位数结果已保存到：{save_path}")

# ========= 轮盘赌 ==========

