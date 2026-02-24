import pandas as pd

# 示例数据
df_formula = pd.read_excel("Virture_samples_10_21_提取元素含量.xlsx")
df_category = pd.DataFrame({'Support': [1, 2, 3, 4, 5]})

# 笛卡尔积（所有组合）
df_result = df_formula.merge(df_category, how='cross')

print(df_result)
df_result.to_excel("Virture_samples_10_27_最终虚拟样本.xlsx")