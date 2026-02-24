import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# all_ele = ["V2O5", "CeO2", "WO3", "CuO", "Fe2O3", "MnO2", "Co2O3"]  # 由于化学式中包含数字，因此每种氧化物用字母代表
all_ele = ["A", "B", "C", "D", "E", "F", "G"]
T50_count = [23, 46, 22, 10, 24, 32, 9]
T90_count = [23, 42, 21, 10, 22, 35, 9]
conv200_count = [23, 49, 23, 10, 24, 37, 9]

# 计算各个元素总的概率
all_elements_prob = [(1 / 3) * x / sum(T50_count) + (1 / 3) * y / sum(T90_count) + (1 / 3) * z / sum(conv200_count) for
                     x, y, z in zip(T50_count, T90_count, conv200_count)]

T50_fenweishu = pd.read_excel("10-21-new_models/oxide_fraction_quantiles_T50.xlsx").to_numpy().tolist()
T90_fenweishu = pd.read_excel("10-21-new_models/oxide_fraction_quantiles_T90.xlsx").to_numpy().tolist()
conv200_fenweishu = pd.read_excel("10-21-new_models/oxide_fraction_quantiles_NOx_Conv_200°C.xlsx").to_numpy().tolist()
print(T50_fenweishu, T90_fenweishu, conv200_fenweishu)

# 绘制饼图，准备数据
labels = all_ele.copy()
sizes = all_elements_prob
# 设置随机种子，以确保每次生成相同的颜色
np.random.seed(9)
# 生成num_colors种暖色、清新颜色
colors = []
num_colors = len(all_ele)

for _ in range(num_colors):
    # 生成RGB颜色值
    r = np.random.uniform(0.7, 1)  # 红色通道范围：0.7-1
    g = np.random.uniform(0.7, 1)  # 绿色通道范围：0.8-1
    b = np.random.uniform(0.7, 1)  # 蓝色通道范围：0.9-1
    colors.append((r, g, b))
# 绘制饼图
for i in range(len(labels)):
    if sizes[i] > 0.03:
        labels[i] = all_ele[i]
    else:
        labels[i] = ''
plt.pie(sizes, labels=labels, colors=colors, autopct=lambda x: '%1.1f%%' % x if x >= 5 else '', startangle=90,
        pctdistance=0.95)
# plt.legend()
plt.axis('equal')

# 显示图表
plt.savefig('element_distribution_pie.png', dpi=500)
# plt.show()


# 轮盘赌选择元素
def roulette(probabilities, num):
    """
    根据概率分布随机选择不重复的元素索引
    :param probabilities: 每个元素的概率 (list or np.array)
    :param num: 需要选择的元素个数
    :return: 选中的元素索引列表
    """
    probabilities = np.array(probabilities)
    probabilities = probabilities / probabilities.sum()  # 归一化
    cumulative = np.cumsum(probabilities)
    result = set()

    while len(result) < num:
        r = np.random.rand()
        idx = np.searchsorted(cumulative, r)
        result.add(idx)

    return list(result)


# ===============================
# 生成每个元素的含量（基于范围随机）
# ===============================
def elements_fraction(elements_range, index):
    """
    根据元素范围生成实际含量（总和不超过50）
    :param elements_range: 每个元素的 [min, max] 范围
    :param index: 已选元素索引
    :return: 各元素的实际含量（四舍五入到0.01）
    """
    raw_values = []
    for idx in index:
        low, high = elements_range[idx]
        val = low + (high - low) * random.random()
        raw_values.append(max(val, 0))

    total = sum(raw_values)
    # 如果总和超过100，则按比例缩放
    if total > 50:
        scaled = [round(v * 50 / total, 2) for v in raw_values]
    else:
        scaled = [round(v, 2) for v in raw_values]

    return scaled


# ===============================
# 生成虚拟样本
# ===============================
def generate_virtual_samples(elements_list, elements_range, elements_prob, num_samples=100000):
    """
    生成虚拟样本
    :param elements_list: 元素名称列表
    :param elements_range: 每个元素的范围 [[min,max],...]
    :param elements_prob: 每个元素被选中的概率
    :param num_samples: 生成样本数
    :return: DataFrame 包含公式字符串
    """
    all_elements = elements_list
    formula_list = []

    for _ in range(num_samples):
        # n_elements = random.randint(4, 7)  # 随机选取 4~7 种元素
        n_elements = 5  # 选择五种元素
        selected_idx = roulette(elements_prob, n_elements)
        selected_elements = [all_elements[i] for i in selected_idx]
        selected_fractions = elements_fraction(elements_range, selected_idx)

        # 拼接成化学式，如 "Fe0.25Co0.25Ni0.25Cu0.25"
        formula = ''.join(f"{el}{frac}" for el, frac in zip(selected_elements, selected_fractions))
        formula_list.append(formula)

    df = pd.DataFrame(formula_list, columns=["formula"])
    return df


elements_range = [[] for i in range(len(all_ele))]
for i in range(len(all_ele)):
    if T50_fenweishu[i][5] == 0:
        elements_range[i].append(T90_fenweishu[i][1])
        elements_range[i].append(T90_fenweishu[i][5])
    elif conv200_fenweishu[i][5] == 0:
        elements_range[i].append(T90_fenweishu[i][1])
        elements_range[i].append(T90_fenweishu[i][5])
    elif T90_fenweishu[i][5] == 0:
        elements_range[i].append(T50_fenweishu[i][1])
        elements_range[i].append(T50_fenweishu[i][5])
    else:
        elements_range[i].append(max(conv200_fenweishu[i][1], T90_fenweishu[i][1],T50_fenweishu[i][1]))
        elements_range[i].append(min(conv200_fenweishu[i][5], T90_fenweishu[i][5],T50_fenweishu[i][5]))

df = generate_virtual_samples(all_ele, elements_range, elements_prob=all_elements_prob, num_samples=10000)
print(df.head())

save_path = "Virture_samples_10_21.xlsx"
df.to_excel(save_path, index=False)
print(f"✅ 虚拟样本已保存至: {save_path}")
