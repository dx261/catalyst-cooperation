import pandas as pd
import numpy as np
import itertools

def get_chemical_formula(dataset):
    """
    Al   Ni   Si
    0.5  0.5  0
    :return: get_chemical_formula from element mol weigh dataframe Al0.5Ni0.5
    """
    elements_columns = dataset.columns
    dataset = dataset.reset_index()
    chemistry_formula = []
    for i in range(dataset.shape[0]):
        single_formula = []
        for col in elements_columns:
            if (dataset.at[i, col]) > 0:
                # element
                single_formula.append(col)
                # ratio
                single_formula.append(str(dataset.at[i, col]))
        chemistry_formula.append("".join(single_formula))
    return chemistry_formula

if __name__ == "__main__":
    # # 1-生成不同金属氧化物组合
    # df = pd.read_excel("data/负载型未中毒催化剂数据库V2.xlsx")
    # features = df.iloc[:, 4:12]
    # feature_col_name = features.columns
    #
    # search_range = {
    #     col: [i / 100 for i in range(0, 11, 2)]  # 粗步长提升效率
    #     for col in feature_col_name[1:]
    # }
    supports = [i for i in range(1, 6)]  # 先根据氧化物生成样本，再考虑基底
    #
    # uniques = list(search_range.values())
    # print(uniques)
    #
    # def valid_ratios(product_gen):
    #     threshold = 1e-5
    #     for x in product_gen:
    #         non_zero_count = len(x) - np.count_nonzero(np.isclose(x, 0.0, atol=threshold))
    #         suma = np.sum(x)
    #         # if non_zero_count == 5 and suma <= 0.9:  # 控制非零元素个数,并给基底留至少10%的空间
    #         if non_zero_count == 5:  # 控制非零元素个数
    #             print(x)
    #             yield x
    #
    # all_element_ratios = list(valid_ratios(itertools.product(*uniques)))
    #
    # result = pd.DataFrame(all_element_ratios, columns=search_range.keys())
    # result.to_csv("data/高通量样本生成-916.csv")

    # # 2-生成不同金属氧化物组合
    # df = pd.read_csv("data/高通量样本生成-916.csv")
    # df = df.iloc[:, 1:]
    # columns = list(df.columns)
    # print(columns)
    #
    # results = []
    #
    # for _, row in df.iterrows():
    #     metals = list(row)
    #     for support in supports:
    #         results.append(metals + [support])
    #
    # # 保存为新表格
    # columns_new = columns + ["support"]
    # print(columns_new)
    # pd.DataFrame(results, columns=columns_new).to_csv("data/高通量样本完整版-9-16.csv", index=False)

    # # 12-30 六十多种元素
    # df = pd.read_excel("优选元素列表.xlsx")
    # elements = df["element"].tolist()
    #
    # MAX_PERCENT = 10  # 比例总和最大 10%
    # STEP = 2  # 步长
    # MAX_K = 2  # 每个样本最多 3 个元素
    #
    # all_element_ratios = []
    #
    # # 1~3 个元素参与
    # for k in range(MAX_K, MAX_K + 1):
    #     for chosen_elements in itertools.combinations(elements, k):
    #
    #         # 给这 k 个元素分配比例
    #         value_range = range(1, MAX_PERCENT + 1, STEP)
    #
    #         for ratios in itertools.product(value_range, repeat=k):
    #             if sum(ratios) <= MAX_PERCENT:
    #
    #                 sample = dict.fromkeys(elements, 0.0)
    #
    #                 for el, r in zip(chosen_elements, ratios):
    #                     sample[el] = r / 100
    #
    #                 all_element_ratios.append(sample)
    #
    # # 转成 DataFrame 并保存
    # result = pd.DataFrame(all_element_ratios)
    # formula = get_chemical_formula(result)
    # formula = pd.DataFrame(formula, columns=["formula"])
    # formula.to_excel("双元素步长2%样本.xlsx")

    # 2-24 20多元素
    origin_elements = [
        "V",
        "Ce",
        "W",
        "Cu",
        "Fe",
        "Mn",
        "Co",
        "Sb",
        "Sn"
    ]

    df = pd.read_excel("优选元素列表.xlsx")
    new_elements = df["element"].tolist()

    STEP = 10
    TOTAL_UNIT = 10  # 因为 100 / 10


    def ratio_generator_positive(n_elements=5, total_unit=10):
        """
        生成 x1+x2+...+xn = total_unit
        且 xi >= 1
        """
        # 转换为 y1+...+yn = total_unit-n_elements
        remain = total_unit - n_elements

        def backtrack(current, remaining, depth):
            if depth == n_elements - 1:
                current.append(remaining)
                yield current.copy()
                current.pop()
                return

            for i in range(remaining + 1):
                current.append(i)
                yield from backtrack(current, remaining - i, depth + 1)
                current.pop()

        for y_units in backtrack([], remain, 0):
            # 每个 +1
            yield [(y + 1) * STEP for y in y_units]


    output_file = "2-24外推元素高通量样本3旧+2新.csv"

    all_columns = list(set(origin_elements + new_elements))
    pd.DataFrame(columns=all_columns).to_csv(output_file, index=False)

    for origin_combo in itertools.combinations(origin_elements, 3):
        for new_combo in itertools.combinations(new_elements, 2):
            elements = list(origin_combo) + list(new_combo)

            rows = []

            for ratios in ratio_generator_positive(5, TOTAL_UNIT):

                row = dict.fromkeys(all_columns, 0)
                for elem, value in zip(elements, ratios):
                    row[elem] = value

                rows.append(row)

            pd.DataFrame(rows).to_csv(
                output_file,
                mode='a',
                header=False,
                index=False
            )

    ratio = pd.read_csv(output_file)
    formula = get_chemical_formula(ratio)
    formula = pd.DataFrame(formula, columns=["formula"])
    formula.to_csv(output_file)

    print("全部组合生成完成")