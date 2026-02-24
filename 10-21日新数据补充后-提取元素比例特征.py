import re

import pandas as pd
import numpy as np

df = pd.read_excel("Virture_samples_10_21.xlsx")


def formula_to_dataframe(input_excel, formula_col="formula", output_excel="parsed_formulas.xlsx"):
    """
    从 Excel 中读取化学式列，解析为元素比例 DataFrame，并保存到新 Excel。

    参数：
        input_excel : str
            输入文件路径 (.xlsx)
        formula_col : str
            存放化学式字符串的列名
        output_excel : str
            输出 Excel 文件路径

    返回：
        pd.DataFrame ：包含元素比例的 DataFrame
    """
    # 1. 读取数据
    df = pd.read_excel(input_excel)
    if formula_col not in df.columns:
        raise ValueError(f"未找到列 '{formula_col}'，请确认列名是否正确。")

    formulas = df[formula_col].dropna().astype(str).tolist()

    # 2. 定义解析函数
    def parse_formula(formula):
        pairs = re.findall(r'([A-Z][a-z]*)([0-9.]+)', formula)
        elements = [el for el, _ in pairs]
        fractions = [float(frac) for _, frac in pairs]
        return elements, fractions

    # 3. 收集所有出现过的元素
    all_elements = sorted(set(sum([parse_formula(f)[0] for f in formulas], [])))

    # 4. 生成每行的元素比例字典
    parsed_rows = []
    for f in formulas:
        elements, fractions = parse_formula(f)
        row_dict = {el: frac for el, frac in zip(elements, fractions)}
        parsed_rows.append(row_dict)

    # 5. 转成 DataFrame，并补全缺失元素
    df_elements = pd.DataFrame(parsed_rows).fillna(0)[all_elements]

    # 6. 合并原表（可选）
    result = pd.concat([df.reset_index(drop=True), df_elements.reset_index(drop=True)], axis=1)

    # 7. 保存结果
    result.to_excel(output_excel, index=False)
    print(f"✅ 已保存到: {output_excel}")
    print(f"共解析 {len(all_elements)} 种元素: {', '.join(all_elements)}")

    return result


# ========= 示例调用 =========
if __name__ == "__main__":
    # 假设 Excel 里有一列名为 'formula'
    result_df = formula_to_dataframe(
        input_excel=r"C:\Users\12832\Desktop\氧化脱硝\Virture_samples_10_21.xlsx",
        formula_col="formula",
        output_excel=r"C:\Users\12832\Desktop\氧化脱硝\Virture_samples_10_27_提取元素含量.xlsx"
    )