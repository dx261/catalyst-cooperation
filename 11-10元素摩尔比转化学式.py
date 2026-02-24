import pandas as pd
import numpy as np

def df_to_formula(df: pd.DataFrame) -> pd.Series:
    """
    将 DataFrame 中每行的元素摩尔比拼接为化学式字符串。
    例：
        Fe | O | C
        2  | 3 | 0   → Fe2O3
        1  | 2 | 1   → FeO2C
    """
    formulas = []

    for _, row in df.iterrows():
        formula = ""
        for elem, val in row.items():
            if val == 0 or pd.isna(val):
                continue
            if abs(val - 1) < 1e-8:
                formula += f"{elem}"
            else:
                # 去掉多余小数
                if isinstance(val, float):
                    val_str = f"{val:.4g}".rstrip("0").rstrip(".")
                else:
                    val_str = str(val)
                formula += f"{elem}{val_str}"
        formulas.append(formula)

    return pd.Series(formulas, name="formula")

if __name__ == "__main__":
    # 元素摩尔比转化学式
    df = pd.read_excel("data/1-19质量分数直接建模.xlsx")
    df = df.iloc[:, :9]
    formula = df_to_formula(df)
    formula.to_excel("data/1-19质量分数直接建模-转质量分数比化学式.xlsx")


    # df = pd.read_excel("data/12-1元素摩尔比转化学式.xlsx")
    # # 去除空字符串和 NaN
    # df = df.replace("", pd.NA)  # 把空字符串转为 NaN
    # df = df.dropna(subset=["formula"])  # 删除空行
    # df = df.reset_index(drop=True)  # 重新编号
    # # formula = formula.drop_duplicates()
    # df.to_excel("data/12-1元素摩尔比转化学式_去空缺值.xlsx", index=False)
