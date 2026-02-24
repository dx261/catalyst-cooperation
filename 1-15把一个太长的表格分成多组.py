import numpy as np
import pandas as pd

df = pd.read_excel("双元素步长2%样本.xlsx")

for i, part in enumerate(np.array_split(df, 10), start=1):
    part.to_excel(f"双元素步长2%样本part{i}.xlsx", index=False)

