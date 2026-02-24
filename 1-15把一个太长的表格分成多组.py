import numpy as np
import pandas as pd

df = pd.read_excel("2-24外推元素高通量样本4旧+1新.xlsx")

for i, part in enumerate(np.array_split(df, 40), start=1):
    part.to_excel(f"2-24data/2-24外推元素高通量样本4旧+1新part{i}.xlsx", index=False)

