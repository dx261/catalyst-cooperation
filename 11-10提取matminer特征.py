import pandas as pd
import numpy as np
import itertools
import warnings
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition.alloy import WenAlloys
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # df = pd.read_excel("data/12-1元素摩尔比转化学式.xlsx")
    # part = "part10"
    # df = pd.read_excel(f"双元素步长2%样本{part}.xlsx")

    df = pd.read_excel("2-24外推元素高通量样本4旧+1新.xlsx")
    df = StrToComposition(reduce=True, target_col_id='composition_obj').featurize_dataframe(df, 'formula')
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              WenAlloys()])
    feature_labels = feature_calculators.feature_labels()
    data = feature_calculators.featurize_dataframe(df, col_id='composition_obj', ignore_errors=True)
    # data.to_excel(f"data/双元素步长2%样本-提取matminer特征{part}.xlsx")
    data.to_excel("2-24外推元素高通量样本4旧+1新-提取matminer特征.xlsx")