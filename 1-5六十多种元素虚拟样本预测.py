import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy
import joblib
from matplotlib.colors import Normalize
from matplotlib.pyplot import colorbar

# T90
best_feature3 = ['Interant f electrons', 'Interant p electrons', 'MagpieData avg_dev NpValence', 'MagpieData maximum GSmagmom', 'MagpieData maximum NfUnfilled', 'MagpieData maximum NpValence', 'MagpieData mean Electronegativity', 'MagpieData mean GSmagmom', 'MagpieData mean NpValence', 'MagpieData mean SpaceGroupNumber', 'MagpieData minimum Electronegativity', 'MagpieData minimum SpaceGroupNumber', 'MagpieData mode Electronegativity', 'MagpieData mode NValence', 'MagpieData mode NfUnfilled', 'MagpieData mode SpaceGroupNumber', 'MagpieData range NpValence', 'MagpieData range SpaceGroupNumber', 'Mean cohesive energy', 'Total weight']
# T50
best_feature2 = ['APE mean', 'Interant p electrons', 'Lambda entropy', 'MagpieData avg_dev CovalentRadius', 'MagpieData maximum NpValence', 'MagpieData mean GSmagmom', 'MagpieData minimum Electronegativity', 'MagpieData minimum GSmagmom', 'MagpieData minimum SpaceGroupNumber', 'MagpieData mode Electronegativity', 'MagpieData mode SpaceGroupNumber', 'MagpieData range Column', 'MagpieData range CovalentRadius', 'MagpieData range GSvolume_pa', 'MagpieData range MendeleevNumber', 'MagpieData range NpValence', 'MagpieData range SpaceGroupNumber', 'Radii gamma', 'Radii local mismatch', 'Total weight']
# NOx_Conv_200°C
best_feature1 = ['Interant p electrons', 'MagpieData avg_dev CovalentRadius', 'MagpieData avg_dev NUnfilled', 'MagpieData avg_dev NpValence', 'MagpieData maximum NpValence', 'MagpieData mean NpValence', 'MagpieData minimum Electronegativity', 'MagpieData minimum SpaceGroupNumber', 'MagpieData mode Electronegativity', 'MagpieData mode SpaceGroupNumber', 'MagpieData range Column', 'MagpieData range CovalentRadius', 'MagpieData range GSvolume_pa', 'MagpieData range MendeleevNumber', 'MagpieData range NUnfilled', 'MagpieData range NpValence', 'MagpieData range SpaceGroupNumber', 'Radii gamma', 'Radii local mismatch', 'Total weight']


if __name__ == '__main__':
    model1 = joblib.load("models/12月元素外推模型_NOx_Conv_200°C_best_model.pkl")
    model2 = joblib.load("models/12月元素外推模型_T50_best_model.pkl")
    model3 = joblib.load("models/12月元素外推模型_T90_best_model.pkl")

    # df_magpie = pd.read_excel("data/单元素样本-提取matminer特征.xlsx")
    part = "part10"
    df_magpie = pd.read_excel(f"data/双元素步长2%样本-提取matminer特征{part}.xlsx")
    formula = df_magpie["formula"]
    col = pd.read_excel("data/人工筛选后特征.xlsx")
    new_col = list(col['features'])

    df_magpie_origin = pd.read_excel("data/12-1提取matminer特征.xlsx").dropna()
    df_target_origin = df_magpie_origin.iloc[:, 2:8]
    print(df_target_origin.columns)

    df_magpie = pd.concat([formula, df_magpie[new_col]], axis=1)
    df_magpie = df_magpie.dropna()
    df_magpie = df_magpie.drop(columns=["formula"])

    X1 = df_magpie[best_feature1]
    X2 = df_magpie[best_feature2]
    X3 = df_magpie[best_feature3]

    target = {}
    target["NOx_Conv_200°C"] = model1.predict(X1)
    target["T50"] = model2.predict(X2)
    target["T90"] = model3.predict(X3)
    target = pd.DataFrame(target)
    target["formula"] = formula
    # target.to_excel("data/单元素样本-预测结果.xlsx")
    target.to_excel(f"data/双元素步长2%样本-预测结果{part}.xlsx")

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca_origin = pca.fit_transform(df_magpie_origin[best_feature3])
    X_pca = pca.transform(X3)

    t90_all = np.concatenate([target["T90"].values, df_target_origin["T90"].values])
    norm = Normalize(vmin=df_target_origin["T90"].values.min(), vmax=df_target_origin["T90"].values.max())

    sc1 = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=target["T90"].values,
        cmap='rainbow',
        norm=norm,
        label="T90_virtual_samples"
    )

    # 3️⃣ 原始样本
    sc2 = plt.scatter(
        X_pca_origin[:, 0], X_pca_origin[:, 1],
        c=df_target_origin["T90"].values,
        cmap='rainbow',
        norm=norm,
        marker='*',
        label="T90_origin"
    )

    cb = plt.colorbar(sc1)
    cb.set_label("T90")

    plt.legend()
    plt.show()
