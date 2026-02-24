import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_T50 = joblib.load("10-21-new_models/best_model_T50.pkl")
model_T90 = joblib.load("10-21-new_models/best_model_T90.pkl")
model_Conv200 = joblib.load("10-21-new_models/best_model_NOx_Conv_200°C.pkl")

std_T50 = joblib.load("10-21-new_models/scaler_T50.pkl")
std_T90 = joblib.load("10-21-new_models/scaler_T90.pkl")
std_Conv200 = joblib.load("10-21-new_models/scaler_NOx_Conv_200°C.pkl")

df = pd.read_excel("Virture_samples_10_27_提取元素含量.xlsx")
features = df.iloc[:, 1:].copy()
features_T50 = std_T50.transform(features)
features_T90 = std_T90.transform(features)
features_Conv200 = std_Conv200.transform(features)
features_T50 = pd.DataFrame(features_T50)
features_T90 = pd.DataFrame(features_T90)
features_Conv200 = pd.DataFrame(features_Conv200)

result_T50 = model_T50.predict(features_T50)
result_T90 = model_T90.predict(features_T90)
result_Conv200 = model_Conv200.predict(features_Conv200)
# result_T50 = model_T50.predict(features)
# result_T90 = model_T90.predict(features)
# result_Conv200 = model_Conv200.predict(features)
result = pd.DataFrame(
    {
        "T50_pred": result_T50,
        "T90_pred": result_T90,
        "Conv200_pred": result_Conv200,
    }
)
final_result = pd.concat([df, result], axis=1)
final_result.to_csv("轮盘赌虚拟样本预测结果_10-27.csv", index=False)