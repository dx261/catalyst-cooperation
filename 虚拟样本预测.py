import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_T50 = pickle.load(open("models/XGBRegressor_T50_.pkl", "rb"))
model_T90 = pickle.load(open("models/ExtraTreesRegressor_T90_.pkl", "rb"))
model_Conv200 = pickle.load(open("models/XGBRegressor_NOx_Conv_200°C_.pkl", "rb"))

std_T50 = joblib.load("std/standard_scaler_T50.pkl")
std_T90 = joblib.load("std/standard_scaler_T90.pkl")
std_Conv200 = joblib.load("std/standard_scaler_NOx_Conv_200°C.pkl")

# df = pd.read_csv("data/高通量样本完整版-7-7.csv")
df = pd.read_csv("data/高通量样本完整版-9-16.csv")
# extra = ["Sb2O3", "SnO2", "Fe2(SO4)3", "Ce(SO4)2", "FeVO4", "CePO4"]
features = df.copy()
# features[extra] = 0
print(std_T50.transform(features))
result_T50 = model_T50.predict(std_T50.transform(features))
result_T90 = model_T90.predict(std_T90.transform(features))
result_Conv200 = model_Conv200.predict(std_Conv200.transform(features))
result = pd.DataFrame(
    {
        "T50_pred": result_T50,
        "T90_pred": result_T90,
        "Conv200_pred": result_Conv200,
    }
)
# result.to_csv("data/虚拟样本预测结果_9-16_步长2.csv", index=False)