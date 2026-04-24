import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

# Use same column names as training
columns = ["Tenure", "WarehouseToHome", "NumberOfDeviceRegistered",
           "PreferedOrderCat", "SatisfactionScore", "MaritalStatus",
           "NumberOfAddress", "Complain", "DaySinceLastOrder", "CashbackAmount"]

data = [[10, 20, 3, 1, 4, 2, 3, 0, 5, 150]]

df = pd.DataFrame(data, columns=columns)

prediction = model.predict(df)

print("Churn Prediction:", prediction)