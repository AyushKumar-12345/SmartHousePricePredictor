import os
import joblib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

if not os.path.exists("dataset/california_housing.csv"):
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["Price"] = housing.target
    os.makedirs("dataset", exist_ok=True)
    df.to_csv("dataset/california_housing.csv", index=False)

df = pd.read_csv("dataset/california_housing.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_.tolist()
feature_importance_dict = dict(zip(X.columns, importances))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

with open("models/importance.json", "w") as f:
    json.dump(feature_importance_dict, f)

print("Model and weights exported successfully.")