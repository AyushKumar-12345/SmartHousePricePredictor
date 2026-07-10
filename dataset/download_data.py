import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

df.to_csv("dataset/california_housing.csv", index=False)
print("Raw data warehouse populated: dataset/california_housing.csv")