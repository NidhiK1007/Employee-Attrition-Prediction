import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
print (df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn Name:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())

print("\nAttrition Percentage:")
print(df["Attrition"].value_counts(normalize=True)*100)

print("\nCategorical Columns:")
print(df.select_dtypes(include="object").columns.tolist())

print("\nUnique Values in Each Column:")
print(df.nunique())

print("\nRemoving irrelevant columns...")
df=df.drop(columns=[
      "EmployeeCount",                  
      "EmployeeNumber",
      "Over18",
      "StandardHours"
])
print("\nNew Dataset Shape:")
print(df.shape)

print("\nseparating input and target data:")
X=df.drop(columns=["Attrition"])
Y=df["Attrition"]
print("Features Shape: ", X.shape)
print("Target Shape: ", Y.shape)

print("\nCategorical values:")
categorical_columns=X.select_dtypes(include=["str","object"]).columns.tolist()
print(categorical_columns)
print("\nNumerical values:")
numerical_columns=X.select_dtypes(include=["int64","float64"]).columns.tolist()
print(numerical_columns)

preprocessor = ColumnTransformer(transformers=[("categorical",
                                                 OneHotEncoder(handle_unknown="ignore"), categorical_columns)],
                                                 remainder="passthrough")
