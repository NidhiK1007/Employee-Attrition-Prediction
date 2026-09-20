import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



df=pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

selected_features = [
    "OverTime",
    "JobRole",
    "BusinessTravel",
    "MaritalStatus",
    "YearsAtCompany",
    "TotalWorkingYears",
    "YearsWithCurrManager",
    "Age",
    "MonthlyIncome",
    "JobLevel",
    "JobSatisfaction",
    "YearsInCurrentRole"
]


X=df[selected_features]
Y=df["Attrition"]

X_train, X_test, Y_train, Y_test= train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)


categorical_columns=X.select_dtypes(include=["str","object"]).columns.tolist()
numerical_columns=X.select_dtypes(include=["int64","float64"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[("categorical",
                                                 OneHotEncoder(handle_unknown="ignore"),
                                                 categorical_columns),
                                                 ("numerical",
                                                  StandardScaler(),
                                                  numerical_columns)
                                                 ],
                                                )


balanced_model = LogisticRegression(max_iter=2000,
                                    class_weight="balanced")

balanced_pipeline = Pipeline(steps=[
                                ("prepocessing", preprocessor),
                                ("model", balanced_model)
])

balanced_pipeline.fit(X_train, Y_train)
print("Balanced Logistic Regression completed")

Y_pred = balanced_pipeline.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(Y_test, Y_pred))


print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))


joblib.dump(balanced_pipeline,"models/employee_attrition_model.pkl")
print("Model saved successfully!")