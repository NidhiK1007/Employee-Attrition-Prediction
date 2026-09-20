import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance


# 1. Load dataset
df = pd.read_csv(
    "dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


# 2. Remove irrelevant columns
df = df.drop(columns=[
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
])


# 3. Separate features and target
X = df.drop(columns=["Attrition"])
Y = df["Attrition"]


# 4. Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# 5. Split data
X_train, X_val, Y_train, Y_val = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)


# 6. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


# 7. Create model
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)


# 8. Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# 9. Train model
pipeline.fit(X_train, Y_train)


# 10. Calculate permutation importance
result = permutation_importance(
    pipeline,
    X_val,
    Y_val,
    scoring="f1_macro",
    n_repeats=10,
    random_state=42
)


# 11. Create feature importance table
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean
})


# 12. Sort features
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# 13. Display all features
print("\nFEATURE IMPORTANCE")
print(importance_df)


# 14. Select top 12 features
selected_features = importance_df.head(12)["Feature"].tolist()


print("\nSELECTED FEATURES")
for feature in selected_features:
    print(feature)