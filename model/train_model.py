import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# Set MLflow tracking uri
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load data
df = pd.read_csv("data/churn_data.csv")
X = df.drop(columns=["Exited"])
y = df["Exited"]

# Preprocessing pipelines
numeric_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
categorical_features = ["Geography", "Gender"]

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Models to try
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
}

# Parameter grids
param_grids = {
    "Random Forest": {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [None, 10, 20],
    },
    "Logistic Regression": {
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ["liblinear", "saga"],
    },
    "Decision Tree": {
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
    },
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train all models
for model_name, model in models.items():
    print(f"Training {model_name}...")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    # Grid Search
    grid_search = GridSearchCV(
        pipeline, param_grids[model_name], cv=5, n_jobs=-1, scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy of {model_name}: {acc}")
    # Chuẩn bị input example và signature
    input_example = X_test.iloc[:1]  # lấy 1 sample input
    signature = infer_signature(X_test, y_pred)
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="Best_customer_churn_predict_model",
            input_example=input_example,
            signature=signature,
        )
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)

# === Chọn best model tự động ===
client = MlflowClient()

experiment = client.get_experiment_by_name("Default")
experiment_id = experiment.experiment_id

runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.accuracy DESC"],
)

best_run = runs[0]
print(f"Best Run ID: {best_run.info.run_id}")
print(f"Best Accuracy: {best_run.data.metrics['accuracy']}")

# Lấy model từ run tốt nhất
best_model_uri = f"runs:/{best_run.info.run_id}/model"

# Đăng ký model tốt nhất
model_details = mlflow.register_model(
    model_uri=best_model_uri, name="Best_customer_churn_predict_model"
)

# Gán alias bestmodel
client.set_registered_model_alias(
    name="Best_customer_churn_predict_model",
    alias="bestmodel",
    version=model_details.version,
)

print("✅ Best model registered and alias 'bestmodel' set!")
