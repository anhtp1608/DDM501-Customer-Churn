import mlflow.sklearn
from mlflow.tracking import MlflowClient

client = MlflowClient()
versions = client.search_model_versions(f"name='Best_customer_churn_predict_model'")
for v in versions:
    print(v.version, v.current_stage, v.aliases)
