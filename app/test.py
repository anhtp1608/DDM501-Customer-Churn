from mlflow.tracking import MlflowClient

client = MlflowClient()
name_model = "name='Best_customer_churn_predict_model'"
versions = client.search_model_versions(name_model)
for v in versions:
    print(v.version, v.current_stage, v.aliases)
