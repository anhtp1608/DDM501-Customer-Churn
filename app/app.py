from flask import Flask, request, render_template
import mlflow.sklearn
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

app = Flask(__name__)

# Load model từ Model Registry stage Production
name_model_lookup = "models:/Best_customer_churn_predict_model@bestmodel"
model = mlflow.sklearn.load_model(name_model_lookup)

# Các cột đầu vào
INPUT_FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    input_data = {
        feature: "" for feature in INPUT_FEATURES
    }  # Khởi tạo input trống mặc định

    if request.method == "POST":
        input_data = {
            feature: request.form[feature]
            for feature in INPUT_FEATURES
        }

        # Xử lý dữ liệu để predict
        input_data_processed = {
            "CreditScore": float(input_data["CreditScore"]),
            "Geography": input_data["Geography"],
            "Gender": input_data["Gender"],
            "Age": float(input_data["Age"]),
            "Tenure": float(input_data["Tenure"]),
            "Balance": float(input_data["Balance"]),
            "NumOfProducts": float(input_data["NumOfProducts"]),
            "HasCrCard": float(input_data["HasCrCard"]),
            "IsActiveMember": float(input_data["IsActiveMember"]),
            "EstimatedSalary": float(input_data["EstimatedSalary"]),
        }

        input_df = pd.DataFrame([input_data_processed])
        prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        features=INPUT_FEATURES,
        prediction=prediction,
        input_data=input_data,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
