import random
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import os

def generate_data():
    random_state = 42
    random.seed(random_state)
    np.random.seed(random_state)

    # Generate raw numeric features
    X, y = make_classification(
        n_samples=10000,
        n_features=6,
        n_informative=4,
        n_redundant=2,
        n_classes=2,
        weights=[0.6, 0.4],  # 60% ở lại, 40% rời bỏ
        random_state=random_state
    )

    # Create DataFrame với các cột số
    df = pd.DataFrame(X, columns=['CreditScore_raw', 'Age_raw', 'Balance_raw',
                                  'EstimatedSalary_raw', 'HasCrCard_raw', 'IsActiveMember_raw'])
    
    df['Exited'] = y

    # Sinh các cột đúng format:
    df['CreditScore'] = (np.abs(df['CreditScore_raw']) * 200).astype(int) + 300
    df['CreditScore'] = df['CreditScore'].clip(300, 850)

    df['Age'] = (np.abs(df['Age_raw']) * 30).astype(int) + 18
    df['Age'] = df['Age'].clip(18, 92)

    # Balance: float dương 0-250000, lấy 2 số thập phân
    df['Balance'] = (np.abs(df['Balance_raw']) * 50000)
    df['Balance'] = df['Balance'].clip(0, 250000)
    df['Balance'] = df['Balance'].round(2)

    # EstimatedSalary: float dương 0-200000, lấy 2 số thập phân
    df['EstimatedSalary'] = (np.abs(df['EstimatedSalary_raw']) * 40000)
    df['EstimatedSalary'] = df['EstimatedSalary'].clip(0, 200000)
    df['EstimatedSalary'] = df['EstimatedSalary'].round(2)

    df['HasCrCard'] = (np.abs(df['HasCrCard_raw']) > 0).astype(int)
    df['IsActiveMember'] = (np.abs(df['IsActiveMember_raw']) > 0).astype(int)

    df['NumOfProducts'] = np.random.randint(1, 5, size=len(df))
    df['Tenure'] = np.random.randint(0, 11, size=len(df))

    df['Geography'] = np.random.choice(['France', 'Spain', 'Germany'], size=len(df))
    df['Gender'] = np.random.choice(['Male', 'Female'], size=len(df))

    df['RowNumber'] = range(1, len(df) + 1)
    df['CustomerId'] = np.random.randint(10000000, 99999999, size=len(df))
    df['Surname'] = np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'], size=len(df))

    # Rearrange columns đúng thứ tự yêu cầu
    df = df[['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender',
             'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
             'IsActiveMember', 'EstimatedSalary', 'Exited']]

    # Ép kiểu cho các cột int
    int_cols = ['RowNumber', 'CustomerId', 'CreditScore', 'Age', 'Tenure',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    df[int_cols] = df[int_cols].astype(int)

    # Tạo thư mục data nếu chưa có
    os.makedirs("data", exist_ok=True)

    # Ghi file csv
    df.to_csv("data/churn_data.csv", index=False)

if __name__ == "__main__":
    generate_data()
