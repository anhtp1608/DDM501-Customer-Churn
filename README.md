Test Mike
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
# hoặc venv\Scripts\activate (Windows)

1/ make install
2/ make data: generate data file churn_data.csv
3/ make train: trai and tune model and choose best model after train and tune
4/ make mlflow-ui: start mlflow server and ui
4/ open another terminal make run: start flask application

