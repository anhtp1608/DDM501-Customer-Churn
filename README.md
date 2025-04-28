Start project:  
git clone https://github.com/anhtp1608/DDM501-Customer-Churn.git  
cd DDM501-Customer-Churn   
python -m venv venv  
source venv/bin/activate   # (Linux/Mac)  
hoặc venv\Scripts\activate (Windows)  

1/ make install  
2/ make data: generate data file churn_data.csv  
3/ make train: train and tunethe model and choose the best model after training and tuning  
4/ make mlflow-ui: start mlflow server and ui  
4/ open another terminal, make run: startthe flask application  

