# Makefile cho Python project LOCAL

# ========== Các biến ==========
PYTHON=python
PIP=pip

# ========== Các target ==========

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) data/generate_data.py

train:
	$(PYTHON) model/train_model.py

run:
	$(PYTHON) app/app.py

mlflow-ui:
	mlflow ui --host 127.0.0.1 --port 5000

format:
	black .

# ========== Help ==========
help:
	@echo "Các lệnh Makefile hỗ trợ:"
	@echo "  make install    - Cài thư viện Python từ requirements.txt"
	@echo "  make data       - Sinh dữ liệu mẫu"
	@echo "  make train      - Train model và log vào MLflow"
	@echo "  make run        - Chạy Flask web app"
	@echo "  make mlflow-ui  - Chạy MLflow UI tại localhost:5000"
	@echo "  make format     - Format code bằng black"

.PHONY: install data train run mlflow-ui format help
