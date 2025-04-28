# ========== Variables ==========
PYTHON=python
PIP=pip

# ========== Targets ==========

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

lint:
	flake8 --select=E,F,W --show-source --statistics $(shell find . -name "*.py")

# ========== Help ==========
help:
	@echo "Makefile hỗ trợ các lệnh sau:"
	@echo "  make install     - Cài đặt dependencies từ requirements.txt"
	@echo "  make data        - Sinh dữ liệu giả lập"
	@echo "  make train       - Train model và log MLflow"
	@echo "  make run         - Chạy Flask web application"
	@echo "  make mlflow-ui   - Chạy MLflow UI local tại http://127.0.0.1:5000"
	@echo "  make format      - Format code theo chuẩn black"
	@echo "  make lint        - Kiểm tra lỗi code theo chuẩn flake8"
	@echo "  make help        - Hiển thị hướng dẫn"

.PHONY: install data train run mlflow-ui format lint help
