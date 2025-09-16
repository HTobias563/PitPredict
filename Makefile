# Makefile for PitPredict

.PHONY: help install install-dev train predict test clean lint format

help:
	@echo "Available commands:"
	@echo "  install      - Install package dependencies"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  train        - Train all models"
	@echo "  predict      - Run prediction example"
	@echo "  predict-race - Predict specific race (RACE=2024_XX)"
	@echo "  predict-2025 - Interactive 2025 predictions"
	@echo "  install-app  - Install web app dependencies"
	@echo "  run-app      - Run Streamlit web application"
	@echo "  test         - Run test suite"
	@echo "  lint         - Run code quality checks"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e .[dev]

train:
	python -m src.pitpredict.cli train

predict:
	python tests/predict_example.py

# Specific race predictions
predict-race:
	@echo "Usage: make predict-race RACE=2024_21"
	@if [ -z "$(RACE)" ]; then echo "Please specify RACE=2024_XX"; exit 1; fi
	python -c "from src.pitpredict.models.final_position_predict import predict_race_positions; predict_race_positions('$(RACE)')"

predict-2025:
	cd src && python predict_2025.py

predict-future:
	@echo "Usage: make predict-future RACE='Netherlands GP 2025' TRACK=netherlands"
	cd src && python predict_future_race.py --race_name "$(RACE)" --track_type $(TRACK)

# Web Application
install-app:
	conda run -n pitpredict_env pip install -r app/requirements.txt

run-app:
	conda run -n pitpredict_env streamlit run app/pitpredict_app.py

run-app-dev:
	streamlit run app/pitpredict_app.py --server.runOnSave true

test:
	pytest tests/

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	
# Docker commands (optional)
docker-build:
	docker build -t pitpredict .

docker-run:
	docker run -it --rm -v $(PWD):/workspace pitpredict

# Production deployment
deploy-model:
	@echo "Deploy trained models to production environment"
	# Add deployment logic here
