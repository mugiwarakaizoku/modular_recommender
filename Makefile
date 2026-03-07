.PHONY:

test:
	pytest

format:
	black src

lint:
	pre-commit run --all-files

mlflow:
	scripts/run_mlflow.sh

setup:
	chmod +x scripts/run_mlflow.sh
	pre-commit install

tensorboard:
	tensorboard --logdir logs/ --port 6006
