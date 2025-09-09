install:
	pip install -r requirements.txt

run-api:
	uvicorn api.app:app --reload

train-baseline:
	python -m src.train_baseline

train-bert:
	python -m src.train_bert

eval:
	python -m src.eval
