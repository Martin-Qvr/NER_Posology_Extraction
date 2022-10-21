.ONESHELL:

init-venv:
	python -m venv .venv
	. .venv/bin/activate
	pip install --upgrade pip
	pip install wheel
	pip install -r requirements.txt
	python -m spacy download fr_core_news_md

clean-venv:
	rm -rf .venv