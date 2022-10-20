.ONESHELL:

init-venv:
	python -m venv .venv
	. .venv/bin/activate
	pip install --upgrade pip
	pip install wheel
	pip install -r requirements.txt

clean-venv:
	rm -rf .venv