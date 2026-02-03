create_environment:
	conda create --name ecomrec python=3.10.0 -y
	@echo ">>> New conda environment created. Activate it with:\nconda activate coecomrec"

requirements:
	if [ -a dev-requirements.txt ]; then pip uninstall -r dev-requirements.txt -y; fi;
	pip install --upgrade pip-tools
	pip install -r dev-requirements.txt
	pip install -e .

precommit:
	pre-commit install
	pre-commit run --all-files

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# run:
#   docker-compose up --build
