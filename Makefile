SHELL=/bin/bash
POETRY_VERSION=1.1.7
PACKAGE = loguru wandb flake8 mypy black pyyaml pytorch-lightning \
		jupytext madgrad albumentations timm

SKLEARN = pip3 uninstall -y scikit-learn \
	&& pip3 install --pre --extra-index https://pypi.anaconda.org/scipy-wheels-nightly/simple scikit-learn

POETRY = curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 - --version ${POETRY_VERSION}\
		&& echo "export PATH=${HOME}/.poetry/bin:${PATH}" > ~/.bashrc \
		&& source ~/.bashrc \
		&& ${HOME}/.poetry/bin/poetry config virtualenvs.create false

config: ## config for colab pro + ssh + vscode (e.g git config, and copy ssh credentials to communicatte with github)
	@sh config.sh

poetry:
	${POETRY}

develop: # usually use this command
	pip3 install -q -U ${PACKAGE}

pip_export:
	pip3 freeze > requirements.txt

poetry_export:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

transfer_pip_to_poetry:
	for package in $(cat requirements.txt); do poetry add "${package}"; done

clean:
	rm -rf output/*/*.ckpt
	rm -rf output/*/wandb
	rm -rf lightning_logs/