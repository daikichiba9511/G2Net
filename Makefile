SHELL=/bin/bash
POETRY_VERSION=1.1.7

SKLEARN = pip3 uninstall -y scikit-learn \
	&& pip3 install --pre --extra-index https://pypi.anaconda.org/scipy-wheels-nightly/simple scikit-learn

POETRY = curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 - --version ${POETRY_VERSION}\
		&& echo "export PATH=${HOME}/.poetry/bin:${PATH}" > ~/.bashrc \
		&& source ~/.bashrc \
		&& ${HOME}/.poetry/bin/poetry config virtualenvs.create false

KAGGLE = bash ./kaggle_api.sh

kaggle_api:
	@${KAGGLE}

config: ## config for colab pro + ssh + vscode (e.g git config, and copy ssh credentials to communicatte with github)
	@sh config.sh

poetry:
	${POETRY}

develop: # usually use this command
	pip3 install -q -U -r requirements.txt

set_gpu:
	@sh config.sh \
	&& pip3 install -q -U -r requirements.txt

init_gpu:
	bash tmp/config2.sh \
	&& sh config.sh \
	&& pip3 install -q -U -r requirements.txt

init_tpu:
	@bash tmp/config2.sh \
	&& pip3 install 'torch>=1.3,<1.9' pytorch-lightning \
	&& pip3 install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl \
	&& sh config.sh \
	&& pip3 install -q -U -r requirements.txt

set_tpu:
	pip3 install 'torch>=1.3,<1.9' pytorch-lightning \
	&& pip3 install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl \
	&& sh config.sh \
	&& pip3 install -q -U -r requirements.txt

confirm_tpu:
	python3 -c "import torch; import pytorch_lightning; import torch_xla; print(torch.__version__, ':', pytorch_lightning.__version__, ':', torch_xla.__version__)"

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

compress:
	tar cvJf g2net-archive.tar.bz2 input/g2net-gravitational-wave-detection/

decompress:
	tar xvjf g2net-archive.tar.bz2