# Code Sandboxes REPL examples

PYTHON ?= python

.PHONY: all eval docker jupyter-server monty google-colab kaggle kaggle-gpu kaggle-live kaggle-gpu-live modal modal-gpu daytona daytona-gpu datalayer

all: eval docker jupyter-server monty google-colab modal daytona datalayer

eval:
	$(PYTHON) eval_sandbox_example.py

docker:
	$(PYTHON) docker_sandbox_example.py

jupyter-server:
	$(PYTHON) jupyter_server_sandbox_example.py

monty:
	$(PYTHON) monty_sandbox_example.py

google-colab:
	$(PYTHON) google_colab_sandbox_example.py

kaggle-live: ## Kaggle REPL on one persistent kernel (agent job + dataset bus)
	@echo "==> Kaggle REPL (live: one batch job holds the kernel; boot takes minutes)"
	KAGGLE_LIVE=1 $(PYTHON) kaggle_sandbox_example.py

kaggle-gpu-live: ## Kaggle live REPL on a GPU (KAGGLE_GPU flavor, default t4)
	@echo "==> Kaggle REPL (live, GPU: $${KAGGLE_GPU:-t4}; a GPU job queues longer before it boots)"
	KAGGLE_LIVE=1 KAGGLE_GPU=$${KAGGLE_GPU:-t4} $(PYTHON) kaggle_sandbox_example.py

kaggle-gpu: ## Kaggle batch REPL on a GPU (KAGGLE_GPU flavor, default t4)
	@echo "==> Kaggle REPL (batch, GPU: $${KAGGLE_GPU:-t4}; each input is a batch job, and a GPU one queues longer)"
	KAGGLE_GPU=$${KAGGLE_GPU:-t4} $(PYTHON) kaggle_sandbox_example.py

kaggle: ## Kaggle REPL — batch job per input by default, live session when RUNTIME_URL is set
	@echo "==> Kaggle REPL"
	@if [ -n "$$RUNTIME_URL" ]; then echo "    mode:    interactive (RUNTIME_URL is set)"; else echo "    mode:    batch (each input is a kaggle.com batch job, about a minute each)"; fi
	@if [ -n "$$KAGGLE_API_TOKEN" ]; then echo "    token:   KAGGLE_API_TOKEN is set"; else echo "    token:   KAGGLE_API_TOKEN is NOT set"; fi
	$(PYTHON) kaggle_sandbox_example.py

modal: ## Modal REPL (CPU)
	@echo "==> Modal REPL (CPU)"
	@echo "    auth: MODAL_TOKEN_ID/MODAL_TOKEN_SECRET or ~/.modal.toml"
	$(PYTHON) modal_sandbox_example.py

modal-gpu: ## Modal REPL on a GPU (MODAL_GPU flavor, default T4)
	@echo "==> Modal REPL (GPU: $${MODAL_GPU:-T4})"
	@echo "    The run fails unless nvidia-smi lists a $${MODAL_GPU:-T4} in the sandbox."
	MODAL_GPU=$${MODAL_GPU:-T4} $(PYTHON) modal_sandbox_example.py --gpu "$${MODAL_GPU:-T4}"

daytona: ## Daytona REPL (CPU)
	@echo "==> Daytona REPL (CPU)"
	@echo "    auth: DAYTONA_API_KEY, or DAYTONA_JWT_TOKEN with DAYTONA_ORGANIZATION_ID"
	@echo "    Definitions persist between lines: the interpreter holds one namespace."
	$(PYTHON) daytona_sandbox_example.py

daytona-gpu: ## Daytona REPL on a GPU (DAYTONA_GPU flavor, default H100)
	@echo "==> Daytona REPL (GPU: $${DAYTONA_GPU:-H100}; built from an image, so slower to start)"
	DAYTONA_GPU=$${DAYTONA_GPU:-H100} $(PYTHON) daytona_sandbox_example.py --gpu "$${DAYTONA_GPU:-H100}"

datalayer:
	$(PYTHON) datalayer_sandbox_example.py
