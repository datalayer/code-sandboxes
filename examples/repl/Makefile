# Code Sandboxes REPL examples

PYTHON ?= python

.PHONY: all eval docker jupyter monty google-colab kaggle modal modal-gpu datalayer

all: eval docker jupyter monty google-colab modal datalayer

eval:
	$(PYTHON) eval_sandbox_example.py

docker:
	$(PYTHON) docker_sandbox_example.py

jupyter:
	$(PYTHON) jupyter_sandbox_example.py

monty:
	$(PYTHON) monty_sandbox_example.py

google-colab:
	$(PYTHON) google_colab_sandbox_example.py

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

datalayer:
	$(PYTHON) datalayer_sandbox_example.py
