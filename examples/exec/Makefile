# Code Sandboxes examples

PYTHON ?= python

.PHONY: all eval docker jupyter monty google-colab kaggle kaggle-gpu modal modal-gpu datalayer

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

kaggle: ## Kaggle example (CPU) — batch job by default, live session when RUNTIME_URL is set
	@echo "==> Kaggle example (CPU)"
	@if [ -n "$$RUNTIME_URL" ]; then echo "    mode:    interactive (RUNTIME_URL is set)"; else echo "    mode:    batch (no RUNTIME_URL — the job runs on kaggle.com, allow a few minutes)"; fi
	@if [ -n "$$KAGGLE_API_TOKEN" ]; then echo "    token:   KAGGLE_API_TOKEN is set"; else echo "    token:   KAGGLE_API_TOKEN is NOT set"; fi
	$(PYTHON) kaggle_sandbox_example.py

kaggle-gpu: ## Kaggle example on a GPU (KAGGLE_GPU flavor, default T4) — batch mode requests it, interactive mode only verifies
	@echo "==> Kaggle example (GPU: $${KAGGLE_GPU:-T4})"
	@echo "    Batch mode requests the accelerator on the job; the run fails"
	@echo "    unless nvidia-smi lists a $${KAGGLE_GPU:-T4} in the runtime."
	$(PYTHON) kaggle_sandbox_example.py --gpu "$${KAGGLE_GPU:-T4}"

modal: ## Modal example (CPU)
	@echo "==> Modal example (CPU)"
	@echo "    auth: MODAL_TOKEN_ID/MODAL_TOKEN_SECRET or ~/.modal.toml"
	$(PYTHON) modal_sandbox_example.py

modal-gpu: ## Modal example on a GPU (MODAL_GPU flavor, default T4)
	@echo "==> Modal example (GPU: $${MODAL_GPU:-T4})"
	@echo "    The run fails unless nvidia-smi lists a $${MODAL_GPU:-T4} in the sandbox."
	MODAL_GPU=$${MODAL_GPU:-T4} $(PYTHON) modal_sandbox_example.py --gpu "$${MODAL_GPU:-T4}"

datalayer:
	$(PYTHON) datalayer_sandbox_example.py
