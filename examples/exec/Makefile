# Code Sandboxes examples

PYTHON ?= python

.PHONY: all eval docker jupyter-server monty google-colab kaggle kaggle-gpu modal modal-gpu daytona daytona-gpu daytona-gpu-spot datalayer

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

daytona: ## Daytona example (CPU)
	@echo "==> Daytona example (CPU)"
	@echo "    auth: DAYTONA_API_KEY, or DAYTONA_JWT_TOKEN with DAYTONA_ORGANIZATION_ID"
	$(PYTHON) daytona_sandbox_example.py

daytona-gpu: ## Daytona example on a GPU (DAYTONA_GPU flavor, default H100)
	@echo "==> Daytona example (GPU: $${DAYTONA_GPU:-H100})"
	@echo "    A GPU sandbox is built from an image rather than the default"
	@echo "    snapshot, so it takes longer to come up. The run fails unless"
	@echo "    nvidia-smi lists a device in the sandbox."
	DAYTONA_GPU=$${DAYTONA_GPU:-H100} $(PYTHON) daytona_sandbox_example.py --gpu "$${DAYTONA_GPU:-H100}"

daytona-gpu-spot: ## Daytona example on a SPOT GPU (DAYTONA_GPU flavor, default H100,H200)
	@echo "==> Daytona example (spot GPU: $${DAYTONA_GPU:-H100,H200})"
	@echo "    Preemptible capacity: cheaper, outside the GPU quota, and"
	@echo "    reclaimed without warning. Creation fails at once when there"
	@echo "    is none free — there is no queue to wait in."
	DAYTONA_GPU=$${DAYTONA_GPU:-H100,H200} $(PYTHON) daytona_sandbox_example.py --gpu "$${DAYTONA_GPU:-H100,H200}" --spot

datalayer:
	$(PYTHON) datalayer_sandbox_example.py
