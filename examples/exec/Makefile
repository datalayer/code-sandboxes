# Code Sandboxes examples

PYTHON ?= python

.PHONY: all eval docker jupyter-server monty google-colab kaggle kaggle-gpu modal modal-gpu daytona daytona-gpu daytona-gpu-spot datalayer e2b coreweave coreweave-gpu cloudflare

all: eval docker jupyter-server monty google-colab modal daytona datalayer e2b coreweave cloudflare

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

e2b: ## E2B example (Firecracker microVM with a Jupyter kernel)
	@echo "==> E2B example"
	@echo "    auth: E2B_API_KEY, and E2B_DOMAIN for a self-hosted cluster"
	@echo "    The kernel holds one namespace, so snippets share their state."
	$(PYTHON) e2b_sandbox_example.py

coreweave: ## CoreWeave example (CPU)
	@echo "==> CoreWeave example (CPU)"
	@echo "    auth: CWSANDBOX_API_KEY, and CWSANDBOX_BASE_URL for another control plane"
	$(PYTHON) coreweave_sandbox_example.py

coreweave-gpu: ## CoreWeave example on a GPU (COREWEAVE_GPU flavor, default H100)
	@echo "==> CoreWeave example (GPU: $${COREWEAVE_GPU:-H100})"
	@echo "    The run fails unless nvidia-smi lists a device in the sandbox."
	@echo "    COREWEAVE_IMAGE names the image, which has to carry what the code needs."
	COREWEAVE_GPU=$${COREWEAVE_GPU:-H100} $(PYTHON) coreweave_sandbox_example.py --gpu "$${COREWEAVE_GPU:-H100}"

cloudflare: ## Cloudflare example (through a deployed sandbox bridge Worker)
	@echo "==> Cloudflare example"
	@echo "    auth: CLOUDFLARE_SANDBOX_API_URL and CLOUDFLARE_SANDBOX_API_KEY"
	@echo "    Deploy the bridge once: npm create cloudflare -- sandbox-bridge \\"
	@echo "        --template=cloudflare/sandbox-sdk/bridge/worker"
	@echo "    Each snippet runs in a process of its own: nothing crosses between them."
	$(PYTHON) cloudflare_sandbox_example.py
