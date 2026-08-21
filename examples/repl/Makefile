# Code Sandboxes REPL examples

PYTHON ?= python

.PHONY: all eval docker jupyter-server monty google-colab kaggle kaggle-gpu kaggle-live kaggle-gpu-live modal modal-gpu daytona daytona-gpu daytona-gpu-spot datalayer e2b coreweave coreweave-gpu cloudflare

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

daytona-gpu-spot: ## Daytona REPL on a SPOT GPU (DAYTONA_GPU flavor, default H100,H200)
	@echo "==> Daytona REPL (spot GPU: $${DAYTONA_GPU:-H100,H200})"
	@echo "    Preemptible capacity: cheaper, outside the GPU quota, and"
	@echo "    reclaimed without warning. Creation fails at once when there"
	@echo "    is none free — there is no queue to wait in."
	DAYTONA_GPU=$${DAYTONA_GPU:-H100,H200} $(PYTHON) daytona_sandbox_example.py --gpu "$${DAYTONA_GPU:-H100,H200}" --spot

datalayer:
	$(PYTHON) datalayer_sandbox_example.py

e2b: ## E2B REPL (Firecracker microVM with a Jupyter kernel)
	@echo "==> E2B REPL"
	@echo "    auth: E2B_API_KEY, and E2B_DOMAIN for a self-hosted cluster"
	@echo "    Definitions persist between lines: the kernel holds one namespace."
	@echo "    The sandbox is given five minutes to live; --minutes says otherwise."
	$(PYTHON) e2b_sandbox_example.py

coreweave: ## CoreWeave REPL (CPU)
	@echo "==> CoreWeave REPL (CPU)"
	@echo "    auth: CWSANDBOX_API_KEY, and CWSANDBOX_BASE_URL for another control plane"
	@echo "    Definitions persist between lines: one session process holds them."
	$(PYTHON) coreweave_sandbox_example.py

coreweave-gpu: ## CoreWeave REPL on a GPU (COREWEAVE_GPU flavor, default H100)
	@echo "==> CoreWeave REPL (GPU: $${COREWEAVE_GPU:-H100})"
	@echo "    COREWEAVE_IMAGE names the image, which has to carry what the code needs."
	COREWEAVE_GPU=$${COREWEAVE_GPU:-H100} $(PYTHON) coreweave_sandbox_example.py --gpu "$${COREWEAVE_GPU:-H100}"

cloudflare: ## Cloudflare REPL (through a deployed sandbox bridge Worker)
	@echo "==> Cloudflare REPL"
	@echo "    auth: CLOUDFLARE_SANDBOX_API_URL and CLOUDFLARE_SANDBOX_API_KEY"
	@echo "    Deploy the bridge once: npm create cloudflare -- sandbox-bridge \\"
	@echo "        --template=cloudflare/sandbox-sdk/bridge/worker"
	@echo "    Each line runs in its own process: definitions do NOT persist."
	$(PYTHON) cloudflare_sandbox_example.py
