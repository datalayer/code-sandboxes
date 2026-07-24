# Code Sandboxes examples

PYTHON ?= python

.PHONY: all eval docker jupyter monty colab kaggle modal modal-gpu datalayer

all: eval docker jupyter monty colab modal datalayer

eval:
	$(PYTHON) eval_sandbox_example.py

docker:
	$(PYTHON) docker_sandbox_example.py

jupyter:
	$(PYTHON) jupyter_sandbox_example.py

monty:
	$(PYTHON) monty_sandbox_example.py

colab:
	$(PYTHON) colab_sandbox_example.py

kaggle:
	$(PYTHON) kaggle_sandbox_example.py

modal:
	@if [ -n "$$MODAL_GPU" ]; then \
		echo "Running modal example with GPU flavor: $$MODAL_GPU"; \
		$(PYTHON) modal_sandbox_example.py --gpu "$$MODAL_GPU"; \
	else \
		$(PYTHON) modal_sandbox_example.py; \
	fi

modal-gpu:
	@echo "Preparing Modal GPU sandbox run..."
	@echo "Run this snippet to confirm GPU availability inside your runtime:"
	@printf '%s\n' 'python - <<'\''PY'\''' 'import shutil, subprocess' 'print("nvidia-smi available:", shutil.which("nvidia-smi") is not None)' 'if shutil.which("nvidia-smi"):' '    print(subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True).stdout or "(no output)")' 'try:' '    import torch  # type: ignore' '    print("torch:", torch.__version__)' '    print("cuda available:", torch.cuda.is_available())' '    print("cuda device count:", torch.cuda.device_count())' 'except Exception as exc:' '    print("torch check unavailable:", exc)' 'PY'
	MODAL_GPU=$${MODAL_GPU:-T4} $(PYTHON) modal_sandbox_example.py --gpu "$${MODAL_GPU:-T4}"

datalayer:
	$(PYTHON) datalayer_sandbox_example.py
