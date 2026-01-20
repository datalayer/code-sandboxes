# Docker image for LocalDockerSandbox
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir jupyter_server ipykernel

RUN useradd -ms /bin/bash sandbox && \
    mkdir -p /workspace && \
    chown -R sandbox:sandbox /workspace

USER sandbox
WORKDIR /workspace

ENV JUPYTER_TOKEN=local-sandbox-token
EXPOSE 8888

CMD ["bash", "-lc", "jupyter server --ip=0.0.0.0 --port=8888 --ServerApp.token=${JUPYTER_TOKEN} --ServerApp.allow_origin=* --ServerApp.allow_remote_access=True --ServerApp.disable_check_xsrf=True --ServerApp.open_browser=False --ServerApp.root_dir=/workspace"]
