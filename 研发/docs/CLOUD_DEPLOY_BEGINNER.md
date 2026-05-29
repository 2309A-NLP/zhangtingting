# RAG App Cloud Deployment Guide For Beginners

This document is written for a first-time deployer.

It follows your current project structure:

- backend: FastAPI
- frontend: Vite + React
- storage/services: MySQL, Redis, Milvus, MinIO
- local model: vLLM + `Qwen2.5-0.5B-Instruct`
- deployment mode: Docker Compose

## 1. Recommended architecture

For your current project, the easiest production-like deployment is:

1. One Ubuntu 22.04 cloud server
2. Docker Engine on the server
3. NVIDIA driver + NVIDIA Container Toolkit on the server
4. All backend services started by Docker Compose
5. Frontend built into static files
6. Caddy used as the public web entry

Public access path:

- `https://your-domain.com/` -> frontend
- `https://your-domain.com/api/v1/...` -> backend API

Internal-only services:

- MySQL
- Redis
- Milvus
- MinIO
- vLLM

## 2. What you need to prepare before deployment

Prepare these things first:

1. A cloud server
2. A domain name
3. SSH access to the server
4. Your project code
5. Your local model files

Recommended cloud server spec:

- OS: Ubuntu 22.04
- CPU: 8 vCPU or higher
- RAM: 16 GB or higher
- GPU: NVIDIA GPU, 8 GB VRAM or higher
- Disk: 200 GB SSD or higher

Important:

- If you want to use vLLM, the server must have a usable NVIDIA GPU.
- The deployment script installs NVIDIA Container Toolkit, but it does not guarantee GPU driver installation on every cloud vendor.
- The safest choice is to buy a GPU image that already has the NVIDIA driver installed, or verify that `nvidia-smi` already works on the server.

## 3. Which ports to open

In the cloud security group / firewall, open only:

- `22` for SSH
- `80` for HTTP
- `443` for HTTPS

Do not expose these ports publicly unless you are debugging:

- `8000`
- `8001`
- `3306` / `3307`
- `6379`
- `19530`
- `9000`
- `9001`

## 4. Production config checklist

Before deploying publicly, modify your `.env`.

These items are especially important:

1. `APP_DEBUG=false`
2. `AUTH_ENABLE_DEV_HEADER=false`
3. Replace `APP_SECRET_KEY`
4. Replace MySQL / MinIO passwords
5. Confirm `VLLM_BASE_URL=http://vllm:8000/v1`
6. Confirm `VLLM_API_KEY=local-vllm-key`
7. Confirm `VLLM_HF_MODEL=/models/Qwen2.5-0.5B-Instruct`
8. Confirm `LOCAL_LLM_MODEL_PATH=./data/models/Qwen2.5-0.5B-Instruct`

If you want the frontend to call the public domain, use:

- `VITE_API_BASE_URL=https://your-domain.com`

## 5. Recommended deployment path

For beginners, I recommend this path:

1. Package code on your local machine
2. Upload the package to the cloud server
3. Upload the model directory to the cloud server
4. Unpack the project on the server
5. Run the deployment script on the server
6. Build the frontend
7. Start Caddy for public HTTPS access
8. Verify backend, vLLM, and website

## 6. Step A: package the code on your local machine

Run this in your project root:

```bash
bash scripts/package_release.sh --output ./rag-app-release.tar.gz --include-models no
```

What this does:

- packages the project as a `tar.gz`
- excludes `.git`, logs, cache directories, and temporary data
- does not include `data/models` by default, because model files are usually large

If you really want one big package, you can include models:

```bash
bash scripts/package_release.sh --output ./rag-app-release-with-models.tar.gz --include-models yes
```

## 7. Step B: upload files to the cloud server

You need to upload:

1. `rag-app-release.tar.gz`
2. your `.env`
3. `data/models`

Example with `scp`:

```bash
scp rag-app-release.tar.gz user@your-server-ip:/tmp/
scp .env user@your-server-ip:/tmp/rag-app.env
scp -r data/models user@your-server-ip:/tmp/rag-models
```

If you are using Windows, tools you can use:

- PowerShell + `scp`
- WinSCP
- Xshell / Xftp

## 8. Step C: connect to the server

SSH into the server:

```bash
ssh user@your-server-ip
```

If it is your first time, do these checks:

```bash
uname -a
cat /etc/os-release
nvidia-smi
```

Expected result:

- OS is Ubuntu 22.04 or a supported CentOS/Rocky family
- `nvidia-smi` shows your GPU normally

If `nvidia-smi` does not work, stop here first and install or fix the NVIDIA driver.

## 9. Step D: create the project directory and unpack code

Run on the server:

```bash
sudo mkdir -p /opt/rag-app
sudo tar -xzf /tmp/rag-app-release.tar.gz -C /opt/rag-app
sudo cp /tmp/rag-app.env /opt/rag-app/.env
sudo mkdir -p /opt/rag-app/data
sudo cp -r /tmp/rag-models /opt/rag-app/data/models
sudo chown -R $USER:$USER /opt/rag-app
```

Now check:

```bash
cd /opt/rag-app
ls
ls data/models
```

You should see:

- `docker-compose.yml`
- `scripts/`
- `app/`
- `frontend/`
- `data/models/Qwen2.5-0.5B-Instruct`

## 10. Step E: run the deployment script

### 10.1 Backend-only deployment

If you only want backend services first:

```bash
cd /opt/rag-app
sudo bash scripts/deploy_cloud_server.sh
```

This script will:

1. install common system packages
2. install Docker
3. install NVIDIA Container Toolkit
4. validate `.env`
5. start Docker Compose services

### 10.2 Public website deployment

If you also want public website access:

```bash
cd /opt/rag-app
sudo bash scripts/deploy_cloud_server.sh \
  --install-gpu yes \
  --install-node yes \
  --build-frontend yes \
  --enable-caddy yes \
  --domain your-domain.com \
  --public-base-url https://your-domain.com
```

This will additionally:

1. install Node.js
2. build `frontend/dist`
3. generate `deploy/Caddyfile`
4. start Caddy by `docker-compose.prod.yml`

## 11. Step F: verify the deployment

### 11.1 Check containers

```bash
cd /opt/rag-app
docker compose ps
```

If you enabled Caddy:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

### 11.2 Check API health

```bash
curl http://127.0.0.1:8000/api/v1/health
```

You want to eventually see:

- `mysql=ok`
- `redis=ok`
- `milvus=ok`
- `llm_local=ok`

### 11.3 Check vLLM

```bash
curl http://127.0.0.1:8001/v1/models -H "Authorization: Bearer local-vllm-key"
```

You should get a model list.

### 11.4 Check website

Open in browser:

```text
https://your-domain.com
```

If the page opens and the frontend can login/chat, the public deployment is working.

## 12. Optional host python environment

Your runtime services already use Docker, so host Python is not required for normal web service startup.

You only need host Python if you want to run local scripts on the server, for example:

- `scripts/run_ragas_eval.py`
- `scripts/load_test_chat.py`
- custom maintenance scripts

If you want that, run:

```bash
cd /opt/rag-app
sudo bash scripts/deploy_cloud_server.sh --install-uv yes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Important:

- this host Python environment is optional
- for online deployment, Docker Compose is the main runtime

## 13. How to update your project later

When you modify the code in the future, the simplest update flow is:

1. package a new archive locally
2. upload it to the server
3. unpack it over `/opt/rag-app`
4. rerun the deployment script

Example:

```bash
sudo tar -xzf /tmp/rag-app-release.tar.gz -C /opt/rag-app
sudo chown -R $USER:$USER /opt/rag-app
cd /opt/rag-app
sudo bash scripts/deploy_cloud_server.sh
```

If frontend changed too:

```bash
cd /opt/rag-app
sudo bash scripts/deploy_cloud_server.sh \
  --install-node yes \
  --build-frontend yes \
  --enable-caddy yes \
  --domain your-domain.com \
  --public-base-url https://your-domain.com
```

## 14. Common beginner mistakes

### Mistake 1: forgetting to upload model files

Symptom:

- `vllm` fails to start
- health shows CPU fallback

Check:

```bash
ls /opt/rag-app/data/models/Qwen2.5-0.5B-Instruct
```

### Mistake 2: `.env` still uses local-machine settings

Typical bad values:

- `APP_DEBUG=true`
- `AUTH_ENABLE_DEV_HEADER=true`
- `VLLM_BASE_URL=http://127.0.0.1:8001/v1`

For Docker internal networking, use:

```text
VLLM_BASE_URL=http://vllm:8000/v1
```

### Mistake 3: security group not open

Symptom:

- browser cannot reach the site
- HTTPS never works

Fix:

- open `80` and `443`
- confirm domain DNS points to the server public IP

### Mistake 4: no NVIDIA driver

Symptom:

- `nvidia-smi` not found
- `docker run --gpus all ... nvidia-smi` fails

Fix:

- install the host NVIDIA driver first
- or use a cloud image with driver preinstalled

## 15. Minimum acceptance checklist

Before you say "deployment is complete", confirm all of these:

1. `docker compose ps` shows core containers are up
2. `curl http://127.0.0.1:8000/api/v1/health` works
3. `curl http://127.0.0.1:8001/v1/models ...` works
4. frontend page opens in browser
5. login works
6. chat works
7. one file upload / knowledge ingestion works
8. HTTPS works on your domain

## 16. What I recommend you do in practice

For your first real deployment, do it in two phases:

### Phase 1

Deploy backend only first:

```bash
sudo bash scripts/deploy_cloud_server.sh
```

Verify:

- API healthy
- vLLM healthy
- manual chat request works

### Phase 2

Then deploy public website:

```bash
sudo bash scripts/deploy_cloud_server.sh \
  --install-node yes \
  --build-frontend yes \
  --enable-caddy yes \
  --domain your-domain.com \
  --public-base-url https://your-domain.com
```

This phased method is much safer for beginners because it separates:

1. backend deployment problems
2. frontend/domain/HTTPS problems

If you want, the next step can be:

- generate a production `.env` template for your current project
- generate a one-command server update script
- generate a rollback checklist
