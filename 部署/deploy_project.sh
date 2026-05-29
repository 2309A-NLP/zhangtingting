#!/usr/bin/env bash
set -euo pipefail

WORKDIR=""
ENV_FILE=""
COMPOSE_FILE=""
INSTALL_DOCKER="yes"
INSTALL_NVIDIA_TOOLKIT="no"
START_VLLM="yes"
REBUILD="yes"

usage() {
  cat <<'EOF'
Usage:
  sudo bash scripts/deploy_project.sh [options]

Options:
  --workdir <path>                 Project root on the server
  --env-file <path>                .env file path, default: <workdir>/.env
  --compose-file <path>            docker-compose.yml path, default: <workdir>/docker-compose.yml
  --install-docker <yes|no>        Install Docker automatically, default: yes
  --install-nvidia-toolkit <yes|no>
                                   Install NVIDIA container toolkit, default: no
  --start-vllm <yes|no>            Start vLLM service together, default: yes
  --rebuild <yes|no>               Rebuild compose services, default: yes
  --help                           Show help

Examples:
  sudo bash scripts/deploy_project.sh --workdir /opt/rag-app
  sudo bash scripts/deploy_project.sh --workdir /opt/rag-app --start-vllm no
  sudo bash scripts/deploy_project.sh --workdir /opt/rag-app --install-nvidia-toolkit yes
EOF
}

log() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --compose-file)
      COMPOSE_FILE="$2"
      shift 2
      ;;
    --install-docker)
      INSTALL_DOCKER="$2"
      shift 2
      ;;
    --install-nvidia-toolkit)
      INSTALL_NVIDIA_TOOLKIT="$2"
      shift 2
      ;;
    --start-vllm)
      START_VLLM="$2"
      shift 2
      ;;
    --rebuild)
      REBUILD="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  fail "Please run with sudo or as root."
fi

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(pwd)"
fi

if [[ -z "$ENV_FILE" ]]; then
  ENV_FILE="$WORKDIR/.env"
fi

if [[ -z "$COMPOSE_FILE" ]]; then
  COMPOSE_FILE="$WORKDIR/docker-compose.yml"
fi

detect_os() {
  if [[ ! -f /etc/os-release ]]; then
    fail "Cannot detect OS. /etc/os-release not found."
  fi

  # shellcheck disable=SC1091
  source /etc/os-release
  case "${ID:-}" in
    ubuntu|debian)
      PKG_INSTALL="apt-get install -y"
      PKG_UPDATE="apt-get update"
      ;;
    centos|rhel|rocky|almalinux)
      PKG_INSTALL="dnf install -y"
      PKG_UPDATE="dnf makecache"
      ;;
    *)
      fail "Unsupported OS: ${ID:-unknown}. Recommended: Ubuntu 22.04."
      ;;
  esac
}

install_base_packages() {
  log "Installing base packages"
  eval "$PKG_UPDATE"
  eval "$PKG_INSTALL curl git tar gzip ca-certificates gnupg lsb-release jq"
}

install_docker() {
  if [[ "$INSTALL_DOCKER" != "yes" ]]; then
    warn "Skipping Docker installation."
    return
  fi

  if command -v docker >/dev/null 2>&1; then
    log "Docker already installed"
  else
    log "Installing Docker"
    curl -fsSL https://get.docker.com | sh
  fi

  systemctl enable docker
  systemctl restart docker

  if ! docker version >/dev/null 2>&1; then
    fail "Docker is installed but daemon is unavailable."
  fi
}

install_nvidia_toolkit() {
  if [[ "$INSTALL_NVIDIA_TOOLKIT" != "yes" ]]; then
    warn "Skipping NVIDIA container toolkit installation."
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found. Install NVIDIA driver first."
  fi

  if command -v nvidia-ctk >/dev/null 2>&1; then
    log "NVIDIA container toolkit already installed"
  else
    log "Installing NVIDIA container toolkit"
    if grep -qiE 'ubuntu|debian' /etc/os-release; then
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
      apt-get update
      apt-get install -y nvidia-container-toolkit
    else
      curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        > /etc/yum.repos.d/nvidia-container-toolkit.repo
      dnf install -y nvidia-container-toolkit
    fi
  fi

  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
}

validate_project_files() {
  [[ -d "$WORKDIR" ]] || fail "Workdir not found: $WORKDIR"
  [[ -f "$COMPOSE_FILE" ]] || fail "docker-compose.yml not found: $COMPOSE_FILE"
  [[ -f "$ENV_FILE" ]] || fail ".env not found: $ENV_FILE"

  if [[ ! -d "$WORKDIR/data/models/Qwen2.5-0.5B-Instruct" ]]; then
    warn "Local model directory not found: data/models/Qwen2.5-0.5B-Instruct"
  fi
}

prepare_directories() {
  log "Preparing runtime directories"
  mkdir -p "$WORKDIR/data/uploads"
  mkdir -p "$WORKDIR/data/parsed"
  mkdir -p "$WORKDIR/data/pdf_parse_cache"
  mkdir -p "$WORKDIR/data/hf-home"
}

start_services() {
  log "Starting services"
  cd "$WORKDIR"

  local services=("mysql" "redis" "etcd" "minio" "milvus")
  if [[ "$START_VLLM" == "yes" ]]; then
    services+=("vllm" "api")
  else
    services+=("api")
  fi

  if [[ "$REBUILD" == "yes" ]]; then
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --build "${services[@]}"
  else
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d "${services[@]}"
  fi
}

print_verify_commands() {
  cat <<EOF

[OK] Deployment script completed.

Next verification commands:

cd $WORKDIR
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps
docker logs rag-api --tail 100
curl http://127.0.0.1:8000/api/v1/health

If vLLM is enabled:
curl http://127.0.0.1:8001/v1/models -H "Authorization: Bearer local-vllm-key"

If API is up but retrieval is abnormal:
1. Check whether data/models/Qwen2.5-0.5B-Instruct exists
2. Check whether .env uses VLLM_BASE_URL=http://vllm:8000/v1
3. Check docker logs rag-api and rag-vllm
EOF
}

detect_os
install_base_packages
install_docker
install_nvidia_toolkit
validate_project_files
prepare_directories
start_services
print_verify_commands
