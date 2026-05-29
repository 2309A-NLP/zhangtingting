#!/usr/bin/env bash
set -euo pipefail

WORKDIR=""
INSTALL_GPU="yes"
INSTALL_UV="no"
INSTALL_NODE="no"
BUILD_FRONTEND="no"
ENABLE_CADDY="no"
DOMAIN=""
PUBLIC_BASE_URL=""
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OS_FAMILY=""

usage() {
  cat <<'EOF'
Usage:
  sudo bash scripts/deploy_cloud_server.sh [options]

Options:
  --workdir <path>             Project root on the server. Default: current project root
  --install-gpu <yes|no>       Install NVIDIA container toolkit. Default: yes
  --install-uv <yes|no>        Install uv and python tooling on host. Default: no
  --install-node <yes|no>      Install Node.js 20 on host. Default: no
  --build-frontend <yes|no>    Build frontend/dist on the server. Default: no
  --enable-caddy <yes|no>      Start Caddy for public web access. Default: no
  --domain <domain>            Public domain used by Caddy, for example rag.example.com
  --public-base-url <url>      Frontend API base URL, for example https://rag.example.com
  --help                       Show help

Examples:
  sudo bash scripts/deploy_cloud_server.sh

  sudo bash scripts/deploy_cloud_server.sh \
    --install-gpu yes \
    --install-node yes \
    --build-frontend yes \
    --enable-caddy yes \
    --domain rag.example.com \
    --public-base-url https://rag.example.com
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
    --install-gpu)
      INSTALL_GPU="$2"
      shift 2
      ;;
    --install-uv)
      INSTALL_UV="$2"
      shift 2
      ;;
    --install-node)
      INSTALL_NODE="$2"
      shift 2
      ;;
    --build-frontend)
      BUILD_FRONTEND="$2"
      shift 2
      ;;
    --enable-caddy)
      ENABLE_CADDY="$2"
      shift 2
      ;;
    --domain)
      DOMAIN="$2"
      shift 2
      ;;
    --public-base-url)
      PUBLIC_BASE_URL="$2"
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
  fail "Please run this script with sudo or as root."
fi

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$PROJECT_ROOT"
fi

detect_os() {
  if [[ ! -f /etc/os-release ]]; then
    fail "Cannot detect operating system. /etc/os-release not found."
  fi

  # shellcheck disable=SC1091
  source /etc/os-release

  case "${ID:-}" in
    ubuntu|debian)
      OS_FAMILY="debian"
      ;;
    centos|rhel|rocky|almalinux)
      OS_FAMILY="rhel"
      ;;
    *)
      fail "Unsupported OS: ${ID:-unknown}. Recommended: Ubuntu 22.04 or CentOS/Rocky 9."
      ;;
  esac
}

run_pkg_install() {
  if [[ "$OS_FAMILY" == "debian" ]]; then
    apt-get update
    apt-get install -y "$@"
  else
    dnf install -y "$@" || yum install -y "$@"
  fi
}

install_base_packages() {
  log "Installing base packages"
  if [[ "$OS_FAMILY" == "debian" ]]; then
    run_pkg_install ca-certificates curl gnupg lsb-release git tar unzip jq
  else
    run_pkg_install ca-certificates curl gnupg2 git tar unzip jq yum-utils
  fi
}

install_docker() {
  if command -v docker >/dev/null 2>&1; then
    log "Docker is already installed"
  else
    log "Installing Docker Engine"
    curl -fsSL https://get.docker.com | sh
  fi

  systemctl enable docker
  systemctl restart docker

  if ! docker version >/dev/null 2>&1; then
    fail "Docker installed but docker daemon is not available."
  fi
}

install_nvidia_toolkit() {
  if [[ "$INSTALL_GPU" != "yes" ]]; then
    warn "Skipping NVIDIA container toolkit installation."
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found on the host. This usually means the NVIDIA driver is not installed."
    warn "Please install the GPU driver first, or use a cloud image with NVIDIA driver preinstalled."
    return
  fi

  if command -v nvidia-ctk >/dev/null 2>&1; then
    log "NVIDIA container toolkit is already installed"
  else
    log "Installing NVIDIA container toolkit"
    if [[ "$OS_FAMILY" == "debian" ]]; then
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
      dnf install -y nvidia-container-toolkit || yum install -y nvidia-container-toolkit
    fi
  fi

  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker

  log "Testing GPU inside Docker"
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
}

install_python_tooling() {
  if [[ "$INSTALL_UV" != "yes" ]]; then
    warn "Skipping host python/uv installation."
    return
  fi

  log "Installing host python tooling"
  if [[ "$OS_FAMILY" == "debian" ]]; then
    run_pkg_install python3 python3-pip python3-venv
  else
    run_pkg_install python3 python3-pip
  fi

  python3 -m pip install --upgrade pip
  python3 -m pip install uv
}

install_node() {
  if [[ "$INSTALL_NODE" != "yes" ]]; then
    warn "Skipping Node.js installation."
    return
  fi

  log "Installing Node.js 20"
  curl -fsSL https://rpm.nodesource.com/setup_20.x | bash - >/dev/null 2>&1 || true
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null 2>&1 || true
  run_pkg_install nodejs
  node --version
  npm --version
}

validate_project() {
  log "Validating project files in $WORKDIR"

  [[ -f "$WORKDIR/docker-compose.yml" ]] || fail "docker-compose.yml not found in $WORKDIR"
  [[ -f "$WORKDIR/.env" ]] || fail ".env not found in $WORKDIR"

  if [[ ! -d "$WORKDIR/data/models/Qwen2.5-0.5B-Instruct" ]]; then
    warn "Model directory data/models/Qwen2.5-0.5B-Instruct not found."
    warn "vLLM cannot start until you upload the local model files."
  fi

  if grep -q '^APP_DEBUG=true' "$WORKDIR/.env"; then
    warn "APP_DEBUG=true detected in .env. Please change it to false before public deployment."
  fi

  if grep -q '^AUTH_ENABLE_DEV_HEADER=true' "$WORKDIR/.env"; then
    warn "AUTH_ENABLE_DEV_HEADER=true detected in .env. Please change it to false before public deployment."
  fi
}

build_frontend_if_needed() {
  if [[ "$BUILD_FRONTEND" != "yes" ]]; then
    warn "Skipping frontend build."
    return
  fi

  command -v npm >/dev/null 2>&1 || fail "npm not found. Re-run with --install-node yes."
  [[ -n "$PUBLIC_BASE_URL" ]] || fail "--public-base-url is required when --build-frontend yes"

  log "Building frontend"
  cat > "$WORKDIR/frontend/.env.production" <<EOF
VITE_API_BASE_URL=$PUBLIC_BASE_URL
EOF

  (
    cd "$WORKDIR/frontend"
    npm ci
    npm run build
  )
}

prepare_caddy() {
  if [[ "$ENABLE_CADDY" != "yes" ]]; then
    warn "Skipping Caddy setup."
    return
  fi

  [[ -n "$DOMAIN" ]] || fail "--domain is required when --enable-caddy yes"
  [[ -d "$WORKDIR/frontend/dist" ]] || fail "frontend/dist not found. Build the frontend first."
  [[ -f "$WORKDIR/deploy/Caddyfile.example" ]] || fail "deploy/Caddyfile.example not found."

  log "Generating deploy/Caddyfile"
  sed "s/__DOMAIN__/$DOMAIN/g" \
    "$WORKDIR/deploy/Caddyfile.example" \
    > "$WORKDIR/deploy/Caddyfile"
}

start_services() {
  log "Starting services with Docker Compose"
  cd "$WORKDIR"

  local compose_files=(-f docker-compose.yml)
  if [[ "$ENABLE_CADDY" == "yes" ]]; then
    compose_files+=(-f docker-compose.prod.yml)
  fi

  docker compose "${compose_files[@]}" pull
  docker compose "${compose_files[@]}" up -d --build
}

print_next_steps() {
  cat <<EOF

[OK] Deployment script finished.

Useful verification commands:
  cd $WORKDIR
  docker compose ps
  docker logs rag-api --tail 100
  curl http://127.0.0.1:8000/api/v1/health
  curl http://127.0.0.1:8001/v1/models -H "Authorization: Bearer local-vllm-key"

If Caddy is enabled:
  Open in browser: https://$DOMAIN

If you still see connection issues:
  1. Check server security group / firewall
  2. Check whether .env uses VLLM_BASE_URL=http://vllm:8000/v1
  3. Check whether data/models/Qwen2.5-0.5B-Instruct exists on the server
EOF
}

detect_os
install_base_packages
install_docker
install_nvidia_toolkit
install_python_tooling
install_node
validate_project
build_frontend_if_needed
prepare_caddy
start_services
print_next_steps
