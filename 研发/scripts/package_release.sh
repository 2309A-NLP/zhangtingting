#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${PROJECT_ROOT}/rag-app-release.tar.gz"
INCLUDE_MODELS="no"
GIT_PULL="no"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/package_release.sh [options]

Options:
  --output <path>          Output tar.gz path. Default: ./rag-app-release.tar.gz
  --include-models <yes|no>
                           Whether to include data/models in the archive.
                           Default: no
  --git-pull <yes|no>      Run git pull --ff-only before packaging. Default: no
  --help                   Show this help.

Examples:
  bash scripts/package_release.sh
  bash scripts/package_release.sh --output /tmp/rag-app.tar.gz --include-models yes
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --include-models)
      INCLUDE_MODELS="$2"
      shift 2
      ;;
    --git-pull)
      GIT_PULL="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$INCLUDE_MODELS" != "yes" && "$INCLUDE_MODELS" != "no" ]]; then
  echo "--include-models must be yes or no" >&2
  exit 1
fi

if [[ "$GIT_PULL" != "yes" && "$GIT_PULL" != "no" ]]; then
  echo "--git-pull must be yes or no" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

if [[ "$GIT_PULL" == "yes" ]]; then
  if [[ -d .git ]]; then
    git pull --ff-only
  else
    echo "Current directory is not a git repository, skip git pull."
  fi
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

TAR_EXCLUDES=(
  --exclude=.git
  --exclude=.idea
  --exclude=.pytest_cache
  --exclude=__pycache__
  --exclude=*.pyc
  --exclude=frontend/node_modules
  --exclude=frontend/dist
  --exclude=frontend/*.log
  --exclude=data/uploads
  --exclude=data/parsed
  --exclude=data/pdf_parse_cache
  --exclude=data/eval
  --exclude=data/minio
)

if [[ "$INCLUDE_MODELS" == "no" ]]; then
  TAR_EXCLUDES+=(--exclude=data/models)
fi

echo "[INFO] Packaging project from: $PROJECT_ROOT"
echo "[INFO] Output archive: $OUTPUT_PATH"
echo "[INFO] Include models: $INCLUDE_MODELS"

tar -czf "$OUTPUT_PATH" "${TAR_EXCLUDES[@]}" -C "$PROJECT_ROOT" .

echo "[OK] Archive created: $OUTPUT_PATH"
