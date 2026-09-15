#!/usr/bin/env bash
#
# setup.sh — one-shot bootstrap for the CITS3200 TruckScenes dashboard.
#
# What it does:
#   1. Makes sure the AWS CLI is available (asks before installing).
#   2. Downloads the MAN TruckScenes mini dataset from S3.
#   3. Makes sure Python 3.11 is available (asks before installing).
#   4. Creates a virtualenv and installs requirements.txt + truckscenes-devkit[all] into it.
#   5. Launches the Flask dashboard.
#
# Anything that installs software on system (AWS CLI, Python 3.11) asks
# for y/N confirmation first. Steps already done are skipped.
#
# Usage:
#   ./setup.sh              # set up everything and launch the dashboard
#   ./setup.sh --no-run      # set up everything but don't launch the dashboard
#   ./setup.sh --force-data  # re-sync the dataset
#   ./setup.sh --yes         # auto-confirm every install prompt (non-interactive/CI use)
#   ./setup.sh --help
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="$SCRIPT_DIR/data"
DATASET_VERSION="v1.2-mini"
DATAROOT="$DATA_DIR/man-truckscenes"
DATASET_DIR="$DATAROOT/$DATASET_VERSION"
S3_URI="s3://man-truckscenes/release/mini/"

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_MIN_MINOR=11
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

DASHBOARD_DIR="$SCRIPT_DIR/dashboard"
DASHBOARD_ENTRYPOINT="run.py"
FLASK_HOST="127.0.0.1"
FLASK_PORT="5000"

RUN_AFTER_SETUP=1
FORCE_DATA=0
AUTO_YES=0

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
COLOR_RESET="\033[0m"; COLOR_GREEN="\033[1;32m"; COLOR_YELLOW="\033[1;33m"
COLOR_RED="\033[1;31m"; COLOR_BLUE="\033[1;34m"

info()  { printf "${COLOR_BLUE}[setup]${COLOR_RESET} %s\n" "$1"; }
ok()    { printf "${COLOR_GREEN}[ ok ]${COLOR_RESET} %s\n" "$1"; }
warn()  { printf "${COLOR_YELLOW}[warn]${COLOR_RESET} %s\n" "$1"; }
fail()  { printf "${COLOR_RED}[fail]${COLOR_RESET} %s\n" "$1"; exit 1; }

# confirm "question text" — returns 0 (yes) or 1 (no). Defaults no.
confirm() {
  local prompt="$1"
  if [ "$AUTO_YES" -eq 1 ]; then
    info "$prompt (auto-confirmed via --yes)"
    return 0
  fi
  if [ ! -t 0 ]; then
    warn "$prompt — no terminal to prompt on and --yes not given, assuming 'no'"
    return 1
  fi
  local reply
  read -r -p "$(printf "${COLOR_YELLOW}[?]${COLOR_RESET} %s [y/N] " "$prompt")" reply
  case "$reply" in
    [yY]|[yY][eE][sS]) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
for arg in "$@"; do
  case "$arg" in
    --no-run)     RUN_AFTER_SETUP=0 ;;
    --force-data) FORCE_DATA=1 ;;
    --yes|-y)     AUTO_YES=1 ;;
    --help|-h)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      fail "Unknown argument: $arg (use --help for usage)"
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Step 0: detect OS / package manager
# ---------------------------------------------------------------------------
OS_NAME="unknown"
PKG_MANAGER="none"

detect_platform() {
  case "$(uname -s)" in
    Linux*)
      OS_NAME="linux"
      if   command -v apt-get  >/dev/null 2>&1; then PKG_MANAGER="apt"
      elif command -v dnf      >/dev/null 2>&1; then PKG_MANAGER="dnf"
      elif command -v yum      >/dev/null 2>&1; then PKG_MANAGER="yum"
      elif command -v pacman   >/dev/null 2>&1; then PKG_MANAGER="pacman"
      elif command -v apk      >/dev/null 2>&1; then PKG_MANAGER="apk"
      fi
      ;;
    Darwin*)
      OS_NAME="macos"
      if command -v brew >/dev/null 2>&1; then PKG_MANAGER="brew"; fi
      ;;
    *)
      OS_NAME="unknown"
      ;;
  esac
  info "Detected platform: $OS_NAME (package manager: $PKG_MANAGER)"
}

pkg_install() {
  # $1 = human-readable name for logging, $2.. = package name(s) per-manager handled by caller
  case "$PKG_MANAGER" in
    apt)     sudo apt-get update -y && sudo apt-get install -y "$@" ;;
    dnf)     sudo dnf install -y "$@" ;;
    yum)     sudo yum install -y "$@" ;;
    pacman)  sudo pacman -Sy --noconfirm "$@" ;;
    apk)     sudo apk add --no-cache "$@" ;;
    brew)    brew install "$@" ;;
    *)       return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Step 1: ensure AWS CLI is present
# ---------------------------------------------------------------------------
ensure_aws_cli() {
  if command -v aws >/dev/null 2>&1; then
    ok "AWS CLI already installed ($(aws --version 2>&1 | head -n1))"
    return
  fi

  warn "AWS CLI not found — it's needed to download the dataset from S3."
  detect_platform

  # Method 1: system package manager
  if [ "$PKG_MANAGER" != "none" ]; then
    if confirm "Install the AWS CLI using $PKG_MANAGER (may prompt for sudo)?"; then
      case "$PKG_MANAGER" in
        apt)    pkg_install awscli || true ;;
        dnf)    pkg_install awscli || true ;;
        yum)    pkg_install awscli || true ;;
        pacman) pkg_install aws-cli || true ;;
        apk)    pkg_install aws-cli || true ;;
        brew)   pkg_install awscli || true ;;
      esac
    else
      info "Skipped $PKG_MANAGER install of AWS CLI."
    fi
  else
    warn "No supported package manager detected."
  fi

  if command -v aws >/dev/null 2>&1; then
    ok "AWS CLI installed via $PKG_MANAGER"
    return
  fi

  # Method 2: official AWS installer script
  if command -v curl >/dev/null 2>&1 && command -v unzip >/dev/null 2>&1 \
     && { [ "$OS_NAME" = "linux" ] || [ "$OS_NAME" = "macos" ]; }; then
    if confirm "Install the AWS CLI using AWS's official installer (downloads from awscli.amazonaws.com, requires sudo)?"; then
      local tmp_dir
      tmp_dir="$(mktemp -d)"
      if [ "$OS_NAME" = "linux" ]; then
        curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$tmp_dir/awscliv2.zip" \
          && unzip -q "$tmp_dir/awscliv2.zip" -d "$tmp_dir" \
          && sudo "$tmp_dir/aws/install"
      elif [ "$OS_NAME" = "macos" ]; then
        curl -fsSL "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "$tmp_dir/AWSCLIV2.pkg" \
          && sudo installer -pkg "$tmp_dir/AWSCLIV2.pkg" -target /
      fi
      rm -rf "$tmp_dir"
    else
      info "Skipped official-installer install of AWS CLI."
    fi
  fi

  if command -v aws >/dev/null 2>&1; then
    ok "AWS CLI installed via official installer"
    return
  fi

  fail "AWS CLI is not installed. Please install it yourself (see https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and re-run this script."
}

# ---------------------------------------------------------------------------
# Step 2: download the dataset
# ---------------------------------------------------------------------------
download_dataset() {
  if [ "$FORCE_DATA" -eq 0 ] && [ -d "$DATASET_DIR" ] \
     && [ -n "$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    ok "Dataset already present at $DATASET_DIR (use --force-data to re-sync)"
    return
  fi

  mkdir -p "$DATA_DIR"
  info "Downloading MAN TruckScenes ($DATASET_VERSION) from S3 — this can take a while..."
  aws s3 sync --no-sign-request "$S3_URI" "$DATA_DIR/" \
    || fail "Dataset download failed. Check your internet connection and try again."

  ok "Dataset downloaded to $DATASET_DIR"
}

# ---------------------------------------------------------------------------
# Step 3: python 3.11 + venv + requirements
# ---------------------------------------------------------------------------
find_python311() {
  for candidate in python3.11 python3.11.0; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

ensure_python311() {
  if PY_BIN="$(find_python311)"; then
    ok "Found $PY_BIN"
    return
  fi

  warn "python3.11 not found — open3d/matplotlib versions needed are incompatible with newer Pythons."
  detect_platform

  if [ "$PKG_MANAGER" = "none" ]; then
    fail "No supported package manager detected to install python3.11. Please install it yourself and re-run this script."
  fi

  if confirm "Install python3.11 using $PKG_MANAGER (may prompt for sudo)?"; then
    case "$PKG_MANAGER" in
      apt)
        sudo apt-get update -y
        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev || true
        ;;
      dnf)    sudo dnf install -y python3.11 || true ;;
      yum)    sudo yum install -y python3.11 || true ;;
      pacman) sudo pacman -Sy --noconfirm python311 || true ;;
      brew)   brew install python@3.11 || true ;;
    esac
  else
    info "Skipped python3.11 install."
  fi

  if PY_BIN="$(find_python311)"; then
    ok "Installed $PY_BIN"
    return
  fi

  fail "python3.11 is not installed. Please install it yourself and re-run this script."
}

ensure_venv() {
  if [ -x "$VENV_DIR/bin/python" ]; then
    ok "Virtualenv already exists at $VENV_DIR"
  else
    info "Creating virtualenv at $VENV_DIR using $PY_BIN..."
    "$PY_BIN" -m venv "$VENV_DIR" \
      || fail "Failed to create the virtualenv. On Debian/Ubuntu you may need: sudo apt-get install python3.11-venv"
    ok "Virtualenv created"
  fi
}

install_requirements() {
  [ -f "$REQUIREMENTS_FILE" ] || fail "Could not find requirements.txt at $REQUIREMENTS_FILE"

  info "Installing Python dependencies"
  "$VENV_DIR/bin/pip" install --upgrade pip -q
  "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE" -q \
    || fail "pip install -r requirements.txt failed"
  "$VENV_DIR/bin/pip" install "truckscenes-devkit[all]" -q \
    || fail "pip install truckscenes-devkit[all] failed"
  ok "Python dependencies installed"
}

# ---------------------------------------------------------------------------
# Step 4: launch the dashboard
# ---------------------------------------------------------------------------
run_dashboard() {
  [ -f "$DASHBOARD_DIR/$DASHBOARD_ENTRYPOINT" ] \
    || fail "Could not find $DASHBOARD_DIR/$DASHBOARD_ENTRYPOINT"

  info "Starting the dashboard at http://$FLASK_HOST:$FLASK_PORT ..."
  cd "$DASHBOARD_DIR"
  exec "$VENV_DIR/bin/python" "$DASHBOARD_ENTRYPOINT"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  info "=== CITS3200 TruckScenes dashboard setup ==="

  ensure_aws_cli
  download_dataset
  ensure_python311
  ensure_venv
  install_requirements

  ok "=== Setup complete ==="

  if [ "$RUN_AFTER_SETUP" -eq 1 ]; then
    run_dashboard
  else
    info "Skipping auto-launch (--no-run given). Start it later with:"
    info "  ./setup.sh"
    info "or manually:"
    info "  source .venv/bin/activate && cd dashboard && python $DASHBOARD_ENTRYPOINT"
  fi
}

main
