#!/usr/bin/env bash
# Frontier Bridge bootstrap: fresh clone -> working `frontier` CLI -> doctor.
#
# Idempotent: safe to re-run. Installs nothing system-wide and asks for no
# sudo — it tells you what is missing (via `frontier doctor`) and how to get
# it. Run from the repo root:
#
#   git clone https://github.com/Brianletort/Frontier-Bridge.git
#   cd Frontier-Bridge && ./scripts/bootstrap.sh
set -euo pipefail

cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Install Python 3.10+ first:"
    echo "  Ubuntu:  sudo apt install -y python3 python3-venv python3-pip"
    echo "  macOS:   brew install python"
    exit 1
fi

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python ${PYVER} found but 3.10+ is required."
    exit 1
fi

say "Python ${PYVER} — creating virtualenv (.venv/)"
if [ ! -x .venv/bin/python ]; then
    if ! python3 -m venv .venv 2>/dev/null; then
        echo "venv module missing. On Ubuntu: sudo apt install -y python3-venv"
        exit 1
    fi
fi

say "Installing Frontier Bridge (editable, with dev extras)"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e ".[dev]"

say "Checking this machine (frontier doctor)"
.venv/bin/frontier doctor || true

say "Done"
echo "Activate with:  source .venv/bin/activate"
echo "Then:           frontier detect -o hardware_profiles/local/my_machine.yaml"
echo "                frontier runbook match"
echo "Optional tools doctor may have flagged (Ubuntu): sudo apt install -y fio bolt"
