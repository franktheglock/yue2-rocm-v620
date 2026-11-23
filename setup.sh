#!/usr/bin/env bash
# Build YuE2 Studio's Python environments and HIP kernels.
#
#   ./setup.sh            # everything: core + flamingo + kernels, then check
#   ./setup.sh core       # .venv          (engine, web UI, MCP, SheetSage2)
#   ./setup.sh flamingo   # .venv-flamingo (Music Flamingo + Parakeet, transformers 5.x)
#   ./setup.sh kernels    # libv620_kernels.so via build.sh (needs hipcc)
#   ./setup.sh check      # verify what's installed, change nothing
#
# Safe to re-run: existing environments are updated in place, and a venv that can already
# import a ROCm build of torch keeps it.
#
# Overrides (environment variables):
#   PYTHON_VERSION=3.12
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.1
#   TORCH_VERSION=2.13.0+rocm7.1   TORCHAUDIO_VERSION=2.11.0+rocm7.1
#   VENV_DIR=.venv   FLAMINGO_VENV_DIR=.venv-flamingo
#   HIPCC=/opt/rocm/bin/hipcc   GPU_ARCH=gfx1030   (passed to build.sh)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.1}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0+rocm7.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0+rocm7.1}"
VENV_DIR="${VENV_DIR:-.venv}"
FLAMINGO_VENV_DIR="${FLAMINGO_VENV_DIR:-.venv-flamingo}"
YUE2_INFER_WHEEL="yue2_infer-0.1.5-py3-none-any.whl"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
die()   { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

require_uv() {
  command -v uv >/dev/null 2>&1 && return
  die "uv is required (it also installs Python ${PYTHON_VERSION} for you).
  Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh
  then open a new shell and re-run ./setup.sh"
}

make_venv() {  # $1 = venv dir
  if [[ -x "$1/bin/python" ]]; then
    ok "$1 exists ($("$1/bin/python" -V 2>&1))"
  else
    uv venv --python "$PYTHON_VERSION" "$1"
    ok "created $1"
  fi
}

has_rocm_torch() {  # $1 = venv dir
  "$1/bin/python" - <<'EOF' >/dev/null 2>&1
import torch, sys
sys.exit(0 if getattr(torch.version, "hip", None) else 1)
EOF
}

install_torch() {  # $1 = venv dir, $2 = also install torchaudio (1/0)
  if has_rocm_torch "$1"; then
    ok "ROCm torch already importable in $1 ($("$1/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null)); keeping it"
    return
  fi
  bold "  installing torch ${TORCH_VERSION} from ${TORCH_INDEX_URL} (large download)"
  uv pip install --python "$1/bin/python" --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}"
  if [[ "$2" == 1 ]]; then
    uv pip install --python "$1/bin/python" --index-url "$TORCH_INDEX_URL" --no-deps "torchaudio==${TORCHAUDIO_VERSION}"
  fi
  has_rocm_torch "$1" || die "torch installed in $1 but it is not a ROCm build (check TORCH_INDEX_URL)"
  ok "torch $("$1/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
}

install_reqs() {  # $1 = venv dir, $2 = requirements file
  uv pip install --python "$1/bin/python" -r "$2" --excludes requirements/torch-exclude.txt
  ok "installed $(basename "$2")"
}

setup_core() {
  bold "==> Core environment (${VENV_DIR})"
  require_uv
  make_venv "$VENV_DIR"
  install_torch "$VENV_DIR" 1
  install_reqs "$VENV_DIR" requirements/core.txt
  if "$VENV_DIR/bin/python" -c 'import yue2' >/dev/null 2>&1; then
    ok "yue2_infer already installed"
  else
    bold "  fetching ${YUE2_INFER_WHEEL} from m-a-p/YuE2-3B"
    local whl
    whl="$("$VENV_DIR/bin/python" - <<EOF
from huggingface_hub import hf_hub_download
print(hf_hub_download("m-a-p/YuE2-3B", "${YUE2_INFER_WHEEL}"))
EOF
)"
    # --no-deps: the wheel pins CUDA torch 2.10 and would replace the ROCm build.
    uv pip install --python "$VENV_DIR/bin/python" --no-deps "$whl"
    ok "yue2_infer installed"
  fi
}

setup_flamingo() {
  bold "==> Analysis environment (${FLAMINGO_VENV_DIR})"
  require_uv
  make_venv "$FLAMINGO_VENV_DIR"
  install_torch "$FLAMINGO_VENV_DIR" 0
  install_reqs "$FLAMINGO_VENV_DIR" requirements/flamingo.txt
}

setup_kernels() {
  bold "==> HIP kernels (libv620_kernels.so)"
  local hipcc="${HIPCC:-}"
  if [[ -z "$hipcc" ]]; then
    for c in "$(command -v hipcc 2>/dev/null || true)" /opt/rocm/bin/hipcc /opt/rocm-*/bin/hipcc; do
      [[ -n "$c" && -x "$c" ]] && { hipcc="$c"; break; }
    done
  fi
  if [[ -z "$hipcc" ]]; then
    if [[ -f libv620_kernels.so ]]; then
      warn "hipcc not found; keeping the existing libv620_kernels.so"
      return
    fi
    fail "hipcc not found, so libv620_kernels.so can't be built (the engine needs it)."
    echo "      Install the ROCm 7.x HIP SDK (provides hipcc), or point HIPCC=/path/to/hipcc, then run: ./setup.sh kernels"
    return 1
  fi
  if [[ -z "${GPU_ARCH:-}" && -x "$VENV_DIR/bin/python" ]]; then
    GPU_ARCH="$("$VENV_DIR/bin/python" -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])' 2>/dev/null || true)"
  fi
  HIPCC="$hipcc" GPU_ARCH="${GPU_ARCH:-}" ./build.sh
}

check() {
  bold "==> Check"
  local status=0
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    if "$VENV_DIR/bin/python" - <<'EOF'
import importlib, sys
ok = True
def line(good, msg):
    global ok
    ok &= good
    mark = "\033[32m✓" if good else "\033[31m✗"
    print(f"  {mark}\033[0m {msg}")
try:
    import torch
    hip = getattr(torch.version, "hip", None)
    line(bool(hip), f"core: torch {torch.__version__} (HIP {hip})")
    gpu = torch.cuda.is_available()
    line(gpu, "core: GPU visible: " + (torch.cuda.get_device_name(0) if gpu else "no"))
except Exception as e:
    line(False, f"core: torch import failed: {e}")
for mod in ("transformers", "yue2", "fastapi", "uvicorn", "mcp", "soundfile", "mir_eval", "pretty_midi", "torchaudio"):
    try:
        m = importlib.import_module(mod)
        line(True, f"core: {mod} {getattr(m, '__version__', '')}".rstrip())
    except Exception as e:
        line(False, f"core: {mod} missing ({type(e).__name__})")
try:
    import ctypes, pathlib
    so = pathlib.Path("libv620_kernels.so")
    ctypes.CDLL(str(so.resolve())) if so.exists() else None
    line(so.exists(), "core: libv620_kernels.so " + ("loads" if so.exists() else "not built (./setup.sh kernels)"))
except OSError as e:
    line(False, f"core: libv620_kernels.so fails to load: {e}")
sys.exit(0 if ok else 1)
EOF
    then :; else status=1; fi
  else
    fail "core: ${VENV_DIR} missing (./setup.sh core)"; status=1
  fi
  if [[ -x "$FLAMINGO_VENV_DIR/bin/python" ]]; then
    if "$FLAMINGO_VENV_DIR/bin/python" - <<'EOF'
import sys
ok = True
def line(good, msg):
    global ok
    ok &= good
    mark = "\033[32m✓" if good else "\033[31m✗"
    print(f"  {mark}\033[0m {msg}")
try:
    import torch
    line(bool(getattr(torch.version, "hip", None)), f"flamingo: torch {torch.__version__}")
except Exception as e:
    line(False, f"flamingo: torch import failed: {e}")
try:
    import transformers
    from transformers import MusicFlamingoForConditionalGeneration, ParakeetForTDT  # noqa: F401
    line(True, f"flamingo: transformers {transformers.__version__} (MusicFlamingo + Parakeet)")
except Exception as e:
    line(False, f"flamingo: transformers lacks MusicFlamingo/Parakeet ({type(e).__name__}: {e})")
sys.exit(0 if ok else 1)
EOF
    then :; else status=1; fi
  else
    warn "flamingo: ${FLAMINGO_VENV_DIR} missing (optional; ./setup.sh flamingo enables From Audio style + lyrics)"
  fi
  if command -v ffmpeg >/dev/null 2>&1; then ok "ffmpeg found"; else fail "ffmpeg missing (needed to decode uploads; e.g. sudo apt install ffmpeg)"; status=1; fi
  echo
  if [[ $status == 0 ]]; then
    bold "All set. Start the Studio with: ${VENV_DIR}/bin/python web_server.py"
    echo "Model weights: YuE2-3B / YuE2-Vae per the README; optional models have Download buttons in the UI."
  else
    bold "Some checks failed (see above)."
  fi
  return $status
}

case "${1:-all}" in
  core)     setup_core ;;
  flamingo) setup_flamingo ;;
  kernels)  setup_kernels ;;
  check)    check ;;
  all)
    setup_core
    setup_flamingo
    setup_kernels || true
    check
    ;;
  -h|--help|help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown command '$1' (use: all, core, flamingo, kernels, check)" ;;
esac
