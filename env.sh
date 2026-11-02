# shellcheck shell=bash
# source this: keeps every cache/install inside ~/MimicLite (uv, uv-managed python, uv tools, HF hub).
export MIMICLITE_ROOT=$HOME/MimicLite
export UV_CACHE_DIR=$MIMICLITE_ROOT/.cache/uv
export UV_PYTHON_INSTALL_DIR=$MIMICLITE_ROOT/.cache/uv-python
export UV_PYTHON_PREFERENCE=only-managed
export UV_TOOL_DIR=$MIMICLITE_ROOT/.cache/uv-tools
export UV_TOOL_BIN_DIR=$MIMICLITE_ROOT/.cache/uv-tools/bin
export HF_HOME=$MIMICLITE_ROOT/.cache/hf
export PATH=$HOME/.local/bin:$PATH
