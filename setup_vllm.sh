#!/bin/bash
# 如果用 sh 执行，自动切换到 bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi
# ============================================================
# vLLM 安装脚本（适用于 RunPod 环境）
#
# 功能：
#   1. 安装 vLLM v0.10.1 预编译 wheel
#   2. 克隆 Soul-AILab 定制 fork（RAS 采样支持）
#   3. 替换 4 个核心文件以启用自定义采样
#   4. 验证安装结果
#
# 用法：
#   bash setup_vllm.sh
#   bash setup_vllm.sh --vllm-version 0.10.1
#   bash setup_vllm.sh --skip-install   # 仅补丁，跳过 pip install
# ============================================================

set -e

# ------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------
VLLM_VERSION="${VLLM_VERSION:-0.10.1}"
FORK_REPO="https://github.com/Soul-AILab/vllm.git"
FORK_TAG="v0.10.1.1-soulxpodcast"
CLONE_DIR="/tmp/vllm-soulxpodcast"
SKIP_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --vllm-version)
            VLLM_VERSION="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--vllm-version VERSION] [--skip-install]"
            echo ""
            echo "Options:"
            echo "  --vllm-version VERSION  vLLM version to install (default: 0.10.1)"
            echo "  --skip-install          Skip pip install, only apply patches"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------
# 颜色输出
# ------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { printf "${GREEN}[INFO]${NC} %s\n" "$1"; }
log_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
log_error() { printf "${RED}[ERROR]${NC} %s\n" "$1"; }

# ------------------------------------------------------------
# 前置检查
# ------------------------------------------------------------
log_info "检查环境..."

if ! python3 -c "import torch" 2>/dev/null; then
    log_error "PyTorch 未安装，请先运行 setup_runpod.sh"
    exit 1
fi

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
log_info "PyTorch: ${TORCH_VERSION}, CUDA: ${CUDA_VERSION}"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log_error "CUDA 不可用，请检查 GPU 环境"
    exit 1
fi

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
log_info "GPU: ${GPU_NAME}"

# ------------------------------------------------------------
# Step 1: 安装 vLLM
# ------------------------------------------------------------
if [ "$SKIP_INSTALL" = true ]; then
    log_warn "跳过 pip install（--skip-install）"
else
    # 检查是否已安装
    INSTALLED_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
    if [ "$INSTALLED_VERSION" = "$VLLM_VERSION" ]; then
        log_warn "vLLM ${VLLM_VERSION} 已安装，跳过"
    else
        if [ -n "$INSTALLED_VERSION" ]; then
            log_warn "已安装 vLLM ${INSTALLED_VERSION}，将升级/降级到 ${VLLM_VERSION}"
        fi
        log_info "Step 1/4: 安装 vLLM ${VLLM_VERSION}..."
        pip install "vllm==${VLLM_VERSION}" 2>&1 | tail -5
        log_info "vLLM 安装完成"
    fi
fi

# ------------------------------------------------------------
# Step 2: 克隆定制 fork
# ------------------------------------------------------------
log_info "Step 2/4: 获取 Soul-AILab 定制 fork..."

if [ -d "$CLONE_DIR" ]; then
    log_warn "临时目录已存在，清理后重新克隆"
    rm -rf "$CLONE_DIR"
fi

git clone --depth 1 --branch "$FORK_TAG" "$FORK_REPO" "$CLONE_DIR" 2>&1 | tail -3
log_info "Fork 克隆完成 (tag: ${FORK_TAG})"

# ------------------------------------------------------------
# Step 3: 替换核心文件（RAS 采样补丁）
# ------------------------------------------------------------
log_info "Step 3/4: 应用 RAS 采样补丁..."

VLLM_PATH=$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
log_info "vLLM 安装路径: ${VLLM_PATH}"

# 备份原始文件
BACKUP_DIR="${VLLM_PATH}/_backup_original"
if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "${BACKUP_DIR}/model_executor/layers"
    cp "${VLLM_PATH}/model_executor/layers/sampler.py" "${BACKUP_DIR}/model_executor/layers/sampler.py"
    cp "${VLLM_PATH}/model_executor/layers/utils.py" "${BACKUP_DIR}/model_executor/layers/utils.py"
    cp "${VLLM_PATH}/model_executor/sampling_metadata.py" "${BACKUP_DIR}/model_executor/sampling_metadata.py"
    cp "${VLLM_PATH}/sampling_params.py" "${BACKUP_DIR}/sampling_params.py"
    log_info "原始文件已备份到 ${BACKUP_DIR}"
else
    log_warn "备份已存在，跳过备份"
fi

# 替换文件
PATCH_FILES=(
    "model_executor/layers/sampler.py"
    "model_executor/layers/utils.py"
    "model_executor/sampling_metadata.py"
    "sampling_params.py"
)

for f in "${PATCH_FILES[@]}"; do
    SRC="${CLONE_DIR}/vllm/${f}"
    DST="${VLLM_PATH}/${f}"
    if [ ! -f "$SRC" ]; then
        log_error "补丁文件不存在: ${SRC}"
        exit 1
    fi
    cp "$SRC" "$DST"
    log_info "  patched: ${f}"
done

log_info "补丁应用完成（共 ${#PATCH_FILES[@]} 个文件）"

# ------------------------------------------------------------
# Step 4: 验证
# ------------------------------------------------------------
log_info "Step 4/4: 验证安装..."

# 基本 import 测试
if python3 -c "
import os
os.environ['VLLM_USE_V1'] = '0'
from vllm import LLM
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
print('import OK')
" 2>&1; then
    log_info "vLLM import 验证通过"
else
    log_error "vLLM import 失败，请检查错误信息"
    exit 1
fi

# 清理临时文件
rm -rf "$CLONE_DIR"
log_info "临时文件已清理"

# ------------------------------------------------------------
# 完成
# ------------------------------------------------------------
echo ""
echo "========================================"
log_info "vLLM ${VLLM_VERSION} + RAS 补丁安装完成"
echo "========================================"
echo ""
echo "启动方式："
echo "  WebUI:  python webui.py --model_path pretrained_models/SoulX-Podcast-1.7B --llm_engine vllm"
echo "  API:    python run_api.py --model pretrained_models/SoulX-Podcast-1.7B --engine vllm"
echo "  CLI:    python cli/podcast.py --model_path pretrained_models/SoulX-Podcast-1.7B --llm_engine vllm"
echo ""
