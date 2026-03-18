#!/bin/bash
# ============================================================
# SoulX-Podcast RunPod 一键部署脚本
# 适用镜像: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# GPU: RTX 4090 (24GB VRAM)
# ============================================================

set -e

WORK_DIR="/workspace/SoulX-Podcast"
MODEL_BASE="base"       # base | dialect | both
ENABLE_API=false        # true: 启动 API 服务; false: 仅安装环境

# ------------------------------------------------------------
# 颜色输出
# ------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ------------------------------------------------------------
# 1. 克隆项目
# ------------------------------------------------------------
log_info "Step 1/5: 克隆项目..."

if [ -d "$WORK_DIR" ]; then
    log_warn "项目目录已存在，跳过克隆"
else
    git clone https://github.com/Soul-AILab/SoulX-Podcast.git "$WORK_DIR"
    log_info "项目克隆完成"
fi

cd "$WORK_DIR"

# ------------------------------------------------------------
# 2. 升级 PyTorch + 安装 Python 依赖
# ------------------------------------------------------------
log_info "Step 2/5: 升级 PyTorch + 安装依赖..."

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
log_info "当前 PyTorch 版本: $TORCH_VERSION"

# 升级 PyTorch 到项目要求的 2.7.1
# torch 2.7.1 仅提供 cu126 和 cu128 的 wheel
log_info "升级 PyTorch 到 2.7.1 (cu126) ..."
pip install --no-cache-dir \
    torch==2.7.1 \
    torchaudio==2.7.1 \
    torchvision==0.22.1 \
    --index-url "https://download.pytorch.org/whl/cu126"

NEW_TORCH=$(python3 -c "import torch; print(torch.__version__)")
log_info "PyTorch 升级完成: $TORCH_VERSION -> $NEW_TORCH"

# 安装其余依赖
pip install --no-cache-dir \
    librosa \
    numpy \
    scipy \
    s3tokenizer \
    diffusers \
    einops \
    gradio \
    "triton>=3.0.0" \
    "transformers==4.57.1" \
    "accelerate==1.10.1" \
    onnxruntime \
    onnxruntime-gpu

# API 依赖
pip install --no-cache-dir \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "python-multipart>=0.0.6" \
    "pydantic>=2.0.0" \
    "aiofiles>=23.2.0"

log_info "依赖安装完成"

# ------------------------------------------------------------
# 3. 下载模型
# ------------------------------------------------------------
log_info "Step 3/5: 下载模型..."

# huggingface_hub 在 step 2 中已随 transformers 安装，仅补装 cli 依赖
pip install --no-cache-dir 'huggingface_hub[cli]>=0.34.0,<1.0'

# 确保 huggingface-cli 在 PATH 中
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
SCRIPTS_DIR=$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))' 2>/dev/null || true)
if [ -n "$SCRIPTS_DIR" ]; then
    export PATH="$SCRIPTS_DIR:$PATH"
fi
hash -r

mkdir -p pretrained_models

download_model() {
    local model_name=$1
    local local_dir=$2

    if [ -d "$local_dir" ] && [ "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        log_warn "模型已存在: $local_dir，跳过下载"
    else
        log_info "正在下载 $model_name ..."
        # 优先用 huggingface-cli，fallback 到 python -m
        if command -v huggingface-cli &>/dev/null; then
            huggingface-cli download --resume-download "$model_name" --local-dir "$local_dir"
        else
            python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$model_name', local_dir='$local_dir')
"
        fi
        log_info "$model_name 下载完成"
    fi
}

case "$MODEL_BASE" in
    base)
        download_model "Soul-AILab/SoulX-Podcast-1.7B" "pretrained_models/SoulX-Podcast-1.7B"
        ;;
    dialect)
        download_model "Soul-AILab/SoulX-Podcast-1.7B-dialect" "pretrained_models/SoulX-Podcast-1.7B-dialect"
        ;;
    both)
        download_model "Soul-AILab/SoulX-Podcast-1.7B" "pretrained_models/SoulX-Podcast-1.7B"
        download_model "Soul-AILab/SoulX-Podcast-1.7B-dialect" "pretrained_models/SoulX-Podcast-1.7B-dialect"
        ;;
esac

# ------------------------------------------------------------
# 4. 环境验证
# ------------------------------------------------------------
log_info "Step 4/5: 环境验证..."

python3 << 'PYEOF'
import torch
import transformers
import s3tokenizer
import triton

print(f"PyTorch:       {torch.__version__}")
print(f"CUDA:          {torch.version.cuda}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:           {props.name}")
    print(f"VRAM:          {props.total_memory / 1024**3:.1f} GB")
    print(f"bfloat16:      {torch.cuda.is_bf16_supported()}")
else:
    print("GPU:           N/A")
print(f"Transformers:  {transformers.__version__}")
print(f"Triton:        {triton.__version__}")
PYEOF

# 验证模型文件完整性
log_info "验证模型文件..."
for model_dir in pretrained_models/SoulX-Podcast-*; do
    if [ -d "$model_dir" ]; then
        for f in flow.pt hift.pt; do
            if [ -f "$model_dir/$f" ]; then
                log_info "  $model_dir/$f  OK"
            else
                log_error "  $model_dir/$f  缺失!"
            fi
        done
    fi
done

# ------------------------------------------------------------
# 5. 快速测试 / 启动服务
# ------------------------------------------------------------
log_info "Step 5/5: 部署完成!"

echo ""
echo "============================================================"
echo "  SoulX-Podcast 环境部署完成"
echo "============================================================"
echo ""
echo "  项目目录: $WORK_DIR"
echo ""
echo "  快速测试 (CLI):"
echo "    # 播客对话生成"
echo "    bash example/infer_dialogue.sh"
echo ""
echo "    # 单句 TTS"
echo "    bash example/infer_tts.sh"
echo ""
echo "  启动 WebUI:"
echo "    python3 webui.py --model_path pretrained_models/SoulX-Podcast-1.7B"
echo ""
echo "  启动 API 服务:"
echo "    python3 run_api.py --model pretrained_models/SoulX-Podcast-1.7B --port 8000"
echo ""
echo "  方言模型 (四川话/河南话/粤语):"
echo "    python3 webui.py --model_path pretrained_models/SoulX-Podcast-1.7B-dialect"
echo ""
echo "============================================================"

if [ "$ENABLE_API" = true ]; then
    log_info "正在启动 API 服务..."
    python3 run_api.py --model pretrained_models/SoulX-Podcast-1.7B --port 8000
fi
