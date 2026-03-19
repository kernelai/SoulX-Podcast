#!/bin/bash
# ============================================================
# SoulX-Podcast RunPod 部署脚本（支持 Network Volume 持久化）
# 适用镜像: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# GPU: RTX 4090 (24GB VRAM)
#
# Network Volume 挂载到 /workspace，venv 和模型存储在其中，
# Pod 删除后再重建无需重新安装依赖和下载模型。
# ============================================================

set -e

WORK_DIR="/workspace/SoulX-Podcast"
VENV_DIR="/workspace/venv"
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

# 持久化缓存到 Network Volume（s3tokenizer ONNX 模型、HuggingFace 等）
export XDG_CACHE_HOME="/workspace/.cache"

# ------------------------------------------------------------
# 2. 创建 venv + 安装依赖（已存在则跳过）
# ------------------------------------------------------------
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    log_info "Step 2/5: 检测到已有 venv，跳过安装"
    source "$VENV_DIR/bin/activate"

    # 快速验证关键包是否可用
    if python3 -c "import torch; import transformers; import s3tokenizer" 2>/dev/null; then
        log_info "venv 验证通过，关键依赖完整"
    else
        log_warn "venv 中部分依赖缺失，重新安装..."
        deactivate 2>/dev/null || true
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    log_info "Step 2/5: 创建 venv + 安装依赖..."

    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    log_info "当前 PyTorch 版本: $TORCH_VERSION"

    # 升级 PyTorch 到项目要求的 2.7.1
    # torch 2.7.1 仅提供 cu126 和 cu128 的 wheel
    log_info "安装 PyTorch 2.7.1 (cu126) ..."
    pip install --no-cache-dir \
        torch==2.7.1 \
        torchaudio==2.7.1 \
        torchvision==0.22.1 \
        --index-url "https://download.pytorch.org/whl/cu126"

    NEW_TORCH=$(python3 -c "import torch; print(torch.__version__)")
    log_info "PyTorch 安装完成: $NEW_TORCH"

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

    # huggingface-cli（模型下载用）+ hf_transfer（RunPod 默认启用高速下载）
    pip install --no-cache-dir 'huggingface_hub[cli]>=0.34.0,<1.0' hf_transfer

    log_info "依赖安装完成（venv 持久化在 $VENV_DIR）"
fi

# ------------------------------------------------------------
# 3. 下载模型
# ------------------------------------------------------------
log_info "Step 3/5: 下载模型..."

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

    # 检查关键模型文件是否存在（而非仅判断目录非空）
    if [ -f "$local_dir/flow.pt" ] && [ -f "$local_dir/hift.pt" ]; then
        log_warn "模型已存在: $local_dir，跳过下载"
    else
        if [ -d "$local_dir" ]; then
            log_warn "模型目录不完整，重新下载: $local_dir"
        fi
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
# 5. 完成
# ------------------------------------------------------------
log_info "Step 5/5: 部署完成!"

echo ""
echo "============================================================"
echo "  SoulX-Podcast 环境部署完成"
echo "============================================================"
echo ""
echo "  项目目录: $WORK_DIR"
echo "  venv 目录: $VENV_DIR"
echo ""
echo "  后续启动 Pod 时只需执行:"
echo "    source /workspace/venv/bin/activate"
echo "    cd /workspace/SoulX-Podcast"
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
