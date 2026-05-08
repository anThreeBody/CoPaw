#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════╗
# ║           CoPaw 一键启动脚本                         ║
# ║  用法：cd /Users/chenmengke/Code/CoPaw && ./start.sh ║
# ╚══════════════════════════════════════════════════════╝
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 颜色 ──────────────────────────────────────────────
BOLD="\033[1m"; GREEN="\033[0;32m"; YELLOW="\033[0;33m"
RED="\033[0;31m"; CYAN="\033[0;36m"; RESET="\033[0m"

ok()   { printf "${GREEN}  ✅ %s${RESET}\n" "$*"; }
info() { printf "${CYAN}  ➜  %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}  ⚠️  %s${RESET}\n" "$*"; }
fail() { printf "${RED}  ❌ %s${RESET}\n" "$*"; exit 1; }
step() { printf "\n${BOLD}${CYAN}[%s] %s${RESET}\n" "$1" "$2"; }

printf "\n${BOLD}🚀 CoPaw 启动脚本${RESET}\n"
printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

# ── Step 1: Conda 环境 ────────────────────────────────
step "1/4" "检查 Conda 环境"

CONDA_INIT="/opt/miniconda3/etc/profile.d/conda.sh"
[[ -f "$CONDA_INIT" ]] || fail "未找到 conda: $CONDA_INIT"
source "$CONDA_INIT"
conda activate copaw_local 2>/dev/null || fail "无法激活 conda 环境 copaw_local"
ok "Conda 环境 copaw_local 已激活"

# ── Step 2: 前端构建（console/dist）────────────────────
step "2/4" "检查前端构建"

DIST_DIR="$REPO_DIR/console/dist"
CONSOLE_DIR="$REPO_DIR/console"

# 判断是否需要重新构建：dist 不存在，或 src 比 dist 新
need_build=false
if [[ ! -f "$DIST_DIR/index.html" ]]; then
    warn "console/dist 不存在，需要构建"
    need_build=true
else
    # src 中有比 dist/index.html 更新的文件
    newer=$(find "$CONSOLE_DIR/src" -newer "$DIST_DIR/index.html" -type f 2>/dev/null | head -1)
    if [[ -n "$newer" ]]; then
        warn "前端源码有更新，需要重新构建"
        need_build=true
    fi
fi

if [[ "$need_build" == true ]]; then
    info "执行 npm run build..."
    cd "$CONSOLE_DIR"
    # 如果 node_modules 不存在先安装依赖
    if [[ ! -d "node_modules" ]]; then
        info "安装前端依赖 npm install..."
        npm install --silent || fail "npm install 失败"
    fi
    npm run build --silent || fail "前端构建失败"
    ok "前端构建完成"
    cd "$REPO_DIR"
else
    ok "前端已是最新，跳过构建"
fi

# ── Step 3: Ollama 服务 ───────────────────────────────
step "3/4" "检查 Ollama 服务（Embedding）"

if curl -s http://localhost:11434 --max-time 2 > /dev/null 2>&1; then
    ok "Ollama 已在运行"
else
    info "启动 Ollama 服务..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    for i in $(seq 1 10); do
        sleep 1
        if curl -s http://localhost:11434 --max-time 1 > /dev/null 2>&1; then
            ok "Ollama 启动成功 (PID=$OLLAMA_PID)"
            break
        fi
        [[ $i -eq 10 ]] && fail "Ollama 启动超时，请检查 /tmp/ollama.log"
    done
fi

# 检查 bge-m3-q4 模型
if ollama list 2>/dev/null | grep -q "bge-m3-q4"; then
    ok "bge-m3-q4 模型已就绪"
else
    warn "bge-m3-q4 模型未找到，尝试创建..."
    GGUF_PATH="$HOME/Models/bge-m3/bge-m3-Q4_K_M.gguf"
    [[ -f "$GGUF_PATH" ]] || fail "GGUF 文件不存在: $GGUF_PATH"
    echo "FROM $GGUF_PATH" > /tmp/Modelfile.bge-m3
    ollama create bge-m3-q4 -f /tmp/Modelfile.bge-m3 > /dev/null 2>&1
    ok "bge-m3-q4 模型创建成功"
fi

# ── Step 4: 启动 CoPaw ────────────────────────────────
step "4/4" "启动 CoPaw"

if curl -s http://localhost:8088/health --max-time 2 > /dev/null 2>&1; then
    warn "CoPaw 已在运行于 http://localhost:8088"
    printf "\n${BOLD}${GREEN}✨ 全部服务已就绪！${RESET}\n"
    printf "   CoPaw:  ${CYAN}http://localhost:8088${RESET}\n"
    printf "   Ollama: ${CYAN}http://localhost:11434${RESET}\n\n"
    exit 0
fi

printf "\n${BOLD}${GREEN}✨ 全部就绪，CoPaw 启动中...${RESET}\n"
printf "   CoPaw:  ${CYAN}http://localhost:8088${RESET}\n"
printf "   Ollama: ${CYAN}http://localhost:11434${RESET}\n"
printf "   按 Ctrl+C 停止\n\n"

exec copaw app --host 127.0.0.1 --port 8088
