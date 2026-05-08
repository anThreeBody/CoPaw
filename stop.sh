#!/usr/bin/env bash
# 用法：cd /Users/chenmengke/Code/CoPaw && ./stop.sh

BOLD="\033[1m"; GREEN="\033[0;32m"; YELLOW="\033[0;33m"
CYAN="\033[0;36m"; RESET="\033[0m"

ok()   { printf "${GREEN}  ✅ %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}  ⚠️  %s${RESET}\n" "$*"; }

printf "\n${BOLD}🛑 CoPaw 停止脚本${RESET}\n"
printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

# 停止 Ollama
if pgrep -x ollama > /dev/null 2>&1; then
    pkill -x ollama
    ok "Ollama 已停止"
else
    warn "Ollama 未在运行"
fi

printf "\n${BOLD}${CYAN}提示：${RESET} CoPaw 请在启动终端按 Ctrl+C 关闭\n\n"
