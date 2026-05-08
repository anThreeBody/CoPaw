# -*- coding: utf-8 -*-
"""
OpenClaw LLM API Client
支持两种 provider:
  - oneapi: OpenAI Completions 格式 (GLM, DeepSeek, ERNIE, Kimi, MiniMax 等)
  - oneapi-claude: Anthropic Messages 格式 (Claude 系列)
"""

import os
import json
import requests

BASE_URL = "https://oneapi-comate.baidu-int.com/v1"
API_KEY = "sk-36zSDm3aFxYH9LOv247e6030411c49DaB5Ce272400927083"
SANDBOX_USERNAME = "chenmengke"


# ─────────────────────────────────────────────
# Provider 1: OpenAI Completions 格式
# ─────────────────────────────────────────────
def chat_openai(
    messages: list[dict],
    model: str = "AUTO",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stream: bool = False,
) -> str:
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "comate_custom_header": json.dumps(
            {
                "username": SANDBOX_USERNAME,
                "source": "hermes",
            },
        ),
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    if stream:
        with requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            full_text = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    print(delta, end="", flush=True)
                    full_text += delta
            print()
            return full_text
    else:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────
# Provider 2: Anthropic Messages 格式
# ─────────────────────────────────────────────
def chat_claude(
    messages: list[dict],
    model: str = "Claude Sonnet 4.5",
    max_tokens: int = 1024,
    system: str | None = None,
) -> str:
    url = f"{BASE_URL}/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
    }
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system:
        payload["system"] = system

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    return result["content"][0]["text"]


def test_model(name: str, fn, *args, **kwargs):
    """运行单个模型测试，捕获异常并打印结果。"""
    try:
        reply = fn(*args, **kwargs)
        status = "OK"
        preview = reply[:80].replace("\n", " ")
        print(f"  [{status}] {name}\n       {preview}\n")
    except Exception as e:
        print(f"  [FAIL] {name}\n       {e}\n")


if __name__ == "__main__":
    prompt = "用一句话介绍你自己。"
    messages = [{"role": "user", "content": prompt}]

    # ── OpenAI 格式模型 ──────────────────────────
    openai_models = [
        "AUTO",
        "glm-5-openclaw",
        "deepseek-v3.1",
        "deepseek-v3.2",
        "ERNIE-5.0",
        "glm-4.7-internal",
        "kimi-k2.5",
        "MiniMax-M2.5",
        "MiniMax-M2.5-internal",
        "Kimi-K2.5-internal",
        "GLM-5",
        "glm-5.1",
    ]

    print("=" * 60)
    print("Provider: oneapi (OpenAI Completions 格式)")
    print("=" * 60)
    for m in openai_models:
        test_model(m, chat_openai, messages, model=m)

    # ── Anthropic 格式模型 ───────────────────────
    claude_models = [
        "Claude Sonnet 4.6",
        "Claude Sonnet 4.5",
        "Claude Haiku 4.5",
    ]

    print("=" * 60)
    print("Provider: oneapi-claude (Anthropic Messages 格式)")
    print("=" * 60)
    for m in claude_models:
        test_model(m, chat_claude, messages, model=m)
