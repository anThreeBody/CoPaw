# -*- coding: utf-8 -*-
"""OpenClaw LLM gateway provider.

This provider wraps Baidu's internal OneAPI gateway (oneapi-comate.baidu-int.com),
which exposes two sub-endpoints:

  * /v1/chat/completions  – OpenAI-compatible format, requires a custom
    ``comate_custom_header`` JSON header carrying the caller's username.
  * /v1/messages          – Anthropic Messages format, standard Anthropic
    headers (x-api-key, anthropic-version).

Both endpoints share the same base URL and API key.  The provider uses
``OpenAIProvider`` for the OpenAI-format models and an embedded
``_OpenClawAnthropicMixin`` for the Anthropic-format Claude models, selecting
the right client based on the ``chat_model`` field.

Usage (two built-in provider instances):

    PROVIDER_OPENCLAW_OPENAI    – chat_model="OpenAIChatModel"
    PROVIDER_OPENCLAW_ANTHROPIC – chat_model="AnthropicChatModel"
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, List

from agentscope.model import ChatModelBase

import anthropic
from openai import APIError, AsyncOpenAI

from qwenpaw.providers.multimodal_prober import (
    ProbeResult,
    _PROBE_IMAGE_B64,
    _IMAGE_PROBE_PROMPT,
    _is_media_keyword_error,
    evaluate_image_probe_answer,
)
from qwenpaw.providers.provider import ModelInfo, Provider

logger = logging.getLogger(__name__)

OPENCLAW_BASE_URL = "https://oneapi-comate.baidu-int.com/v1"

# Default username injected into comate_custom_header.
# Users can override via generate_kwargs: {"_openclaw_username": "yourname"}
_DEFAULT_USERNAME = ""


class OpenClawOpenAIProvider(Provider):
    """OpenClaw gateway provider using OpenAI-compatible Chat Completions format.

    Adds the mandatory ``comate_custom_header`` to every request.
    The username is read from ``generate_kwargs["_openclaw_username"]``
    (provider-level or model-level); falls back to ``_DEFAULT_USERNAME``.

    Private generate_kwargs key ``_openclaw_username`` is stripped before
    passing kwargs to the model instance so it never leaks to the API.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _username(self) -> str:
        return str(
            self.generate_kwargs.get("_openclaw_username", _DEFAULT_USERNAME),
        )

    def _comate_header(self, username: str | None = None) -> str:
        return json.dumps(
            {
                "username": username or self._username(),
                "source": "hermes",
            },
            ensure_ascii=False,
        )

    def _openai_client(self, timeout: float = 5) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={"comate_custom_header": self._comate_header()},
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        """Check connectivity by listing /models."""
        try:
            await self._openai_client(timeout).models.list(timeout=timeout)
            return True, ""
        except APIError:
            return False, f"API error when connecting to `{self.base_url}`"
        except Exception:
            return (
                False,
                f"Unknown exception when connecting to `{self.base_url}`",
            )

    async def fetch_models(self, timeout: float = 5) -> List[ModelInfo]:
        """Discover models via /models endpoint (may not be available)."""
        try:
            payload = await self._openai_client(timeout).models.list(
                timeout=timeout,
            )
            rows = getattr(payload, "data", []) or []
            seen: set[str] = set()
            models: List[ModelInfo] = []
            for row in rows:
                mid = str(getattr(row, "id", "") or "").strip()
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                mname = str(getattr(row, "name", "") or mid).strip() or mid
                models.append(ModelInfo(id=mid, name=mname))
            return models
        except Exception:
            return []

    async def check_model_connection(
        self,
        model_id: str,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        """Send a minimal streaming request to validate the model."""
        model_id = (model_id or "").strip()
        if not model_id:
            return False, "Empty model ID"
        try:
            res = await self._openai_client(timeout).chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                stream=True,
                timeout=timeout,
            )
            async for _ in res:
                break
            return True, ""
        except APIError:
            return False, f"API error when connecting to model '{model_id}'"
        except Exception:
            return (
                False,
                f"Unknown exception when connecting to model '{model_id}'",
            )

    def get_chat_model_instance(self, model_id: str) -> ChatModelBase:
        from agentscope.credential._openai import OpenAICredential
        from agentscope.model import OpenAIChatModel

        from .openai_chat_model_compat import OpenAIChatModelCompat

        # Strip private keys before forwarding to the model instance.
        gkw = {
            k: v
            for k, v in self.get_effective_generate_kwargs(model_id).items()
            if not k.startswith("_openclaw_")
        }

        credential = OpenAICredential(
            id=f"qwenpaw-{self.id}",
            api_key=self.api_key,
            base_url=self.base_url,
        )
        parameters = OpenAIChatModel.Parameters(
            max_tokens=gkw.pop("max_tokens", None),
            temperature=gkw.pop("temperature", None),
            top_p=gkw.pop("top_p", None),
        )

        return OpenAIChatModelCompat(
            credential=credential,
            model=model_id,
            parameters=parameters,
            stream=True,
            default_headers={
                "comate_custom_header": self._comate_header(),
            },
            extra_generate_kwargs=gkw or None,
        )

    async def probe_model_multimodal(
        self,
        model_id: str,
        timeout: float = 10,
        image_only: bool = False,
    ) -> ProbeResult:
        """Probe multimodal capability via OpenAI-compatible image_url format."""
        img_ok, img_msg = await self._probe_image_support(model_id, timeout)
        if not img_ok:
            return ProbeResult(
                supports_image=False,
                supports_video=False,
                image_message=img_msg,
                video_message="Skipped: image probe failed",
            )
        if image_only:
            return ProbeResult(
                supports_image=img_ok,
                supports_video=False,
                image_message=img_msg,
                video_message="Skipped: image_only=True",
            )
        return ProbeResult(
            supports_image=img_ok,
            supports_video=False,
            image_message=img_msg,
            video_message="Video probe skipped for OpenClaw",
        )

    async def _probe_image_support(
        self,
        model_id: str,
        timeout: float = 15,
    ) -> tuple[bool, str]:
        logger.info(
            "Image probe start: model=%s url=%s",
            model_id,
            self.base_url,
        )
        start_time = time.monotonic()
        client = self._openai_client(timeout)
        try:
            res = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_PROBE_IMAGE_B64}",
                                },
                            },
                            {"type": "text", "text": _IMAGE_PROBE_PROMPT},
                        ],
                    },
                ],
                max_tokens=200,
                timeout=timeout,
            )
            answer = (res.choices[0].message.content or "").lower().strip()
            return evaluate_image_probe_answer(answer, model_id, start_time)
        except APIError as e:
            elapsed = time.monotonic() - start_time
            logger.warning(
                "Image probe error: model=%s type=%s msg=%s %.2fs",
                model_id,
                type(e).__name__,
                e,
                elapsed,
            )
            status = getattr(e, "status_code", None)
            if status == 400 or _is_media_keyword_error(e):
                return False, f"Image not supported: {e}"
            return False, f"Probe inconclusive: {e}"
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.warning(
                "Image probe error: model=%s type=%s msg=%s %.2fs",
                model_id,
                type(e).__name__,
                e,
                elapsed,
            )
            return False, f"Probe failed: {e}"


class OpenClawAnthropicProvider(Provider):
    """OpenClaw gateway provider using Anthropic Messages format.

    Targets the /v1/messages endpoint for Claude-series models.
    Uses ``x-api-key`` / ``anthropic-version`` headers (standard Anthropic).
    No ``comate_custom_header`` is needed for this endpoint.

    Note: The Anthropic SDK appends ``/v1`` to the base_url internally.
    OpenClaw's base_url already contains ``/v1``, so we strip it before
    passing to any Anthropic SDK client to avoid ``/v1/v1/messages``.
    """

    def _anthropic_base_url(self) -> str:
        """Strip trailing /v1 so the Anthropic SDK doesn't double-add it."""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base

    def _client(self, timeout: float = 5) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key=self.api_key,
            base_url=self._anthropic_base_url(),
            timeout=timeout,
        )

    @staticmethod
    def _normalize_models_payload(payload: Any) -> List[ModelInfo]:
        if isinstance(payload, dict):
            rows = payload.get("data", [])
        else:
            rows = getattr(payload, "data", payload)

        models: List[ModelInfo] = []
        seen: set[str] = set()
        for row in rows or []:
            mid = str(getattr(row, "id", "") or "").strip()
            mname = str(getattr(row, "display_name", "") or mid).strip() or mid
            if not mid or mid in seen:
                continue
            seen.add(mid)
            models.append(ModelInfo(id=mid, name=mname))
        return models

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        try:
            await self._client(timeout).models.list()
            return True, ""
        except anthropic.APIError:
            return False, "Anthropic API error"
        except Exception:
            return (
                False,
                f"Unknown exception when connecting to `{self.base_url}`",
            )

    async def fetch_models(self, timeout: float = 5) -> List[ModelInfo]:
        try:
            payload = await self._client(timeout).models.list()
            return self._normalize_models_payload(payload)
        except Exception:
            return []

    async def check_model_connection(
        self,
        model_id: str,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        target = (model_id or "").strip()
        if not target:
            return False, "Empty model ID"
        try:
            client = self._client(timeout)
            resp = await client.messages.create(
                model=target,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
                stream=True,
            )
            async for _ in resp:
                break
            return True, ""
        except anthropic.APIError:
            return False, f"Model '{model_id}' is not reachable or usable"
        except Exception:
            return (
                False,
                f"Unknown exception when connecting to model '{model_id}'",
            )

    def get_chat_model_instance(self, model_id: str) -> ChatModelBase:
        from agentscope.credential import AnthropicCredential
        from agentscope.formatter import AnthropicChatFormatter
        from agentscope.model import AnthropicChatModel

        base_url = self._anthropic_base_url()
        gkw = self.get_effective_generate_kwargs(model_id)

        max_tokens = gkw.pop("max_tokens", 16384)
        params_kwargs: dict[str, Any] = {"max_tokens": max_tokens}
        for key in ("thinking_enable", "thinking_budget"):
            if key in gkw:
                params_kwargs[key] = gkw.pop(key)

        class _BedrockCleaningFormatter(AnthropicChatFormatter):
            """Formatter that normalises messages before sending to Bedrock.

            Two transformations are applied to the formatted message list:

            1. **Strip thinking blocks** – When OpenClaw routes to AWS
               Bedrock, Bedrock may auto-enable extended thinking for Claude
               models.  In multi-turn conversations the returned ``thinking``
               content blocks must be passed back with their original
               ``signature``.  However the signature is frequently absent or
               empty in the stored history, causing Bedrock to return::

                   ValidationException:
                       messages.N.content.0.thinking.signature: Field required

               Since we never explicitly request extended thinking, stripping
               thinking blocks from *history* messages before the API call is
               safe and avoids the validation error.

            2. **Convert role="tool" to Anthropic tool_result format** – Some
               code paths produce OpenAI-style messages with ``role: "tool"``.
               Bedrock only allows ``"user"`` or ``"assistant"`` roles, so
               these are converted to the Anthropic ``role: "user"`` +
               ``type: "tool_result"`` format.
            """

            async def format(self, msgs: list) -> list[dict]:
                formatted = await super().format(msgs)
                formatted = _strip_thinking_from_history(formatted)
                return _normalize_tool_messages_for_bedrock(formatted)

        credential = AnthropicCredential(
            api_key=self.api_key or "",
            base_url=base_url,
        )

        return AnthropicChatModel(
            credential=credential,
            model=model_id,
            parameters=AnthropicChatModel.Parameters(**params_kwargs),
            stream=True,
            formatter=_BedrockCleaningFormatter(),
        )

    async def probe_model_multimodal(
        self,
        model_id: str,
        timeout: float = 10,
        image_only: bool = False,  # pylint: disable=unused-argument
    ) -> ProbeResult:
        """Probe image support via Anthropic Messages format.
        Video is not supported by Anthropic protocol."""
        img_ok, img_msg = await self._probe_image_support(model_id, timeout)
        return ProbeResult(
            supports_image=img_ok,
            supports_video=False,
            image_message=img_msg,
            video_message="Video not supported by Anthropic protocol",
        )

    async def _probe_image_support(
        self,
        model_id: str,
        timeout: float = 10,
    ) -> tuple[bool, str]:
        logger.info(
            "Image probe start: model=%s url=%s",
            model_id,
            self.base_url,
        )
        start_time = time.monotonic()
        client = self._client(timeout)
        try:
            resp = await client.messages.create(
                model=model_id,
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _PROBE_IMAGE_B64,
                                },
                            },
                            {"type": "text", "text": _IMAGE_PROBE_PROMPT},
                        ],
                    },
                ],
            )
            answer = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    answer += block.text
            return evaluate_image_probe_answer(answer, model_id, start_time)
        except anthropic.APIError as e:
            elapsed = time.monotonic() - start_time
            logger.warning(
                "Image probe error: model=%s type=%s msg=%s %.2fs",
                model_id,
                type(e).__name__,
                e,
                elapsed,
            )
            status = getattr(e, "status_code", None)
            if status == 400 or _is_media_keyword_error(e):
                return False, f"Image not supported: {e}"
            return False, f"Probe inconclusive: {e}"
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.warning(
                "Image probe error: model=%s type=%s msg=%s %.2fs",
                model_id,
                type(e).__name__,
                e,
                elapsed,
            )
            return False, f"Probe failed: {e}"


def _strip_thinking_from_history(
    messages: list[dict],
) -> list[dict]:
    """Remove ``thinking`` content blocks from all assistant messages.

    Bedrock requires that thinking blocks in history carry a valid
    ``signature``.  Since we don't explicitly enable extended thinking,
    the presence of thinking blocks in history is unexpected, and stripping
    them avoids the Bedrock ``ValidationException``.

    Args:
        messages: Formatted message dicts ready to send to the API.

    Returns:
        A new list where every assistant message has its thinking blocks
        removed.  Messages without thinking blocks are returned as-is
        (same object, not copied).
    """
    result = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, list):
                filtered = [
                    b
                    for b in content
                    if not (
                        isinstance(b, dict) and b.get("type") == "thinking"
                    )
                ]
                if len(filtered) != len(content):
                    msg = {**msg, "content": filtered or None}
        result.append(msg)
    return result


def _normalize_tool_messages_for_bedrock(
    messages: list[dict],
) -> list[dict]:
    """Convert OpenAI-style ``role="tool"`` messages to Anthropic format.

    Bedrock only allows ``"user"`` or ``"assistant"`` roles.  Some code paths
    in model_factory produce messages with ``role: "tool"`` (OpenAI tool-result
    format).  This function converts them to the Anthropic equivalent:
    ``role: "user"`` with a ``type: "tool_result"`` content block.

    Args:
        messages: Formatted message dicts ready to send to the API.

    Returns:
        A new list where every ``role="tool"`` message has been converted.
        Other messages are returned as-is (same object, not copied).
    """
    result = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            tool_call_id = msg.get("tool_call_id", "")
            if isinstance(content, list):
                tool_content = content
            else:
                tool_content = [{"type": "text", "text": str(content)}]
            msg = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": tool_content,
                    },
                ],
            }
        result.append(msg)
    return result
