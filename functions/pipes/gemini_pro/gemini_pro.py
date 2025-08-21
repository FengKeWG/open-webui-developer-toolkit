"""
title: OpenAI-Compatible Anti-Truncation (Manifold)
author: WindGuest
version: 0.2.0
license: MIT
requirements:
  - aiohttp>=3.8.0
description: 为 OpenAI 兼容 /v1/chat/completions 供应商提供“防截断+自动续写”，以 manifold 形式导出模型清单，可在模型管理中直接选择使用。
"""

import asyncio
import json
from typing import AsyncIterator, Dict, List, Optional, Tuple, Callable, Awaitable

import aiohttp
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        # 上游对接
        API_BASE_URL: str = Field(
            default="https://api.xxx.top/v1", description="不含 /chat/completions"
        )
        API_KEY: str = Field(default="", description="供应商 Token（不含 Bearer）")
        API_KEY_HEADER: str = Field(
            default="Authorization",
            description="鉴权头名，如 Authorization / X-API-Key",
        )
        API_KEY_PREFIX: str = Field(
            default="Bearer ", description="鉴权前缀（含尾部空格），如需关闭请置空"
        )

        # 模型清单（manifold）
        MODEL_IDS: str = Field(
            default="gemini-2.5-pro-thinking", description="以分号 ; 分隔，如 a;b;c"
        )
        NAME_PREFIX: str = Field(default="JUHEAI/", description="展示名前缀")

        # 防截断策略
        MAX_CONSECUTIVE_RETRIES: int = Field(
            default=6, description="最大自动续写/重试次数"
        )
        RETRY_DELAY_MS: int = Field(default=750, description="重试间隔（毫秒）")
        INJECT_END_MARK: bool = Field(default=True, description="注入结尾标记提示")
        END_MARK: str = Field(default="[done]", description="强制结尾标记")
        SWALLOW_REASONING: bool = Field(
            default=True, description="忽略 delta.reasoning（适配 thinking 模型）"
        )
        TIMEOUT_SECONDS: int = Field(default=120, description="单次请求超时（秒）")
        DEBUG_MODE: bool = Field(default=False, description="调试日志")

    def __init__(self):
        # manifold：以“模型清单”的方式出现在下拉选择里
        self.type = "manifold"
        self.id = "openai_compat_anti_truncation_manifold"
        self.name = "OpenAI-Compatible Anti-Truncation"
        self.valves = self.Valves()

    def pipes(self) -> List[dict]:
        models = [
            m.strip() for m in (self.valves.MODEL_IDS or "").split(";") if m.strip()
        ]
        prefix = self.valves.NAME_PREFIX or ""
        out = []
        for m in models:
            out.append(
                {
                    "id": m,  # 纯上游ID → 传给上游
                    "name": f"{prefix}{m}",  # 带前缀的展示名 → 显示在UI
                }
            )
        return out or [
            {"id": "configure_me", "name": "Configure MODEL_IDS in Function"}
        ]

    def _log_debug(self, *args):
        if self.valves.DEBUG_MODE:
            print("[AT][DEBUG]", *args)

    def _log_error(self, *args):
        print("[AT][ERROR]", *args)

    async def pipe(
        self,
        body: Dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> AsyncIterator[str] | str:

        if not self.valves.API_KEY:
            return "Error: API_KEY 未配置"

        streaming_requested = bool(body.get("stream", True))
        end_mark = self.valves.END_MARK

        # 将 UI 选择到的模型名还原为上游 id（去掉可选前缀）
        upstream_model = self._resolve_upstream_model(body.get("model", ""))

        # 注入“以 [done] 结束”的 system
        messages = self._messages_with_end_mark(
            body.get("messages", []), end_mark, self.valves.INJECT_END_MARK
        )

        # 透传常用生成参数（模型管理里设的会进来）
        gen_cfg = {}
        for k in (
            "temperature",
            "top_p",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
        ):
            if k in body:
                gen_cfg[k] = body[k]

        async def emit(level: str, msg: str, done: bool = False):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "complete" if done else "in_progress",
                            "level": level,
                            "description": msg,
                            "done": done,
                        },
                    }
                )

        await emit("info", "连接上游中 ...", False)

        accumulated_text = ""
        retry_count = 0

        async def run_once(
            messages_payload: List[dict],
        ) -> Tuple[Optional[AsyncIterator[str]], Dict, Optional[str]]:
            url = f"{self.valves.API_BASE_URL.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            if self.valves.API_KEY_HEADER:
                headers[self.valves.API_KEY_HEADER] = (
                    f"{self.valves.API_KEY_PREFIX}{self.valves.API_KEY}".strip()
                )

            payload: Dict = {
                "model": upstream_model,
                "messages": messages_payload,
                "stream": True,
                **gen_cfg,
            }

            try:
                timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS)
                session = aiohttp.ClientSession(timeout=timeout)
                resp = await session.post(url, headers=headers, json=payload)
            except Exception as e:
                return None, {}, f"请求失败: {e}"

            if resp.status // 100 != 2:
                err = await resp.text()
                await session.close()
                return None, {}, f"HTTP {resp.status}: {err[:500]}"

            meta = {"finish_reason": None, "blocked": False, "drop": False}

            async def iterator() -> AsyncIterator[str]:
                nonlocal accumulated_text
                buf = b""
                got = False
                reasoning_state = -1  # -1: not started, 0: in reasoning, 1: answered
                try:
                    async for chunk in resp.content.iter_chunked(2048):
                        if not chunk:
                            continue
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            s = line.decode("utf-8", "ignore").strip()
                            if not s or not s.startswith("data:"):
                                continue
                            data_text = s[5:].strip()
                            if data_text == "[DONE]":
                                continue
                            if data_text.startswith('{"error"'):
                                meta["blocked"] = True
                                return
                            try:
                                obj = json.loads(data_text)
                            except Exception:
                                continue
                            choices = obj.get("choices") or []
                            if not choices:
                                continue
                            ch = choices[0]
                            if ch.get("finish_reason"):
                                meta["finish_reason"] = ch["finish_reason"]
                                # 若在思考中直接结束，补发关闭标签
                                if (
                                    reasoning_state == 0
                                    and not self.valves.SWALLOW_REASONING
                                ):
                                    yield "</think>\n\n"
                                    reasoning_state = 1

                            delta = ch.get("delta") or {}
                            # 角色字段忽略
                            if "role" in delta:
                                pass
                            # 思考过程（reasoning）按需输出（不计入 accumulated_text）
                            reasoning_text = self._extract_reasoning_text(
                                delta.get("reasoning")
                            )
                            if reasoning_text and not self.valves.SWALLOW_REASONING:
                                if reasoning_state == -1:
                                    reasoning_state = 0
                                    yield "<think>"
                                yield reasoning_text
                            # 正式输出
                            text = delta.get("content")
                            if text:
                                if (
                                    reasoning_state == 0
                                    and not self.valves.SWALLOW_REASONING
                                ):
                                    # 从思考切换到正文时，关闭思考块
                                    yield "</think>\n\n"
                                    reasoning_state = 1
                                got = True
                                yield text
                                accumulated_text += text
                finally:
                    if not meta["finish_reason"] and got:
                        meta["drop"] = True
                    # 若流意外结束仍在思考态，补一个关闭标签
                    if reasoning_state == 0 and not self.valves.SWALLOW_REASONING:
                        try:
                            yield "</think>\n\n"
                        except Exception:
                            pass
                    try:
                        await resp.release()
                    except Exception:
                        pass
                    try:
                        await session.close()
                    except Exception:
                        pass

            return iterator(), meta, None

        async def stream_driver() -> AsyncIterator[str]:
            nonlocal retry_count, accumulated_text
            base_msgs = list(messages)

            while True:
                it, meta, err = await run_once(base_msgs)
                if err:
                    await emit("error", f"上游错误：{err}", True)
                    yield f"\n[proxy error] {err}\n"
                    return
                if it is None:
                    await emit("error", "无流迭代器", True)
                    yield "\n[proxy error] 无流迭代器\n"
                    return

                async for piece in it:
                    yield piece

                trimmed = accumulated_text.strip()
                finish_reason = meta.get("finish_reason")
                need_retry = False

                if meta.get("blocked") or meta.get("drop"):
                    need_retry = True
                elif finish_reason in (None, "content_filter", "length"):
                    need_retry = True
                elif finish_reason == "stop":
                    if self.valves.END_MARK and not trimmed.endswith(
                        self.valves.END_MARK
                    ):
                        need_retry = True

                if not need_retry:
                    await emit("info", "完成", True)
                    return

                if retry_count >= self.valves.MAX_CONSECUTIVE_RETRIES:
                    await emit(
                        "error",
                        f"重试次数超限（{self.valves.MAX_CONSECUTIVE_RETRIES}）",
                        True,
                    )
                    yield f"\n[proxy error] 重试次数超限（{self.valves.MAX_CONSECUTIVE_RETRIES}），终止。\n"
                    return

                retry_count += 1
                await emit(
                    "warning",
                    f"防截断拼接第 {retry_count} 次 ...（如无需拼接，可点击停止按钮）",
                    False,
                )
                base_msgs = self._build_retry_messages(
                    messages, accumulated_text, self.valves.END_MARK
                )
                await asyncio.sleep(self.valves.RETRY_DELAY_MS / 1000.0)

        if streaming_requested:
            return stream_driver()
        else:
            # 非流式聚合
            result = await self._aggregate_non_stream(messages, upstream_model, gen_cfg)
            await emit("info", "完成", True)
            return result

    # ---------- 非流式聚合 ----------
    async def _aggregate_non_stream(
        self, messages: List[dict], model: str, gen_cfg: Dict
    ) -> str:
        acc = ""
        retry = 0
        while True:
            text, finish, err = await self._call_non_stream_once(
                messages, model, gen_cfg
            )
            if err:
                return f"[proxy error] {err}"
            acc += text or ""
            trim = acc.strip()
            need = False
            if finish in (None, "content_filter", "length"):
                need = True
            elif (
                finish == "stop"
                and self.valves.END_MARK
                and not trim.endswith(self.valves.END_MARK)
            ):
                need = True
            if not need:
                return acc
            if retry >= self.valves.MAX_CONSECUTIVE_RETRIES:
                return (
                    acc
                    + f"\n[proxy error] 重试次数超限（{self.valves.MAX_CONSECUTIVE_RETRIES}）。"
                )
            retry += 1
            messages = self._build_retry_messages(messages, acc, self.valves.END_MARK)
            await asyncio.sleep(self.valves.RETRY_DELAY_MS / 1000.0)

    async def _call_non_stream_once(
        self, messages: List[dict], model: str, gen_cfg: Dict
    ) -> Tuple[str, Optional[str], Optional[str]]:
        url = f"{self.valves.API_BASE_URL.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.valves.API_KEY_HEADER:
            headers[self.valves.API_KEY_HEADER] = (
                f"{self.valves.API_KEY_PREFIX}{self.valves.API_KEY}".strip()
            )
        payload = {"model": model, "messages": messages, "stream": False, **gen_cfg}
        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status // 100 != 2:
                        return (
                            "",
                            None,
                            f"HTTP {resp.status}: {(await resp.text())[:500]}",
                        )
                    data = await resp.json()
        except Exception as e:
            return "", None, f"请求失败: {e}"
        try:
            ch = (data.get("choices") or [])[0]
            msg = ch.get("message") or {}
            return msg.get("content") or "", ch.get("finish_reason"), None
        except Exception as e:
            return "", None, f"解析失败: {e}"

    # ---------- 工具 ----------
    def _resolve_upstream_model(self, model_field: str) -> str:
        # 1) 去掉 Pipe 前缀（例如 gemini_pro.）
        if "." in model_field:
            model_field = model_field.split(".", 1)[1]
        # 2) 去掉用于展示的可选名称前缀（例如 JUHEAI/）
        prefix = getattr(self.valves, "NAME_PREFIX", "") or ""
        if prefix and model_field.startswith(prefix):
            model_field = model_field[len(prefix) :]
        return model_field

    def _messages_with_end_mark(
        self, messages: List[dict], end_mark: str, inject: bool
    ) -> List[dict]:
        msgs = list(messages or [])
        if inject and end_mark:
            inject_line = f"Your message must end with {end_mark} to signify the end of your output."
            msgs.insert(0, {"role": "system", "content": inject_line})
        return msgs

    def _build_retry_messages(
        self, base_messages: List[dict], accumulated_text: str, end_mark: str
    ) -> List[dict]:
        retry_msgs = list(base_messages)
        retry_msgs.append({"role": "assistant", "content": accumulated_text})
        retry_msgs.append(
            {
                "role": "user",
                "content": f"Continue exactly where you left off without any preamble or repetition. End your message with {end_mark}.",
            }
        )
        return retry_msgs

    def _extract_reasoning_text(self, reasoning) -> str:
        """
        从供应商的 delta.reasoning 结构中尽量提取可展示文本。
        常见格式：
        - 字符串
        - 对象：{"content": "..."}
        - 对象：{"tokens": [{"content": "..."}, ...]}
        """
        if not reasoning:
            return ""
        if isinstance(reasoning, dict):
            content = reasoning.get("content")
            if isinstance(content, str):
                return content
            tokens = reasoning.get("tokens")
            if isinstance(tokens, list):
                return "".join(
                    t.get("content", "") for t in tokens if isinstance(t, dict)
                )
        if isinstance(reasoning, str):
            return reasoning
        return ""
