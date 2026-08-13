"""调用 OpenAI Responses 兼容的外部 API（api.aixoras.com）。

和本地 vLLM 的区别：没有 guided_choice / guided_json，也拿不到 logprobs，
所以标签和风险分都要模型自己以 JSON 形式输出，客户端再做校验。

Token 从环境变量读，不要写进代码：
    export NEWAPI_TOKEN=...
"""

import json
import os
import random
import time

import requests

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ResponsesClient:
    def __init__(self, base_url, model, token=None, timeout=180, max_retries=5):
        self.url = base_url.rstrip("/") + "/responses"
        self.model = model
        self.token = token or os.environ.get("NEWAPI_TOKEN")
        if not self.token:
            raise SystemExit("缺少 token，先 export NEWAPI_TOKEN=...")
        self.timeout = timeout
        self.max_retries = max_retries
        # 文档没写图片怎么传，两种常见写法都试一遍，成功后记住
        self.image_style = None

    # ------------------------------------------------------------------
    def _image_part(self, b64, style):
        url = f"data:image/jpeg;base64,{b64}"
        if style == "str":
            return {"type": "input_image", "image_url": url}
        return {"type": "input_image", "image_url": {"url": url}}

    def _post(self, payload):
        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    self.url, json=payload, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.token}",
                             "Content-Type": "application/json"})
            except requests.RequestException as e:
                last = e
                time.sleep(min(2 ** attempt, 30) + random.random())
                continue
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in RETRY_STATUS:
                # 429 按文档是限流，退避后重试
                time.sleep(min(2 ** attempt, 30) + random.random())
                continue
            raise RuntimeError(last)
        raise RuntimeError(f"重试 {self.max_retries} 次仍失败：{last}")

    # ------------------------------------------------------------------
    @staticmethod
    def _text_of(resp):
        """从 Responses 结构里取出文本，兼容 output_text 快捷字段。"""
        if isinstance(resp.get("output_text"), str) and resp["output_text"]:
            return resp["output_text"]
        parts = []
        for item in resp.get("output") or []:
            for c in item.get("content") or []:
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    parts.append(c["text"])
        return "".join(parts)

    def call(self, text, b64=None, instructions=None,
             max_output_tokens=1024, json_mode=False, temperature=0.0):
        content = [{"type": "input_text", "text": text}]
        styles = [self.image_style] if self.image_style else ["str", "obj"]
        if b64 is None:
            styles = [None]

        last = None
        for style in styles:
            parts = list(content)
            if b64 is not None:
                parts.append(self._image_part(b64, style))
            payload = {
                "model": self.model,
                "input": [{"role": "user", "content": parts}],
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "stream": False,
            }
            if instructions:
                payload["instructions"] = instructions
            if json_mode:
                payload["text"] = {"format": {"type": "json_object"}}
            try:
                resp = self._post(payload)
            except RuntimeError as e:
                last = e
                continue
            if b64 is not None:
                self.image_style = style
            return self._text_of(resp)
        raise RuntimeError(f"图片两种传法都失败：{last}")


class ChatClient:
    """OpenAI /v1/chat/completions 兼容服务，用于 llama-server。

    和 ResponsesClient 暴露同样的 call() 接口，所以 predict_api 两边通用。
    结构化输出走 response_format 的 json_schema，llama.cpp 会翻译成 GBNF 语法，
    保证返回的 JSON 一定符合 schema，不用再写容错解析。
    """

    def __init__(self, base_url, model, token=None, timeout=600, max_retries=3,
                 enable_thinking=False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.token = token or os.environ.get("NEWAPI_TOKEN") or "EMPTY"
        self.timeout = timeout
        self.max_retries = max_retries
        # Qwen3.5/3.6 默认开思考，会把 token 全烧在 reasoning_content 里，
        # content 返回空字符串。抽客观特征不需要推理，默认关掉。
        self.enable_thinking = enable_thinking

    def call(self, text, b64=None, instructions=None, schema=None,
             max_output_tokens=1024, json_mode=False, temperature=0.0):
        content = [{"type": "text", "text": text}]
        if b64 is not None:
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"}})
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": content})

        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_output_tokens,
                   "chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema, "strict": True}}
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}

        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    self.url, json=payload, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.token}",
                             "Content-Type": "application/json"})
            except requests.RequestException as e:
                last = e
                time.sleep(min(2 ** attempt, 20) + random.random())
                continue
            if r.status_code == 200:
                ch = r.json()["choices"][0]
                msg = ch.get("message") or {}
                content = (msg.get("content") or "").strip()
                if content:
                    return content
                # content 空但有推理内容，说明 token 全烧在思考上了
                think = (msg.get("reasoning_content") or "").strip()
                raise RuntimeError(
                    f"回复的 content 为空，finish_reason={ch.get('finish_reason')}，"
                    f"reasoning_content 长度 {len(think)}。"
                    "多半是模型在思考模式下把 max_tokens 用完了——"
                    "确认 enable_thinking=False 已生效，或者把 max_tokens 调大。"
                    + (f" 推理片段：{think[:200]}" if think else ""))
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in RETRY_STATUS:
                time.sleep(min(2 ** attempt, 20) + random.random())
                continue
            raise RuntimeError(last)
        raise RuntimeError(f"重试 {self.max_retries} 次仍失败：{last}")


def loads_lenient(text):
    """模型偶尔会在 JSON 外面裹一层 ```json 或者说明文字，这里兜一下。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i:j + 1])
    raise ValueError(f"无法解析成 JSON: {text[:200]}")
