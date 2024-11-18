# SPDX-FileCopyrightText: 2024 Zsbyqx20 <112002598+Zsbyqx20@users.noreply.github.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Sequence, TypeVar, cast

from anthropic import Anthropic, AsyncAnthropic
from anthropic._exceptions import AuthenticationError, PermissionDeniedError
from anthropic.types.content_block import ContentBlock
from anthropic.types.message import Message
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_use_block import ToolUseBlock

from .base import (
    BaseClient,
    BaseMessage,
    MessageInput,
    Role,
    ServiceType,
)
from .message import ImageMessage, TextMessage, convert_to_message
from .utils import async_retry, retry

T = TypeVar("T")


def process_content_block_default(block: ContentBlock) -> str:
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ToolUseBlock):
        raise NotImplementedError("Tool use blocks are not supported")
    raise ValueError(f"Unsupported content block type: {type(block)}")


class AnthropicClient(BaseClient):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        super().__init__(api_key, base_url, ServiceType.ANTHROPIC)

    def _init_client(self):
        self._client = Anthropic(api_key=self.api_key, base_url=self.base_url)
        self._async_client = AsyncAnthropic(
            api_key=self.api_key, base_url=self.base_url
        )

    def _create_anthropic_content(self, msg: BaseMessage) -> dict[str, Any]:
        if isinstance(msg, TextMessage):
            return {"type": "text", "text": msg.content}
        elif isinstance(msg, ImageMessage):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{msg._format.value}",
                    "data": msg.content,
                },
            }
        raise ValueError(f"Unsupported message type: {type(msg)}")

    def _merge_user_messages(self, contents: list[dict[str, Any]]) -> dict[str, Any]:
        if not contents:
            return {}

        if all(item["type"] == "text" for item in contents):
            return {
                "role": "user",
                "content": " ".join(item["text"] for item in contents),
            }

        if len(contents) == 1:
            content = contents[0]
            if content["type"] == "text":
                return {"role": "user", "content": content["text"]}
            return {"role": "user", "content": [content]}

        return {"role": "user", "content": contents.copy()}

    def organize_messages(
        self, messages: Sequence[MessageInput]
    ) -> list[dict[str, Any]]:
        processed_messages = [
            msg if isinstance(msg, BaseMessage) else convert_to_message(msg)
            for msg in messages
        ]

        self._validate_message_sequence(processed_messages)

        organized_msgs: list[dict[str, Any]] = []
        current_user_content: list[dict[str, Any]] = []

        def flush_user_content():
            if current_user_content:
                organized_msgs.append(self._merge_user_messages(current_user_content))
                current_user_content.clear()

        for msg in processed_messages:
            if msg.role == Role.USER:
                current_user_content.append(self._create_anthropic_content(msg))
            else:
                flush_user_content()
                organized_msgs.append({
                    "role": msg.role.value,
                    "content": msg.content
                    if isinstance(msg, TextMessage)
                    else str(msg.content),
                })

        flush_user_content()
        return organized_msgs

    def _process_response(self, response: Message) -> tuple[ContentBlock, Any]:
        content = response.content
        if not content:
            raise ValueError("No response from Anthropic")

        return content[0], response.usage

    def _prepare_request(
        self,
        messages: list[MessageInput],
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.5,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = self.organize_messages(messages)
        if len(prompt) > 1 and prompt[0]["role"] == "system":
            system_msg = prompt[0]
            prompt = prompt[1:]
        else:
            system_msg = None
        params = {
            "model": model,
            "messages": prompt,
            "temperature": temperature,
            "max_tokens": 4096,
            **kwargs,
        }
        if system_msg:
            params["system"] = system_msg["content"]
        return prompt, params

    @retry(max_attempts=3, skip_exceptions=(AuthenticationError, PermissionDeniedError))
    def call(
        self,
        messages: list[MessageInput],
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.5,
        callback: Callable[[ContentBlock], T] | None = None,
        **kwargs: Any,
    ):
        _, params = self._prepare_request(messages, model, temperature, **kwargs)

        response = cast(Message, self._client.messages.create(**params))
        resp_obj, _ = self._process_response(response)
        if callback is not None:
            return callback(resp_obj)
        return process_content_block_default(resp_obj)

    @async_retry(
        max_attempts=3, skip_exceptions=(AuthenticationError, PermissionDeniedError)
    )
    async def async_call(
        self,
        messages: list[MessageInput],
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.5,
        callback: Callable[[ContentBlock], T] | None = None,
        **kwargs: Any,
    ):
        _, params = self._prepare_request(messages, model, temperature, **kwargs)

        response = cast(Message, await self._async_client.messages.create(**params))
        resp_obj, _ = self._process_response(response)
        if callback is not None:
            return callback(resp_obj)
        return process_content_block_default(resp_obj)
