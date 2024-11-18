# SPDX-FileCopyrightText: 2024 Zsbyqx20 <112002598+Zsbyqx20@users.noreply.github.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Sequence, TypeVar, cast

from openai import AsyncOpenAI, AuthenticationError, OpenAI, PermissionDeniedError
from openai.types.chat import ChatCompletion

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


class OpenAIClient(BaseClient):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        super().__init__(api_key, base_url, ServiceType.OPENAI)

    def _init_client(self):
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _create_openai_content(self, msg: BaseMessage) -> dict[str, Any]:
        if isinstance(msg, TextMessage):
            return {"type": "text", "text": msg.content}
        elif isinstance(msg, ImageMessage):
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{msg._format.value};base64,{msg.content}",
                    "detail": msg.detail.value,
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
                current_user_content.append(self._create_openai_content(msg))
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

    def _process_response(self, response: ChatCompletion) -> tuple[str, Any]:
        resp_obj = response.choices[0].message.content
        resp_usage = response.usage

        if resp_obj is None:
            raise ValueError("No response from OpenAI")
        if resp_usage is None:
            raise ValueError("No usage data from OpenAI")

        return resp_obj, resp_usage

    def _prepare_request(
        self,
        messages: list[MessageInput],
        model: str,
        temperature: float,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = self.organize_messages(messages)
        params = {
            "model": model,
            "messages": prompt,  # type: ignore
            "temperature": temperature,
            **kwargs,
        }
        return prompt, params

    @retry(max_attempts=3, skip_exceptions=(AuthenticationError, PermissionDeniedError))
    def call(
        self,
        messages: list[MessageInput],
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        callback: Callable[[str], T] | None = None,
        **kwargs: Any,
    ):
        _, params = self._prepare_request(messages, model, temperature, **kwargs)

        response = cast(
            ChatCompletion,
            self._client.chat.completions.create(**params),
        )
        resp_obj, _ = self._process_response(response)
        if callback is not None:
            return callback(resp_obj)
        return resp_obj

    @async_retry(
        max_attempts=3, skip_exceptions=(AuthenticationError, PermissionDeniedError)
    )
    async def async_call(
        self,
        messages: list[MessageInput],
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        callback: Callable[[str], T] | None = None,
        **kwargs: Any,
    ):
        _, params = self._prepare_request(messages, model, temperature, **kwargs)

        response = await self._async_client.chat.completions.create(**params)
        resp_obj, _ = self._process_response(response)
        if callback is not None:
            return callback(resp_obj)
        return resp_obj
