# SPDX-FileCopyrightText: 2024 Zsbyqx20 <112002598+Zsbyqx20@users.noreply.github.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar, Union

from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T")


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"


class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class ImageFormat(Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    GIF = "gif"
    UNKNOWN = "unknown"


class ImageDetail(Enum):
    HIGH = "high"
    LOW = "low"
    AUTO = "auto"


class ServiceType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


MessageInput = Union[str, dict[str, Any], "BaseMessage"]


class BaseMessage(Generic[T]):
    def __init__(
        self,
        content: T,
        role: Role = Role.USER,
        content_type: ContentType = ContentType.TEXT,
    ):
        self._role = role
        self._content = content
        self._content_type = content_type

    @property
    def role(self):
        return self._role

    @property
    def content(self):
        return self._content

    @property
    def content_type(self):
        return self._content_type


class BaseClient(ABC):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        service_type: ServiceType = ServiceType.OPENAI,
    ):
        self.service_type = service_type
        self.api_key = api_key or os.getenv(f"{service_type.value.upper()}_API_KEY")
        self.base_url = base_url or os.getenv(f"{service_type.value.upper()}_BASE_URL")
        if self.api_key is None:
            raise ValueError(f"{service_type.value.upper()}_API_KEY is not set")
        self._init_client()

    @abstractmethod
    def _init_client(self):
        raise NotImplementedError

    @abstractmethod
    def organize_messages(self, messages: list[BaseMessage]):
        raise NotImplementedError

    def _validate_message_sequence(self, messages: list[BaseMessage]):
        system_indices = [
            i for i, msg in enumerate(messages) if msg.role == Role.SYSTEM
        ]
        if len(system_indices) > 1:
            raise ValueError("Multiple system messages are not allowed")
        if system_indices and system_indices[0] != 0:
            raise ValueError("System message must be the first message if present")
        if messages and messages[-1].role == Role.ASSISTANT:
            raise ValueError("Message sequence cannot end with an assistant message")
