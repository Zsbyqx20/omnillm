# SPDX-FileCopyrightText: 2024 Zsbyqx20 <112002598+Zsbyqx20@users.noreply.github.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from omnillm.base import (
    BaseClient,
    BaseMessage,
    ContentType,
    ImageDetail,
    ImageFormat,
    Role,
    ServiceType,
)


def test_content_type_enum():
    assert ContentType.TEXT.value == "text"
    assert ContentType.IMAGE.value == "image"


def test_role_enum():
    assert Role.USER.value == "user"
    assert Role.SYSTEM.value == "system"
    assert Role.ASSISTANT.value == "assistant"


def test_image_format_enum():
    assert ImageFormat.PNG.value == "png"
    assert ImageFormat.JPEG.value == "jpeg"
    assert ImageFormat.WEBP.value == "webp"
    assert ImageFormat.GIF.value == "gif"
    assert ImageFormat.UNKNOWN.value == "unknown"


def test_image_detail_enum():
    assert ImageDetail.HIGH.value == "high"
    assert ImageDetail.LOW.value == "low"
    assert ImageDetail.AUTO.value == "auto"


def test_service_type_enum():
    assert ServiceType.OPENAI.value == "openai"
    assert ServiceType.ANTHROPIC.value == "anthropic"


class TestBaseMessage:
    def test_init_default_values(self):
        message = BaseMessage("Hello, world!")
        assert message.content == "Hello, world!"
        assert message.role == Role.USER
        assert message.content_type == ContentType.TEXT

    def test_init_custom_values(self):
        message = BaseMessage(
            content="Test message",
            role=Role.ASSISTANT,
            content_type=ContentType.IMAGE,
        )
        assert message.content == "Test message"
        assert message.role == Role.ASSISTANT
        assert message.content_type == ContentType.IMAGE

    def test_init_with_int_content(self):
        message = BaseMessage(content=42)
        assert message.content == 42
        assert message.role == Role.USER
        assert message.content_type == ContentType.TEXT

    def test_init_with_list_content(self):
        content = ["item1", "item2", "item3"]
        message = BaseMessage(content=content)
        assert message.content == content
        assert isinstance(message.content, list)
        assert message.role == Role.USER
        assert message.content_type == ContentType.TEXT

    def test_init_with_dict_content(self):
        content = {"key": "value", "nested": {"data": 123}}
        message = BaseMessage(content=content)
        assert message.content == content
        assert isinstance(message.content, dict)
        assert message.role == Role.USER
        assert message.content_type == ContentType.TEXT

    def test_init_with_custom_class(self):
        class CustomData:
            def __init__(self, value):
                self.value = value

        custom_obj = CustomData("test")
        message = BaseMessage(content=custom_obj)
        assert message.content == custom_obj
        assert isinstance(message.content, CustomData)
        assert message.role == Role.USER
        assert message.content_type == ContentType.TEXT


class TestBaseClient:
    @pytest.fixture
    def mock_concrete_client(self, mocker):
        """Fixture to create a mock concrete client"""

        class ConcreteClient(BaseClient):
            _init_client = mocker.Mock()
            organize_messages = mocker.Mock(
                return_value=[
                    {"role": "user", "content": {"type": "text", "text": "test"}}
                ]
            )

        return ConcreteClient

    def test_init_with_explicit_values(self, mock_concrete_client):
        client = mock_concrete_client(
            api_key="test_key",
            base_url="https://api.test.com",
            service_type=ServiceType.OPENAI,
        )

        # Verify client initialization
        assert client.api_key == "test_key"
        assert client.base_url == "https://api.test.com"
        assert client.service_type == ServiceType.OPENAI

        # Verify _init_client was called
        client._init_client.assert_called_once()

    def test_init_with_env_vars(self, mock_concrete_client, monkeypatch):
        # Mock environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "env_test_key")

        client = mock_concrete_client()
        assert client.api_key == "env_test_key"
        client._init_client.assert_called_once()

    def test_init_missing_api_key(self, mock_concrete_client, monkeypatch):
        # Remove environment variable if exists
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            mock_concrete_client()

    def test_abstract_methods(self):
        # Verify that BaseClient has the required abstract methods
        abstract_methods = BaseClient.__abstractmethods__
        assert "_init_client" in abstract_methods
        assert "organize_messages" in abstract_methods
