# SPDX-FileCopyrightText: 2024 Zsbyqx20 <112002598+Zsbyqx20@users.noreply.github.com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import pytest
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_use_block import ToolUseBlock
from dotenv import load_dotenv
from PIL import Image

from omnillm.anthropic import AnthropicClient, process_content_block_default
from omnillm.base import BaseMessage, Role, ServiceType
from omnillm.message import ImageMessage, TextMessage

load_dotenv()


class TestProcessContentBlock:
    def test_process_text_block(self):
        block = TextBlock(text="Hello, world!", type="text")
        result = process_content_block_default(block)
        assert result == "Hello, world!"

    def test_process_tool_use_block(self):
        block = ToolUseBlock(id="1", input="", name="test", type="tool_use")
        with pytest.raises(
            NotImplementedError, match="Tool use blocks are not supported"
        ):
            process_content_block_default(block)

    def test_process_unsupported_block(self):
        class UnsupportedBlock:
            def __init__(self):
                self.type = "unsupported"

        with pytest.raises(ValueError, match="Unsupported content block type"):
            process_content_block_default(UnsupportedBlock())  # type: ignore


class TestAnthropicClient:
    @pytest.fixture
    def client(self):
        return AnthropicClient(api_key="test_key")

    @pytest.fixture
    def event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
        loop.close()

    @pytest.fixture
    def sample_image(self):
        return Image.new("RGB", (100, 100), color="red")

    def test_service_type(self, client):
        assert client.service_type == ServiceType.ANTHROPIC

    def test_init_client_extended(self, mocker):
        mock_anthropic = mocker.patch("omnillm.anthropic.Anthropic")
        mock_async_anthropic = mocker.patch("omnillm.anthropic.AsyncAnthropic")

        client = AnthropicClient(api_key="test_key", base_url="https://test.com")
        assert client._client == mock_anthropic.return_value
        assert client._async_client == mock_async_anthropic.return_value

        mock_anthropic.assert_called_with(
            api_key="test_key", base_url="https://test.com"
        )
        mock_async_anthropic.assert_called_with(
            api_key="test_key", base_url="https://test.com"
        )

        client = AnthropicClient(api_key="test_key")
        mock_anthropic.assert_called_with(
            api_key="test_key", base_url=os.getenv("ANTHROPIC_BASE_URL")
        )
        mock_async_anthropic.assert_called_with(
            api_key="test_key", base_url=os.getenv("ANTHROPIC_BASE_URL")
        )

    def test_create_anthropic_content_text(self, client):
        msg = TextMessage("Hello, world!", role=Role.USER)
        content = client._create_anthropic_content(msg)

        assert content == {"type": "text", "text": "Hello, world!"}

    def test_create_anthropic_content_image(self, client, sample_image, mocker):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )
        msg = ImageMessage(sample_image)
        content = client._create_anthropic_content(msg)

        assert content == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "mock_base64",
            },
        }

    def test_create_anthropic_content_unsupported(self, client):
        class UnsupportedMessage(BaseMessage):
            def __init__(self):
                super().__init__("content", Role.USER)

        with pytest.raises(ValueError, match="Unsupported message type"):
            client._create_anthropic_content(UnsupportedMessage())

    def test_merge_user_messages_empty(self, client):
        result = client._merge_user_messages([])
        assert result == {}

    def test_merge_user_messages_single_text(self, client):
        contents = [{"type": "text", "text": "Hello"}]
        result = client._merge_user_messages(contents)

        assert result == {"content": "Hello", "role": "user"}

    def test_merge_user_messages_multiple_text(self, client):
        contents = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
        ]
        result = client._merge_user_messages(contents)

        assert result == {"content": "Hello world", "role": "user"}

    def test_merge_user_messages_single_image(self, client):
        contents = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "mock_data",
                },
            }
        ]
        result = client._merge_user_messages(contents)

        assert result == {"content": [contents[0]], "role": "user"}

    def test_organize_messages_single_text(self, client):
        messages = [TextMessage("Hello, world!", role=Role.USER)]

        organized = client.organize_messages(messages)

        assert organized == [{"role": "user", "content": "Hello, world!"}]

    def test_organize_messages_mixed_content(self, client, sample_image, mocker):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )

        messages = [
            TextMessage("Hello", role=Role.USER),
            ImageMessage(sample_image),
            TextMessage("How are you?", role=Role.ASSISTANT),
            TextMessage("I'm good!", role=Role.USER),
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "mock_base64",
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "How are you?"},
            {"role": "user", "content": "I'm good!"},
        ]

    def test_organize_messages_with_dict_input(self, client):
        messages = [
            {"role": "user", "type": "text", "content": "Hello"},
            {"role": "assistant", "type": "text", "content": "Hi there!"},
            {"role": "user", "type": "text", "content": "How are you?"},
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

    def test_messages_ending_with_assistant(self, client):
        messages = [
            TextMessage("Hello", role=Role.USER),
            TextMessage("Hi there!", role=Role.ASSISTANT),
        ]

        with pytest.raises(
            ValueError, match="Message sequence cannot end with an assistant message"
        ):
            client.organize_messages(messages)

    def test_multiple_system_messages(self, client):
        messages = [
            TextMessage("System1", role=Role.SYSTEM),
            TextMessage("Hello", role=Role.USER),
            TextMessage("System2", role=Role.SYSTEM),
        ]

        with pytest.raises(
            ValueError, match="Multiple system messages are not allowed"
        ):
            client.organize_messages(messages)

    def test_system_message_not_first(self, client):
        messages = [
            TextMessage("Hello", role=Role.USER),
            TextMessage("System", role=Role.SYSTEM),
        ]

        with pytest.raises(
            ValueError, match="System message must be the first message if present"
        ):
            client.organize_messages(messages)

    def test_multiple_user_text_messages(self, client):
        messages = [
            TextMessage("Hello", role=Role.USER),
            TextMessage("How are you?", role=Role.USER),
            TextMessage("Nice day!", role=Role.USER),
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {
                "role": "user",
                "content": "Hello How are you? Nice day!",
            }
        ]

    def test_mixed_user_text_and_image_messages(self, client, sample_image, mocker):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )

        messages = [
            TextMessage("Hello", role=Role.USER),
            ImageMessage(sample_image),
            TextMessage("This is an image", role=Role.USER),
            ImageMessage(sample_image),
            TextMessage("Another text", role=Role.USER),
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "mock_base64",
                        },
                    },
                    {"type": "text", "text": "This is an image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "mock_base64",
                        },
                    },
                    {"type": "text", "text": "Another text"},
                ],
            }
        ]

    def test_unsupported_message_type(self, client):
        class UnsupportedMessage(BaseMessage):
            def __init__(self):
                super().__init__("content", Role.USER)

        messages = [UnsupportedMessage()]

        with pytest.raises(ValueError, match="Unsupported message type"):
            client.organize_messages(messages)

    def test_organize_messages_single_image(self, client, sample_image, mocker):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )

        messages = [ImageMessage(sample_image)]

        organized = client.organize_messages(messages)

        assert organized == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "mock_base64",
                        },
                    }
                ],
            }
        ]

    def test_organize_messages_single_text_content(self, client):
        messages = [
            TextMessage("Hello", role=Role.USER),
            TextMessage("World", role=Role.USER),
            TextMessage("Test", role=Role.ASSISTANT),
            TextMessage("Single text", role=Role.USER),
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {"role": "user", "content": "Hello World"},
            {"role": "assistant", "content": "Test"},
            {"role": "user", "content": "Single text"},
        ]

    def test_call_method(self, client, mocker):
        mock_messages = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.content = [TextBlock(text="Test response", type="text")]
        mock_response.usage = mocker.MagicMock()
        mock_messages.create.return_value = mock_response
        client._client = mocker.MagicMock(messages=mock_messages)

        messages = [TextMessage("Test message", role=Role.USER)]
        result = client.call(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            temperature=0.7,
            extra_param="test",
        )

        assert result == "Test response"
        mock_messages.create.assert_called_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_async_call_method(self, client, mocker):
        mock_messages = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.content = [TextBlock(text="Test response", type="text")]
        mock_response.usage = mocker.MagicMock()
        mock_messages.create = mocker.AsyncMock(return_value=mock_response)
        client._async_client = mocker.MagicMock(messages=mock_messages)

        messages = [TextMessage("Test message", role=Role.USER)]
        result = await client.async_call(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            temperature=0.7,
            extra_param="test",
        )

        assert result == "Test response"
        mock_messages.create.assert_called_once()

    def test_call_method_no_response(self, client, mocker):
        mock_messages = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.content = []
        mock_messages.create.return_value = mock_response
        client._client = mocker.MagicMock(messages=mock_messages)

        messages = [TextMessage("Test message", role=Role.USER)]
        with pytest.raises(ValueError, match="No response from Anthropic"):
            client.call(messages=messages)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_async_call_method_no_response(self, client, mocker):
        mock_messages = mocker.MagicMock()
        mock_messages.create = mocker.AsyncMock(return_value=None)
        client._async_client = mocker.MagicMock(messages=mock_messages)

    def test_prepare_request_no_system_message(self, client):
        messages = [TextMessage("Hello", role=Role.USER)]
        prompt, params = client._prepare_request(
            messages=messages, model="test-model", temperature=0.7
        )

        assert prompt == [{"role": "user", "content": "Hello"}]
        assert params == {
            "model": "test-model",
            "messages": prompt,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        assert "system" not in params

    def test_prepare_request_with_system_message(self, client):
        messages = [
            TextMessage("System prompt", role=Role.SYSTEM),
            TextMessage("User message", role=Role.USER),
        ]
        prompt, params = client._prepare_request(
            messages=messages, model="test-model", temperature=0.7
        )

        assert prompt == [{"role": "user", "content": "User message"}]
        assert params == {
            "model": "test-model",
            "messages": prompt,
            "temperature": 0.7,
            "max_tokens": 4096,
            "system": "System prompt",
        }

    def test_prepare_request_with_additional_params(self, client):
        messages = [TextMessage("Hello", role=Role.USER)]
        prompt, params = client._prepare_request(
            messages=messages,
            model="test-model",
            temperature=0.7,
            extra_param="test",
            another_param=123,
        )

        assert prompt == [{"role": "user", "content": "Hello"}]
        assert params == {
            "model": "test-model",
            "messages": prompt,
            "temperature": 0.7,
            "max_tokens": 4096,
            "extra_param": "test",
            "another_param": 123,
        }
