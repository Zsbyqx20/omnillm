# SPDX-FileCopyrightText: 2024 Zsbyqx20 <112002598+Zsbyqx20@users.noreply.github.com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import pytest
from dotenv import load_dotenv
from PIL import Image

from omnillm.base import BaseMessage, ImageDetail, ImageFormat, Role, ServiceType
from omnillm.message import ImageMessage, TextMessage
from omnillm.openai import OpenAIClient

load_dotenv()


class TestOpenAIClient:
    @pytest.fixture
    def client(self):
        return OpenAIClient(api_key="test_key")

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
        assert client.service_type == ServiceType.OPENAI

    def test_init_client_extended(self, mocker):
        mock_openai = mocker.patch("omnillm.openai.OpenAI")
        mock_async_openai = mocker.patch("omnillm.openai.AsyncOpenAI")

        client = OpenAIClient(api_key="test_key", base_url="https://test.com")
        assert client._client == mock_openai.return_value
        assert client._async_client == mock_async_openai.return_value

        mock_openai.assert_called_with(api_key="test_key", base_url="https://test.com")
        mock_async_openai.assert_called_with(
            api_key="test_key", base_url="https://test.com"
        )

        client = OpenAIClient(api_key="test_key")
        mock_openai.assert_called_with(
            api_key="test_key", base_url=os.getenv("OPENAI_BASE_URL")
        )
        mock_async_openai.assert_called_with(
            api_key="test_key", base_url=os.getenv("OPENAI_BASE_URL")
        )

    def test_create_openai_content_text(self, client):
        content = client._create_openai_content(
            TextMessage("Hello, world!", role=Role.USER)
        )
        assert content == {"type": "text", "text": "Hello, world!"}

    def test_create_openai_content_image(self, client, sample_image, mocker):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )
        content = client._create_openai_content(ImageMessage(sample_image))
        assert content == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,mock_base64", "detail": "auto"},
        }

    def test_create_openai_content_unsupported(self, client):
        with pytest.raises(ValueError, match="Unsupported message type"):
            client._create_openai_content(BaseMessage("content", Role.USER))

    def test_merge_user_messages_empty(self, client):
        result = client._merge_user_messages([])
        assert result == {}

    def test_merge_user_messages_single_text(self, client):
        result = client._merge_user_messages([{"type": "text", "text": "Hello"}])
        assert result == {"content": "Hello", "role": "user"}

    def test_merge_user_messages_multiple_text(self, client):
        result = client._merge_user_messages([
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
        ])
        assert result == {"content": "Hello world", "role": "user"}

    def test_merge_user_messages_single_image(self, client, sample_image, mocker):
        contents = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,mock_base64",
                    "detail": "auto",
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
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "auto",
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
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "auto",
                        },
                    },
                    {"type": "text", "text": "This is an image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "auto",
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
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "auto",
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
        mock_chat = mocker.MagicMock()
        mock_completion = mocker.MagicMock()
        mock_completion.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content="Test response"))
        ]
        mock_completion.usage = mocker.MagicMock()
        mock_chat.completions.create.return_value = mock_completion
        client._client = mocker.MagicMock(chat=mock_chat)

        messages = [TextMessage("Test message", role=Role.USER)]
        result = client.call(
            messages=messages, model="test-model", temperature=0.7, extra_param="test"
        )

        assert result == "Test response"
        mock_chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Test message"}],
            temperature=0.7,
            extra_param="test",
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_async_call_method(self, client, mocker):
        mock_chat = mocker.MagicMock()
        mock_completion = mocker.MagicMock()
        mock_completion.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content="Test response"))
        ]
        mock_completion.usage = mocker.MagicMock()
        mock_chat.completions.create = mocker.AsyncMock(return_value=mock_completion)
        client._async_client = mocker.MagicMock(chat=mock_chat)

        messages = [TextMessage("Test message", role=Role.USER)]
        result = await client.async_call(
            messages=messages, model="test-model", temperature=0.7, extra_param="test"
        )

        assert result == "Test response"
        mock_chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Test message"}],
            temperature=0.7,
            extra_param="test",
        )

    def test_call_method_no_response(self, client, mocker):
        mock_chat = mocker.MagicMock()
        mock_completion = mocker.MagicMock()
        mock_completion.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content=None))
        ]
        mock_completion.usage = mocker.MagicMock()
        mock_chat.completions.create.return_value = mock_completion
        client._client = mocker.MagicMock(chat=mock_chat)

        messages = [TextMessage("Test message", role=Role.USER)]
        with pytest.raises(ValueError, match="No response from OpenAI"):
            client.call(messages=messages)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_async_call_method_no_response(self, client, mocker):
        mock_chat = mocker.MagicMock()
        mock_completion = mocker.MagicMock()
        mock_completion.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content=None))
        ]
        mock_completion.usage = mocker.MagicMock()
        mock_chat.completions.create = mocker.AsyncMock(return_value=mock_completion)
        client._async_client = mocker.MagicMock(chat=mock_chat)

        messages = [TextMessage("Test message", role=Role.USER)]
        with pytest.raises(ValueError, match="No response from OpenAI"):
            await client.async_call(messages=messages)

    def test_call_method_with_callback(self, client, mocker):
        mock_chat = mocker.MagicMock()
        mock_completion = mocker.MagicMock()
        mock_completion.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content="Test response"))
        ]
        mock_completion.usage = mocker.MagicMock()
        mock_chat.completions.create.return_value = mock_completion
        client._client = mocker.MagicMock(chat=mock_chat)

        def custom_callback(response: str) -> list[str]:
            return response.split()

        messages = [TextMessage("Test message", role=Role.USER)]
        result = client.call(messages=messages, callback=custom_callback)

        assert result == ["Test", "response"]
        mock_chat.completions.create.assert_called_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_async_call_method_with_callback(self, client, mocker):
        mock_chat = mocker.MagicMock()
        mock_completion = mocker.MagicMock()
        mock_completion.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content="Test response"))
        ]
        mock_completion.usage = mocker.MagicMock()
        mock_chat.completions.create = mocker.AsyncMock(return_value=mock_completion)
        client._async_client = mocker.MagicMock(chat=mock_chat)

        def custom_callback(response: str) -> list[str]:
            return response.split()

        messages = [TextMessage("Test message", role=Role.USER)]
        result = await client.async_call(messages=messages, callback=custom_callback)

        assert result == ["Test", "response"]
        mock_chat.completions.create.assert_called_once()

    def test_merge_user_messages_empty_contents(self, client):
        """Test _merge_user_messages with empty contents list."""
        result = client._merge_user_messages([])
        assert result == {}

    def test_organize_messages_multiple_images(self, client, sample_image, mocker):
        # Mock base64 encoding for multiple images
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )

        messages = [
            ImageMessage(sample_image),
            ImageMessage(
                sample_image, format=ImageFormat.JPEG, detail=ImageDetail.HIGH
            ),
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "auto",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,mock_base64",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

    def test_organize_messages_text_image_alternating(
        self, client, sample_image, mocker
    ):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )

        messages = [
            TextMessage("First text", role=Role.USER),
            ImageMessage(sample_image),
            TextMessage("Response", role=Role.ASSISTANT),
            TextMessage("Second text", role=Role.USER),
            ImageMessage(sample_image, format=ImageFormat.JPEG),
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First text"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "auto",
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "Response"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Second text"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,mock_base64",
                            "detail": "auto",
                        },
                    },
                ],
            },
        ]

    def test_organize_messages_image_with_different_formats(
        self, client, sample_image, mocker
    ):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )

        messages = [
            ImageMessage(sample_image, format=ImageFormat.PNG, detail=ImageDetail.LOW),
            ImageMessage(
                sample_image, format=ImageFormat.JPEG, detail=ImageDetail.HIGH
            ),
            ImageMessage(
                sample_image, format=ImageFormat.WEBP, detail=ImageDetail.AUTO
            ),
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,mock_base64",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/webp;base64,mock_base64",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]

    def test_organize_messages_system_with_images(self, client, sample_image, mocker):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )

        messages = [
            TextMessage("System prompt", role=Role.SYSTEM),
            ImageMessage(sample_image),
            TextMessage("Description", role=Role.USER),
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {"role": "system", "content": "System prompt"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "auto",
                        },
                    },
                    {"type": "text", "text": "Description"},
                ],
            },
        ]

    def test_organize_messages_with_raw_inputs(self, client, sample_image, mocker):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )

        messages = [
            "Hello",  # raw string
            sample_image,  # raw PIL Image
            TextMessage("Response", role=Role.ASSISTANT),
            "Follow-up question",  # raw string
            sample_image,  # raw PIL Image
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "auto",
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "Response"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Follow-up question"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "auto",
                        },
                    },
                ],
            },
        ]

    def test_organize_messages_mixed_inputs(self, client, sample_image, mocker):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )

        # Set format on the image for testing format detection
        sample_image.format = "JPEG"

        messages = [
            {
                "role": "system",
                "type": "text",
                "content": "System prompt",
            },  # dict input
            sample_image,  # raw PIL Image with JPEG format
            "Describe this image",  # raw string
            {  # dict input for image
                "role": "user",
                "type": "image",
                "content": sample_image,
                "format": "webp",
                "detail": "high",
            },
            TextMessage("What do you think?", role=Role.USER),  # TextMessage object
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {"role": "system", "content": "System prompt"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,mock_base64",
                            "detail": "auto",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/webp;base64,mock_base64",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "What do you think?"},
                ],
            },
        ]

    def test_organize_messages_with_dict_and_raw_inputs(
        self, client, sample_image, mocker
    ):
        mocker.patch(
            "omnillm.message.ImageMessage.content",
            property(lambda self: "mock_base64"),
        )

        messages = [
            {  # dict input for text
                "role": "user",
                "type": "text",
                "content": "First message",
            },
            sample_image,  # raw PIL Image
            {  # dict input for image
                "role": "user",
                "type": "image",
                "content": sample_image,
                "format": "jpeg",
                "detail": "low",
            },
            "Final question",  # raw string
        ]

        organized = client.organize_messages(messages)

        assert organized == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First message"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,mock_base64",
                            "detail": "auto",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,mock_base64",
                            "detail": "low",
                        },
                    },
                    {"type": "text", "text": "Final question"},
                ],
            },
        ]
