# SPDX-FileCopyrightText: 2024 Zsbyqx20 <112002598+Zsbyqx20@users.noreply.github.com>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import pytest
from PIL import Image

from omnillm.base import ContentType, ImageDetail, ImageFormat, Role
from omnillm.message import (
    ImageMessage,
    MessageDict,
    TextMessage,
    convert_to_message,
)


class TestTextMessage:
    def test_init_default_values(self):
        message = TextMessage("Hello, world!")
        assert message.content == "Hello, world!"
        assert message.role == Role.USER
        assert message.content_type == ContentType.TEXT

    def test_init_custom_values(self):
        message = TextMessage(
            content="Test message",
            role=Role.ASSISTANT,
        )
        assert message.content == "Test message"
        assert message.role == Role.ASSISTANT
        assert message.content_type == ContentType.TEXT


class TestImageMessage:
    @pytest.fixture
    def sample_image(self):
        img = Image.new("RGB", (100, 100), color="red")
        return img

    def test_init_with_pil_image(self, sample_image):
        message = ImageMessage(sample_image)
        assert isinstance(message._content, Image.Image)
        assert message.role == Role.USER
        assert message.content_type == ContentType.IMAGE
        assert message._format == ImageFormat.PNG
        assert message._detail == ImageDetail.AUTO

    def test_init_with_custom_format_and_detail(self, sample_image):
        message = ImageMessage(
            sample_image, format=ImageFormat.JPEG, detail=ImageDetail.HIGH
        )
        assert message._format == ImageFormat.JPEG
        assert message._detail == ImageDetail.HIGH

    def test_content_property(self, sample_image):
        message = ImageMessage(sample_image)
        content = message.content

        assert isinstance(content, str)
        decoded = base64.b64decode(content)
        img = Image.open(io.BytesIO(decoded))
        assert isinstance(img, Image.Image)

    def test_detail_property(self, sample_image):
        message = ImageMessage(sample_image, detail=ImageDetail.HIGH)
        assert message.detail == ImageDetail.HIGH

    def test_init_with_valid_image_url(self, mocker):
        mock_image = Image.new("RGB", (100, 100), color="blue")
        mocker.patch(
            "omnillm.message.is_valid_image_url",
            return_value=(mock_image, ImageFormat.PNG),
        )

        message = ImageMessage("http://example.com/image.png")
        assert isinstance(message._content, Image.Image)
        assert message._format == ImageFormat.PNG

    def test_init_with_unknown_format_url(self, mocker):
        mock_image = Image.new("RGB", (100, 100), color="blue")
        mocker.patch(
            "omnillm.message.is_valid_image_url",
            return_value=(mock_image, ImageFormat.UNKNOWN),
        )

        message = ImageMessage("http://example.com/image.unknown")
        assert isinstance(message._content, Image.Image)
        assert (
            message._format == ImageFormat.PNG
        )  # Should default to PNG when format is UNKNOWN


class TestConvertDictToMessage:
    @pytest.fixture
    def sample_image(self):
        return Image.new("RGB", (100, 100), color="red")

    def test_convert_to_message_text(self, sample_image):
        msg_dict: MessageDict = {
            "role": "user",
            "type": "text",
            "content": "Hello, world!",
        }

        message = convert_to_message(msg_dict)  # type: ignore

        assert isinstance(message, TextMessage)
        assert message.content == "Hello, world!"
        assert message.role == Role.USER
        assert message.content_type == ContentType.TEXT

    def test_convert_to_message_image(self, sample_image):
        msg_dict: MessageDict = {
            "role": "user",
            "type": "image",
            "content": sample_image,
            "format": "png",
            "detail": "auto",
        }

        message = convert_to_message(msg_dict)  # type: ignore

        assert isinstance(message, ImageMessage)
        assert isinstance(message._content, Image.Image)
        assert message.role == Role.USER
        assert message.content_type == ContentType.IMAGE
        assert message._format == ImageFormat.PNG
        assert message._detail == ImageDetail.AUTO

    def test_convert_str_to_message(self):
        message = convert_to_message("Hello, world!")  # type: ignore

        assert isinstance(message, TextMessage)
        assert message.content == "Hello, world!"
        assert message.role == Role.USER
        assert message.content_type == ContentType.TEXT

    def test_invalid_text_message_format(self):
        msg_dict: MessageDict = {
            "role": "user",
            "type": "text",
            "content": 123,  # type: ignore
        }

        with pytest.raises(ValueError, match="Invalid text message format"):
            convert_to_message(msg_dict)  # type: ignore

    def test_invalid_content_type(self):
        msg_dict = {
            "role": "user",
            "type": "invalid",
            "content": "test",
        }

        with pytest.raises(ValueError, match="'invalid' is not a valid ContentType"):
            convert_to_message(msg_dict)  # type: ignore

    def test_invalid_image_content_type(self):
        msg_dict: MessageDict = {
            "role": "user",
            "type": "image",
            "content": 123,  # type: ignore
            "format": "png",
            "detail": "auto",
        }

        with pytest.raises(
            ValueError, match="Image content must be either string or PIL.Image"
        ):
            convert_to_message(msg_dict)  # type: ignore

    def test_convert_unsupported_type_to_message(self):
        with pytest.raises(ValueError, match="Unsupported message type: <class 'int'>"):
            convert_to_message(123)  # type: ignore
