import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import requests
from PIL import Image

from omnillm.base import ImageFormat
from omnillm.utils import async_retry, is_valid_image_url, retry


class TestRetryDecorator:
    def test_successful_execution(self):
        mock_func = MagicMock()
        mock_func.return_value = "success"
        decorated_func = retry()(mock_func)

        result = decorated_func()

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.timeout(1)
    def test_retry_on_failure(self):
        mock_func = MagicMock()
        mock_func.side_effect = [ValueError(), ValueError(), "success"]
        decorated_func = retry(max_attempts=3, delay=0)(mock_func)

        result = decorated_func()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_max_attempts_exceeded(self):
        mock_func = MagicMock()
        mock_func.side_effect = ValueError("test error")
        decorated_func = retry(max_attempts=3, delay=0)(mock_func)

        with pytest.raises(ValueError, match="test error"):
            decorated_func()

        assert mock_func.call_count == 3

    def test_skip_exceptions(self):
        mock_func = MagicMock()
        mock_func.side_effect = KeyError("skip this")
        decorated_func = retry(skip_exceptions=(KeyError,))(mock_func)

        with pytest.raises(KeyError, match="skip this"):
            decorated_func()

        assert mock_func.call_count == 1


class TestAsyncRetryDecorator:
    @pytest.fixture
    def event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
        loop.close()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_successful_execution(self):
        mock_func = MagicMock()
        mock_func.return_value = "success"
        async_mock = AsyncMock(return_value="success")
        decorated_func = async_retry()(async_mock)

        result = await decorated_func()

        assert result == "success"
        assert async_mock.call_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.timeout(1)
    async def test_retry_on_failure(self):
        async_mock = AsyncMock(side_effect=[ValueError(), ValueError(), "success"])
        decorated_func = async_retry(max_attempts=3, delay=0)(async_mock)

        result = await decorated_func()

        assert result == "success"
        assert async_mock.call_count == 3

    @pytest.mark.asyncio(loop_scope="session")
    async def test_max_attempts_exceeded(self):
        async_mock = AsyncMock(side_effect=ValueError("test error"))
        decorated_func = async_retry(max_attempts=3, delay=0)(async_mock)

        with pytest.raises(ValueError, match="test error"):
            await decorated_func()

        assert async_mock.call_count == 3

    @pytest.mark.asyncio(loop_scope="session")
    async def test_skip_exceptions(self):
        async_mock = AsyncMock(side_effect=KeyError("skip this"))
        decorated_func = async_retry(skip_exceptions=(KeyError,))(async_mock)

        with pytest.raises(KeyError, match="skip this"):
            await decorated_func()

        assert async_mock.call_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_backoff_delay(self):
        async_mock = AsyncMock(side_effect=[ValueError(), ValueError(), "success"])
        sleep_mock = AsyncMock()

        with patch("asyncio.sleep", sleep_mock):
            decorated_func = async_retry(max_attempts=3, delay=1.0, backoff_factor=2.0)(
                async_mock
            )

            await decorated_func()

        assert sleep_mock.call_count == 2
        sleep_mock.assert_has_calls([call(1.0), call(2.0)])


class TestIsValidImageUrl:
    @pytest.fixture(autouse=True)
    def setup_timeouts(self):
        self.request_timeout = 0.1
        return self.request_timeout

    @pytest.fixture
    def mock_response(self):
        mock = MagicMock()
        mock.headers = {"content-type": "image/jpeg"}
        return mock

    @pytest.fixture
    def sample_image(self):
        return Image.new("RGB", (100, 100), color="red")

    def test_valid_image_url(self, mock_response, sample_image):
        with patch("requests.head") as mock_head, patch(
            "requests.get"
        ) as mock_get, patch("PIL.Image.open") as mock_open:
            mock_head.return_value = mock_response
            mock_get.return_value.content = b"fake_image_data"

            mock_image = MagicMock()
            mock_image.format = "JPEG"
            mock_image.copy.return_value = sample_image
            mock_open.return_value = mock_image

            image, format = is_valid_image_url("http://example.com/image.jpg")

            assert isinstance(image, Image.Image)
            assert format == ImageFormat.JPEG

    def test_invalid_content_type(self, mock_response):
        mock_response.headers = {"content-type": "text/plain"}

        with patch("requests.head") as mock_head:
            mock_head.return_value = mock_response

            with pytest.raises(ValueError, match="Not an image"):
                is_valid_image_url("http://example.com/not_image.txt")

    def test_connection_error(self):
        with patch("requests.head", side_effect=requests.ConnectionError()):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                is_valid_image_url("http://invalid-url.com/image.jpg")

    def test_timeout_error(self):
        with patch("requests.head", side_effect=requests.Timeout()):
            with pytest.raises(TimeoutError, match="Request timed out"):
                is_valid_image_url("http://example.com/image.jpg")

    def test_unknown_image_format(self, mock_response):
        with patch("requests.head") as mock_head, patch(
            "requests.get"
        ) as mock_get, patch("PIL.Image.open") as mock_open:
            mock_head.return_value = mock_response
            mock_get.return_value.content = b"fake_image_data"

            mock_image = MagicMock()
            mock_image.format = "UNKNOWN_FORMAT"
            mock_image.copy.return_value = Image.new("RGB", (100, 100))
            mock_open.return_value = mock_image

            image, format = is_valid_image_url("http://example.com/image.jpg")

            assert isinstance(image, Image.Image)
            assert format == ImageFormat.UNKNOWN

    def test_invalid_image_data(self, mock_response):
        with patch("requests.head") as mock_head, patch("requests.get") as mock_get:
            mock_head.return_value = mock_response
            mock_get.return_value.content = b"invalid_image_data"
            mock_head.timeout = self.request_timeout
            mock_get.timeout = self.request_timeout

            with pytest.raises(ValueError, match="Invalid image data"):
                is_valid_image_url("http://example.com/invalid.jpg")

    def test_request_exception(self, mock_response):
        with patch("requests.head") as mock_head:
            mock_head.timeout = self.request_timeout
            mock_head.side_effect = requests.RequestException("General request error")

            with pytest.raises(
                RuntimeError, match="Request error: General request error"
            ):
                is_valid_image_url("http://example.com/image.jpg")
