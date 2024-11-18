# SPDX-FileCopyrightText: 2024 Zsbyqx20 <112002598+Zsbyqx20@users.noreply.github.com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import time
from functools import wraps
from typing import Awaitable, Callable, ParamSpec, TypeVar

import requests
from PIL import Image

from .base import ImageFormat

RT = TypeVar("RT")
P = ParamSpec("P")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    skip_exceptions: tuple[type[Exception], ...] = (),
) -> Callable[[Callable[P, RT]], Callable[P, RT]]:
    def decorator(func: Callable[P, RT]) -> Callable[P, RT]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> RT:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if isinstance(e, skip_exceptions):
                        raise
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            raise last_exception  # type: ignore

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    skip_exceptions: tuple[type[Exception], ...] = (),
) -> Callable[[Callable[P, Awaitable[RT]]], Callable[P, Awaitable[RT]]]:
    def decorator(func: Callable[P, Awaitable[RT]]) -> Callable[P, Awaitable[RT]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> RT:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if isinstance(e, skip_exceptions):
                        raise
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

            raise last_exception  # type: ignore

        return wrapper

    return decorator


@retry(max_attempts=3)
def is_valid_image_url(url: str, timeout: int = 5) -> tuple[Image.Image, ImageFormat]:
    try:
        response = requests.head(url, timeout=timeout)
        content_type = response.headers.get("content-type", "")

        if not content_type.startswith("image/"):
            raise ValueError(f"Not an image. Content-Type: {content_type}")

        response = requests.get(url, timeout=timeout)
        image_data = io.BytesIO(response.content)

        img = Image.open(image_data)
        image_format = img.format.lower() if img.format else "unknown"

        img_copy = img.copy()
        img.verify()

        try:
            return img_copy, ImageFormat[image_format.upper()]
        except KeyError:
            return img_copy, ImageFormat.UNKNOWN

    except requests.ConnectionError:
        raise ConnectionError("Failed to connect to the server")
    except requests.Timeout:
        raise TimeoutError("Request timed out")
    except requests.RequestException as e:
        raise RuntimeError(f"Request error: {str(e)}")
    except (IOError, SyntaxError) as e:
        raise ValueError(f"Invalid image data: {str(e)}")
