"""Read Fish error responses and catch them by type.

Run: uv run python examples/fish_errors.py (offline, no key needed)

When a call to Fish fails, you get an HTTP status number and a response body.
The body is not always the same shape. Sometimes it is JSON with a `message`
field, sometimes it uses `detail`, and sometimes it is empty.

The kit turns all of these into Python exception classes, so your code can say
"if the key is wrong, do this; if we are rate limited, wait" without
remembering which status number means what:

- `FishAuthError`: 401, 402 or 403. The key is wrong, missing, or out of
  credit. Retrying will not help.
- `FishRateLimitError`: 429. Too many requests. Wait, then retry.
- `FishUpstreamError`: a 5xx error. Fish itself had a problem. Retrying may
  work.
- `FishHttpError`: the parent class of all of the above, and the type used
  for any other status, such as 404.

This example uses made-up responses so it runs without a key. You will see one
line per response, showing which branch caught it.
"""

from fish_audio_suite_kit import (
    FishAuthError,
    FishHttpError,
    FishRateLimitError,
    FishUpstreamError,
    parse_fish_error,
    retry_after_s,
)

# Four failed responses, each as `(status, body, headers)`, which is what any
# HTTP client gives you. Note the different body shapes, and the empty body.
RESPONSES: list[tuple[int, bytes, dict[str, str]]] = [
    (402, b'{"message": "Insufficient credits", "status": 402}', {}),
    (429, b'{"message": "Too many requests"}', {"Retry-After": "7"}),
    (503, b"", {}),
    (404, b'{"detail": "Voice not found"}', {}),
]


def main() -> None:
    for status, body, headers in RESPONSES:
        # `parse_fish_error` reads whatever body shape came back and gives you a
        # `status` and a readable `message`. An empty body becomes `"HTTP 503"`.
        parsed_body = parse_fish_error(status, body)

        # `retry_after_s` reads the `Retry-After` header, if there is one, as a
        # number of seconds. `FishHttpError.from_status` then picks the right
        # exception class for the status and stores that wait on it.
        fish_error = FishHttpError.from_status(
            parsed_body.status, parsed_body.message, retry_after=retry_after_s(headers)
        )

        # In a real app the error would be raised inside your Fish call. It is
        # raised here by hand so you can see how each `except` branch catches it.
        try:
            raise fish_error
        # Put the specific classes first. `FishHttpError` is their parent, so if
        # it came first it would catch everything.
        except FishAuthError as error:
            print(f"{status}: check the key or the credit balance ({error.message})")
        except FishRateLimitError as error:
            print(f"{status}: wait {error.retry_after}s, then retry")
        except FishUpstreamError as error:
            # `retryable` is `True` for 429 and 5xx: the same request may succeed later.
            print(f"{status}: Fish is down, retryable={error.retryable}")
        except FishHttpError as error:
            print(f"{status}: {error.message}, retryable={error.retryable}")


if __name__ == "__main__":
    main()
