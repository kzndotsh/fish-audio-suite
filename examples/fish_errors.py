"""Read Fish error responses and catch them by type.

Run: uv run python examples/fish_errors.py (offline, no key needed)
"""

from fish_audio_suite_kit import (
    FishAuthError,
    FishHttpError,
    FishRateLimitError,
    FishUpstreamError,
    parse_fish_error,
    retry_after_s,
)

# (status, body, headers) as an HTTP client hands them over.
RESPONSES: list[tuple[int, bytes, dict[str, str]]] = [
    (402, b'{"message": "Insufficient credits", "status": 402}', {}),
    (429, b'{"message": "Too many requests"}', {"Retry-After": "7"}),
    (503, b"", {}),
    (404, b'{"detail": "Voice not found"}', {}),
]


def main() -> None:
    for status, body, headers in RESPONSES:
        parsed = parse_fish_error(status, body)
        error = FishHttpError.from_status(
            parsed.status, parsed.message, retry_after=retry_after_s(headers)
        )
        try:
            raise error
        except FishAuthError as exc:
            print(f"{status}: check the key or the credit balance ({exc.message})")
        except FishRateLimitError as exc:
            print(f"{status}: wait {exc.retry_after}s, then retry")
        except FishUpstreamError as exc:
            print(f"{status}: Fish is down, retryable={exc.retryable}")
        except FishHttpError as exc:
            print(f"{status}: {exc.message}, retryable={exc.retryable}")


if __name__ == "__main__":
    main()
