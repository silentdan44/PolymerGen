"""JSON Lines command-line runner for the polymer library API."""

import json
import sys

from .api import API_VERSION, handle_request
from .utils import logger as library_logger


def _error_response(code: str, message: str) -> dict:
    return {
        "api_version": API_VERSION,
        "request_id": None,
        "ok": False,
        "error": {"code": code, "message": message},
    }


def main() -> None:
    """Read one JSON request per line and write one response per line."""
    # The interactive library logger defaults to stdout; reserve stdout for
    # machine-readable responses when running the protocol process.
    for handler in library_logger.handlers:
        if hasattr(handler, "setStream"):
            handler.setStream(sys.stderr)
    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
        except json.JSONDecodeError as exc:
            response = _error_response("INVALID_JSON", f"Line {line_number}: {exc.msg}")
        except Exception:  # Defensive boundary around protocol processing.
            response = _error_response("INTERNAL_ERROR", "Request failed; see runner stderr logs")
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
