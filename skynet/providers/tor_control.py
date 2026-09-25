"""Best-effort Tor circuit rotation over the control socket.

A dirty Tor exit can answer with HTTP 403, which the HTTP layer cannot tell
apart from a rejected API key. Rotating the circuit on a proxy-level 403 turns
a fatal auth failure back into a retryable network condition. The whole module
is optional: when there is no control socket (a local run, or a proxy that is
not Tor) it reports failure and the caller keeps its normal classification.
"""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

log = logging.getLogger("skynet.providers.tor_control")

DEFAULT_CONTROL_SOCKET = "/run/tor/control"
DEFAULT_COOKIE_PATH = "/run/tor/control.authcookie"


def _disabled() -> bool:
    return os.getenv("SKYNET_TOR_CONTROL_ENABLED", "true").strip().casefold() in {"0", "false", "no", "off"}


def _read_cookie_hex(path: str) -> str | None:
    # The auth cookie is 32 raw bytes; AUTHENTICATE wants it hex-encoded.
    try:
        data = Path(path).read_bytes().strip()
    except OSError:
        return None
    return data.hex() if data else None


def _talk(sock: socket.socket, line: str) -> str:
    sock.sendall((line + "\r\n").encode())
    collected = ""
    sock.settimeout(3.0)
    try:
        while True:
            data = sock.recv(4096)
            if not data:
                break
            collected += data.decode("utf-8", "replace")
            if collected.rstrip().endswith("250 OK"):
                break
    except (TimeoutError, OSError):
        pass
    return collected


def newnym(control_socket: str | None = None, cookie_path: str | None = None) -> bool:
    """Ask Tor for a fresh circuit. True on ``250 OK``, else False.

    Never raises: a missing control socket or an unreadable cookie is a normal
    condition for a deployment that does not proxy through Tor.
    """
    if _disabled():
        return False
    socket_path = control_socket or os.getenv("SKYNET_TOR_CONTROL_SOCKET", DEFAULT_CONTROL_SOCKET)
    cookie = _read_cookie_hex(cookie_path or os.getenv("SKYNET_TOR_CONTROL_COOKIE", DEFAULT_COOKIE_PATH))
    if not cookie or not Path(socket_path).exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect(socket_path)
        _talk(sock, "PROTOCOLINFO 1")
        if "250 OK" not in _talk(sock, f"AUTHENTICATE {cookie}"):
            return False
        rotated = "250 OK" in _talk(sock, "SIGNAL NEWNYM")
        _talk(sock, "QUIT")
        return rotated
    except OSError as exc:
        log.warning("tor control rotation failed: %s", exc)
        return False
    finally:
        sock.close()
