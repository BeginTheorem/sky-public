"""Tor circuit rotation: cookie auth, silent failure, and the disabled switch."""

from __future__ import annotations

import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from skynet.providers import tor_control


class _FakeTor:
    """A minimal Tor control server: replies 250 OK to every command line."""

    def __init__(self, path: Path) -> None:
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        self._server.listen(1)
        self.lines: list[str] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        conn, _ = self._server.accept()
        with conn:
            buffer = b""
            while True:
                data = conn.recv(1)
                if not data:
                    break
                buffer += data
                if buffer.endswith(b"\r\n"):
                    self.lines.append(buffer.decode().strip())
                    conn.sendall(b"250 OK\r\n")
                    buffer = b""

    def __enter__(self) -> _FakeTor:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.close()


class TorControlTests(unittest.TestCase):
    def test_newnym_is_disabled_by_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SKYNET_TOR_CONTROL_ENABLED": "false"}):
            self.assertFalse(
                tor_control.newnym(str(Path(directory) / "control"), str(Path(directory) / "cookie"))
            )

    def test_newnym_is_false_without_a_socket_or_cookie(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(
                tor_control.newnym(str(Path(directory) / "missing"), str(Path(directory) / "cookie"))
            )

    def test_newnym_authenticates_with_the_hex_cookie_and_signals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cookie = root / "control.authcookie"
            cookie.write_bytes(b"0123456789abcdef0123456789abcdef")
            control = root / "control"
            with _FakeTor(control) as tor, patch.dict(os.environ, {"SKYNET_TOR_CONTROL_ENABLED": "true"}):
                self.assertTrue(tor_control.newnym(str(control), str(cookie)))
                tor._thread.join(timeout=5)
            self.assertIn("SIGNAL NEWNYM", tor.lines)
            auth = next(line for line in tor.lines if line.startswith("AUTHENTICATE"))
            self.assertEqual(auth, "AUTHENTICATE " + cookie.read_bytes().hex())
