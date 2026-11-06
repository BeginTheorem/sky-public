"""Inline keyboards for the owner's control channel.

Pure data only: a keyboard here is a ``reply_markup`` dict and every callback carries an
*action*, never text — so a tap can route an intent (refresh the card, answer a question)
but can never smuggle a command into the shell. Nothing in this module talks to the
network, which is what keeps the routing in ``telegram_bot`` testable without a token.

Callback grammar (one action, then positional args, ``:``-separated):

- ``ui:status`` / ``ui:logs:<n>`` / ``ui:metrics`` / ``ui:alerts`` / ``ui:provider``
- ``q:<question_id>:<index>`` — the owner's answer to a queued question (this is also how a
  sudo approval is granted: the question carries ``Разрешить`` / ``Запретить`` as options)
- ``pv:<provider>`` — switch one provider rung on or off. It carries the DESIRED state only:
  the reactor copies it into the applied chain at the next heartbeat, so a tap can never
  reshape the ladder under a run that is already in flight
- ``confirm:<action>`` / ``cancel`` — the pre-existing confirmation path, unchanged
"""

from __future__ import annotations

from typing import Any

CALLBACK_LIMIT = 64  # Telegram's hard limit on callback_data


def parse_callback(data: str) -> tuple[str, list[str]]:
    """Split ``callback_data`` into its kind and positional arguments.

    An unknown or malformed payload yields ``("", [])``: the caller answers it with a
    short toast instead of guessing at an action.
    """
    parts = list(str(data or "").split(":"))
    if not parts or not parts[0]:
        return "", []
    return parts[0], parts[1:]


def _button(label: str, data: str) -> dict[str, str]:
    return {"text": label, "callback_data": data[:CALLBACK_LIMIT]}


def _rows(rows: list[list[tuple[str, str]]]) -> dict[str, Any]:
    return {"inline_keyboard": [[_button(label, data) for label, data in row] for row in rows]}


def status_keyboard() -> dict[str, Any]:
    """The card's controls: refresh in place, then the three detail views."""
    return _rows([
        [("🔄 Обновить", "ui:status"), ("📜 Логи", "ui:logs")],
        [("📊 Метрики", "ui:metrics"), ("⚠️ Алерты", "ui:alerts")],
        [("🧠 Провайдеры", "ui:provider")],
    ])


def logs_keyboard(limit: int | None = None) -> dict[str, Any]:
    """Log window sizes, so the owner picks the depth instead of typing a number."""
    window = int(limit or 0)
    sizes = [3, 5, 10, 20]
    row = [(f"{size}{' ✅' if size == window else ''}", f"ui:logs:{size}") for size in sizes]
    return _rows([[("🔄 Обновить", f"ui:logs:{window or 5}")], row, [("🧭 Статус", "ui:status")]])


def question_keyboard(question_id: str, options: list[Any]) -> dict[str, Any] | None:
    """One button per option of a queued owner question; ``None`` when there are no options.

    The index is positional and the question id is the store's own, so the tap resolves
    through the same ``store.answer_question`` the text ``/answer`` path uses — one
    authority for answers, not two.
    """
    clean = [str(option).strip() for option in options if str(option).strip()]
    if not clean or not question_id:
        return None
    return _rows([[(option[:32], f"q:{question_id}:{index}")] for index, option in enumerate(clean[:8])])


def providers_keyboard(entries: list[tuple[str, str]]) -> dict[str, Any]:
    """One toggle per provider rung, two to a row.

    ``entries`` are ``(label, provider_name)`` pairs where the label already carries the
    ✅/⛔ the owner reads; the callback carries the provider NAME only, so a tap can name
    a rung but can never smuggle text towards the shell.
    """
    rows = [[(label, f"pv:{name}") for label, name in entries[index : index + 2]] for index in range(0, len(entries), 2)]
    rows.append([("🔄 Обновить", "ui:provider"), ("🧭 Статус", "ui:status")])
    return _rows(rows)


def help_keyboard() -> dict[str, Any]:
    """A card of the controls that need no typing."""
    return _rows([
        [("🧭 Статус", "ui:status"), ("📜 Логи", "ui:logs")],
        [("📊 Метрики", "ui:metrics"), ("⚠️ Алерты", "ui:alerts")],
        [("🧠 Провайдеры", "ui:provider")],
    ])
