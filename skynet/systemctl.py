"""Build the ``systemctl`` invocation for this process.

The harness has two shapes. In production it is an unprivileged, lingering
*user* service owned by the service account, so every control action must go
through the user manager (``systemctl --user``, reachable via
``XDG_RUNTIME_DIR``). The legacy shape ran as a root *system* service and used
the system bus. A ``SKYNET_SYSTEMCTL`` override (ops and tests) wins and may
carry arguments, so it is split with :func:`shlex.split` instead of being taken
as a single executable name.

No ``sudo`` is ever emitted: the service account has no administrative rights.
"""

from __future__ import annotations

import os
import shlex


def systemctl_command(*args: str) -> list[str]:
    """Return the ``systemctl`` argv for ``args`` on the correct bus.

    ``SKYNET_SYSTEMCTL`` overrides the executable (and may carry flags such as
    ``--user``); otherwise the bus follows the effective uid: a non-root process
    controls its own user units, a root process the system ones.
    """
    override = os.getenv("SKYNET_SYSTEMCTL")
    if override:
        return [*shlex.split(override), *args]
    command = ["systemctl"]
    if os.geteuid() != 0:
        command.append("--user")
    command.extend(args)
    return command
