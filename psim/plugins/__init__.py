"""Application plugins (trusted code shipped with the application).

Each module in this package that defines `register(registry)` is imported at
startup and may register additional object types. Plugins are part of the
installed application; projects can never load or execute plugin code.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil

log = logging.getLogger(__name__)
loaded_plugins: list[str] = []
plugin_errors: list[str] = []


def load_plugins(registry) -> None:
    for info in pkgutil.iter_modules(__path__):
        name = f"{__name__}.{info.name}"
        try:
            module = importlib.import_module(name)
            register = getattr(module, "register", None)
            if callable(register):
                register(registry)
                loaded_plugins.append(info.name)
        except Exception as exc:  # a broken plugin must not prevent start-up
            msg = f"Plugin '{info.name}' failed to load: {exc}"
            plugin_errors.append(msg)
            log.warning(msg)
