"""Signed formula-process packages (.psimproc).

Package layout (JSON)::

    {"format": "psim-process-package", "version": 1,
     "process": <FormulaProcessSpec.to_dict()>,
     "key_id": "<signer key id>", "signature": "<base64 Ed25519 signature>"}

The signature covers the canonical JSON encoding (sorted keys, compact
separators) of ``{"format", "version", "process"}``.
"""
from __future__ import annotations

import json
import os

from .formula_process import FormulaProcessSpec
from .permissions import AccessController, DeveloperToken
from .process_base import ProcessError

PACKAGE_FORMAT = "psim-process-package"
PACKAGE_VERSION = 1
EXTENSION = ".psimproc"


def canonical_bytes(spec: dict) -> bytes:
    body = {"format": PACKAGE_FORMAT, "version": PACKAGE_VERSION, "process": spec}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def packages_dir(base: str) -> str:
    d = os.path.join(base, "processes")
    os.makedirs(d, exist_ok=True)
    return d


def write_package(path: str, spec: FormulaProcessSpec, access: AccessController, token: DeveloperToken) -> None:
    data = spec.to_dict()
    key_id, sig = access.sign(token, canonical_bytes(data))
    payload = {"format": PACKAGE_FORMAT, "version": PACKAGE_VERSION, "process": data, "key_id": key_id,
               "signature": sig}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_package(path: str, access: AccessController) -> tuple[FormulaProcessSpec, bool, str]:
    """Return (spec, verified, message). Raises ProcessError for malformed files."""
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ProcessError(f"{os.path.basename(path)}: not a readable package ({exc})") from None
    if not isinstance(payload, dict) or payload.get("format") != PACKAGE_FORMAT:
        raise ProcessError(f"{os.path.basename(path)}: not a PSim process package")
    if payload.get("version") != PACKAGE_VERSION:
        raise ProcessError(f"{os.path.basename(path)}: unsupported package version {payload.get('version')}")
    spec = FormulaProcessSpec.from_dict(payload.get("process"))
    ok, msg = access.verify(canonical_bytes(payload["process"]), str(payload.get("key_id", "")),
                            str(payload.get("signature", "")))
    return spec, ok, msg
