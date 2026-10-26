"""Role-based access to process authoring, and package signing.

Ordinary users can select, configure and inspect approved processes. Only an
authorized developer can create, edit, validate, publish or withdraw
formula-process definitions.

Enforcement is in application logic, not only in the GUI:

* Developer credentials (a salted PBKDF2 verifier and an Ed25519 signing key
  encrypted with the developer passphrase) are created with the command-line
  tool ``python -m psim.devtools init-developer``; the GUI cannot create them.
* ``AccessController.unlock(passphrase)`` returns a session ``DeveloperToken``.
  The process registry is sealed after start-up; registering, publishing or
  withdrawing processes afterwards requires a valid token.
* Developer-authored processes are distributed as signed packages. A package
  is loaded only if its Ed25519 signature verifies against a trusted public
  key (``trusted_keys/``); project files can only reference process ids.

Limits: on a locally installed application a determined user who controls
the machine can modify the installation itself (e.g. its code or trusted
keys). These measures prevent accidental or casual modification of approved
models and detect tampered packages; they are not protection against reverse
engineering.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from enum import Enum

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

PBKDF2_ITERATIONS = 300_000


class Role(str, Enum):
    USER = "user"
    DEVELOPER = "developer"


class PermissionDenied(PermissionError):
    pass


@dataclass(frozen=True, eq=False)
class DeveloperToken:
    """Opaque session token; valid only while it is the controller's active token."""

    key_id: str
    author: str
    _nonce: bytes

    def __repr__(self) -> str:
        return f"DeveloperToken(key_id={self.key_id!r})"


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=True)


def key_id_for(public_raw: bytes) -> str:
    return hashlib.sha256(public_raw).hexdigest()[:16]


def _default_base() -> str:
    from ..project.autosave import data_dir
    return data_dir()


class AccessController:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base = base_dir or _default_base()
        self._token: DeveloperToken | None = None
        self._private: Ed25519PrivateKey | None = None

    # ---- paths -----------------------------------------------------------------------------------------
    @property
    def credentials_path(self) -> str:
        return os.path.join(self.base, "developer", "credentials.json")

    @property
    def trusted_dir(self) -> str:
        return os.path.join(self.base, "trusted_keys")

    def has_credentials(self) -> bool:
        return os.path.exists(self.credentials_path)

    # ---- role ---------------------------------------------------------------------------------------------
    @property
    def role(self) -> Role:
        return Role.DEVELOPER if self._token is not None else Role.USER

    @property
    def token(self) -> DeveloperToken | None:
        return self._token

    def is_valid(self, token: DeveloperToken | None) -> bool:
        return token is not None and self._token is not None and token is self._token

    def require(self, token: DeveloperToken | None, action: str = "This action") -> None:
        if not self.is_valid(token):
            raise PermissionDenied(f"{action} requires the developer role (Tools ▸ Developer Mode)")

    def unlock(self, passphrase: str) -> DeveloperToken:
        if not self.has_credentials():
            raise PermissionDenied("No developer credentials are installed on this computer. "
                                   "Create them with: python -m psim.devtools init-developer")
        with open(self.credentials_path, encoding="utf-8") as fh:
            cred = json.load(fh)
        derived = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), _unb64(cred["salt"]), int(cred["iterations"]))
        if not hmac.compare_digest(derived, _unb64(cred["verifier"])):
            raise PermissionDenied("Incorrect developer passphrase")
        try:
            key = serialization.load_pem_private_key(cred["private_key"].encode(), password=passphrase.encode())
        except (ValueError, TypeError):
            raise PermissionDenied("The developer signing key could not be decrypted") from None
        if not isinstance(key, Ed25519PrivateKey):
            raise PermissionDenied("Unsupported developer key type")
        self._private = key
        self._token = DeveloperToken(cred["key_id"], cred.get("author", ""), secrets.token_bytes(16))
        return self._token

    def lock(self) -> None:
        self._token = None
        self._private = None

    # ---- signing ------------------------------------------------------------------------------------------
    def sign(self, token: DeveloperToken, data: bytes) -> tuple[str, str]:
        self.require(token, "Signing a process package")
        return token.key_id, _b64(self._private.sign(data))

    def trusted_keys(self) -> dict[str, Ed25519PublicKey]:
        keys: dict[str, Ed25519PublicKey] = {}
        dirs = [self.trusted_dir, os.path.join(os.path.dirname(__file__), "trusted_keys")]
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".pub"):
                    continue
                try:
                    with open(os.path.join(d, fn), encoding="utf-8") as fh:
                        raw = _unb64(fh.read().strip().split()[-1])
                    keys[key_id_for(raw)] = Ed25519PublicKey.from_public_bytes(raw)
                except (OSError, ValueError):
                    continue
        return keys

    def verify(self, data: bytes, key_id: str, signature: str) -> tuple[bool, str]:
        key = self.trusted_keys().get(key_id)
        if key is None:
            return False, f"signed by an untrusted key ({key_id})"
        try:
            key.verify(_unb64(signature), data)
            return True, "signature verified"
        except (InvalidSignature, ValueError):
            return False, "invalid signature (the package was modified after signing)"

    # ---- setup (command-line tool only) ----------------------------------------------------------------------
    def create_credentials(self, passphrase: str, author: str, overwrite: bool = False) -> str:
        if len(passphrase) < 10:
            raise ValueError("The developer passphrase must have at least 10 characters")
        if self.has_credentials() and not overwrite:
            raise FileExistsError(f"Developer credentials already exist: {self.credentials_path}")
        key = Ed25519PrivateKey.generate()
        pub_raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        salt = secrets.token_bytes(16)
        cred = {
            "version": 1, "author": author, "kdf": "pbkdf2-sha256", "iterations": PBKDF2_ITERATIONS,
            "salt": _b64(salt),
            "verifier": _b64(hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, PBKDF2_ITERATIONS)),
            "key_id": key_id_for(pub_raw), "public_key": _b64(pub_raw),
            "private_key": key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                             serialization.BestAvailableEncryption(passphrase.encode())).decode(),
        }
        os.makedirs(os.path.dirname(self.credentials_path), exist_ok=True)
        with open(self.credentials_path, "w", encoding="utf-8") as fh:
            json.dump(cred, fh, indent=1)
        try:
            os.chmod(self.credentials_path, 0o600)
        except OSError:
            pass
        self.trust_public_key(_b64(pub_raw), f"{author or 'developer'}")
        return cred["key_id"]

    def public_key_text(self) -> str:
        with open(self.credentials_path, encoding="utf-8") as fh:
            cred = json.load(fh)
        return f"psim-ed25519 {cred['public_key']}"

    def trust_public_key(self, public_b64: str, label: str = "") -> str:
        raw = _unb64(public_b64.strip().split()[-1])
        Ed25519PublicKey.from_public_bytes(raw)  # validates the key
        kid = key_id_for(raw)
        os.makedirs(self.trusted_dir, exist_ok=True)
        safe = "".join(c for c in label if c.isalnum() or c in "-_") or "key"
        with open(os.path.join(self.trusted_dir, f"{safe}-{kid}.pub"), "w", encoding="utf-8") as fh:
            fh.write(f"psim-ed25519 {_b64(raw)}\n")
        return kid


_controller: AccessController | None = None


def access_controller() -> AccessController:
    global _controller
    if _controller is None:
        _controller = AccessController()
    return _controller
