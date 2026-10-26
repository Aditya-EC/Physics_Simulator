"""Developer command-line tools (not available from the GUI).

    python -m psim.devtools init-developer [--author NAME] [--force]
    python -m psim.devtools show-public-key
    python -m psim.devtools trust-key <public-key-text-or-file> [--label NAME]
    python -m psim.devtools list-trusted
    python -m psim.devtools verify <package.psimproc>

`init-developer` creates the developer credentials (passphrase verifier and an
encrypted Ed25519 signing key) used by Tools ▸ Developer Mode, and trusts the
new key on this computer. Distribute the public key (show-public-key) to the
installations that should accept your process packages (trust-key).
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys


def _controller():
    from PySide6.QtCore import QCoreApplication

    from . import APP_NAME, ORG_NAME
    QCoreApplication.setOrganizationName(ORG_NAME)
    QCoreApplication.setApplicationName(APP_NAME)
    from .processes.permissions import AccessController
    return AccessController()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m psim.devtools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-developer")
    p.add_argument("--author", default=getpass.getuser())
    p.add_argument("--force", action="store_true", help="replace existing credentials")
    p.add_argument("--passphrase-env", help="read the passphrase from this environment variable (automation)")
    sub.add_parser("show-public-key")
    p = sub.add_parser("trust-key")
    p.add_argument("key")
    p.add_argument("--label", default="")
    sub.add_parser("list-trusted")
    p = sub.add_parser("verify")
    p.add_argument("package")
    args = ap.parse_args(argv)
    ac = _controller()
    if args.cmd == "init-developer":
        if args.passphrase_env:
            pw = os.environ.get(args.passphrase_env, "")
        else:
            pw = getpass.getpass("New developer passphrase (min. 10 characters): ")
            if pw != getpass.getpass("Repeat passphrase: "):
                print("error: passphrases do not match", file=sys.stderr)
                return 1
        try:
            kid = ac.create_credentials(pw, args.author, overwrite=args.force)
        except (ValueError, FileExistsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Developer credentials created at {ac.credentials_path}\nSigning key id: {kid}")
        return 0
    if args.cmd == "show-public-key":
        print(ac.public_key_text())
        return 0
    if args.cmd == "trust-key":
        text = open(args.key, encoding="utf-8").read() if os.path.exists(args.key) else args.key
        print(f"Trusted key {ac.trust_public_key(text, args.label)}")
        return 0
    if args.cmd == "list-trusted":
        for kid in ac.trusted_keys():
            print(kid)
        return 0
    from .processes.packages import read_package
    spec, ok, msg = read_package(args.package, ac)
    print(f"{spec.process_id} {spec.version}: {msg}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
