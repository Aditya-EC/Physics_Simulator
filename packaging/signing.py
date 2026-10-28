"""Code signing for the distributable packages (used by build_dist.py).

Windows   Authenticode with signtool (Windows SDK): SHA-256 file digest and an RFC 3161 timestamp, from a
          PFX certificate. Signs the two executables and the native core, and (through Inno Setup's SignTool
          directive) the installer and its uninstaller.
macOS     codesign of the PSim Studio.app bundle, inside out (every Mach-O file, then the bundle), with the
          hardened runtime and a secure timestamp; then notarization with notarytool and stapling.

Signing only happens when credentials are configured; without them the packages are built unsigned
(Windows) or ad-hoc signed (macOS, required for Apple silicon) and the build says so. Settings come from
environment variables (CI passes them from repository secrets):

    PSIM_WIN_CERT_PFX         path to a .pfx/.p12 with the code-signing certificate and its private key
    PSIM_WIN_CERT_PASSWORD    its password
    PSIM_WIN_TIMESTAMP_URL    RFC 3161 timestamp server (default http://timestamp.digicert.com; "" = none)
    PSIM_MACOS_SIGN_IDENTITY  "Developer ID Application: Name (TEAMID)" in the keychain (default "-": ad hoc)
    PSIM_MACOS_NOTARY_APPLE_ID, PSIM_MACOS_NOTARY_PASSWORD (app-specific password), PSIM_MACOS_NOTARY_TEAM_ID
                              notarize when all three are set (requires a Developer ID identity)

`python packaging/signing.py selftest <file.exe>` (Windows) signs a copy of an executable with a throwaway
self-signed certificate and checks the signature, so CI exercises the signing path without real credentials.
"""
from __future__ import annotations

import base64
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ENTITLEMENTS = os.path.join(HERE, "macos", "entitlements.plist")
DEFAULT_TIMESTAMP_URL = "http://timestamp.digicert.com"


class SigningError(RuntimeError):
    pass


def _run(cmd: list[str], secret: str | None = None, **kw) -> subprocess.CompletedProcess:
    shown = " ".join("***" if secret and a == secret else a for a in cmd)
    print("+", shown, flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if res.returncode != 0:
        out = (res.stdout + res.stderr).replace(secret, "***") if secret else res.stdout + res.stderr
        raise SigningError(f"{os.path.basename(cmd[0])} failed ({res.returncode}): {out.strip()}")
    return res


def run_quiet(cmd: list[str], secret: str | None = None) -> None:
    """Run a build command whose arguments may contain a secret: echoed and reported with it masked."""
    _run(cmd, secret=secret or None)


# ---- Windows -------------------------------------------------------------------------------------------------------
@dataclass
class WindowsSigning:
    pfx: str
    password: str
    timestamp_url: str = DEFAULT_TIMESTAMP_URL

    @classmethod
    def from_env(cls) -> "WindowsSigning | None":
        pfx = os.environ.get("PSIM_WIN_CERT_PFX", "")
        if not pfx:
            return None
        if not os.path.isfile(pfx):
            raise SigningError(f"PSIM_WIN_CERT_PFX does not name a file: {pfx}")
        return cls(pfx, os.environ.get("PSIM_WIN_CERT_PASSWORD", ""),
                   os.environ.get("PSIM_WIN_TIMESTAMP_URL", DEFAULT_TIMESTAMP_URL))

    def sign_command(self, signtool: str) -> list[str]:
        cmd = [signtool, "sign", "/fd", "SHA256", "/f", self.pfx]
        if self.password:
            cmd += ["/p", self.password]
        if self.timestamp_url:
            cmd += ["/tr", self.timestamp_url, "/td", "SHA256"]
        return cmd

    def sign(self, files: list[str]) -> None:
        _run(self.sign_command(find_signtool()) + list(files), secret=self.password or None)

    def inno_sign_tool(self) -> str:
        """The command for Inno Setup's /S option ($f is the file Inno Setup signs: setup and uninstaller)."""
        parts = [f'"{p}"' if " " in p else p for p in self.sign_command(find_signtool())]
        return " ".join(parts) + " $f"


def find_signtool() -> str:
    found = shutil.which("signtool")
    if found:
        return found
    kits = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Windows Kits", "10", "bin")
    candidates = sorted(glob.glob(os.path.join(kits, "10.*", "x64", "signtool.exe")))
    if not candidates:
        raise SigningError("signtool.exe not found (install the Windows SDK)")
    return candidates[-1]                      # the newest SDK


def pfx_thumbprint(pfx: str, password: str) -> str:
    ps = (f"$c = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2('{pfx}', "
          f"$env:PSIM_PFX_PW); $c.Thumbprint")
    res = _run(["powershell", "-NoProfile", "-Command", ps], env={**os.environ, "PSIM_PFX_PW": password})
    return res.stdout.strip().upper()


def signer_thumbprint(path: str) -> str:
    """Thumbprint of the certificate that signed `path` (empty when it is not signed)."""
    ps = f"(Get-AuthenticodeSignature -LiteralPath '{path}').SignerCertificate.Thumbprint"
    return _run(["powershell", "-NoProfile", "-Command", ps]).stdout.strip().upper()


def verify_windows(files: list[str], pfx: str, password: str) -> None:
    """Every file carries a signature by the certificate in `pfx` (trust is not checked: a self-signed test
    certificate is not trusted, and a real one is checked by Windows itself)."""
    want = pfx_thumbprint(pfx, password)
    for f in files:
        got = signer_thumbprint(f)
        if got != want:
            raise SigningError(f"{f}: signed by {got or 'nobody'}, expected certificate {want}")
    print(f"verified Authenticode signatures ({len(files)} files, certificate {want})")


def make_test_certificate(directory: str) -> tuple[str, str]:
    """A throwaway self-signed code-signing certificate as (pfx path, password). Windows only."""
    pfx = os.path.join(directory, "psim-test-signing.pfx")
    password = base64.b32encode(os.urandom(15)).decode()
    ps = ("$c = New-SelfSignedCertificate -Type CodeSigningCert -Subject 'CN=PSim Studio signing test' "
          "-CertStoreLocation Cert:\\CurrentUser\\My -NotAfter (Get-Date).AddDays(2); "
          "$pw = ConvertTo-SecureString -String $env:PSIM_PFX_PW -Force -AsPlainText; "
          f"Export-PfxCertificate -Cert $c -FilePath '{pfx}' -Password $pw | Out-Null; "
          "Remove-Item -LiteralPath ('Cert:\\CurrentUser\\My\\' + $c.Thumbprint)")
    _run(["powershell", "-NoProfile", "-Command", ps], env={**os.environ, "PSIM_PFX_PW": password})
    return pfx, password


def windows_selftest(exe: str) -> None:
    """Sign a copy of `exe` with a throwaway certificate (no timestamp) and check the signature."""
    with tempfile.TemporaryDirectory() as tmp:
        pfx, password = make_test_certificate(tmp)
        copy = os.path.join(tmp, os.path.basename(exe))
        shutil.copy2(exe, copy)
        if signer_thumbprint(copy):
            raise SigningError(f"{exe} is already signed; the self-test expects an unsigned file")
        WindowsSigning(pfx, password, timestamp_url="").sign([copy])
        verify_windows([copy], pfx, password)
    print("Windows signing self-test passed")


# ---- macOS ---------------------------------------------------------------------------------------------------------
@dataclass
class MacSigning:
    identity: str = "-"                        # "-" = ad hoc
    apple_id: str = ""
    password: str = ""
    team_id: str = ""

    @classmethod
    def from_env(cls) -> "MacSigning":
        return cls(os.environ.get("PSIM_MACOS_SIGN_IDENTITY", "") or "-",
                   os.environ.get("PSIM_MACOS_NOTARY_APPLE_ID", ""),
                   os.environ.get("PSIM_MACOS_NOTARY_PASSWORD", ""),
                   os.environ.get("PSIM_MACOS_NOTARY_TEAM_ID", ""))

    @property
    def ad_hoc(self) -> bool:
        return self.identity == "-"

    @property
    def notarize(self) -> bool:
        return bool(self.apple_id and self.password and self.team_id) and not self.ad_hoc

    def codesign_args(self) -> list[str]:
        if self.ad_hoc:
            return ["codesign", "--force", "--sign", "-"]
        # hardened runtime and a secure timestamp: both required for notarization
        return ["codesign", "--force", "--options", "runtime", "--timestamp", "--entitlements", ENTITLEMENTS,
                "--sign", self.identity]


def _is_macho(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
    except OSError:
        return False
    return magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf",
                     b"\xfe\xed\xfa\xce", b"\xbe\xba\xfe\xca")


def macho_files(app: str) -> list[str]:
    """Mach-O files inside the bundle, deepest first (nested code is signed before what contains it)."""
    found = []
    for base, _dirs, files in os.walk(app):
        for f in files:
            p = os.path.join(base, f)
            if not os.path.islink(p) and _is_macho(p):
                found.append(p)
    return sorted(found, key=lambda p: (-p.count(os.sep), p))


def sign_macos_app(app: str, cfg: MacSigning) -> None:
    files = macho_files(app)
    args = cfg.codesign_args()
    for i in range(0, len(files), 50):          # in batches: one codesign call per file would take minutes
        _run(args + files[i:i + 50])
    _run(args + [app])
    kind = "ad hoc" if cfg.ad_hoc else cfg.identity
    print(f"signed {app} ({len(files)} Mach-O files; {kind})")


def verify_macos_app(app: str, cfg: MacSigning) -> None:
    _run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", app])
    if not cfg.ad_hoc:
        # Gatekeeper's assessment; it only accepts a notarized (stapled) Developer ID app
        res = subprocess.run(["spctl", "--assess", "--type", "execute", "--verbose=2", app],
                             capture_output=True, text=True)
        print((res.stdout + res.stderr).strip())
        if cfg.notarize and res.returncode != 0:
            raise SigningError("Gatekeeper rejects the notarized app")
    print(f"verified code signature of {app}")


def notarize_macos(app: str, zip_path: str, cfg: MacSigning) -> None:
    """Submit the zipped app to Apple's notary service, wait for the verdict and staple the ticket to the app."""
    _run(["xcrun", "notarytool", "submit", zip_path, "--apple-id", cfg.apple_id, "--password", cfg.password,
          "--team-id", cfg.team_id, "--wait"], secret=cfg.password)
    _run(["xcrun", "stapler", "staple", app])
    _run(["xcrun", "stapler", "validate", app])
    print(f"notarized and stapled {app}")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "selftest":
        if os.name != "nt":
            print("the signing self-test is for Windows (macOS builds are ad-hoc signed and verified anyway)")
            return 0
        windows_selftest(argv[1])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
