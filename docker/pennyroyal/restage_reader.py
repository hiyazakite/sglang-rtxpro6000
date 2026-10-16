#!/usr/bin/env python3
"""Re-stage the NVMe PLE reader the way ``uv pip install --target`` would.

``tools/ple_nvme/install.sh`` builds the reader with uv, cargo and rust, none of
which exist outside the canonical image's discarded build stage. The reader's
Rust source is byte-identical between the revision the parent image bakes and
the revision ``Dockerfile.pennyroyal-local`` checks out, so its compiled ``_io``
extension is still the one a fresh build would produce and only the package's
Python moved. This lays the checked-out Python over that extension and rewrites
the packaging metadata to match, leaving what a real ``--target`` install of the
new revision would have left.
"""

from __future__ import annotations

import base64
import hashlib
import re
import shutil
from pathlib import Path

REPO_ROOT = Path("/opt/pennyroyal")
PLUGIN_DIR = REPO_ROOT / ".ple-nvme"
PACKAGE = "sglang_ssd_stream"
SOURCE_DIR = REPO_ROOT / "tools" / "ple_nvme" / "ssd_stream" / "src" / PACKAGE
PYPROJECT = REPO_ROOT / "tools" / "ple_nvme" / "ssd_stream" / "pyproject.toml"


def record_entry(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    if path.name == "RECORD":
        return f"{relative},,"
    data = path.read_bytes()
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    )
    return f"{relative},sha256={digest},{len(data)}"


def declared_version() -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.MULTILINE)
    if match is None:
        raise SystemExit(f"{PYPROJECT}: no static project version to report")
    return match.group(1)


def main() -> None:
    version = declared_version()

    staged = PLUGIN_DIR / PACKAGE
    dist_infos = sorted(PLUGIN_DIR.glob(f"{PACKAGE}-*.dist-info"))
    if not staged.is_dir() or len(dist_infos) != 1:
        raise SystemExit(
            f"{PLUGIN_DIR}: unexpected reader layout {sorted(p.name for p in PLUGIN_DIR.iterdir())}"
        )

    for source in sorted(SOURCE_DIR.iterdir()):
        if source.is_file():
            shutil.copy2(source, staged / source.name)

    # shutil.move, not Path.rename: the reader sits in a lower overlayfs layer,
    # and renaming a directory across that boundary raises EXDEV.
    wanted = PLUGIN_DIR / f"{PACKAGE}-{version}.dist-info"
    if dist_infos[0] != wanted:
        shutil.move(dist_infos[0], wanted)

    metadata = wanted / "METADATA"
    text, count = re.subn(
        r"^Version:.*$",
        f"Version: {version}",
        metadata.read_text(),
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit(f"{metadata}: no Version line to rewrite")
    metadata.write_text(text)

    entries = sorted(path for path in PLUGIN_DIR.rglob("*") if path.is_file())
    (wanted / "RECORD").write_text(
        "".join(f"{record_entry(PLUGIN_DIR, path)}\n" for path in entries)
    )

    print(f"reader re-staged at {version}: {len(entries)} files recorded")


if __name__ == "__main__":
    main()
