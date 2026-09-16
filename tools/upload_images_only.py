#!/usr/bin/env python3
"""Upload only image files from an existing LoD package to ModelScope."""
import argparse
from getpass import getpass
import hashlib
import json
from pathlib import Path
import sys

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff")
ALLOW_PATTERNS = [
    pattern
    for extension in EXTENSIONS
    for suffix in (extension, extension.upper())
    for pattern in ("*" + suffix, "**/*" + suffix)
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_images(folder):
    folder = folder.resolve()
    assets = folder / "asset"
    if not assets.is_dir():
        raise ValueError("Expected an asset/ directory inside: " + str(folder))
    images = [
        p for p in assets.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]
    if not images:
        raise ValueError("No images found under: " + str(assets))
    for path in images:
        if assets.resolve() not in path.resolve().parents:
            raise ValueError("Image points outside asset/: " + str(path))
    manifest_path = folder / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        expected = {}
        for entry in manifest["files"]:
            relative = entry["path"]
            if Path(relative).suffix.lower() not in EXTENSIONS:
                continue
            path = (folder / relative).resolve()
            if assets.resolve() not in path.parents:
                raise ValueError("Invalid image path in manifest: " + relative)
            expected[relative] = entry["sha256"]
        actual = {p.relative_to(folder).as_posix(): p for p in images}
        if set(actual) != set(expected):
            raise ValueError("Images differ from manifest; rebuild or check the package.")
        for relative, path in actual.items():
            if sha256(path) != expected[relative]:
                raise ValueError("Image checksum mismatch: " + relative)
    return assets, len(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True,
                        help="Existing package directory containing asset/ (v1 or v2).")
    parser.add_argument("--repo-id", default="detecpolo/Learning-to-Detect")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        assets, count = check_images(args.folder)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1
    print("Verified %d images under %s" % (count, assets))
    print("Only image files will be uploaded to asset/ in %s." % args.repo_id)
    print("JSON, README and manifest files are excluded. Existing remote files are not deleted.")
    if args.check_only:
        return 0
    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        print("Install the SDK first: python3 -m pip install -U modelscope", file=sys.stderr)
        return 1
    token = getpass("ModelScope Token: ")
    if not token:
        print("No token entered.", file=sys.stderr)
        return 1
    HubApi().upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(assets),
        path_in_repo="asset",
        allow_patterns=ALLOW_PATTERNS,
        token=token,
    )
    print("Image upload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
