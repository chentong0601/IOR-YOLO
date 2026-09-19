"""Read-only helpers for dataset acquisition and audit."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    """Read PNG/JPEG dimensions; raises ValueError for unreadable/unsupported files."""
    with path.open("rb") as stream:
        header = stream.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) == 24:
            width = int.from_bytes(header[16:20], "big")
            height = int.from_bytes(header[20:24], "big")
            if width and height:
                return width, height
            raise ValueError("invalid PNG dimensions")
        if not header.startswith(b"\xff\xd8"):
            raise ValueError("unsupported image format or invalid header")
        stream.seek(2)
        sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                       0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while True:
            marker_start = stream.read(1)
            if not marker_start:
                raise ValueError("JPEG dimensions not found")
            if marker_start != b"\xff":
                continue
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if not marker:
                raise ValueError("truncated JPEG")
            code = marker[0]
            if code in {0xD8, 0xD9, 0x01} or 0xD0 <= code <= 0xD7:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                raise ValueError("truncated JPEG segment")
            length = int.from_bytes(length_bytes, "big")
            if length < 2:
                raise ValueError("invalid JPEG segment")
            if code in sof_markers:
                payload = stream.read(5)
                if len(payload) != 5:
                    raise ValueError("truncated JPEG SOF")
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                if width and height:
                    return width, height
                raise ValueError("invalid JPEG dimensions")
            stream.seek(length - 2, 1)
