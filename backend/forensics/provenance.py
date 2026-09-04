"""
Stage 4: Provenance analysis.

Real C2PA / Content Credentials verification requires parsing signed
manifest data embedded in the file (JUMBF boxes in JPEG/PNG/etc.) and
validating certificate chains. That library integration is the natural
"swap-in" point post-hackathon (e.g. the official `c2pa-python` bindings).

For the MVP we do a lightweight, honest check:
  1. Look for the raw C2PA marker bytes in the file so we can at least say
     "a manifest appears to be present" vs. "no manifest found".
  2. Never claim verification we haven't actually performed.
"""
from __future__ import annotations

from typing import Any

C2PA_MARKERS = [b"c2pa", b"C2PA", b"urn:c2pa"]


def analyze_provenance(raw_bytes: bytes) -> dict[str, Any]:
    manifest_present = any(marker in raw_bytes for marker in C2PA_MARKERS)

    if manifest_present:
        return {
            "manifest_found": True,
            "verified": False,  # honest: byte-sniffing != cryptographic verification
            "note": (
                "A possible C2PA/Content Credentials manifest marker was found in the file, "
                "but this MVP does not yet validate the certificate chain. Wire in a C2PA "
                "verification library (e.g. c2pa-python) to confirm signer identity and "
                "edit history."
            ),
        }

    return {
        "manifest_found": False,
        "verified": False,
        "note": (
            "No provenance manifest detected. This is common and does NOT imply the image "
            "is fake - most cameras, apps, and platforms do not attach Content Credentials."
        ),
    }
