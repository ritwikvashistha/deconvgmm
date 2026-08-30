"""Custody checks for conservatively retained upstream license notices."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTICE = ROOT / "THIRD_PARTY_NOTICES.md"


def _fenced_text_after(document: str, heading: str) -> str:
    section = document.split(heading, 1)[1]
    return section.split("```text\n", 1)[1].split("\n```", 1)[0]


def test_conservative_upstream_notices_match_pinned_license_digests() -> None:
    document = NOTICE.read_text(encoding="utf-8")
    astroml = _fenced_text_after(document, "## astroML XDGMM") + "\n"
    bovy = _fenced_text_after(
        document, "## Original extreme-deconvolution software"
    )

    assert hashlib.sha256(astroml.encode()).hexdigest() == (
        "829eccd5a3dc1dafa02fdfe6b810ff7a8d7c0dc97630eb3658d3cb8900e55384"
    )
    assert hashlib.sha256(bovy.encode()).hexdigest() == (
        "e52808797a9bd901b30bbd0a42d2189090f9390803fa8102c1afbb9919f3c18e"
    )


def test_notice_records_unknown_history_and_nonendorsement() -> None:
    document = NOTICE.read_text(encoding="utf-8")

    assert "exact historical inputs are no longer known" in document
    assert (
        "not claims that these exact revisions were the historical inputs"
        in document
    )
    assert (
        "do not imply that the upstream authors endorse this project" in document
    )
