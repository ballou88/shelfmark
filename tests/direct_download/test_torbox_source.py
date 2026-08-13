"""Tests for the Torbox transport in the direct-download cascade."""

from pathlib import Path

import shelfmark.release_sources.direct_download as dd
from shelfmark.release_sources import BrowseRecord

# A real Anna's Archive md5 (public-domain title), matching the id format the
# Torbox transport turns into an /md5/ page URL.
_MD5 = "f204c941e622f16a08a0b2ccbaf43129"


def _book() -> BrowseRecord:
    return BrowseRecord(id=_MD5, title="A Book", source="direct", size="1 mb")


def _never_called(*args, **kwargs):
    msg = "TorboxClient must not be constructed when the prerequisites are missing"
    raise AssertionError(msg)


def _capture_warnings(monkeypatch) -> list[str]:
    """Collect warnings directly; the app logger does not reach pytest's caplog."""
    warnings: list[str] = []
    monkeypatch.setattr(
        dd.logger,
        "warning",
        lambda message, *args: warnings.append(str(message) % args if args else str(message)),
    )
    return warnings


def test_skips_when_no_aa_mirror_is_configured(monkeypatch, tmp_path):
    """An unset mirror would otherwise build a bare '/md5/<id>' path Torbox rejects."""
    monkeypatch.setattr(dd.network, "get_aa_base_url", lambda: "")
    monkeypatch.setattr("shelfmark.download.clients.torbox.TorboxClient.__init__", _never_called)
    warnings = _capture_warnings(monkeypatch)

    result = dd._try_torbox_aa_download(_book(), tmp_path / "book.epub", None, None, None)

    assert result is None
    assert any("no Anna's Archive mirror is configured" in w for w in warnings)


def test_skips_and_explains_when_api_key_is_missing(monkeypatch, tmp_path):
    """The key field is hidden unless Torbox is the torrent client, so say where it lives."""
    monkeypatch.setattr(dd.network, "get_aa_base_url", lambda: "https://mirror.example")
    monkeypatch.setattr("shelfmark.download.clients.torbox.config.get", lambda key, default="": "")
    warnings = _capture_warnings(monkeypatch)

    result = dd._try_torbox_aa_download(_book(), tmp_path / "book.epub", None, None, None)

    assert result is None
    assert any("TORBOX_API_KEY" in w and "Download Clients" in w for w in warnings)


def test_sends_the_md5_page_url_and_writes_the_payload(monkeypatch, tmp_path):
    """Torbox resolves Anna's Archive pages itself, so it must receive the /md5/ page."""
    from io import BytesIO

    monkeypatch.setattr(dd.network, "get_aa_base_url", lambda: "https://mirror.example")
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.config.get",
        lambda key, default="": {"TORBOX_API_KEY": "api-key"}.get(key, default),
    )
    captured: dict[str, object] = {}

    def _fake_download_web_url(self, url, name, size="", *args, **kwargs):
        del self, args, kwargs
        captured.update(url=url, name=name, size=size)
        payload = BytesIO(b"x" * 50_000)
        payload.seek(0, 2)  # Callers read the size from the stream position.
        return payload

    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.TorboxClient.download_web_url", _fake_download_web_url
    )
    book_path = tmp_path / "book.epub"

    result = dd._try_torbox_aa_download(_book(), book_path, None, None, None)

    expected_url = f"https://mirror.example/md5/{_MD5}"
    assert captured["url"] == expected_url
    assert result == expected_url
    assert book_path.read_bytes() == b"x" * 50_000


def test_rejects_a_too_small_payload(monkeypatch, tmp_path):
    """A truncated body is an error page, not a book; fall back to the other sources."""
    from io import BytesIO

    monkeypatch.setattr(dd.network, "get_aa_base_url", lambda: "https://mirror.example")
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.config.get",
        lambda key, default="": {"TORBOX_API_KEY": "api-key"}.get(key, default),
    )

    def _tiny(self, url, name, size="", *args, **kwargs):
        del self, url, name, size, args, kwargs
        payload = BytesIO(b"nope")
        payload.seek(0, 2)
        return payload

    monkeypatch.setattr("shelfmark.download.clients.torbox.TorboxClient.download_web_url", _tiny)
    book_path = tmp_path / "book.epub"

    assert dd._try_torbox_aa_download(_book(), book_path, None, None, None) is None
    assert not book_path.exists() or Path(book_path).stat().st_size == 0
