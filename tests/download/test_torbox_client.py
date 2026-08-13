"""Unit tests for the Torbox download client."""

from io import BytesIO
from unittest.mock import MagicMock

import pytest
import requests

from shelfmark.download.clients import DownloadState
from shelfmark.download.clients.torbox import TorboxClient


def _response(payload):
    response = MagicMock()
    response.json.return_value = payload
    return response


def _error_response(payload, status):
    """A rejection: Torbox pairs a non-2xx status with a JSON body naming the cause."""
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        f"{status} Server Error: for url: https://api.torbox.app/"
    )
    return response


def _client(monkeypatch):
    config_values = {
        "PROWLARR_TORRENT_CLIENT": "torbox",
        "TORBOX_API_KEY": "api-key",
    }
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.config.get",
        lambda key, default="": config_values.get(key, default),
    )
    TorboxClient._downloads.clear()
    return TorboxClient()


def test_is_configured(monkeypatch):
    assert TorboxClient.is_configured() is False
    _client(monkeypatch)
    assert TorboxClient.is_configured() is True


def test_connection_accepts_active_subscription(monkeypatch):
    client = _client(monkeypatch)
    mock_get = MagicMock(
        return_value=_response(
            {
                "success": True,
                "data": {"email": "reader@example.com", "is_subscribed": True, "plan": 2},
            }
        )
    )
    monkeypatch.setattr("shelfmark.download.clients.torbox.requests.get", mock_get)

    assert client.test_connection() == (True, "Connected to Torbox as 'reader@example.com'")
    assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer api-key"}


def test_add_download_posts_magnet_and_stores_state(monkeypatch):
    client = _client(monkeypatch)
    mock_post = MagicMock(return_value=_response({"success": True, "data": {"torrent_id": 42}}))
    monkeypatch.setattr("shelfmark.download.clients.torbox.requests.post", mock_post)

    assert client.add_download("magnet:?xt=urn:btih:hash", "A Book") == "42"
    assert mock_post.call_args.kwargs["data"] == {
        "magnet": "magnet:?xt=urn:btih:hash",
        "name": "A Book",
    }
    assert "42" in client._downloads


def test_get_status_starts_http_download_when_torrent_is_cached(monkeypatch):
    client = _client(monkeypatch)
    state = client._ensure_state("42")
    mock_get = MagicMock(
        return_value=_response(
            {
                "success": True,
                "data": {"download_state": "cached", "files": [{"id": 7, "name": "book.epub"}]},
            }
        )
    )
    monkeypatch.setattr("shelfmark.download.clients.torbox.requests.get", mock_get)
    started_with = []
    monkeypatch.setattr(
        client, "_maybe_start_download_thread", lambda *args: started_with.extend(args)
    )

    status = client.get_status("42")

    assert status.state == DownloadState.DOWNLOADING
    assert status.progress == 50.0
    assert started_with == [state, [{"id": 7, "name": "book.epub"}]]
    assert mock_get.call_args.kwargs["params"] == {"id": "42", "bypass_cache": "true"}


def test_download_files_requests_tokenized_link_and_filters_books(monkeypatch, tmp_path):
    client = _client(monkeypatch)
    state = client._ensure_state("42")
    state.target_dir = tmp_path
    mock_get = MagicMock(
        return_value=_response({"success": True, "data": "https://cdn.example/book"})
    )
    monkeypatch.setattr("shelfmark.download.clients.torbox.requests.get", mock_get)
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.download_url",
        lambda *args, **kwargs: BytesIO(b"contents"),
    )

    client._download_files(state, [{"id": 1, "name": "cover.jpg"}, {"id": 2, "name": "book.epub"}])

    assert (tmp_path / "book.epub").read_bytes() == b"contents"
    assert not (tmp_path / "cover.jpg").exists()
    assert mock_get.call_args.kwargs["params"] == {
        "token": "api-key",
        "torrent_id": "42",
        "file_id": 2,
    }
    assert state.phase == "complete"


def test_remove_uses_torbox_delete_operation(monkeypatch):
    client = _client(monkeypatch)
    mock_post = MagicMock(return_value=_response({"success": True, "data": None}))
    monkeypatch.setattr("shelfmark.download.clients.torbox.requests.post", mock_post)

    assert client.remove("42") is True
    assert mock_post.call_args.kwargs["json"] == {"torrent_id": 42, "operation": "delete"}


def test_download_web_url_creates_then_fetches_cdn_link(monkeypatch):
    client = _client(monkeypatch)
    mock_post = MagicMock(return_value=_response({"success": True, "data": {"id": 42}}))
    mock_get = MagicMock(
        side_effect=[
            _response({"success": True, "data": {"download_state": "completed"}}),
            _response({"success": True, "data": "https://cdn.example/book"}),
        ]
    )
    monkeypatch.setattr("shelfmark.download.clients.torbox.requests.post", mock_post)
    monkeypatch.setattr("shelfmark.download.clients.torbox.requests.get", mock_get)
    expected = BytesIO(b"contents")
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.download_url", lambda *args, **kwargs: expected
    )

    assert client.download_web_url("https://source.example/book.epub", "A Book") is expected
    assert mock_post.call_args_list[0].kwargs["data"] == {
        "link": "https://source.example/book.epub",
        "name": "A Book",
    }
    assert mock_post.call_args_list[0].args[0].endswith("/webdl/createwebdownload")
    assert mock_get.call_args_list[1].kwargs["params"] == {"token": "api-key", "web_id": "42"}
    assert mock_post.call_args_list[1].kwargs["json"] == {"webdl_id": 42, "operation": "delete"}


def test_download_web_url_does_not_create_job_when_cancelled(monkeypatch):
    from threading import Event

    client = _client(monkeypatch)
    cancel_flag = Event()
    cancel_flag.set()
    mock_post = MagicMock()
    monkeypatch.setattr("shelfmark.download.clients.torbox.requests.post", mock_post)

    assert (
        client.download_web_url(
            "https://source.example/book.epub", "A Book", cancel_flag=cancel_flag
        )
        is None
    )
    mock_post.assert_not_called()


def test_download_web_url_gives_up_once_the_deadline_passes(monkeypatch):
    """A job Torbox never finishes must not pin the calling download worker forever."""
    client = _client(monkeypatch)
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.requests.post",
        MagicMock(return_value=_response({"success": True, "data": {"id": 42}})),
    )
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.requests.get",
        MagicMock(return_value=_response({"success": True, "data": {"download_state": "queued"}})),
    )
    # Advance a fake clock past the wait budget instead of really sleeping.
    clock = {"now": 0.0}
    monkeypatch.setattr("shelfmark.download.clients.torbox.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.time.sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    with pytest.raises(RuntimeError, match="did not finish"):
        client.download_web_url("https://source.example/book.epub", "A Book")


def test_create_web_download_allows_torbox_time_to_fetch_the_origin(monkeypatch):
    """Creating a web download is not a quick job accept - Torbox fetches the origin.

    Observed live: a cold Anna's Archive link blew past the 30s API timeout, and
    because a client-side timeout does not cancel the job, the download was lost
    *and* an orphan was left burning the account's 25/day AA link quota.
    """
    client = _client(monkeypatch)
    mock_post = MagicMock(return_value=_response({"success": True, "data": {"id": 42}}))
    mock_get = MagicMock(
        side_effect=[
            _response({"success": True, "data": {"download_state": "completed"}}),
            _response({"success": True, "data": "https://cdn.example/book"}),
        ]
    )
    monkeypatch.setattr("shelfmark.download.clients.torbox.requests.post", mock_post)
    monkeypatch.setattr("shelfmark.download.clients.torbox.requests.get", mock_get)
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.download_url", lambda *args, **kwargs: BytesIO(b"x")
    )

    client.download_web_url("https://annas-archive.gl/md5/abc", "A Book")

    create_timeout = mock_post.call_args_list[0].kwargs["timeout"]
    assert create_timeout >= 120, (
        f"createwebdownload timeout {create_timeout}s is too short for a cold origin fetch"
    )


def test_rejection_surfaces_torbox_detail_not_the_http_status(monkeypatch):
    """Torbox pairs a 500 with the real reason; the status alone tells users nothing.

    Verified against the live API: posting an unsupported link returns HTTP 500 with
    {"success": false, "detail": "The site you are trying to download from is not
    supported.", "error": "UNSUPPORTED_SITE"}.
    """
    client = _client(monkeypatch)
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.requests.post",
        MagicMock(
            return_value=_error_response(
                {
                    "success": False,
                    "detail": "The site you are trying to download from is not supported.",
                    "error": "UNSUPPORTED_SITE",
                    "data": None,
                },
                500,
            )
        ),
    )

    with pytest.raises(RuntimeError, match="not supported"):
        client.download_web_url("https://unsupported.example/book.epub", "A Book")


def test_connection_reports_torbox_detail_for_a_bad_key(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.requests.get",
        MagicMock(
            return_value=_error_response(
                {"success": False, "detail": "Invalid or expired API key.", "data": None}, 403
            )
        ),
    )

    success, message = client.test_connection()

    assert success is False
    assert "Invalid or expired API key." in message
    assert "403" not in message


def test_get_status_reports_error_when_torbox_torrent_fails(monkeypatch):
    client = _client(monkeypatch)
    state = client._ensure_state("42")
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.requests.get",
        MagicMock(
            return_value=_response(
                {"success": True, "data": {"download_state": "error", "progress": 0.2}}
            )
        ),
    )

    status = client.get_status("42")

    assert status.state == DownloadState.ERROR
    assert state.phase == "error"
    # A terminal failure must stick, so later polls do not re-query a dead torrent.
    assert client.get_status("42").state == DownloadState.ERROR


def test_get_status_tolerates_null_speed_and_progress(monkeypatch):
    """Torbox sends null speed/eta before a torrent starts; that is not a failure."""
    client = _client(monkeypatch)
    client._ensure_state("42")
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.requests.get",
        MagicMock(
            return_value=_response(
                {
                    "success": True,
                    "data": {
                        "download_state": "downloading",
                        "progress": None,
                        "download_speed": None,
                        "eta": None,
                        "name": "A Book",
                    },
                }
            )
        ),
    )

    status = client.get_status("42")

    assert status.state == DownloadState.DOWNLOADING
    assert status.progress == 0.0
    assert status.download_speed is None


def test_get_status_treats_unknown_torrent_as_queued(monkeypatch):
    """Torbox briefly returns no record for a just-added torrent."""
    client = _client(monkeypatch)
    client._ensure_state("42")
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.requests.get",
        MagicMock(return_value=_response({"success": True, "data": None})),
    )

    status = client.get_status("42")

    assert status.state == DownloadState.QUEUED
    assert status.complete is False


def test_get_status_starts_download_when_files_are_present_without_ready_state(monkeypatch):
    """`download_present` is Torbox's authoritative 'files are fetchable' signal."""
    client = _client(monkeypatch)
    state = client._ensure_state("42")
    monkeypatch.setattr(
        "shelfmark.download.clients.torbox.requests.get",
        MagicMock(
            return_value=_response(
                {
                    "success": True,
                    "data": {
                        "download_state": "uploading",
                        "download_present": True,
                        "files": [{"id": 7, "name": "book.epub"}],
                    },
                }
            )
        ),
    )
    started_with = []
    monkeypatch.setattr(
        client, "_maybe_start_download_thread", lambda *args: started_with.extend(args)
    )

    status = client.get_status("42")

    assert status.state == DownloadState.DOWNLOADING
    assert started_with == [state, [{"id": 7, "name": "book.epub"}]]
