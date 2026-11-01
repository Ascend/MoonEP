"""Server-dedup environment flags and the ``qp_count=2`` clamp.

The Buffer's dedup domain is fixed at construction, while the shmem bootstrap
runs later and holds no Buffer handle: ``_create_context`` publishes
``ASCEND_MOONEP_DEDUP_MODE`` / ``ASCEND_MOONEP_SERVER_DEDUP`` and
``ShmemRuntime.init_with_buffer`` / ``init_with_ip_port`` read them back to
enforce the server domain's QP requirement (the server kernels only support two
active QPs per peer pair).

Covered here, host-only (no device allocation, no collective):

  1. publishing / reading the dedup flags;
  2. the QP-count policy: unchanged for rank-level dedup and for non-UDMA
     bootstraps, forced to 2 with a warning under server-level dedup.

Run:
    python -m pytest -s test_npu/buffer_test/test_dedup_env_qp.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from npu_patch import shmem_runtime as sr


_ENV_NAMES = (
    sr.DEDUP_MODE_ENV,
    sr.SERVER_DEDUP_ENV,
    "ASCEND_MOONEP_QP_COUNT",
)


@pytest.fixture(autouse=True)
def clean_dedup_env(monkeypatch):
    """Every case starts without published flags or a QP override.

    ``publish_dedup_env`` writes ``os.environ`` directly (that is its job), so
    teardown also pops the variables instead of relying on ``monkeypatch`` undo
    records alone.
    """
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in _ENV_NAMES:
        os.environ.pop(name, None)


def test_publish_roundtrip():
    sr.publish_dedup_env("server")
    assert os.environ[sr.DEDUP_MODE_ENV] == "server"
    assert os.environ[sr.SERVER_DEDUP_ENV] == "1"
    assert sr.server_dedup_enabled() is True

    sr.publish_dedup_env("rank")
    assert os.environ[sr.DEDUP_MODE_ENV] == "rank"
    assert os.environ[sr.SERVER_DEDUP_ENV] == "0"
    assert sr.server_dedup_enabled() is False


def test_publish_rejects_unknown_mode():
    with pytest.raises(ValueError):
        sr.publish_dedup_env("board")


def test_unset_flags_mean_rank_dedup():
    assert sr.server_dedup_enabled() is False


def test_flag_wins_over_mode(monkeypatch):
    # A launcher may flip only the boolean flag; it stays authoritative.
    monkeypatch.setenv(sr.DEDUP_MODE_ENV, "rank")
    monkeypatch.setenv(sr.SERVER_DEDUP_ENV, "1")
    assert sr.server_dedup_enabled() is True

    monkeypatch.setenv(sr.DEDUP_MODE_ENV, "server")
    monkeypatch.setenv(sr.SERVER_DEDUP_ENV, "0")
    assert sr.server_dedup_enabled() is False


def test_mode_alone_enables_server_dedup(monkeypatch):
    monkeypatch.setenv(sr.DEDUP_MODE_ENV, "server")
    assert sr.server_dedup_enabled() is True


def test_qp_count_defaults_and_env(monkeypatch):
    # Unset: two QPs per peer pair (the historical default).
    assert sr.resolve_qp_count(True, 0) == 2
    # Explicit request wins over the environment.
    monkeypatch.setenv("ASCEND_MOONEP_QP_COUNT", "4")
    assert sr.resolve_qp_count(True, 0) == 4
    assert sr.resolve_qp_count(True, 1) == 1
    # QP configuration only exists on the UDMA path.
    assert sr.resolve_qp_count(False, 0) == 0
    assert sr.resolve_qp_count(False, 4) == 4


def test_rank_dedup_keeps_configured_qp_count():
    sr.publish_dedup_env("rank")
    assert sr.resolve_qp_count(True, 4) == 4
    assert sr.resolve_qp_count(True, 1) == 1


def test_server_dedup_forces_two_qps_with_warning():
    sr.publish_dedup_env("server")
    with pytest.warns(UserWarning, match="only supports qp_count=2"):
        assert sr.resolve_qp_count(True, 4) == 2
    with pytest.warns(UserWarning, match="only supports qp_count=2"):
        assert sr.resolve_qp_count(True, 1) == 2
    # Two is the supported value: nothing is corrected, nothing is warned.
    assert sr.resolve_qp_count(True, 2) == 2


def test_server_dedup_overrides_qp_env(monkeypatch):
    monkeypatch.setenv("ASCEND_MOONEP_QP_COUNT", "8")
    sr.publish_dedup_env("server")
    with pytest.warns(UserWarning, match="only supports qp_count=2"):
        assert sr.resolve_qp_count(True, 0) == 2


def test_server_dedup_leaves_non_udma_bootstrap_alone():
    sr.publish_dedup_env("server")
    # No QPs are created without enable_udma, so there is nothing to clamp.
    assert sr.resolve_qp_count(False, 0) == 0