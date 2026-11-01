"""VMM huge-page policy: request normalization, per-scope defaults, ordering.

Covers the host-side invariants without needing an NPU allocation:

  1. request normalization (``normalize_vmm_page_size``) and the two-scope
     resolution (``resolve_vmm_page_size`` / ``env_vmm_page_size``):
     ``ASCEND_MOONEP_VMM_PAGE_SIZE`` governs MoonEP's dispatch/combine chunks
     and ``ASCEND_MOONEP_FRAMEWORK_VMM_PAGE_SIZE`` the training framework's
     parameters (falling back to the former), while an explicit request always
     wins;
  2. the Buffer-level dedup default (``api.buffer_vmm_page_size``): server-level
     dedup defaults to 1 GiB pages, rank-level dedup to 2 MiB;
  3. descriptor ordering for a heap that mixes 1 GiB and 2 MiB pages.
     ACLSHMEM validates each descriptor's cumulative ``segment_offset``
     against *that buffer's* granularity (``validate_buffer_mapping``), so
     ``symmetric_descriptors()`` must emit large-page ranges first.

The actual allocation path is exercised by ``test_buffer_vmm.py``: ``Buffer``
applies one resolved page policy to payload chunks and the metadata/sync/op-
workspace arena alike.

Run:
    python -m pytest -s test_npu/buffer_test/test_buffer_vmm_page_size.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from npu_patch import buffer_c
from npu_patch.buffer_c import (
    VMM_PAGE_SIZE_1G,
    VMM_PAGE_SIZE_2M,
    VMM_SCOPE_FRAMEWORK,
    VMM_SCOPE_MOONEP,
    VmmAllocation,
    env_vmm_page_size,
    normalize_vmm_page_size,
    register_symmetric_descriptor,
    reset_symmetric_descriptors,
    resolve_vmm_page_size,
    symmetric_descriptors,
)

GIB = 1 << 30
MIB = 1 << 20


@pytest.mark.parametrize(
    "requested,expected",
    [
        (None, "default"),
        ("", "default"),
        ("default", "default"),
        ("1g", "1g"),
        ("1G", "1g"),
        (" 2M ", "2m"),
        ("2m", "2m"),
    ],
)
def test_normalize_page_size(requested, expected):
    assert normalize_vmm_page_size(requested) == expected


@pytest.mark.parametrize("bad", ["512k", "2g", "huge", "1g,2m"])
def test_normalize_page_size_rejects_unknown(bad):
    with pytest.raises(ValueError):
        normalize_vmm_page_size(bad)


def test_normalize_page_size_rejects_non_str():
    with pytest.raises(TypeError):
        normalize_vmm_page_size(1)


@pytest.fixture
def clean_page_size_env(monkeypatch):
    """No scope override leaks in from the launching shell."""
    monkeypatch.delenv("ASCEND_MOONEP_VMM_PAGE_SIZE", raising=False)
    monkeypatch.delenv("ASCEND_MOONEP_FRAMEWORK_VMM_PAGE_SIZE", raising=False)


def test_resolve_precedence(clean_page_size_env, monkeypatch):
    # Nothing requested, nothing exported: the caller's default (2 MiB normally).
    assert resolve_vmm_page_size() == VMM_PAGE_SIZE_2M
    assert resolve_vmm_page_size(default=VMM_PAGE_SIZE_1G) == VMM_PAGE_SIZE_1G
    # The moonep-scope variable overrides the default, and is the framework
    # scope's fallback.
    monkeypatch.setenv("ASCEND_MOONEP_VMM_PAGE_SIZE", "1g")
    assert resolve_vmm_page_size() == VMM_PAGE_SIZE_1G
    assert resolve_vmm_page_size(scope=VMM_SCOPE_FRAMEWORK) == VMM_PAGE_SIZE_1G
    # An explicit per-allocation request always wins over the environment.
    assert resolve_vmm_page_size("2m") == VMM_PAGE_SIZE_2M


def test_framework_scope_is_independent(clean_page_size_env, monkeypatch):
    monkeypatch.setenv("ASCEND_MOONEP_VMM_PAGE_SIZE", "2m")
    monkeypatch.setenv("ASCEND_MOONEP_FRAMEWORK_VMM_PAGE_SIZE", "1g")
    assert env_vmm_page_size(VMM_SCOPE_MOONEP) == VMM_PAGE_SIZE_2M
    assert env_vmm_page_size(VMM_SCOPE_FRAMEWORK) == VMM_PAGE_SIZE_1G
    # MoonEP chunks follow the moonep variable, framework params their own.
    assert resolve_vmm_page_size() == VMM_PAGE_SIZE_2M
    assert resolve_vmm_page_size(scope=VMM_SCOPE_FRAMEWORK) == VMM_PAGE_SIZE_1G
    assert resolve_vmm_page_size(
        "2m", scope=VMM_SCOPE_FRAMEWORK) == VMM_PAGE_SIZE_2M


def test_env_page_size_normalizes_and_rejects(clean_page_size_env, monkeypatch):
    monkeypatch.setenv("ASCEND_MOONEP_FRAMEWORK_VMM_PAGE_SIZE", " 1G ")
    assert env_vmm_page_size(VMM_SCOPE_FRAMEWORK) == VMM_PAGE_SIZE_1G
    monkeypatch.setenv("ASCEND_MOONEP_FRAMEWORK_VMM_PAGE_SIZE", "512k")
    with pytest.raises(ValueError):
        env_vmm_page_size(VMM_SCOPE_FRAMEWORK)
    with pytest.raises(ValueError):
        resolve_vmm_page_size(scope=VMM_SCOPE_FRAMEWORK)


def test_unknown_scope_rejected(clean_page_size_env):
    with pytest.raises(ValueError):
        env_vmm_page_size("gpu")


def test_buffer_page_size_dedup_defaults(clean_page_size_env, monkeypatch):
    from npu_patch.api import buffer_vmm_page_size

    # Rank-level dedup keeps the 2 MiB default.
    assert buffer_vmm_page_size("rank") == VMM_PAGE_SIZE_2M
    assert buffer_vmm_page_size("rank", "1g") == VMM_PAGE_SIZE_1G
    # Server-level dedup defaults to 1 GiB, switchable back to 2 MiB.
    assert buffer_vmm_page_size("server") == VMM_PAGE_SIZE_1G
    assert buffer_vmm_page_size("server", "2m") == VMM_PAGE_SIZE_2M
    # The framework scope never inherits the server-dedup default.
    assert resolve_vmm_page_size(scope=VMM_SCOPE_FRAMEWORK) == VMM_PAGE_SIZE_2M
    # A process-wide override beats the dedup-domain default.
    monkeypatch.setenv("ASCEND_MOONEP_VMM_PAGE_SIZE", "2m")
    assert buffer_vmm_page_size("server") == VMM_PAGE_SIZE_2M
    with pytest.raises(ValueError):
        buffer_vmm_page_size("board")


@pytest.fixture
def clean_registry():
    if buffer_c._SHMEM_BACKING_LOCKED:
        pytest.skip("buffer-backed shmem heap already initialized in this process")
    reset_symmetric_descriptors()
    yield
    # Fake allocations own no VA/handle: mark them destroyed so the registry
    # reset does not try to release anything.
    for allocation in list(buffer_c._VMM_REGISTRY):
        allocation._destroyed = True
    reset_symmetric_descriptors()


def _fake_allocation(va, size, granularity, policy):
    """A registry entry with no real VA/handle (sorting/validation only)."""
    return VmmAllocation(va, 0, 0, size, page_size=policy,
                         granularity=granularity)


def test_mixed_page_size_descriptors_large_pages_first(clean_registry):
    # Creation order deliberately interleaves the two page sizes: the heap
    # layout must still put every 1 GiB range before the 2 MiB ranges.
    small_first = _fake_allocation(0x00200000, 2 * MIB, 2 * MIB,
                                   VMM_PAGE_SIZE_2M)
    large = _fake_allocation(0x40000000, GIB, GIB, VMM_PAGE_SIZE_1G)
    small_second = _fake_allocation(0x00600000, 6 * MIB, 2 * MIB,
                                    VMM_PAGE_SIZE_2M)
    for allocation in (small_first, large, small_second):
        register_symmetric_descriptor(allocation)

    descriptors = symmetric_descriptors()
    assert [d[0] for d in descriptors] == [
        large.source_va, small_first.source_va, small_second.source_va,
    ]

    # Replay ACLSHMEM's per-descriptor offset check on the emitted order.
    granularity = {
        large.source_va: GIB,
        small_first.source_va: 2 * MIB,
        small_second.source_va: 2 * MIB,
    }
    offset = 0
    for addr, size, mem_handle, fabric in descriptors:
        gran = granularity[addr]
        assert offset % gran == 0, (
            f"segment_offset {offset} not aligned to {gran} for VA {addr:#x}")
        assert size % gran == 0
        assert mem_handle == 0 and fabric == b""
        offset += size


def test_same_granularity_keeps_creation_order(clean_registry):
    first = _fake_allocation(0x40000000, GIB, GIB, VMM_PAGE_SIZE_1G)
    second = _fake_allocation(0x80000000, GIB, GIB, VMM_PAGE_SIZE_1G)
    for allocation in (first, second):
        register_symmetric_descriptor(allocation)

    descriptors = symmetric_descriptors()
    assert [d[0] for d in descriptors] == [first.source_va, second.source_va]


def test_reset_clears_registry(clean_registry):
    register_symmetric_descriptor(
        _fake_allocation(0x00200000, 2 * MIB, 2 * MIB, VMM_PAGE_SIZE_2M))
    assert len(symmetric_descriptors()) == 1
    for allocation in list(buffer_c._VMM_REGISTRY):
        allocation._destroyed = True
    reset_symmetric_descriptors()
    assert symmetric_descriptors() == ()