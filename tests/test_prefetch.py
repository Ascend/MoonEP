"""
Remote expert prefetch correctness test.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_prefetch.py

Coverage axes:
  - shapes: single tile, multi tile, rectangular tiles, production
    7168x3072 (>2^31-element local expert offsets at local ids >= 98).
  - plans: hand-picked (holes, duplicates, self/remote slots, multiple
    owners), empty, asymmetric send/receive, random multi-seed.
  - pipeline: tiles-per-CTA both below and far above the smem stage count,
    so the producer/consumer state and the deferred consumer_release wrap
    many times within one launch.
  - fused inputs: three independent weight pools and optional three scale
    pools share one launch.
  - lifecycle: collective completion and repeated launches on one Buffer.
"""

import os

import pytest
import torch
import torch.distributed as dist

from moonep import Buffer
from moonep.buffer import create_nvl_dist_tensor, pad_to_granularity
from moonep.prefetch import launch_prefetch
from tests.kernel_test_utils import local_device_index


K3_H, K3_I = 3584, 3072


def _random_experts(epn, num_slots, seed):
    """Random local ids with ~40% holes and possible duplicates."""
    g = torch.Generator()
    g.manual_seed(seed)
    live = torch.rand(num_slots, generator=g) < 0.6
    ids = torch.randint(0, epn, (num_slots,), generator=g, dtype=torch.int32)
    return torch.where(live, ids, torch.full_like(ids, -1)).tolist()


# Contain both aligned cases and rank padding cases without enlarging epn.
CASES = [
    {
        "name": "single_tile_one_buffer",
        "epn": 64,  # 2 MiB bf16 rank chunk at 128x128.
        "H": 128,
        "Hp": 128,
        "num_sms": 1,
        "owner_offset": 1,
        "experts": lambda epn: [epn - 1],
    },
    {
        "name": "multi_tile_with_unused_slot",
        "epn": 16,  # 2 MiB bf16 rank chunk at 256x256.
        "H": 256,
        "Hp": 256,
        "num_sms": 8,
        "owner_offset": 1,
        "pattern": "self_remote",
        "experts": lambda epn: [0, epn - 1, epn // 2, -1],
    },
    {
        "name": "rectangular_wide_tiles_duplicates",
        "epn": 32,  # 6 MiB bf16 rank chunk at 256x384.
        "H": 256,
        "Hp": 384,
        "num_sms": 16,
        "owner_offset": 2,
        "pattern": "multiple_owners",
        "experts": lambda epn: [epn - 1, 0, epn // 2, epn // 2, -1],
    },
    {
        "name": "more_sms_than_tiles",
        "epn": 16,
        "H": 512,
        "Hp": 128,
        "num_sms": 32,
        "owner_offset": 3,
        "pattern": "asymmetric",
        "experts": lambda epn: [epn - 1, 1, -1],
    },
    # All -1: total_tiles == 0. The kernel must pass the warp split and the
    # final cp_async_bulk_wait_group(0) without issuing a single load/store,
    # leaving every sentinel intact.
    {
        "name": "empty_plan",
        "epn": 64,  # 2 MiB bf16 rank chunk at 128x128.
        "H": 128,
        "Hp": 128,
        "num_sms": 8,
        "owner_offset": 1,
        "repeats": 2,
        "experts": lambda epn: [-1, -1, -1, -1],
    },
    # 96 tiles on 2 CTAs = 48 tiles per CTA >> stages (<= 6): the load
    # pipeline and the deferred consumer_release wrap many times within one
    # launch. The other cases give each CTA at most ~2 tiles and never wrap.
    {
        "name": "pipeline_wraparound",
        "epn": 16,
        "H": 512,
        "Hp": 384,
        "num_sms": 2,
        "owner_offset": 1,
        "repeats": 3,
        "experts": lambda epn: [3, epn - 1, 3, 0, epn // 2, epn - 1, 7, 1],
    },
    # Production-like expert shape where local_expert * H * Hp crosses 2^31
    # elements (local ids >= 98 at 7168x3072). Catches 32-bit offset
    # arithmetic anywhere in the TMA descriptor / addressing chain.
    {
        "name": "i64_offset_7168x3072",
        "epn": 104,
        "H": 7168,
        "Hp": 3072,
        "num_sms": 32,
        "owner_offset": 1,
        "experts": lambda epn: [epn - 1, 98, -1],
    },
    # ---- Quantized experts -------------------------------------------------
    {
        "name": "mxfp4_packed_k3_gate_up",
        "epn": 8,
        "H": 2 * K3_I, 
        "Hp": K3_H // 2,
        "num_sms": 32,
        "owner_offset": 1,
        "dtype": torch.uint8,
        "experts": lambda epn: [epn - 1, 0, -1],
    },
    {
        "name": "mxfp4_packed_k3_down",
        "epn": 8,
        "H": K3_H, 
        "Hp": K3_I // 2, 
        "num_sms": 32,
        "owner_offset": 2,
        "pattern": "multiple_owners",
        "dtype": torch.uint8,
        "experts": lambda epn: [1, epn - 1, 1, -1],
    },
    # ue8m0 block scales, re-tiled. Hp=4096 pads the scale extent so an epn=8
    # rank chunk is VMM-granularity aligned.
    {
        "name": "mxfp4_sf_k3_gate_up_retiled",
        "epn": 8,
        "H": 128,
        "Hp": 4096,  # 4 MiB uint8 rank chunk at epn=8.
        "num_sms": 16,
        "owner_offset": 1,
        "dtype": torch.uint8,
        "scales": True,
        "experts": lambda epn: [epn - 1, 2, 2, -1],
    },
    {
        "name": "mxfp4_sf_k3_down_retiled",
        "epn": 8,
        "H": 128,
        "Hp": 4096,  # 4 MiB uint8 rank chunk at epn=8.
        "num_sms": 16,
        "owner_offset": 3,
        "dtype": torch.uint8,
        "scales": True,
        "experts": lambda epn: [0, epn - 1, -1],
    },
    # Random plans (holes, duplicates, uneven coverage) with 16 active slots.
    {
        "name": "random_plan_s1",
        "epn": 32,
        "H": 256,
        "Hp": 128,
        "num_sms": 16,
        "owner_offset": 2,
        "experts": lambda epn: _random_experts(epn, 16, seed=1),
    },
    {
        "name": "random_plan_s2",
        "epn": 32,
        "H": 256,
        "Hp": 128,
        "num_sms": 16,
        "owner_offset": 1,
        "experts": lambda epn: _random_experts(epn, 16, seed=2),
    },
    {
        "name": "random_plan_s3",
        "epn": 32,
        "H": 256,
        "Hp": 128,
        "num_sms": 16,
        "owner_offset": 3,
        "experts": lambda epn: _random_experts(epn, 16, seed=3),
    },
    {
        "name": "strided_small_epn",
        "epn": 3,
        "H": 128,
        "Hp": 128,
        "num_sms": 1,
        "owner_offset": 1,
        "experts": lambda epn: [2, 0, -1],
    },
    {
        "name": "strided_non_row_aligned_mixed_pools",
        "epn": 3,
        "H": 384,
        "Hp": 640,
        "num_sms": 2,
        "owner_offset": 1,
        "pattern": "multiple_owners",
        "mixed_strides": True,
        "offset_bytes": 128,
        "repeats": 3,
        "experts": lambda epn: [2, 0, 2],
    },
    {
        "name": "strided_same_shape_different_stride",
        "epn": 3,
        "H": 384,
        "Hp": 640,
        "num_sms": 2,
        "owner_offset": 1,
        "extra_rank_pages": 1,
        "experts": lambda epn: [2, 0, -1],
    },
    {
        "name": "strided_empty_plan",
        "epn": 3,
        "H": 128,
        "Hp": 128,
        "num_sms": 8,
        "owner_offset": 1,
        "repeats": 2,
        "experts": lambda epn: [-1, -1, -1],
    },
    {
        "name": "strided_epn1",
        "epn": 1,
        "H": 128,
        "Hp": 128,
        "num_sms": 1,
        "owner_offset": 1,
        "repeats": 2,
        "experts": lambda epn: [0],
    },
    {
        "name": "strided_weights_fp32_scales",
        "epn": 3,
        "H": 256,
        "Hp": 256,
        "num_sms": 2,
        "owner_offset": 1,
        "pattern": "multiple_owners",
        "dtype": torch.uint8,
        "scales": True,
        "scale_dtype": torch.float32,
        "scale_shape": (64, 64),
        "mixed_strides": True,
        "offset_bytes": 128,
        "repeats": 3,
        "experts": lambda epn: [2, 0, -1],
    },
]


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_prefetch requires torchrun with at least 2 GPUs")

    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(local_device_index())
    R = dist.get_world_size()
    if R < 2:
        if owns_process_group:
            dist.destroy_process_group()
        pytest.skip("test_prefetch requires at least 2 GPUs")

    yield rank, R

    dist.barrier(device_ids=[local_device_index()])
    if owns_process_group:
        dist.destroy_process_group()


def _fill_random(shape, dtype, gen):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
    info = torch.iinfo(dtype)
    return torch.randint(
        info.min, info.max, shape, dtype=dtype, device="cuda", generator=gen
    )


def _sentinel_for(dtype):
    if dtype.is_floating_point:
        return -123.0
    return 0xAB if dtype == torch.uint8 else 123


def _remote_owner(rank, R, owner_offset):
    owner_offset %= R
    if owner_offset == 0:
        owner_offset = 1
    return (rank + owner_offset) % R


def _expert_seed(owner, local_expert, epn, H, Hp, tensor_id):
    return (
        2026 + owner * 100_003 + local_expert + epn * 13 + H * 17 + Hp * 19
        + tensor_id * 1_000_003
    )


def _expert_value(owner, local_expert, epn, H, Hp, dtype, tensor_id):
    gen = torch.Generator(device="cuda").manual_seed(
        _expert_seed(owner, local_expert, epn, H, Hp, tensor_id)
    )
    return _fill_random((H, Hp), dtype, gen)


def _make_local_experts(rank, epn, H, Hp, dtype, tensor_id):
    local = torch.empty(epn, H, Hp, dtype=dtype, device="cuda")
    for local_expert in range(epn):
        local[local_expert].copy_(
            _expert_value(rank, local_expert, epn, H, Hp, dtype, tensor_id)
        )
    return local


def _make_plan(case, R, epn):
    local_ids = case["experts"](epn)
    assert len(local_ids) <= epn
    plan = torch.full((R, epn), -1, dtype=torch.int32)
    pattern = case.get("pattern", "ring")
    for dst in range(R):
        for b, local_expert in enumerate(local_ids):
            if local_expert < 0:
                continue
            if pattern == "ring":
                owner = _remote_owner(dst, R, case["owner_offset"])
            elif pattern == "self_remote":
                owner = dst if b % 2 == 0 else _remote_owner(
                    dst, R, case["owner_offset"]
                )
            elif pattern == "multiple_owners":
                owner = (dst + 1 + b) % R
            elif pattern == "asymmetric":
                if dst != 0:
                    continue
                owner = 1
            else:
                raise ValueError(f"unknown prefetch pattern: {pattern}")
            plan[dst, b] = owner * epn + local_expert
    return plan.to("cuda")


def _make_pool(rank, R, epn, H, Hp, dtype, case, tensor_id):
    offset_bytes = case.get("offset_bytes", 0)
    extra_pages = tensor_id if case.get("mixed_strides", False) else case.get("extra_rank_pages", 0)
    stride_bytes = pad_to_granularity(epn * H * Hp * dtype.itemsize + offset_bytes)
    stride_bytes += extra_pages * pad_to_granularity(1)
    rank_stride = stride_bytes // dtype.itemsize
    backing = create_nvl_dist_tensor([rank_stride], dtype, rank, R).view(R, rank_stride)
    pool = backing.as_strided(
        (R, epn, H, Hp),
        (rank_stride, H * Hp, Hp, 1),
        storage_offset=offset_bytes // dtype.itemsize,
    )
    return pool, backing


def run_case(rank, R, case):
    epn = case["epn"]
    H = case["H"]
    Hp = case["Hp"]
    num_sms = case["num_sms"]
    dtype = case.get("dtype", torch.bfloat16)
    dev = "cuda"

    assert H % 128 == 0 and Hp % 128 == 0, \
        f"{case['name']}: H/Hp must be multiples of 128"

    local_weights = tuple(
        _make_local_experts(rank, epn, H, Hp, dtype, tensor_id)
        for tensor_id in range(3)
    )
    prefetch_buffers, weight_backings = zip(*(
        _make_pool(rank, R, epn, H, Hp, dtype, case, tensor_id)
        for tensor_id in range(3)
    ))
    if case.get("scales", False):
        scale_dtype = case.get("scale_dtype", dtype)
        scale_H, scale_Hp = case.get("scale_shape", (H, Hp))
        local_scales = tuple(
            _make_local_experts(rank, epn, scale_H, scale_Hp, scale_dtype, tensor_id)
            for tensor_id in range(3, 6)
        )
        scale_buffers, scale_backings = zip(*(
            _make_pool(rank, R, epn, scale_H, scale_Hp, scale_dtype, case, tensor_id)
            for tensor_id in range(3, 6)
        ))
        scales = tuple(zip(local_scales, scale_buffers, strict=True))
    else:
        scale_buffers = ()
        scale_backings = ()
        scales = None
    experts_to_copy = _make_plan(case, R, epn)
    buffer = Buffer(
        S=1,
        H=H,
        K=1,
        E=epn * R,
        num_ep_ranks=R,
        num_sms=num_sms,
        token_padding=1,
        explicitly_destroy=True,
    )
    try:
        ctx = buffer._require_ctx()
        ok = True
        for _ in range(case.get("repeats", 1)):
            for backing in (*weight_backings, *scale_backings):
                backing[rank].fill_(_sentinel_for(backing.dtype))
            torch.cuda.synchronize()
            dist.barrier(device_ids=[local_device_index()])

            launch_prefetch(
                local_weights,
                prefetch_buffers,
                experts_to_copy,
                rank=rank,
                num_sms=num_sms,
                meta_buf=ctx["meta_buf"],
                meta_stride=int(ctx["meta_chunk_padded"]),
                barrier_off=int(ctx["BARRIER_OFF"]),
                grid_sync_bar=ctx["grid_sync_bar"],
                scales=scales,
            )
            torch.cuda.synchronize()

            for tensor_id, prefetch_buffer in enumerate(
                (*prefetch_buffers, *scale_buffers)
            ):
                sentinel = _sentinel_for(prefetch_buffer.dtype)
                for b in range(epn):
                    expert = int(experts_to_copy[rank, b].item())
                    if expert < 0:
                        expected = torch.full_like(
                            prefetch_buffer[rank, b], sentinel
                        )
                    else:
                        owner, local_expert = divmod(expert, epn)
                        expected = _expert_value(
                            owner, local_expert, epn, *prefetch_buffer.shape[2:],
                            prefetch_buffer.dtype, tensor_id
                        )
                    if not torch.equal(prefetch_buffer[rank, b], expected):
                        ok = False
                        diff = (
                            prefetch_buffer[rank, b].float() - expected.float()
                        ).abs().max().item()
                        print(
                            f"[rank {rank}] {case['name']} mismatch: "
                            f"tensor={tensor_id}, slot={b}, expert={expert}, "
                            f"max_abs_diff={diff}"
                        )
            for pool, backing in zip(
                (*prefetch_buffers, *scale_buffers), (*weight_backings, *scale_backings), strict=True
            ):
                begin = pool.storage_offset()
                end = begin + pool[rank].numel()
                sentinel = _sentinel_for(backing.dtype)
                padding_ok = bool((backing[rank, :begin] == sentinel).all()) and bool(
                    (backing[rank, end:] == sentinel).all()
                )
                if not padding_ok:
                    ok = False
                    print(f"[rank {rank}] {case['name']}: pool padding was modified")
    finally:
        buffer.destroy()

    ok_tensor = torch.tensor([int(ok)], dtype=torch.int32, device=dev)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    assert ok_tensor.item() == 1, f"{case['name']} failed"

    if rank == 0:
        print(
            f"  [PASS] {case['name']}: epn={epn}, "
            f"H={H}, Hp={Hp}, dtype={dtype}, num_sms={num_sms}"
        )

    dist.barrier(device_ids=[local_device_index()])


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_prefetch_case(dist_env, case):
    rank, R = dist_env
    if rank == 0:
        print(f"[test_prefetch] Running case={case['name']} with R={R}")
    run_case(rank, R, case)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
