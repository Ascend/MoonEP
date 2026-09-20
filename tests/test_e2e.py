"""
End-to-end smoke test for the public Buffer dispatch/combine API.

Verifies sync/async dispatch, separate prefetch, combine behavior, and the
public plan-reuse dispatch path.

Run with:
    torchrun --nproc_per_node=8 -m pytest tests/test_e2e.py
"""

import torch
import torch.distributed as dist

from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
from moonep import Buffer, MoonEPCommPlan
from tests.kernel_test_utils import (
    clone_dedup_plan_fields,
    local_device_index,
    dedup_plan_fields_equal,
    dedup_plan_semantic_errors,
)


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(local_device_index())
    return rank, dist.get_world_size()


def make_inputs(rank, S, H, K, E, seed=0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed + rank)
    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g)
    weights = torch.rand(S, K, dtype=torch.float32, device=dev, generator=g)
    topk = torch.randint(0, E, (S, K), dtype=torch.int32, device=dev, generator=g)
    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    return hidden, weights, topk, tpe


def weight_base(experts, H, Hp, offset, dev):
    expert = experts.to(dtype=torch.float32, device=dev).view(-1, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return (offset + expert * 17.0 + row * 0.125 + col * 0.0078125).to(
        torch.bfloat16
    )


def make_dist_pool(rank, R, epn, H, Hp, dtype):
    padded_epn = pad_dim0_for_alignment([epn, H, Hp], dtype)
    assert padded_epn == epn, (
        f"public pool chunk [epn={epn}, H={H}, H'={Hp}] must be VMM aligned"
    )
    return create_nvl_dist_tensor([epn, H, Hp], dtype, rank, R).view(
        R, epn, H, Hp
    )


def make_prefetch_args(rank, R, epn, H, Hp):
    dev = "cuda"
    local_ids = torch.arange(rank * epn, (rank + 1) * epn, device=dev)
    args = {}
    for name, offset in (("gate", 0.0), ("up", 1.0), ("down", 2.0)):
        args[f"local_{name}_weight"] = weight_base(
            local_ids, H, Hp, offset, dev
        ).contiguous()
        pool = make_dist_pool(rank, R, epn, H, Hp, torch.bfloat16)
        pool[rank].zero_()
        args[f"{name}_prefetch_buffer"] = pool
    return args


def assert_prefetched(rank, H, Hp, prefetch_args, experts_to_copy):
    assert experts_to_copy.dim() == 2
    for b in range(experts_to_copy.shape[1]):
        expert = int(experts_to_copy[rank, b].item())
        if expert < 0:
            continue
        expert_id = torch.tensor([expert], device="cuda")
        for name, offset in (("gate", 0.0), ("up", 1.0), ("down", 2.0)):
            pool = prefetch_args[f"{name}_prefetch_buffer"]
            expected = weight_base(expert_id, H, Hp, offset, pool.device)[0]
            assert torch.equal(pool[rank, b], expected), (
                f"{name} prefetch buffer {b} does not match expert {expert}"
            )


def fill_prefetch_slots(rank, prefetch_args, value):
    for name in ("gate", "up", "down"):
        prefetch_args[f"{name}_prefetch_buffer"][rank].fill_(value)


def assert_prefetch_slots_equal(rank, prefetch_args, value):
    for name in ("gate", "up", "down"):
        slots = prefetch_args[f"{name}_prefetch_buffer"][rank]
        assert torch.equal(slots, torch.full_like(slots, value)), (
            f"{name} prefetch slots changed without prefetch_weight"
        )


def assert_dedup_plan_semantic_equal(actual, expected):
    errors = dedup_plan_semantic_errors("dedup plan", actual, expected)
    assert not errors, "; ".join(errors[:5])


def grad_base(rank, epn, H, Hp, offset, dev):
    expert = (
        rank * epn + torch.arange(epn, dtype=torch.float32, device=dev)
    ).view(epn, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return offset + expert * 17.0 + row * 0.125 + col * 0.0078125


def reduce_base(R, epn, H, Hp, offset, dev):
    src = torch.arange(R, dtype=torch.float32, device=dev).view(R, 1, 1, 1)
    slot = torch.arange(epn, dtype=torch.float32, device=dev).view(1, epn, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, 1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, 1, Hp)
    return offset + (src + 1.0) * 101.0 + slot * 11.0 + row * 0.03125 + col * 0.00390625


def make_grad_reduce_args(rank, R, epn, H, Hp, offsets):
    dev = "cuda"
    args = {}
    for i, name in enumerate(("gate", "up", "down")):
        args[f"local_{name}_grad"] = grad_base(
            rank, epn, H, Hp, offsets[i], dev
        ).contiguous()
        pool = make_dist_pool(rank, R, epn, H, Hp, torch.float32)
        pool[rank].copy_(reduce_base(R, epn, H, Hp, offsets[i + 3], dev)[rank])
        args[f"{name}_reduce_buffer"] = pool
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_device_index()])
    return args


def expected_local_grad(rank, R, epn, H, Hp, grad_offset, reduce_offset, experts_to_copy):
    dev = "cuda"
    expected = grad_base(rank, epn, H, Hp, grad_offset, dev)
    reduce_vals = reduce_base(R, experts_to_copy.shape[1], H, Hp, reduce_offset, dev)
    local_start = rank * epn
    local_end = local_start + epn
    for src_rank in range(R):
        for b in range(experts_to_copy.shape[1]):
            expert = int(experts_to_copy[src_rank, b].item())
            if local_start <= expert < local_end:
                expected[expert - local_start].add_(reduce_vals[src_rank, b])
    return expected


def assert_grad_reduced(rank, R, epn, H, Hp, experts_to_copy, args, offsets):
    pairs = [
        ("gate", args["local_gate_grad"], args["gate_reduce_buffer"], offsets[0], offsets[3]),
        ("up", args["local_up_grad"], args["up_reduce_buffer"], offsets[1], offsets[4]),
        ("down", args["local_down_grad"], args["down_reduce_buffer"], offsets[2], offsets[5]),
    ]
    for name, local_grad, reduce_buffer, grad_offset, reduce_offset in pairs:
        expected = expected_local_grad(
            rank, R, epn, H, Hp, grad_offset, reduce_offset, experts_to_copy
        )
        assert torch.allclose(
            local_grad, expected, rtol=0.0, atol=1e-5
        ), f"{name} local grad reduce mismatch"

        original_reduce = reduce_base(
            R, experts_to_copy.shape[1], H, Hp, reduce_offset, reduce_buffer.device
        )
        for b in range(experts_to_copy.shape[1]):
            expert = int(experts_to_copy[rank, b].item())
            if expert >= 0:
                assert torch.equal(
                    reduce_buffer[rank, b],
                    torch.zeros_like(reduce_buffer[rank, b]),
                ), f"{name} consumed reduce slot ({rank}, {b}) was not cleared"
            else:
                assert torch.equal(
                    reduce_buffer[rank, b],
                    original_reduce[rank, b],
                ), f"{name} unused reduce slot ({rank}, {b}) changed"


def assert_raises_assertion(expected_substr, fn):
    try:
        fn()
    except AssertionError as exc:
        assert expected_substr in str(exc), (
            f"expected assertion containing {expected_substr!r}, got {exc!r}"
        )
    else:
        raise AssertionError(f"expected AssertionError containing {expected_substr!r}")


def test_e2e():
    rank, R = setup()
    # epn=8 keeps both bf16 weight and fp32 grad chunks VMM-granularity aligned.
    S, H, K, E = 256, 1024, 4, R * 8
    epn = E // R
    Hp = 128
    num_sms = 32
    buffer = Buffer(S, H, K, E, R, num_sms=num_sms)
    sync_prefetch_args = make_prefetch_args(rank, R, epn, H, Hp)
    async_prefetch_args = make_prefetch_args(rank, R, epn, H, Hp)
    reuse_prefetch_args = make_prefetch_args(rank, R, epn, H, Hp)
    no_prefetch_args = make_prefetch_args(rank, R, epn, H, Hp)

    hidden, weights, topk, tpe = make_inputs(rank, S, H, K, E)

    # --- Sync reference ---
    (h_sync, w_sync, cu_sync, plan_sync) = buffer.dispatch(
        hidden, weights, topk, tpe,
    )
    h_sync_cpu = h_sync.clone()
    w_sync_cpu = w_sync.clone() if w_sync is not None else None
    cu_sync_cpu = cu_sync.clone()
    # Snapshot the plan's tensors so a later dispatch can't mutate them.
    assert isinstance(plan_sync, MoonEPCommPlan)
    plan_snapshot = plan_sync.clone()
    buffer.prefetch_weight(plan=plan_snapshot, **sync_prefetch_args)
    torch.cuda.synchronize()
    assert_prefetched(rank, H, Hp, sync_prefetch_args, plan_snapshot.experts_to_copy)

    # --- Async variant (fresh inputs to avoid NVL aliasing with sync run) ---
    hidden2, weights2, topk2, tpe2 = make_inputs(rank, S, H, K, E)
    assert torch.equal(hidden, hidden2)
    assert torch.equal(weights, weights2)

    (h_a, w_a, cu_a, plan_a, _dispatch_event) = buffer.dispatch(
        hidden2, weights2, topk2, tpe2, async_finish=True,
    )
    prefetch_event = buffer.prefetch_weight(
        plan=plan_a, async_finish=True, **async_prefetch_args,
    )
    # Caller must explicitly wait before reading.
    prefetch_event.wait(torch.cuda.current_stream())
    h_a_snap = h_a.clone()
    w_a_snap = w_a.clone() if w_a is not None else None
    torch.cuda.synchronize()
    assert_prefetched(rank, H, Hp, async_prefetch_args, plan_a.experts_to_copy)

    # Compare
    assert torch.equal(h_sync_cpu, h_a_snap), "dispatch hidden mismatch"
    assert torch.equal(w_sync_cpu, w_a_snap), "dispatch weights mismatch"
    assert torch.equal(cu_sync_cpu, cu_a), "cu_seqlens mismatch"
    for name in ("dst", "experts_to_copy", "zero_fill_ranges", "remote_stats"):
        assert torch.equal(getattr(plan_snapshot, name), getattr(plan_a, name)), \
            f"plan.{name} mismatch"
    assert_dedup_plan_semantic_equal(plan_a, plan_snapshot)

    # --- Public plan-reuse path: planning is skipped, caller passes the
    # saved plan back. ---
    reuse_dedup_before = clone_dedup_plan_fields(plan_snapshot)
    (h_reuse, w_reuse, cu_reuse, plan_reuse) = buffer.dispatch(
        hidden, plan=plan_snapshot,
    )
    buffer.prefetch_weight(plan=plan_snapshot, **reuse_prefetch_args)
    torch.cuda.synchronize()
    assert torch.equal(h_reuse.clone(), h_sync_cpu), "plan-reuse hidden buffer mismatch"
    assert w_reuse is None, "plan-reuse hidden-only dispatch should not return weights buffer"
    assert cu_reuse is None, "plan-reuse path should skip planning outputs"
    assert plan_reuse is plan_snapshot, "plan-reuse should echo the input plan back"
    assert dedup_plan_fields_equal(plan_snapshot, reuse_dedup_before), \
        "plan-reuse dispatch should not rebuild or mutate dedup structures"
    assert_prefetched(rank, H, Hp, reuse_prefetch_args, plan_snapshot.experts_to_copy)

    # --- Public plan-reuse path without prefetch: same hidden scatter, but
    # prefetch slots must remain untouched. This matches backward redispatch.
    sentinel = 7.0
    fill_prefetch_slots(rank, no_prefetch_args, sentinel)
    no_prefetch_dedup_before = clone_dedup_plan_fields(plan_snapshot)
    (h_no_prefetch, w_no_prefetch, cu_no_prefetch, plan_no_prefetch) = buffer.dispatch(
        hidden, plan=plan_snapshot,
    )
    torch.cuda.synchronize()
    assert torch.equal(h_no_prefetch.clone(), h_sync_cpu), "no-prefetch hidden buffer mismatch"
    assert w_no_prefetch is None, "no-prefetch hidden-only dispatch should not return weights buffer"
    assert cu_no_prefetch is None, "no-prefetch plan-reuse path should skip planning outputs"
    assert dedup_plan_fields_equal(plan_snapshot, no_prefetch_dedup_before), \
        "no-prefetch plan-reuse dispatch should not rebuild or mutate dedup structures"
    assert plan_no_prefetch is plan_snapshot, "no-prefetch plan-reuse should echo the input plan back"
    assert_prefetch_slots_equal(rank, no_prefetch_args, sentinel)

    (h_no_weights, _, _, _) = buffer.dispatch(
        hidden, plan=plan_snapshot,
    )
    torch.cuda.synchronize()
    assert torch.equal(h_no_weights.clone(), h_sync_cpu), "dispatch without weights hidden mismatch"

    (h_no_prefetch_async, _, _, _, ev_no_prefetch) = buffer.dispatch(
        hidden, plan=plan_snapshot, async_finish=True,
    )
    ev_no_prefetch.wait(torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(
        h_no_prefetch_async.clone(), h_sync_cpu
    ), "async no-prefetch hidden mismatch"

    # --- Combine: compare sync vs async for the symmetric path ---
    # Use sync_dispatch's NVL population as the "expert output" proxy.
    # Re-run sync dispatch first to restore NVL buffer to a known state.
    h_for_combine, w_for_combine, _, _ = buffer.dispatch(
        hidden, weights, topk, tpe,
    )

    out_sync, _, _ = buffer.combine(plan=plan_snapshot, hidden_nvsh=h_for_combine)
    out_sync_snap = out_sync.clone()
    torch.cuda.synchronize()

    h_for_combine, w_for_combine, _, _ = buffer.dispatch(
        hidden, weights, topk, tpe,
    )
    out_with_weights, gathered_weights, _ = buffer.combine(
        plan=plan_snapshot,
        hidden_nvsh=h_for_combine,
        route_weights_nvs=w_for_combine,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_sync_snap, out_with_weights), "combine weight-gather hidden mismatch"
    assert torch.equal(gathered_weights, weights), "combine route_weights_sk gather mismatch"

    # --- zero_copy round-trip: dispatch returns NVL views, the "FFN" is an
    # identity on the shard, combine consumes the views in place. Must match
    # the zero_copy=False result bit-exactly.
    h_zc, w_zc, _, plan_zc = buffer.dispatch(
        hidden, weights, topk, tpe, zero_copy=True, router_weights_zero_copy=True,
    )
    assert h_zc.data_ptr() == buffer._require_ctx()['hidden_buf_local'].data_ptr(), \
        "dispatch(zero_copy=True) must return the NVL shard view"
    assert torch.equal(h_zc, h_for_combine), \
        "zero_copy dispatch hidden view mismatch vs copied tensor"
    assert torch.equal(w_zc, w_for_combine), \
        "zero_copy dispatch weights view mismatch vs copied tensor"
    out_zc, gathered_weights_zc, _ = buffer.combine(
        plan=plan_zc,
        hidden_nvsh=h_zc,
        route_weights_nvs=w_zc,
        zero_copy=True,
        router_weights_zero_copy=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_sync_snap, out_zc), "zero_copy combine hidden mismatch"
    assert torch.equal(gathered_weights_zc, weights), \
        "zero_copy combine route_weights_sk gather mismatch"
    assert_raises_assertion(
        "alias",
        lambda: buffer.combine(
            plan=plan_zc,
            hidden_nvsh=h_for_combine,
            zero_copy=True,
        ),
    )

    # --- Combine + sync grad_reduce ---
    h_for_combine, _, _, _ = buffer.dispatch(
        hidden, weights, topk, tpe,
    )
    grad_offsets = (1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
    grad_args = make_grad_reduce_args(rank, R, epn, H, Hp, grad_offsets)
    out_grad_sync, _, _ = buffer.combine(
        plan=plan_snapshot,
        hidden_nvsh=h_for_combine,
    )
    buffer.reduce_grad(plan=plan_snapshot, **grad_args)
    torch.cuda.synchronize()
    assert torch.equal(out_sync_snap, out_grad_sync), "sync grad_reduce combine output mismatch"
    assert_grad_reduced(rank, R, epn, H, Hp, plan_snapshot.experts_to_copy, grad_args, grad_offsets)

    # Re-run dispatch to re-populate NVL for async combine
    h_for_combine, _, _, _ = buffer.dispatch(
        hidden, weights, topk, tpe,
    )
    async_grad_offsets = (11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0)
    async_grad_args = make_grad_reduce_args(rank, R, epn, H, Hp, async_grad_offsets)
    out_async, _, _combine_ev = buffer.combine(
        plan=plan_snapshot,
        hidden_nvsh=h_for_combine,
        async_finish=True,
    )
    reduce_ev = buffer.reduce_grad(
        plan=plan_snapshot,
        async_finish=True,
        **async_grad_args,
    )
    reduce_ev.wait(torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert torch.equal(out_sync_snap, out_async), "combine output mismatch"
    assert_grad_reduced(
        rank, R, epn, H, Hp, plan_snapshot.experts_to_copy, async_grad_args, async_grad_offsets
    )

    if rank == 0:
        print("[test_e2e] PASS: public API sync/async, separate prefetch, and plan reuse match.")

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_e2e()
