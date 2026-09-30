#!/usr/bin/env python3
"""Joint UDMA test: dispatch -> combine without an expert/GMM stage.

The dispatch result stays in Buffer's symmetric storage and is passed to
``Buffer.combine`` in zero-copy mode. The combine launcher selects UDMA through
``ASCEND_MOONEP_COMBINE_USE_UDMA``. Therefore this test covers the framework-level
dispatch and UDMA combine path as one operation chain. Both dispatch routing
and the packed combine execution plan are generated on CPU; no device planner
or prepare_combine_plan kernel is launched.

Golden result (there is no expert computation between the two operations):

* every token is dispatched K times, so combined hidden is ``K * hidden``;
* every route weight is gathered back to its original ``[S, K]`` position.

Examples:

  python test_npu/urma/test_dispatch_combine_udma_joint.py --R 8 --S 16 --K 4 --H 128
  python test_npu/urma/test_dispatch_combine_udma_joint.py --R 8 --S 64 --K 8 --H 128 --no-weights
  python test_npu/urma/test_dispatch_combine_udma_joint.py --prod-shape --max-checks 64
  python test_npu/urma/test_dispatch_combine_udma_joint.py --R 8 --dedup-mode server

Multi-node: run once on each host with the same --R/--master-addr/--master-port,
--nproc-per-node 8, and a distinct --node-rank. This script spawns local workers;
do not wrap it in torchrun. The control plane uses HCCL; payload uses SHMEM UDMA.
"""
from __future__ import annotations


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem
import argparse
import struct
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Reuse exactly the same data and communication-plan generation as the UDMA
# dispatch single-op test.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_moonep_dispatch_udma_with_plan as dispatch_test


PRECISION_THR = 1e-2
UDMA_QP_COUNT = 2
PROD_R = 8
PROD_S = 4096
PROD_H = 7168
PROD_K = 16
PROD_E = 896


def validate_joint_args(args) -> None:
    dispatch_test.validate_args(args)
    if args.K > 16:
        raise ValueError("UDMA combine requires K <= 16")
    if args.max_checks < 0:
        raise ValueError("max-checks must be nonnegative")
    if args.R > 1024 or args.R * args.S * args.K >= 2**30:
        raise ValueError("combine requires R<=1024 and R*S*K<2^30")
    if not (args.R == 1 or (args.R % 2 == 0 and
            (args.R <= 8 or (args.R < 128 and args.R % 8 == 0) or
             (args.R >= 128 and args.R % 16 == 0)))):
        raise ValueError("unsupported combine world size")


def cpu_combine_layout(n, ranks):
    """Execution-plan v5 ABI; CPU tests compare offsets with the native header."""
    align = lambda size: (size + 511) & ~511
    result, offset = {}, 512
    for name, size in (('tx_rows', n * 16), ('source_counts', ranks * 64),
                       ('contributor', 8 * n * 16), ('relay_send', 8 * n * 8),
                       ('relay_ranges', 8 * ranks * 64), ('final_routes', n * 128)):
        result[name] = offset
        offset += align(size)
    result['bytes'] = offset
    return result


def make_cpu_combine_plan(args, rank, raw=None):
    """Build every payload table on CPU from the same golden route permutation.

    No SHMEM traffic or device planner is used. Each process independently
    reconstructs source-major prefixes, including remote ingress scratch offsets.
    """
    n, r, k = args.S * args.K, args.R, args.K
    server = args.dedup_mode == 'server'
    if raw is None:
        raw = dispatch_test.make_raw_destinations(args)
    raw = np.asarray(raw, dtype=np.int64).reshape(r, args.S, k)
    if raw.min() < 0 or raw.max() >= r * n:
        raise ValueError('golden route address out of range')
    q = cpu_combine_layout(n, r)
    storage = np.zeros(q['bytes'], dtype=np.uint8)
    struct.pack_into('<8Q', storage, 0, 0x4d45504300000005, n, args.H, k, r, rank, int(server), 0)
    def table(name, rows, columns):
        return np.ndarray((rows, columns), dtype='<u4', buffer=storage, offset=q[name])
    tx = table('tx_rows', n, 4)
    sc = table('source_counts', r, 16)
    contributor = table('contributor', 8 * n, 4)
    relay = table('relay_send', 8 * n, 2)
    ranges = table('relay_ranges', 8 * r, 16)
    final = table('final_routes', n, 32)
    scratch = [0] * r
    requests = [0] * 8
    tx_cursor = group_cursor = 0
    invalid = 0xffffffff
    for source in range(r):
        rows = raw[source]
        peers = rows // n
        domains = peers // 8 if server else peers
        primary = np.ones((args.S, k), dtype=bool)
        for j in range(1, k):
            primary[:, j] = np.all(domains[:, :j] != domains[:, j:j+1], axis=1)
        active = primary | ((peers // 8 == source // 8) if server else False)
        counts = np.bincount(peers[active], minlength=r)
        final_base = np.cumsum(counts) - counts
        final_cursor = final_base.copy()
        tx_begin, scratch_begin = tx_cursor, scratch[rank]
        request_begin = requests.copy()
        direct = 0
        # Remote sources without routes in this domain cannot add local TX,
        # contributor or relay entries. Only the local source needs every Final token.
        domain = rank // 8 if server else rank
        tokens = range(args.S) if source == rank else np.flatnonzero(np.any(domains == domain, axis=1))
        for token in tokens:
            groups = {}
            for j in range(k):
                groups.setdefault(int(domains[token, j]), []).append(j)
            # Build descriptors only for the domain containing this rank.
            members = groups.get(rank // 8 if server else rank, [])
            group, head, count = invalid, invalid, 1
            representative = int(peers[token, members[0]]) if members else -1
            local_source = server and source // 8 == rank // 8
            if len(members) > 1:
                if representative == rank:
                    group = group_cursor
                    group_cursor += 1
                if not local_source:
                    head, count = scratch[representative], len(members)
                    # Position of the representative in this rank's source-major TX stream.
                    own_before = sum(bool(active[token, j]) and int(peers[token, j]) == rank
                                     for j in range(members[0]))
                    for index, j in enumerate(members):
                        address = int(rows[token, j])
                        if representative == rank:
                            if head + index >= 8 * n:
                                raise ValueError('contributor capacity exceeded')
                            contributor[head + index] = (tx_cursor + own_before if index == 0 else 0,
                                count if index == 0 else 0, address if server else address % n, 0)
                        if address // n == rank:
                            lane = representative % 8 if server else 0
                            if requests[lane] >= n:
                                raise ValueError('relay request capacity exceeded')
                            relay[lane * n + requests[lane]] = (address % n, head + index)
                            requests[lane] += 1
                    scratch[representative] += count
            token_final = []
            for j in range(k):
                if not active[token, j]:
                    continue
                peer, loff = int(peers[token, j]), int(rows[token, j] % n)
                reduced = peer == representative and len(members) > 1 and not local_source
                if peer == rank:
                    if tx_cursor >= n:
                        raise ValueError('TX row capacity exceeded')
                    tx[tx_cursor] = (loff, group if primary[token, j] else invalid,
                                     head if reduced else invalid, count if reduced else 1)
                    tx_cursor += 1
                    direct += not reduced
                if source == rank:
                    # Rank-local dedup must consume its reduced self result, not the raw representative.
                    local_raw = peer == rank and not reduced
                    token_final.append((0x80000000 | loff) if local_raw else int(final_cursor[peer]))
                final_cursor[peer] += 1
            if source == rank:
                final[token, 0] = len(token_final)
                final[token, 1:1+len(token_final)] = token_final
        sc[source, :7] = (tx_cursor - tx_begin, int(final_base[rank]), tx_begin, direct,
                           scratch_begin, scratch[rank] - scratch_begin, 0)
        for lane in range(8 if server else 1):
            ranges[lane * r + source, :2] = (request_begin[lane], requests[lane] - request_begin[lane])
    return torch.from_numpy(storage), q


def attach_cpu_combine_plan(ctx, plan, storage):
    """Upload a CPU snapshot and bind production ownership/version/event guards."""
    from npu_patch.planning import CombineExecutionPlan, _key, _routes, _stamp
    from npu_patch import ShmemRuntime
    torch.npu.synchronize()
    ShmemRuntime.barrier()
    # Fresh collective preparation: no payload is in flight while flags are reset.
    ctx['combine_workspace'].zero_()
    storage = storage.to(device=plan.dst.device)
    stream = torch.npu.current_stream()
    ready = torch.npu.Event()
    ready.record(stream)
    for tensor in (storage, ctx['combine_workspace'], *_routes(plan)):
        tensor.record_stream(stream)
    execution = CombineExecutionPlan(storage, id(plan), ctx['combine_workspace'], _key(ctx),
        tuple(map(_stamp, _routes(plan))), _stamp(storage),
        ctx.get('_combine_plan_generation', 0), ready, ctx)
    object.__setattr__(plan, 'combine_execution_plan', execution)
    torch.npu.synchronize()
    ShmemRuntime.barrier()
    execution.validate(ctx, plan)
    return execution


def check_precision(
    real: torch.Tensor,
    golden: torch.Tensor,
    threshold: float = PRECISION_THR,
) -> dict[str, int | float | bool]:
    actual = real.float().cpu()
    expected = golden.float().cpu()
    absolute_error = (actual - expected).abs()
    relative_error = absolute_error / (expected.abs() + 1e-8)
    failed = (absolute_error >= threshold) & (relative_error >= threshold)
    return {
        "ok": not bool(failed.any().item()),
        "max_ae": float(absolute_error.max().item()),
        "fail_both": int(failed.sum().item()),
    }


def select_token_indices(s: int, max_checks: int, seed: int) -> torch.Tensor:
    if max_checks <= 0 or max_checks >= s:
        return torch.arange(s)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 4242)
    return torch.randperm(s, generator=generator)[:max_checks].sort().values


def make_comm_plan(args, rank: int, device: torch.device):
    from npu_patch import MoonEPCommPlan

    dst, dup_groups, dup_loffs, dup_counts, _ = dispatch_test.make_plan(
        args, rank
    )
    E = dispatch_test.num_experts(args)
    B = dispatch_test.num_reserved_slots(args)
    NvS = dispatch_test.num_output_slots(args)
    return MoonEPCommPlan(
        dst=dst.to(device=device),
        experts_to_copy=torch.full(
            (args.R, B), -1, dtype=torch.int32, device=device
        ),
        zero_fill_ranges=torch.zeros(
            (E + B, 2), dtype=torch.int32, device=device
        ),
        remote_stats=torch.zeros(2, dtype=torch.int32, device=device),
        dup_groups=dup_groups.to(device=device),
        dup_loffs=dup_loffs.to(device=device),
        dup_counts=dup_counts.to(device=device),
        N=NvS,
        R=args.R,
        E=E,
        B=B,
        NvS=NvS,
        K=args.K,
    )


def run_rank(local_rank: int, args) -> None:
    validate_joint_args(args)
    rank = dispatch_test.global_rank_for_local(local_rank, args)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.R)
    os.environ["ASCEND_MOONEP_COMBINE_USE_UDMA"] = "1"

    import torch_npu
    from npu_patch._device import local_device_index
    from npu_patch import Buffer, ShmemRuntime
    # The dedicated ``npu_patch.dispatch_udma`` module was removed; the
    # UDMA dispatch path is now driven by ``ASCEND_MOONEP_DISPATCH_USE_UDMA``
    # (set by Buffer construction on UDMA-capable hardware) through the shared
    # ``npu_patch.dispatch`` launcher, which is what the sibling
    # ``test_moonep_dispatch_udma_with_plan`` already uses.
    from npu_patch.dispatch import (
        launch_dispatch,
        materialize_dispatch_outputs,
    )

    # Must come *after* the import above: a function-local ``import`` makes the
    # name local to this whole scope, so any earlier call would raise
    # UnboundLocalError.
    device_id = local_device_index(local_rank)
    torch.npu.set_device(device_id)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="hccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            rank=rank,
            world_size=args.R,
            device_id=torch.device(f"npu:{device_id}"),
        )

    buffer = None
    hidden_nvsh = None
    route_weights_nvs = None
    hidden_sk = None
    route_weights_sk = None
    plan = None
    ctx = None
    failed = False
    try:
        E = dispatch_test.num_experts(args)
        B = dispatch_test.num_reserved_slots(args)
        buffer = Buffer(
            args.S,
            args.H,
            args.K,
            E,
            args.R,
            B=B,
            explicitly_destroy=True,
            dispatch_use_udma=True,
            combine_use_udma=True,
        )
        ctx = buffer._require_ctx()
        if not (ctx.get("_dispatch_use_udma", False)
                and ctx.get("_combine_use_udma", False)):
            raise RuntimeError("Joint test requires both UDMA dispatch and UDMA combine")
        if not ShmemRuntime.is_initialized():
            init_shmem(args=args, 
                group=None,
                enable_udma=True,
                qp_count=UDMA_QP_COUNT,
            )

        buffer._ensure_shmem_views()
        ctx = buffer._require_ctx()
        ctx["active_qp_count"] = UDMA_QP_COUNT

        # Bounded binary fractions keep every K<=16 partial sum exact in BF16;
        # the K*hidden oracle therefore does not depend on reduction grouping.
        token = torch.arange(args.S, dtype=torch.int64)[:, None]
        dimension = torch.arange(args.H, dtype=torch.int64)[None, :]
        hidden_cpu = ((rank * 5 + token * 3 + dimension) % 8).float().div(16).to(torch.bfloat16)
        hidden = hidden_cpu.to(device=f"npu:{device_id}")
        route_weights = (
            dispatch_test.make_weights(rank, args).to(
                device=f"npu:{device_id}"
            )
            if not args.no_weights
            else None
        )
        plan = make_comm_plan(args, rank, hidden.device)
        cpu_execution, _ = make_cpu_combine_plan(args, rank)
        attach_cpu_combine_plan(ctx, plan, cpu_execution)
        del cpu_execution
        print(f'[rank {rank}] CPU golden dispatch + combine execution plan ready; '
              f'dedup={args.dedup_mode}', flush=True)

        ShmemRuntime.barrier()
        launch_dispatch(ctx, hidden, route_weights, plan)
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        # The combine input must be the exact symmetric shard populated by
        # dispatch.  Pointer checks make accidental copy-path coverage fail.
        hidden_nvsh, route_weights_nvs = materialize_dispatch_outputs(
            ctx,
            with_weights=route_weights is not None,
            zero_copy=True,
        )
        if hidden_nvsh.data_ptr() != ctx["hidden_buf_local"].data_ptr():
            raise AssertionError("UDMA dispatch hidden output is not zero-copy")
        if route_weights_nvs is not None and (
            route_weights_nvs.data_ptr()
            != ctx["weights_buf_local"].data_ptr()
        ):
            raise AssertionError("UDMA dispatch weight output is not zero-copy")

        # On the dispatch side, materialize_dispatch_outputs (which does not support this switch)
        # produces the aliased tensors; the combine side declares router_weights_zero_copy explicitly to skip the self-copy.
        hidden_sk, route_weights_sk, done = buffer.combine(
            plan=plan,
            hidden_nvsh=hidden_nvsh,
            route_weights_nvs=route_weights_nvs,
            async_finish=False,
            inter_rank_sync=True,
            zero_copy=True,
            router_weights_zero_copy=True,
        )
        if done is not None:
            raise AssertionError("synchronous UDMA combine must return done=None")
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        check_idx = select_token_indices(args.S, args.max_checks, args.seed)
        golden_hidden = (hidden_cpu.float() * float(args.K)).to(torch.bfloat16)
        hidden_stats = check_precision(
            hidden_sk.cpu()[check_idx], golden_hidden[check_idx]
        )
        print(
            f"[rank {rank}] UDMA joint hidden: ok={hidden_stats['ok']} "
            f"max_ae={hidden_stats['max_ae']:.5f} "
            f"fail_both={hidden_stats['fail_both']} "
            f"checked={check_idx.numel()}/{args.S}",
            flush=True,
        )
        failed = not bool(hidden_stats["ok"])

        if route_weights is None:
            if route_weights_sk is not None:
                raise AssertionError(
                    "UDMA combine returned weights when input weights are absent"
                )
        else:
            if route_weights_sk is None:
                raise AssertionError("UDMA combine did not return route weights")
            weight_stats = check_precision(
                route_weights_sk.cpu()[check_idx], route_weights.cpu()[check_idx]
            )
            print(
                f"[rank {rank}] UDMA joint weights: ok={weight_stats['ok']} "
                f"max_ae={weight_stats['max_ae']:.5f} "
                f"fail_both={weight_stats['fail_both']} "
                f"checked={check_idx.numel()}/{args.S}",
                flush=True,
            )
            failed = failed or not bool(weight_stats["ok"])

        print(
            f"[rank {rank}] UDMA dispatch+combine "
            f"{'PASS' if not failed else 'FAIL'}",
            flush=True,
        )
    finally:
        # Release all aliases before Buffer tears down its symmetric VMM.
        hidden_nvsh = None
        route_weights_nvs = None
        hidden_sk = None
        route_weights_sk = None
        plan = None
        ctx = None
        # Finalize before destroying: the R4 contract requires ShmemRuntime.finalize() to run
        # before Buffer.destroy() (tear down the symmetric mapping first, then release Buffer's VMM
        # physical memory; freeing the memory before finalize would be a use-after-free). This
        # matches the release order of test_moonep_dispatch_udma_with_plan and of
        # finalize_shmem_and_release in kernel_test_utils.
        try:
            if ShmemRuntime.is_initialized():
                ShmemRuntime.finalize()
        finally:
            if buffer is not None:
                buffer.destroy()
            if dist.is_initialized():
                dist.destroy_process_group()

    if failed:
        raise RuntimeError(
            f"rank {rank} UDMA dispatch+combine joint precision failed"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="UDMA dispatch -> UDMA combine joint test (no GMM)"
    )
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--S", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--E", type=int, default=None)
    parser.add_argument("--seed", type=int, default=dispatch_test.SEED)
    parser.add_argument("--first-device", type=int, default=0, help="Deprecated and ignored: the per-rank device index is resolved from ASCEND_RT_VISIBLE_DEVICES via npu_patch._device.")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29521)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument(
        "--plan-mode",
        choices=("random", "cyclic"),
        default="random",
    )
    parser.add_argument(
        "--max-checks",
        type=int,
        default=None,
        help="token rows to check; 0 means all (prod-shape default: 64)",
    )
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument('--dedup-mode', choices=('rank', 'server'), default='rank')
    parser.add_argument(
        "--prod-shape",
        action="store_true",
        help="use R=8, S=4096, K=16, H=7168, E=896",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Compatibility fields consumed by dispatch_test.validate_args.
    args.mode = "kernel"
    args.plan_source = 'golden'
    args.bootstrap = "ipport" if args.nproc_per_node and args.R > args.nproc_per_node else "uid"
    args.shmem_ip_port = ""
    args.profile = False
    args.profile_iters = 1
    args.profile_warmup = 0
    args.active_qp_count = UDMA_QP_COUNT

    if args.prod_shape:
        args.R = PROD_R
        args.S = PROD_S
        args.H = PROD_H
        args.K = PROD_K
        args.E = PROD_E
        if args.max_checks is None:
            args.max_checks = 64
    if args.max_checks is None:
        args.max_checks = 0

    validate_joint_args(args)
    local_count = dispatch_test.local_process_count(args)
    print(
        f"UDMA dispatch+combine joint: R={args.R} S={args.S} H={args.H} "
        f"K={args.K} E={dispatch_test.num_experts(args)} "
        f"weights={not args.no_weights} plan_mode={args.plan_mode} dedup_mode={args.dedup_mode} "
        f"combine_plan=cpu-golden "
        f"qp_count={UDMA_QP_COUNT} max_checks={args.max_checks or 'all'}",
        flush=True,
    )

    if local_count == 1:
        run_rank(0, args)
    else:
        mp.spawn(run_rank, args=(args,), nprocs=local_count, join=True)

    print("ALL RANKS PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
