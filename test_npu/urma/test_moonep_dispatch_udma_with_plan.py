"""MoonEP Dispatch accuracy test: uses either a handcrafted plan or a plan produced by the real Planning operator.

``--plan-source golden`` (default) builds the plan on the CPU and checks Dispatch single-operator accuracy.
``--plan-source kernel`` generates the plan with the real Planning operator, then checks Planning -> Dispatch.
Both sources support ``--dedup-mode rank|server``.

By default all R ranks are spawned on one host. For multi-node runs, launch this
script once per node with the same bootstrap endpoint, then pass a unique
--node-rank and the number of local processes via --nproc-per-node. The ipport
bootstrap is SHMEM-only and does not create a torch.distributed process group.

Run (single node, default ``--bootstrap uid`` -> ``ShmemRuntime.init_with_buffer``;
the script spawns R ranks internally via mp.spawn; note this script has **no** ``--B`` argument):
    python test_npu/urma/test_moonep_dispatch_udma_with_plan.py --mode kernel --R 8 --S 8192 --K 8 --H 7168 \
        --E 384 --plan-mode cyclic
(multi-node example, launched once per node, each passing a unique --node-rank:)
    python test_npu/urma/test_moonep_dispatch_udma_with_plan.py --mode kernel --R 16 --S 8192 --K 8 --H 7168 \
        --E 384 --plan-mode cyclic --nproc-per-node 8 --node-rank ${NODE_RANK} \
        --master-addr <node0_ip> --master-port 29521
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os
import random
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


SHMEM_HEAP_ALIGN_SIZE = 1024 * 1024 * 1024
OP_WORKSPACE_BYTES = 16 * 1024 * 1024
MAX_UB_ROW_BYTES = 128 * 1024
SEED = 20260803


def align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def num_experts(args) -> int:
    return args.E if args.E is not None else args.R * 4


def num_reserved_slots(args) -> int:
    return num_experts(args) // args.R


def required_shmem_bytes(args) -> int:
    """Exact Buffer symmetric chunks plus op-owned workspaces."""
    N = args.S * args.K
    E = num_experts(args)
    align8 = lambda value: align_up(value, 8)
    dup_rank_counts_off = align8(E)
    dup_groups_off = dup_rank_counts_off + 2 * args.R * 8
    dup_loffs_off = dup_groups_off + align8(N * 3)
    planning_workspace_bytes = (dup_loffs_off + align8(N)) * 4
    # dispatch_sync_buffer: token control area + four rounds of local sync area for the 16 token AIVs.
    dispatch_sync_buffer_bytes = args.R * 16 * 6 * 512 + 16 * 4 * 512
    # combine_sync_buffer: independent combine grant/done cacheline pool
    # [R * kCombineMaxGroupWidth(16) * 2 signal slots * 512].
    combine_sync_buffer_bytes = args.R * 16 * 2 * 512
    return (
        N * args.H * 2
        + N * 4
        + 3 * 4
        + planning_workspace_bytes
        + dispatch_sync_buffer_bytes
        + combine_sync_buffer_bytes
        + 2 * OP_WORKSPACE_BYTES
    )


def effective_shmem_bytes(args) -> int:
    required = required_shmem_bytes(args)
    if required <= SHMEM_HEAP_ALIGN_SIZE:
        return SHMEM_HEAP_ALIGN_SIZE
    return align_up(required + SHMEM_HEAP_ALIGN_SIZE, SHMEM_HEAP_ALIGN_SIZE)


def effective_shmem_ip_port(args) -> str:
    if args.shmem_ip_port:
        return args.shmem_ip_port
    shmem_port = args.master_port + 1
    if shmem_port > 65535:
        raise ValueError(
            "--shmem-ip-port is required when --master-port is 65535"
        )
    return f"tcp://{args.master_addr}:{shmem_port}"


def validate_args(args) -> None:
    if args.R <= 0 or args.S <= 0 or args.K <= 0 or args.H <= 0:
        raise ValueError("R/S/K/H must be positive")
    E = num_experts(args)
    if E <= 0 or E % args.R != 0:
        raise ValueError("E must be positive and divisible by R")
    if (3 * E * args.R) % 4 != 0:
        raise ValueError("3*E*R must be divisible by 4 for Buffer planning scratch")
    if args.H % 8 != 0:
        raise ValueError("H must be multiple of 8")
    if args.H * 2 > MAX_UB_ROW_BYTES:
        raise ValueError(f"H is too large for UB row buffer: H*2={args.H * 2} > {MAX_UB_ROW_BYTES}")
    if args.R * args.S * args.K >= 2**31:
        raise ValueError("R*S*K must fit in positive int32 dst encoding")
    if args.plan_source == "kernel" and args.mode != "kernel":
        raise ValueError("--plan-source kernel requires --mode kernel")
    if args.dedup_mode == "server" and args.R % 8 != 0:
        raise ValueError("server dedup requires a complete eight-rank server")
    if args.dedup_mode == "server" and args.K < 2:
        raise ValueError("server dedup accuracy requires K >= 2 to exercise forwarding")
    needs_master_endpoint = args.bootstrap == "uid" or not args.shmem_ip_port
    if needs_master_endpoint:
        if not args.master_addr:
            raise ValueError("master-addr must not be empty")
        if args.master_port <= 0 or args.master_port > 65535:
            raise ValueError("master-port must be in [1, 65535]")
    if args.nproc_per_node is None:
        if args.node_rank != 0:
            raise ValueError("--node-rank requires --nproc-per-node")
        return
    if args.nproc_per_node <= 0:
        raise ValueError("nproc-per-node must be positive")
    if args.R % args.nproc_per_node != 0:
        raise ValueError("R must be divisible by nproc-per-node")
    num_nodes = args.R // args.nproc_per_node
    if args.node_rank < 0 or args.node_rank >= num_nodes:
        raise ValueError(f"node-rank must be in [0, {num_nodes})")


def global_rank_for_local(local_rank: int, args) -> int:
    if args.nproc_per_node is None:
        return local_rank
    return args.node_rank * args.nproc_per_node + local_rank


def local_process_count(args) -> int:
    return args.R if args.nproc_per_node is None else args.nproc_per_node


def num_output_slots(args) -> int:
    return args.S * args.K


def selected_output_modes(args) -> tuple[str, ...]:
    if args.output_mode == "both":
        return ("zero-copy", "copy")
    return (args.output_mode,)


@dataclass(frozen=True)
class RouteEntry:
    src_rank: int
    token_id: int
    topk_id: int


def make_hidden(rank: int, args) -> torch.Tensor:
    row_base = torch.arange(args.S, dtype=torch.float32).unsqueeze(1) + float(rank * args.S)
    col = (torch.arange(args.H, dtype=torch.float32) % 16).unsqueeze(0) / 64.0
    return (row_base + col).to(torch.bfloat16).contiguous()


def expected_hidden_row(src_rank: int, token_id: int, args) -> torch.Tensor:
    base = float(src_rank * args.S + token_id)
    col = (torch.arange(args.H, dtype=torch.float32) % 16) / 64.0
    return (base + col).to(torch.bfloat16)


def make_weights(rank: int, args) -> torch.Tensor:
    base = rank * args.S * args.K
    return (torch.arange(args.S * args.K, dtype=torch.float32) + float(base)).reshape(args.S, args.K).contiguous()


def expected_weight(src_rank: int, token_id: int, topk_id: int, args) -> float:
    """Reproduce make_weights' fp32 arithmetic for large global route ids."""
    base = torch.tensor(src_rank * args.S * args.K, dtype=torch.float32)
    local_route = torch.tensor(token_id * args.K + topk_id, dtype=torch.float32)
    return float((base + local_route).item())


def make_raw_destinations(args) -> torch.Tensor:
    """Build the not-yet-deduplicated global destination slot permutation, the shared input of the plan and the output checks."""
    slots = num_output_slots(args)
    total_routes = args.R * slots

    # cyclic keeps the cyclic permutation; random uses a fixed-seed global random permutation.
    if getattr(args, "plan_mode", "random") == "cyclic":
        # Cyclic order-preserving distribution: target_slot == source_slot,
        # target_rank = (src_rank - ((token + topk) % R) + R) % R.
        # Global route index global_idx = src_rank*slots + token*K + topk,
        # and each route's dst = target_rank*slots + slot (unique destination slot -> a global permutation).
        src_ranks = torch.arange(args.R).reshape(-1, 1, 1)
        token_ids = torch.arange(args.S).reshape(1, -1, 1)
        topk_ids = torch.arange(args.K).reshape(1, 1, -1)
        slot_idx = token_ids * args.K + topk_ids
        target_ranks = (
            src_ranks - ((token_ids + topk_ids) % args.R) + args.R
        ) % args.R
        destination_by_route = (
            target_ranks * slots + slot_idx
        ).reshape(-1).to(torch.int32).contiguous()
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        destination_by_route = torch.randperm(
            total_routes,
            generator=generator,
            dtype=torch.int32,
        )
    return destination_by_route


def make_plan(args, local_rank: int):
    slots = num_output_slots(args)
    server_dedup = getattr(args, "dedup_mode", "rank") == "server"
    destination_by_route = make_raw_destinations(args)

    # Build this source rank's dst directly from its contiguous route slice.
    source_start = local_rank * slots
    local_raw_dst = destination_by_route[source_start:source_start + slots]
    local_dest_ranks = torch.div(
        local_raw_dst,
        slots,
        rounding_mode="floor",
    ).reshape(args.S, args.K)
    # Within a token, only the first positively encoded route to a target rank or server is kept.
    dedup_domains = local_dest_ranks // 8 if server_dedup else local_dest_ranks
    canonical = torch.ones((args.S, args.K), dtype=torch.bool)
    # Later routes in the same domain are negatively encoded and left to the duplicate expansion stage.
    for topk_id in range(1, args.K):
        canonical[:, topk_id] = torch.all(
            dedup_domains[:, :topk_id] != dedup_domains[:, topk_id].unsqueeze(1),
            dim=1,
        )
    dst_local = torch.where(
        canonical.reshape(-1),
        local_raw_dst,
        -local_raw_dst - 1,
    ).contiguous()

    # Invert only this destination rank's NvS entries. The previous reference
    # scanned all R*NvS routes in Python for every process.
    destination_start = local_rank * slots
    destination_end = destination_start + slots
    local_destination_mask = (
        (destination_by_route >= destination_start)
        & (destination_by_route < destination_end)
    )
    source_route_indices = torch.nonzero(
        local_destination_mask,
        as_tuple=False,
    ).reshape(-1)
    if int(source_route_indices.numel()) != slots:
        raise AssertionError("random route permutation does not cover every local output slot")
    local_loffs = (
        destination_by_route[source_route_indices] - destination_start
    ).to(torch.int64)
    received_source_routes = torch.empty(slots, dtype=torch.int32)
    received_source_routes[local_loffs] = source_route_indices.to(torch.int32)

    expected_local = {}
    for loff, source_route in enumerate(received_source_routes.tolist()):
        src_rank, route_in_rank = divmod(source_route, slots)
        token_id, topk_id = divmod(route_in_rank, args.K)
        expected_local[loff] = RouteEntry(src_rank, token_id, topk_id)

    # server expansion needs every route of this server; the source index order also fixes primary and the source
    # prefix.
    if server_dedup:
        server_start = (local_rank // 8) * 8 * slots
        server_mask = (destination_by_route >= server_start) & (destination_by_route < server_start + 8 * slots)
        sorted_source_routes = torch.nonzero(server_mask, as_tuple=False).reshape(-1)
        sorted_loffs = destination_by_route[sorted_source_routes]
    else:
        sorted_source_routes, sorted_loffs = torch.sort(received_source_routes)
    global_tokens = torch.div(
        sorted_source_routes,
        args.K,
        rounding_mode="floor",
    )
    _, token_counts = torch.unique_consecutive(
        global_tokens,
        return_counts=True,
    )
    token_starts = torch.cumsum(token_counts, dim=0) - token_counts
    duplicate_groups = token_counts > 1
    # Each server group belongs only to the card hosting the primary; secondary keeps the full destination encoding.
    if server_dedup:
        duplicate_groups &= sorted_loffs[token_starts] // slots == local_rank
    group_counts = token_counts[duplicate_groups]
    group_starts = token_starts[duplicate_groups]
    duplicate_counts = group_counts - 1
    duplicate_starts = torch.cumsum(duplicate_counts, dim=0) - duplicate_counts

    dup_groups_local = torch.full((slots, 3), -1, dtype=torch.int32)
    group_count = int(group_counts.numel())
    if group_count > 0:
        dup_groups_local[:group_count, 0] = (sorted_loffs[group_starts] % slots).to(torch.int32)
        dup_groups_local[:group_count, 1] = duplicate_starts.to(torch.int32)
        dup_groups_local[:group_count, 2] = duplicate_counts.to(torch.int32)

    duplicate_members = torch.repeat_interleave(duplicate_groups, token_counts)
    duplicate_members[token_starts] = False
    duplicate_loffs = sorted_loffs[duplicate_members].to(torch.int32)
    duplicate_count = int(duplicate_loffs.numel())
    dup_capacity = 8 * slots if server_dedup else slots
    dup_loffs_local = torch.full((dup_capacity,), -1, dtype=torch.int32)
    dup_loffs_local[:duplicate_count] = duplicate_loffs
    dup_counts_local = torch.tensor(
        [group_count, duplicate_count],
        dtype=torch.int32,
    )
    # P[src] is the start of this card groups for each source, letting server forwarding wait on that source rank
    # completion flag.
    if server_dedup:
        group_sources = global_tokens[group_starts] // args.S
        groups_per_source = torch.bincount(group_sources, minlength=args.R)
        source_prefixes = groups_per_source.cumsum(dim=0) - groups_per_source
        dup_counts_local = torch.cat((dup_counts_local, source_prefixes.to(torch.int32)))

    return dst_local, dup_groups_local, dup_loffs_local, dup_counts_local, expected_local


def make_planning_topk(args) -> torch.Tensor:
    """Build a balanced topk that contains cross-card same-server routes, ensuring the server forwarding path is covered."""
    group_width = 8 if args.R % 8 == 0 else args.R
    num_servers = args.R // group_width
    experts_per_rank = num_experts(args) // args.R
    src_ranks = torch.arange(args.R, dtype=torch.int32).view(-1, 1, 1)
    tokens = torch.arange(args.S, dtype=torch.int32).view(1, -1, 1)
    topk_ids = torch.arange(args.K, dtype=torch.int32).view(1, 1, -1)
    dest_servers = (src_ranks // group_width + tokens % num_servers) % num_servers
    dest_ranks = dest_servers * group_width + (tokens + topk_ids) % group_width
    local_experts = (tokens // num_servers + topk_ids // group_width) % experts_per_rank
    return (dest_ranks * experts_per_rank + local_experts).contiguous()


def expected_from_raw_dst(raw_dst: torch.Tensor, rank: int, args) -> dict[int, RouteEntry]:
    """Invert the raw destination addresses of the independent CPU reference into all source routes this card should receive."""
    slots = num_output_slots(args)
    raw = raw_dst.reshape(-1).to(torch.int64)
    source_routes = torch.nonzero(raw // slots == rank, as_tuple=False).reshape(-1)
    local_loffs = raw[source_routes] % slots
    if source_routes.numel() != slots or not torch.equal(torch.sort(local_loffs).values, torch.arange(slots)):
        raise AssertionError(f"rank={rank} CPU reference did not cover every output slot exactly once")

    expected = {}
    # Recover source rank, token, and topk from the raw route index; never use the operator dst as the answer.
    for loff, source_route in zip(local_loffs.tolist(), source_routes.tolist()):
        src_rank, local_route = divmod(source_route, slots)
        token_id, topk_id = divmod(local_route, args.K)
        expected[loff] = RouteEntry(src_rank, token_id, topk_id)
    return expected


def selected_loffs(expected_for_rank: dict[int, RouteEntry], args) -> list[int]:
    loffs = sorted(expected_for_rank)
    if args.max_checks <= 0 or len(loffs) <= args.max_checks:
        return loffs
    rng = random.Random(args.seed + 911)
    return sorted(rng.sample(loffs, args.max_checks))


def validate_golden_plan(
    args,
    dst: torch.Tensor,
    dup_groups: torch.Tensor,
    dup_loffs: torch.Tensor,
    dup_counts: torch.Tensor,
    expected: dict[int, RouteEntry],
    *,
    rank: int,
) -> None:
    slots = num_output_slots(args)
    server_dedup = getattr(args, "dedup_mode", "rank") == "server"
    seen_destinations = set()
    # Per token, check that only the first route in each dedup domain is positively encoded.
    for token_id in range(args.S):
        routes_by_dest = {}
        for topk_id in range(args.K):
            encoded = int(dst[token_id * args.K + topk_id])
            raw = encoded if encoded >= 0 else -encoded - 1
            dest_rank, loff = divmod(raw, slots)
            if dest_rank >= args.R or raw in seen_destinations:
                raise AssertionError(f"invalid or repeated dst entry at token={token_id}, topk={topk_id}")
            seen_destinations.add(raw)
            domain = dest_rank // 8 if server_dedup else dest_rank
            routes_by_dest.setdefault(domain, []).append((topk_id, encoded))
        for members in routes_by_dest.values():
            members.sort()
            if members[0][1] < 0 or any(encoded >= 0 for _, encoded in members[1:]):
                raise AssertionError(f"invalid canonical dst encoding for token={token_id}")

    group_count = int(dup_counts[0])
    dup_count = int(dup_counts[1])
    dup_capacity = 8 * slots if server_dedup else slots
    if not (0 <= group_count <= slots and 0 <= dup_count <= dup_capacity):
        raise AssertionError(f"invalid dup_counts={dup_counts.tolist()}")
    if len(expected) != slots:
        raise AssertionError(f"expected {slots} populated rows, got {len(expected)}")

    # server duplicates may span other cards of this server; verify full addresses and source prefixes against the raw
    # routes.
    if server_dedup:
        raw_routes = make_raw_destinations(args).reshape(args.R, args.S, args.K)
        server_routes = raw_routes // (8 * slots) == rank // 8
        route_counts = server_routes.sum(dim=2)
        first_routes = server_routes.to(torch.int64).argmax(dim=2)
        primary_dst = raw_routes.gather(2, first_routes.unsqueeze(2)).squeeze(2)
        owned_groups = (route_counts > 1) & (primary_dst // slots == rank)
        group_tokens = torch.nonzero(owned_groups, as_tuple=False).tolist()
        groups_per_source = owned_groups.sum(dim=1)
        prefixes = groups_per_source.cumsum(dim=0) - groups_per_source
        expected_counts = torch.cat((
            torch.tensor([len(group_tokens), int((route_counts[owned_groups] - 1).sum())]), prefixes,
        )).to(torch.int32)
        if not torch.equal(dup_counts, expected_counts):
            raise AssertionError("server dup_counts or source prefixes do not match raw routes")
        dup_start = 0
        # Replay the full destination addresses of same-source tokens per group, covering cross-card duplicates,
        # ordering, and the compact prefix.
        for gid, (src, token) in enumerate(group_tokens):
            members = raw_routes[src, token][server_routes[src, token]].tolist()
            primary, start, count = map(int, dup_groups[gid].tolist())
            expected_group = (members[0] % slots, dup_start, len(members) - 1)
            if (primary, start, count) != expected_group:
                raise AssertionError(f"server group {gid} does not match raw routes")
            entry = expected[primary]
            if (entry.src_rank, entry.token_id, entry.topk_id) != (src, token, int(first_routes[src, token])):
                raise AssertionError(f"server group {gid} primary points to a different source token")
            if dup_loffs[start:start + count].tolist() != members[1:]:
                raise AssertionError(f"server group {gid} duplicate destinations do not match raw routes")
            dup_start += count
        return

    consumed_loffs = set()
    for gid in range(group_count):
        primary, dup_start, dup_n = map(int, dup_groups[gid].tolist())
        if primary not in expected:
            raise AssertionError(f"group {gid} has invalid primary loff {primary}")
        if dup_start < 0 or dup_n <= 0 or dup_start + dup_n > dup_count:
            raise AssertionError(f"group {gid} has invalid duplicate range")

        primary_entry = expected[primary]
        group_entries = [primary_entry]
        for index in range(dup_start, dup_start + dup_n):
            duplicate = int(dup_loffs[index])
            if duplicate in consumed_loffs or duplicate not in expected:
                raise AssertionError(f"group {gid} has invalid duplicate loff {duplicate}")
            duplicate_entry = expected[duplicate]
            if (duplicate_entry.src_rank, duplicate_entry.token_id) != (
                primary_entry.src_rank,
                primary_entry.token_id,
            ):
                raise AssertionError(f"group {gid} mixes rows from different source tokens")
            group_entries.append(duplicate_entry)
            consumed_loffs.add(duplicate)
        if primary_entry.topk_id != min(entry.topk_id for entry in group_entries):
            raise AssertionError(f"group {gid} primary is not the canonical top-k route")

    if len(consumed_loffs) != dup_count:
        raise AssertionError(
            f"dup_counts reports {dup_count} duplicate rows, validated {len(consumed_loffs)}"
        )


def run_rank(local_rank: int, args) -> None:
    validate_args(args)
    rank = global_rank_for_local(local_rank, args)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.R)

    if args.mode == "golden":
        dst, dup_groups, dup_loffs, dup_counts, expected = make_plan(args, rank)
        validate_golden_plan(args, dst, dup_groups, dup_loffs, dup_counts, expected, rank=rank)
        print(
            f"[rank {rank}] golden plan validated: groups={int(dup_counts[0])} "
            f"dups={int(dup_counts[1])} checked_rows={len(selected_loffs(expected, args))}",
            flush=True,
        )
        return

    import torch_npu
    from kernel_test_utils import init_shmem
    from npu_patch._device import local_device_index
    from npu_patch import (
        Buffer,
        MoonEPCommPlan,
        ShmemRuntime,
    )
    from npu_patch.dispatch import launch_dispatch, materialize_dispatch_outputs
    from npu_patch.launchers import launch_planning
    from npu_patch.planning import allocate_planning_outputs
    from mte.test_planning import assert_dup_plan_equal, assert_tensor_equal, planning_reference

    # Must come *after* the import above: a function-local ``import`` makes the
    # name local to this whole scope, so any earlier call would raise
    # UnboundLocalError.  Kept out of the golden path so plan-only runs need no NPU.
    device_id = local_device_index(local_rank)
    torch.npu.set_device(device_id)
    # UID bootstrap needs HCCL to broadcast the UID. ipport bootstraps directly
    # through SHMEM and deliberately does not create a torch process group.
    if args.bootstrap == "uid":
        dist.init_process_group(
            backend="hccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            rank=rank,
            world_size=args.R,
        )

    buffer = None
    hidden_shared = None
    weights_shared = None
    hidden_out = None
    weight_out = None
    ctx = None
    try:
        # buffer-backed: construct Buffer first (registers the VMM descriptor), then ShmemRuntime.init_with_ip_port (wrapped by init_shmem).
        # shmem initialization always goes through the buffer-backed path.
        E = num_experts(args)
        B = num_reserved_slots(args)
        buffer = Buffer(
            args.S,
            args.H,
            args.K,
            E,
            args.R,
            B=B,
            explicitly_destroy=True,
            rank=rank if args.bootstrap == "ipport" else None,
            use_udma_dispatch=True,
        )
        # UDMA enablement: created_qp_count = active_qp_count if <4, else 6
        created_qp_count = args.active_qp_count if args.active_qp_count < 4 else 6
        if not ShmemRuntime.is_initialized():
            if args.bootstrap == "ipport":
                init_shmem(args=args,
                    ip_port=effective_shmem_ip_port(args),
                    rank=rank,
                    world_size=args.R,
                    enable_udma=True,
                    qp_count=created_qp_count,
                )
            else:
                init_shmem(args=args,
                    group=None,
                    enable_udma=True,
                    qp_count=created_qp_count,
                )

        buffer._ensure_shmem_views()
        ctx = buffer._require_ctx()
        hidden_shared = ctx["hidden_buf_local"]
        weights_shared = ctx["weights_buf_local"].view(torch.float32)

        hidden = make_hidden(rank, args)
        hidden_npu = hidden.to(device=f"npu:{device_id}")
        weights_npu = None
        if not args.no_weights:
            weights_npu = make_weights(rank, args).to(device=f"npu:{device_id}")
        device = hidden_npu.device

        # The real Planning uses the same topk input; an independent CPU reference checks the plan and the final data.
        if args.plan_source == "kernel":
            topk_all = make_planning_topk(args)
            reference = planning_reference(topk_all, E, B, num_output_slots(args), rank, args.dedup_mode)
            expected = expected_from_raw_dst(reference.raw_dst, rank, args)
            plan, cu_seqlens = allocate_planning_outputs(ctx, dedup_mode=args.dedup_mode)
            topk_npu = topk_all[rank].reshape(-1).to(device=device)
            tpe_npu = reference.tokens_per_expert[rank].to(device=device)
            ShmemRuntime.barrier()
            launch_planning(ctx, topk_npu, tpe_npu, cu_seqlens, plan)
            torch_npu.npu.synchronize()
            ShmemRuntime.barrier()
            assert_tensor_equal(f"rank={rank}/dst", plan.dst, reference.dst[rank].reshape(-1))
            assert_tensor_equal(f"rank={rank}/cu_seqlens", cu_seqlens, reference.cu_seqlens[rank])
            assert_tensor_equal(f"rank={rank}/experts_to_copy", plan.experts_to_copy, reference.experts_to_copy)
            assert_dup_plan_equal(
                f"rank={rank}", plan.dup_groups, plan.dup_loffs, plan.dup_counts,
                reference.dup_groups, reference.dup_loffs, reference.dup_counts,
            )
        else:
            # The CPU golden directly builds rank/server plans, verifying Dispatch independently.
            dst, dup_groups, dup_loffs, dup_counts, expected = make_plan(args, rank)
            plan = MoonEPCommPlan(
                dst=dst.to(device=device),
                experts_to_copy=torch.full((args.R, B), -1, dtype=torch.int32, device=device),
                zero_fill_ranges=torch.zeros((E + B, 2), dtype=torch.int32, device=device),
                remote_stats=torch.zeros(2, dtype=torch.int32, device=device),
                dup_groups=dup_groups.to(device=device),
                dup_loffs=dup_loffs.to(device=device),
                dup_counts=dup_counts.to(device=device),
                N=num_output_slots(args), R=args.R, E=E, B=B,
                NvS=num_output_slots(args), K=args.K,
            )

        if args.clear_symmetric:
            hidden_shared.zero_()
            weights_shared.zero_()
            torch_npu.npu.synchronize()
        ShmemRuntime.barrier()
        launch_dispatch(ctx, hidden_npu, weights_npu, plan)
        torch_npu.npu.synchronize()
        ShmemRuntime.barrier()

        for output_mode in selected_output_modes(args):
            zero_copy = output_mode == "zero-copy"
            hidden_out, weight_out = materialize_dispatch_outputs(
                ctx,
                with_weights=weights_npu is not None,
                zero_copy=zero_copy,
            )
            torch_npu.npu.synchronize()
            if zero_copy:
                if hidden_out.data_ptr() != hidden_shared.data_ptr():
                    raise AssertionError("zero-copy hidden output must alias SHMEM")
                if (
                    not args.no_weights
                    and weight_out.data_ptr() != weights_shared.data_ptr()
                ):
                    raise AssertionError("zero-copy weight output must alias SHMEM")
            else:
                if hidden_out.data_ptr() == hidden_shared.data_ptr():
                    raise AssertionError("copy hidden output must use ordinary NPU storage")
                if (
                    not args.no_weights
                    and weight_out.data_ptr() == weights_shared.data_ptr()
                ):
                    raise AssertionError("copy weight output must use ordinary NPU storage")

            for loff in selected_loffs(expected, args):
                entry = expected[loff]
                actual_h = hidden_out[loff].cpu()
                expect_h = expected_hidden_row(entry.src_rank, entry.token_id, args)
                if not torch.equal(actual_h, expect_h):
                    mismatch_count = int((actual_h != expect_h).sum().item())
                    raise AssertionError(
                        f"rank={rank} output_mode={output_mode} loff={loff} "
                        f"hidden mismatch {mismatch_count}/{args.H}; "
                        f"source=(rank={entry.src_rank}, token={entry.token_id}, "
                        f"topk={entry.topk_id}) expect_first={float(expect_h[0])} "
                        f"got_first={float(actual_h[0])}"
                    )

                if args.no_weights:
                    if weight_out is not None:
                        raise AssertionError(
                            "weight_out must be None when route weights are omitted"
                        )
                else:
                    actual_w = float(weight_out[loff].cpu().item())
                    expect_w = expected_weight(
                        entry.src_rank, entry.token_id, entry.topk_id, args
                    )
                    if actual_w != expect_w:
                        raise AssertionError(
                            f"output_mode={output_mode} loff={loff} "
                            f"weight expected {expect_w}, got {actual_w}"
                        )

    finally:
        hidden_out = None
        weight_out = None
        hidden_shared = None
        weights_shared = None
        ctx = None
        try:
            if ShmemRuntime.is_initialized():
                ShmemRuntime.finalize()
        finally:
            if buffer is not None:
                buffer.destroy()
            if dist.is_initialized():
                dist.destroy_process_group()
    # The real-Planning verdict labels the dedup domain separately, so multi-node summaries can cross-check the
    # verification path.
    if args.plan_source == "kernel":
        print(f"[rank {rank}] Planning+Dispatch {args.dedup_mode} PASS", flush=True)
    print(f"[rank {rank}] MoonEPDispatch PASS", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="MoonEP Planning/Dispatch accuracy test")
    parser.add_argument("--mode", choices=("golden", "kernel"), default="golden")
    parser.add_argument(
        "--plan-source", choices=("golden", "kernel"), default="golden",
        help="golden: build the plan on CPU; kernel: the Planning operator generates the plan (both support rank/server)",
    )
    parser.add_argument("--dedup-mode", choices=("rank", "server"), default="rank")
    parser.add_argument(
        "--bootstrap",
        choices=("uid", "ipport"),
        default="uid",
        help="SHMEM bootstrap; ipport does not create a torch.distributed process group",
    )
    parser.add_argument("--first-device", type=int, default=0, help="Deprecated and ignored: the per-rank device index is resolved from ASCEND_RT_VISIBLE_DEVICES via npu_patch._device.")
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="zero-based node index; requires --nproc-per-node",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="enable multi-node mode and spawn this many local NPU processes",
    )
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help=(
            "UID rendezvous host; with ipport it is only used to derive the default "
            "SHMEM endpoint when --shmem-ip-port is omitted"
        ),
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "28889")),
        help=(
            "UID rendezvous port; with ipport, SHMEM defaults to this port plus one"
        ),
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=None,
        help=(
            "SHMEM ipport endpoint, for example tcp://10.0.0.1:28890; "
            "defaults to tcp://MASTER_ADDR:(MASTER_PORT+1)"
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--prod-shape",
        action="store_true",
        help="Production 8-card config: R=8, S=4096, K=16, H=7168, E=896.",
    )
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--S", type=int, default=2)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument(
        "--E",
        type=int,
        default=None,
        help="expert count used to size Buffer metadata; defaults to 4*R",
    )
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument(
        "--output-mode",
        choices=("zero-copy", "copy", "both"),
        default="zero-copy",
        help="validate SHMEM aliases, ordinary copied outputs, or both",
    )
    parser.add_argument(
        "--clear-symmetric",
        action="store_true",
        help="clear the SHMEM output tensors before dispatch (debug only)",
    )
    parser.add_argument(
        "--max-checks",
        type=int,
        default=0,
        help="0 means check every populated local row; use a positive value for large shapes.",
    )
    parser.add_argument(
        "--active-qp-count",
        type=int,
        default=2,
        choices=(1, 2, 3, 4),
        help="active UDMA QP count for dispatch kernel (default: 2)",
    )
    parser.add_argument(
        "--plan-mode",
        choices=("random", "cyclic"),
        default="random",
        help=(
            "plan distribution: random permutation (default) or cyclic ordered "
            "(target_slot == source_slot)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prod_shape:
        args.R = 8
        args.S = 4096
        args.K = 16
        args.H = 7168
        args.E = 896
        args.max_checks = 0
    validate_args(args)
    local_count = local_process_count(args)
    first_global_rank = global_rank_for_local(0, args)
    last_global_rank = global_rank_for_local(local_count - 1, args)
    master_endpoint = (
        f"{args.master_addr}:{args.master_port}"
        if args.bootstrap == "uid"
        else "unused"
    )
    print(
        f"MoonEPDispatch Test: mode={args.mode}, bootstrap={args.bootstrap}, "
        f"first_device={args.first_device}, seed={args.seed}, prod_shape={args.prod_shape}, "
        f"R={args.R}, S={args.S}, K={args.K}, H={args.H}, "
        f"E={num_experts(args)}, B={num_reserved_slots(args)}, "
        f"slots={num_output_slots(args)}, "
        f"with_weights={not args.no_weights}, output_mode={args.output_mode}, "
        f"clear_symmetric={args.clear_symmetric}, max_checks={args.max_checks}, "
        f"active_qp_count={args.active_qp_count}, plan_source={args.plan_source}, "
        f"dedup_mode={args.dedup_mode}, plan_mode={args.plan_mode}, "
        f"effective_shmem={effective_shmem_bytes(args)}, "
        f"shmem_ip_port={effective_shmem_ip_port(args) if args.bootstrap == 'ipport' else 'unused'}, "
        f"master={master_endpoint}, "
        f"node_rank={args.node_rank}, "
        f"local_processes={local_count}, global_ranks={first_global_rank}-{last_global_rank}",
        flush=True,
    )
    if local_count == 1:
        run_rank(0, args)
    else:
        mp.spawn(run_rank, args=(args,), nprocs=local_count, join=True)
    if args.nproc_per_node is None:
        print("\nALL RANKS PASS")
    else:
        print(f"\nNODE {args.node_rank} LOCAL RANKS PASS")


if __name__ == "__main__":
    main()
