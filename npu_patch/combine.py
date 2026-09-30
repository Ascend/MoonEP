"""Plan-driven MoonEP combine launch helpers for Ascend.

The public boundary mirrors ``dispatch.launch_dispatch``: callers pass ``ctx``
plus output tensors and a plan, while symmetric input shards are owned by the
Buffer and looked up from ``ctx``.  The Ascend kernel fuses prologue accumulate
and K-sum into one launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Workspace sizing remains usable without loading the NPU runtime.
# Runtime dependencies are imported only by launch and input-validation helpers.
if TYPE_CHECKING:
    import torch
    from .planning import MoonEPCommPlan


# ---- Workspace ABI and core roles (mirrors kernels/types.h) ----
# Combine only; does not change num_sms or launch core counts for dispatch/planning.
COMBINE_PUT_CORES = 64
COMBINE_PUT_SOURCE_LANES = 16
def combine_put_roles(ranks: int) -> list:
    """Core ownership in the production kernel: relay/reduce/sender/final-only."""
    if ranks >= 128:
        return [0] * 8 + [1] * 16 + [2] * 32 + [3] * 8
    return [0] * 8 + [1] * 32 + [2] * 24


def combine_put_workspace_bytes(n: int, h: int, ranks: int) -> int:
    # Total symmetric workspace bytes per rank; N=n=S*K, H=h, R=ranks.
    # Matches the order and capacities in MakeCombinePutLayout in
    # include/ascend_moonep/kernels/types.h; reserves maximum slots regardless of deduplication or weights.
    if min(n, h, ranks) <= 0:
        raise ValueError("n, h and ranks must be positive")
    align = lambda value: (value + 511) & ~511
    batches = (n + 63) // 64
    return (
        # BF16 data: final_hidden[N,H], send_hidden[N,H], relay_hidden[8*N,H].
        2 * align(n * h * 2) + align(8 * n * h * 2)
        # Float weights[N]; N entries each for reverse/reverse_send/final_map/group_of_loff/tx_rows.
        + align(n * 4) + 5 * align(n * 64)
        # R entries each for source_counts/counts_send; 8*N each for contributor/relay_send/relay_publish.
        + 2 * align(ranks * 64) + 3 * align(8 * n * 64)
        # Two reserved count tables; source/reduce/final ready have 8*R/R*B/2*R slots, respectively.
        + 1024 + (8 * ranks + ranks * batches + 2 * ranks) * 512
        # R weight_ready slots and 64 core-private magic source slots; 512 bytes per slot, first 8 used.
        + ranks * 512 + COMBINE_PUT_CORES * 512
        + 2 * align(8 * ranks * 64) + ranks * 512
        # One rank-local final gate: core 0 fans in final_ready then broadcasts it.
        + 512
        # N independent 128-byte final descriptors; N/K are active, N covers K=1.
        + align(n * 128)
    )


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.dtype != dtype or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous {dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must be shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _check_combine_inputs(
    ctx: dict,
    output_sh: torch.Tensor,
    plan: MoonEPCommPlan,
    output_sk: torch.Tensor | None,
) -> None:
    import torch
    from .launchers import _env_use_udma
    from .planning import MoonEPCommPlan

    if not isinstance(plan, MoonEPCommPlan):
        raise TypeError("plan must be a MoonEPCommPlan")

    use_udma = _env_use_udma("combine")
    required_ctx_keys = [
        "rank", "R", "E", "B", "S", "H", "K", "NvS", "kernels",
    ]
    if use_udma:
        required_ctx_keys.extend((
            "hidden_buf_local", "weights_buf_local", "combine_sync_buffer",
            "combine_workspace",
        ))
    else:
        required_ctx_keys.extend(("hidden_buf", "weights_buf"))
    missing = [key for key in required_ctx_keys if key not in ctx]
    if missing:
        raise KeyError(f"ctx is missing required keys: {missing}")

    rank = int(ctx["rank"])
    R = int(ctx["R"])
    S = int(ctx["S"])
    H = int(ctx["H"])
    K = int(ctx["K"])
    NvS = int(ctx["NvS"])
    if NvS != S * K:
        raise ValueError(f"Ascend combine requires NvS=S*K, got {NvS} != {S*K}")
    if rank < 0 or rank >= R:
        raise ValueError(f"ctx rank must be in [0, {R}), got {rank}")
    if K > 16:
        raise ValueError("K must be <= 16")
    if use_udma:
        if H % 8 != 0:
            raise ValueError("UDMA combine requires H to be a multiple of 8")
        # Phase 1/3 tile large rows through UB; cap one bf16 token row at 8 MiB.
        if H * torch.bfloat16.itemsize > 8 * 1024 * 1024:
            raise ValueError(
                "UDMA combine requires one bf16 token row to be <=8 MiB"
            )
    else:
        if H % 16 != 0:
            raise ValueError("H must be a multiple of 16")
        if H * 10 + 256 > 192 * 1024:
            raise ValueError(
                "H too large for single-pass combine UB (max ~19680)"
            )

    _check_tensor(output_sh, "output_sh", (S, H), torch.bfloat16, output_sh.device)
    if output_sk is not None:
        _check_tensor(
            output_sk,
            "output_sk",
            (S, K),
            torch.float32,
            output_sh.device,
        )

    E = int(ctx["E"])
    B = int(ctx["B"])
    if (
        plan.N != S * K
        or plan.R != R
        or plan.E != E
        or plan.B != B
        or plan.K != K
        or plan.NvS != NvS
    ):
        raise ValueError(
            "plan scalar metadata must match ctx: "
            f"plan(N={plan.N}, R={plan.R}, E={plan.E}, B={plan.B}, "
            f"K={plan.K}, NvS={plan.NvS}) vs "
            f"ctx(N={S*K}, R={R}, E={E}, B={B}, K={K}, NvS={NvS})"
        )
    _check_tensor(plan.dst, "plan.dst", (S * K,), torch.int32, output_sh.device)
    _check_tensor(
        plan.dup_groups,
        "plan.dup_groups",
        (NvS, 3),
        torch.int32,
        output_sh.device,
    )
    _check_tensor(
        plan.dup_loffs,
        "plan.dup_loffs",
        (8 * NvS if plan.dup_counts.numel() == plan.R + 2 else NvS,),
        torch.int32,
        output_sh.device,
    )
    _check_tensor(
        plan.dup_counts,
        "plan.dup_counts",
        (plan.R + 2 if plan.dup_counts.numel() == plan.R + 2 else 2,),
        torch.int32,
        output_sh.device,
    )
    hidden_key = "hidden_buf_local" if use_udma else "hidden_buf"
    _check_tensor(
        ctx[hidden_key],
        f"ctx.{hidden_key}",
        (NvS, H),
        torch.bfloat16,
        output_sh.device,
    )
    if output_sk is not None:
        weights_key = "weights_buf_local" if use_udma else "weights_buf"
        weights_buf = ctx[weights_key].view(torch.float32)
        _check_tensor(
            weights_buf,
            f"ctx.{weights_key}(fp32 view)",
            (NvS,),
            torch.float32,
            output_sh.device,
        )

    if use_udma:
        _check_tensor(
            ctx["combine_workspace"],
            "ctx.combine_workspace",
            (combine_put_workspace_bytes(NvS, H, plan.R),),
            torch.uint8,
            output_sh.device,
        )
        sync_buffer = ctx["combine_sync_buffer"]
        if sync_buffer.dtype != torch.uint8 or not sync_buffer.is_contiguous():
            raise ValueError(
                "ctx.combine_sync_buffer must be contiguous torch.uint8"
            )
        if sync_buffer.device != output_sh.device:
            raise ValueError(
                "ctx.combine_sync_buffer must be on "
                f"{output_sh.device}, got {sync_buffer.device}"
            )


def launch_combine(
    ctx: dict,
    output_sh: torch.Tensor,
    plan: MoonEPCommPlan,
    output_sk: torch.Tensor | None = None,
) -> None:
    """Launch fused prologue + K-sum into caller-owned ``output_sh`` / ``output_sk``.

    Planning is intentionally outside this function.  ``plan`` must already
    contain canonical ``dst`` and the three duplicate tables.

    ``output_sk`` is the user-facing ``[S, K]`` gather buffer when present.
    AlignUp(K, 8) padding for the native C++ ABI lives in
    ``launch_combine_op`` (same Buffer call path as dispatch →
    ``kernels['*'].compute``).
    """
    _check_combine_inputs(ctx, output_sh, plan, output_sk)
    combine_op = ctx["kernels"].get("combine")
    if combine_op is None:
        raise KeyError("ctx['kernels']['combine'] is required")
    combine_op.compute(
        ctx,
        output_sh,
        plan,
        output_sk=output_sk,
    )


# ---- Launch an immutable execution plan produced by planning.py ----
def launch_combine_prepared(ctx, output_sh, plan, execution_plan, output_sk=None):
    import torch
    import torch_npu
    from .planning import CombineExecutionPlan, _routes, _key

    if not isinstance(execution_plan, CombineExecutionPlan):
        raise TypeError('execution_plan must be returned by prepare_combine_plan')
    from .launchers import _env_use_udma
    if not _env_use_udma('combine'):
        raise ValueError('execution plans require UDMA combine')
    execution_plan.validate(ctx, plan)
    stream = torch_npu.npu.current_stream()
    stream.wait_event(execution_plan.ready)
    weights = ctx['weights_buf_local'].view(torch.float32) if output_sk is not None else None
    torch.classes.AscendMoonEP.CombineUrma().compute_prepared(
        ctx['hidden_buf_local'], weights, ctx['combine_sync_buffer'], ctx['combine_workspace'],
        *_routes(plan), output_sh, output_sk, execution_plan.storage, *_key(ctx))
    for tensor in (execution_plan.storage, *_routes(plan), ctx['hidden_buf_local'], weights,
                   ctx['combine_workspace'], ctx['combine_sync_buffer'], output_sh,
                   output_sk):
        if tensor is not None:
            tensor.record_stream(stream)
    # Also orders subsequent reuse on another stream. Overlapping calls sharing
    # the Buffer workspace remain unsupported, just like legacy combine.
    last_use = torch_npu.npu.Event()
    last_use.record(stream)
    execution_plan.ready = last_use
