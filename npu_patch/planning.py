"""
Local, CUDA-free mirror of the ``moonep/planning.py`` data structures.

``MoonEPCommPlan`` and ``allocate_planning_outputs`` are ported verbatim except
they depend only on torch (device comes from ``ctx``), so this module imports
cleanly without CUDA/CuTe. The planning *kernel* itself is provided by the
``PlanningOp`` (AscendC kernel migration).

This module also defines the shape conventions of the Ascend rank/server dedup plans.
"""

from dataclasses import dataclass, field

import torch

# An Ascend server contains 8 ranks; the server dedup protocol sizes its capacity accordingly.
RANKS_PER_SERVER = 8


@dataclass(frozen=True, slots=True)
class MoonEPCommPlan:
    dst: torch.Tensor
    experts_to_copy: torch.Tensor
    zero_fill_ranges: torch.Tensor
    remote_stats: torch.Tensor
    N: int
    R: int
    E: int
    B: int
    NvS: int
    K: int

    # Dedup structures materialized by the dispatch builder and consumed by the
    # dispatch epilogue / combine prologue. dup_counts = [n_groups, n_dup_loffs];
    # only the compact prefix of dup_groups/dup_loffs is valid.
    # Server mode additionally stores full destination encodings and [G,D,P...]; the counts shape tells the modes apart.
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor
    # Produced by PlanningOp for UDMA combine; clone() deliberately drops it.
    combine_execution_plan: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        N = int(self.N)
        R = int(self.R)
        E = int(self.E)
        B = int(self.B)
        NvS = int(self.NvS)
        server_dedup = self.dedup_mode == "server"
        assert NvS >= N and (not server_dedup or NvS == N), "invalid NvS for the selected dedup mode"
        # Server mode requires a complete 8-rank board; rank mode keeps the original small-scale test capability.
        if server_dedup:
            assert R > 0 and R % RANKS_PER_SERVER == 0
        assert self.dst.dtype == torch.int32 and self.dst.is_contiguous()
        assert self.dst.numel() == N
        assert self.experts_to_copy.dtype == torch.int32 and self.experts_to_copy.is_contiguous()
        assert tuple(self.experts_to_copy.shape) == (R, B)
        assert self.remote_stats.dtype == torch.int32 and self.remote_stats.is_contiguous()
        assert tuple(self.remote_stats.shape) == (2,)
        assert self.zero_fill_ranges.dtype == torch.int32 and self.zero_fill_ranges.is_contiguous()
        assert tuple(self.zero_fill_ranges.shape) == (E + B, 2)
        assert self.dup_groups.dtype == torch.int32 and self.dup_groups.is_contiguous()
        assert tuple(self.dup_groups.shape) == (NvS, 3)
        assert self.dup_loffs.dtype == torch.int32 and self.dup_loffs.is_contiguous()
        dup_capacity = RANKS_PER_SERVER * NvS if server_dedup else NvS
        count_size = R + 2 if server_dedup else 2
        assert tuple(self.dup_loffs.shape) == (dup_capacity,)
        assert self.dup_counts.dtype == torch.int32 and self.dup_counts.is_contiguous()
        assert tuple(self.dup_counts.shape) == (count_size,)

    @property
    def dedup_mode(self) -> str:
        """Distinguish the protocol from the counts shape, avoiding a separate mode field that could go stale."""
        return "server" if tuple(self.dup_counts.shape) == (self.R + 2,) else "rank"

    def clone(self) -> "MoonEPCommPlan":
        """Deep-clone every plan tensor (mirrors ``moonep.planning``; the tests
        snapshot plans with ``plan.clone()``)."""
        return type(self)(
            dst=self.dst.clone(),
            experts_to_copy=self.experts_to_copy.clone(),
            zero_fill_ranges=self.zero_fill_ranges.clone(),
            remote_stats=self.remote_stats.clone(),
            dup_groups=self.dup_groups.clone(),
            dup_loffs=self.dup_loffs.clone(),
            dup_counts=self.dup_counts.clone(),
            N=self.N,
            R=self.R,
            E=self.E,
            B=self.B,
            NvS=self.NvS,
            K=self.K,
        )


# ---- Immutable combine execution-plan ownership and validation ----
# Reuse across hidden/weight updates is allowed; routing changes require all
# ranks to prepare again, in the same collective order.
def _routes(plan):
    return (plan.dst, plan.dup_groups, plan.dup_loffs, plan.dup_counts)


def _stamp(tensor):
    try:
        version = tensor._version
    except RuntimeError:  # Inference tensors have no version counter.
        version = None
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype, tensor.device, version)


def _key(ctx):
    return tuple(int(ctx[name]) for name in ('S', 'H', 'K', 'R', 'rank'))


@dataclass
class CombineExecutionPlan:
    storage: torch.Tensor
    routing_id: int  # Avoid plan -> execution -> plan cycles retaining large tensors.
    workspace: torch.Tensor
    key: tuple
    route_stamps: tuple
    storage_stamp: tuple
    generation: int
    ready: object
    owner_ctx: dict
    prepare_begin: object = None
    prepare_end: object = None
    fused: bool = False

    def validate(self, ctx, plan):
        if ctx is not self.owner_ctx or _key(ctx) != self.key:
            raise ValueError('execution plan belongs to a different Buffer/topology/shape')
        if ctx['combine_workspace'].data_ptr() != self.workspace.data_ptr():
            raise ValueError('combine workspace changed; collectively prepare a new plan')
        if id(plan) != self.routing_id or tuple(map(_stamp, _routes(plan))) != self.route_stamps:
            raise ValueError('routing changed; all ranks must prepare a new execution plan')
        if ctx.get('_combine_plan_generation', 0) != self.generation:
            raise ValueError('planning was rerun; all ranks must prepare a new execution plan')
        if _stamp(self.storage) != self.storage_stamp:
            raise ValueError('execution plan storage was modified')


# Optional timing hooks: record on the current stream without synchronizing.
# Benchmark code reads and reports the samples outside timed combine calls.
def begin_base_planning_timing(ctx, plan):
    import torch_npu
    stream = torch_npu.npu.current_stream()
    begin = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    ctx['_planning_timing_events'] = dict(
        begin=begin, end=end, stream=stream, routing_id=id(plan),
        generation=ctx.get('_combine_plan_generation', 0))
    begin.record(stream)


def end_base_planning_timing(ctx):
    events = ctx['_planning_timing_events']
    events['end'].record(events['stream'])


def begin_fused_combine_plan():
    import torch_npu
    stream = torch_npu.npu.current_stream()
    begin = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    begin.record(stream)
    return begin, end, stream


def prepare_combine_plan(ctx, plan, *, storage=None, fused_timing=None):
    """Attach fused planning output, or prepare explicitly hand-built routes.

    Also available for callers which construct dst/dup_* manually. All ranks
    must call after routing changes; this reads metadata only, not hidden/weights.
    """
    import torch_npu
    from .launchers import _env_use_udma
    if not _env_use_udma('combine'):
        raise ValueError('execution plans require UDMA combine')
    object.__setattr__(plan, 'combine_execution_plan', None)
    fused = fused_timing is not None
    begin, end, stream = fused_timing if fused else begin_fused_combine_plan()
    tensors = _routes(plan)
    if fused:
        if storage is None:
            raise ValueError('fused planner did not return execution storage')
    else:
        storage = torch.classes.AscendMoonEP.Planning().prepare_combine_plan(
            ctx['combine_workspace'], *tensors, *_key(ctx))
    end.record(stream)
    for tensor in (storage, *tensors, ctx['combine_workspace']):
        tensor.record_stream(stream)
    execution = CombineExecutionPlan(storage, id(plan), ctx['combine_workspace'], _key(ctx),
        tuple(map(_stamp, tensors)), _stamp(storage),
        ctx.get('_combine_plan_generation', 0), end, ctx, begin, end)
    execution.fused = fused
    object.__setattr__(plan, 'combine_execution_plan', execution)
    return execution


def _round4(n: int) -> int:
    return (n + 3) & ~3


def allocate_planning_outputs(ctx: dict, *, dedup_mode: str = "rank"):
    """Allocate a ``(MoonEPCommPlan, cu_seqlens)`` pair on ctx's NPU device.

    The plan-owned dedup tensors are allocated here; the Ascend Planning kernel
    materializes their compact prefixes together with ``dup_counts``.
    """
    E = ctx['E']
    B = ctx.get('B', 0)
    S = ctx['S']
    K = ctx['K']
    N = S * K
    NvS = ctx['NvS']
    R = ctx['R']
    if dedup_mode not in ('rank', 'server'):
        raise ValueError("dedup_mode must be 'rank' or 'server'")
    server_dedup = dedup_mode == 'server'
    # Server mode addresses physical boards; an incomplete board cannot be treated as one dedup domain.
    if server_dedup and (R <= 0 or R % RANKS_PER_SERVER != 0):
        raise ValueError("server planning requires R divisible by 8")
    if NvS < N or (server_dedup and NvS != N):
        raise ValueError("invalid NvS for the selected dedup mode")
    dev = ctx['weights_buf'].device

    # Over-allocate output tensors to a multiple of 4 and then slice, avoiding
    # an out-of-bounds full-vector write at the tail (returned shapes unchanged).
    dst = torch.empty(_round4(N), dtype=torch.int32, device=dev)[:N]
    cu_seqlens = torch.empty(_round4(E + B), dtype=torch.int32, device=dev)[:E + B]
    experts_to_copy = torch.empty(_round4(ctx['R'] * B), dtype=torch.int32, device=dev)[
        :ctx['R'] * B
    ].view(ctx['R'], B)
    zero_fill_ranges = torch.empty(
        _round4((E + B) * 2), dtype=torch.int32, device=dev
    )[:(E + B) * 2].view(E + B, 2)
    remote_stats = torch.empty(_round4(2), dtype=torch.int32, device=dev)[:2]
    dup_groups = torch.empty(
        _round4(NvS * 3), dtype=torch.int32, device=dev
    )[:NvS * 3].view(NvS, 3)
    # Only server mode enlarges the secondary list and reserves a group start for each source.
    dup_capacity = RANKS_PER_SERVER * NvS if server_dedup else NvS
    count_size = R + 2 if server_dedup else 2
    dup_loffs = torch.empty(_round4(dup_capacity), dtype=torch.int32, device=dev)[:dup_capacity]
    dup_counts = torch.empty(_round4(count_size), dtype=torch.int32, device=dev)[:count_size]

    plan = MoonEPCommPlan(
        dst=dst,
        experts_to_copy=experts_to_copy,
        zero_fill_ranges=zero_fill_ranges,
        remote_stats=remote_stats,
        dup_groups=dup_groups,
        dup_loffs=dup_loffs,
        dup_counts=dup_counts,
        N=N,
        R=ctx['R'],
        E=E,
        B=B,
        NvS=NvS,
        K=K,
    )
    return plan, cu_seqlens
