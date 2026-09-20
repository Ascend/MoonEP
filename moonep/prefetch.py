"""
Remote expert prefetch kernel.

Every rank pushes its own local expert weights into the prefetch slots of the
peer ranks that need them. The three weight projections and, when present,
their three scale tensors share one persistent kernel launch, one compacted
owner-slot table, and one trailing cross-rank completion barrier.

Copy engine: persistent, warp-specialized 2D TMA pipeline:

  - warp 0: GMEM -> SMEM 2D TMA load
  - warps 1..6: two SMEM -> GMEM 2D TMA store warps per projection

Each tensor keeps its own TMA descriptor and tile count. A compile-time tensor
loop selects the descriptor while reusing the same producer/consumer loop
body. Tiles default to 128 x 128 elements and widen to 128 x 256 for unscaled
weights when every trailing extent permits it. Scale tensors are re-tiled to
``[128, nbytes // 128]`` per expert before entering the kernel.
"""

import functools

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass import BFloat16, Int8, Int32, Int64, Uint8, Uint32
from cutlass.cute.runtime import make_ptr

from moonep._common import cross_rank_barrier
from moonep.planning import match_any_b32


_ELEM_TYPES = {
    torch.bfloat16: (BFloat16, 2),
    torch.int8: (Int8, 1),
    torch.uint8: (Uint8, 1),
}


class PrefetchKernel:
    """Persistent multi-tensor 2D TMA local-expert prefetch push."""

    NUM_THREADS = 224
    PRODUCER_WARP = 0
    FIRST_STORE_WARP = 1
    STORE_WARPS_PER_PROJECTION = 2
    STORE_WARPS = 6
    LAST_STORE_WARP = FIRST_STORE_WARP + STORE_WARPS
    M_BLOCK = 128
    N_BLOCK = 128

    def __init__(
        self,
        EPN: int,
        R: int,
        tensor_shapes: tuple[tuple[int, int], ...],
        rank_strides: tuple[int, ...],
        has_scales: bool,
        meta_stride: int,
        num_sms: int,
        smem_budget: int,
        elem_ty=BFloat16,
        elem_bytes: int = 2,
    ):
        self.EPN = EPN
        self.R = R
        self.tensor_shapes = tensor_shapes
        self.rank_strides = rank_strides
        self.has_scales = has_scales
        self.num_tensors = 6 if has_scales else 3
        self.meta_stride = meta_stride
        self.num_sms = num_sms
        self.elem_ty = elem_ty
        self.elem_bytes = elem_bytes
        if not has_scales and all(Hp % 256 == 0 for _, Hp in tensor_shapes):
            self.N_BLOCK = 256
        self.stages = self._pick_stages(smem_budget)
        if self.stages == 0:
            raise RuntimeError(
                "prefetch: not enough per-block shared memory for one "
                f"{self.M_BLOCK}x{self.N_BLOCK} x {elem_bytes}B tile under "
                f"budget {smem_budget} B"
            )

    def _smem_bytes(self, stages: int) -> int:
        def _round_up(n: int, a: int) -> int:
            return (n + a - 1) // a * a

        tile_bytes = self.M_BLOCK * self.N_BLOCK * self.elem_bytes
        # sexp[R*EPN] + off[EPN+1] + alist[EPN] + acnt[1]
        # + slist[R*EPN] + cur[EPN+1]
        scan = _round_up(
            (2 * self.R * self.EPN + 3 * self.EPN + 3) * 4, 128)
        return (
            _round_up(stages * tile_bytes, 128)
            + _round_up(stages * 2 * 8, 16)
            + scan
            + 256
        )

    def _pick_stages(self, smem_budget: int) -> int:
        for stages in (6, 5, 4, 3, 2):
            if self._smem_bytes(stages) <= smem_budget:
                return stages
        return 0

    @cute.jit
    def _make_tma_atoms(
        self,
        local_ptr: cute.Pointer,
        pool_ptr: cute.Pointer,
        H: cutlass.Constexpr[int],
        Hp: cutlass.Constexpr[int],
        rank_stride: cutlass.Constexpr[int],
    ):
        EPN = cutlass.const_expr(self.EPN)
        R = cutlass.const_expr(self.R)
        M_BLOCK = cutlass.const_expr(self.M_BLOCK)
        N_BLOCK = cutlass.const_expr(self.N_BLOCK)

        Hp64 = cutlass.Int64(Hp)
        gmem_src = cute.make_tensor(
            local_ptr,
            cute.make_layout(
                (EPN * H, Hp),
                stride=(Hp64, cutlass.Int64(1)),
            ),
        )
        gmem_dst = cute.make_tensor(
            pool_ptr,
            cute.make_layout(
                (EPN * H, Hp, R),
                stride=(Hp64, cutlass.Int64(1), cutlass.Int64(rank_stride)),
            ),
        )

        smem_tile_layout = cute.make_ordered_layout(
            (M_BLOCK, N_BLOCK),
            order=(1, 0),
        )
        cta_tiler = (M_BLOCK, N_BLOCK)
        tma_g2s, tma_src = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gmem_src,
            smem_tile_layout,
            cta_tiler,
        )
        tma_s2g, tma_dst = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            gmem_dst,
            smem_tile_layout,
            cta_tiler,
        )
        return tma_g2s, tma_src, tma_s2g, tma_dst

    @cute.jit
    def __call__(
        self,
        local_ptrs: tuple[cute.Pointer, ...],
        pool_ptrs: tuple[cute.Pointer, ...],
        experts_ptr: cute.Pointer,          # int32 [R, EPN]
        meta_ptr: cute.Pointer,             # int32 [R * meta_stride]
        bar_ptr: cute.Pointer,              # int32 [1]
        rank: Int32,
        barrier_off: Int32,
        stream: cuda.CUstream,
    ):
        tma_g2s_atoms = []
        tma_sources = []
        tma_s2g_atoms = []
        tma_destinations = []
        for tensor_id in cutlass.range_constexpr(self.num_tensors):
            (
                tma_g2s,
                tma_src,
                tma_s2g,
                tma_dst,
            ) = self._make_tma_atoms(
                local_ptrs[tensor_id],
                pool_ptrs[tensor_id],
                self.tensor_shapes[tensor_id][0],
                self.tensor_shapes[tensor_id][1],
                self.rank_strides[tensor_id],
            )
            tma_g2s_atoms.append(tma_g2s)
            tma_sources.append(tma_src)
            tma_s2g_atoms.append(tma_s2g)
            tma_destinations.append(tma_dst)

        R = cutlass.const_expr(self.R)
        EPN = cutlass.const_expr(self.EPN)
        meta_stride = cutlass.const_expr(self.meta_stride)
        experts = cute.make_tensor(
            experts_ptr,
            cute.make_layout((R * EPN,)),
        )
        meta = cute.make_tensor(
            meta_ptr,
            cute.make_layout((R * meta_stride,)),
        )
        bar = cute.make_tensor(bar_ptr, cute.make_layout((1,)))

        smem_bytes = self._smem_bytes(self.stages)
        self.kernel(
            tma_g2s_atoms,
            tma_sources,
            tma_s2g_atoms,
            tma_destinations,
            experts,
            meta,
            bar,
            rank,
            barrier_off,
        ).launch(
            grid=(self.num_sms, 1, 1),
            block=(self.NUM_THREADS, 1, 1),
            smem=smem_bytes,
            stream=stream,
            cooperative=True,
        )

    @cute.kernel
    def kernel(
        self,
        tma_g2s_atoms: list[cute.CopyAtom],
        tma_sources: list[cute.Tensor],
        tma_s2g_atoms: list[cute.CopyAtom],
        tma_destinations: list[cute.Tensor],
        experts: cute.Tensor,
        meta: cute.Tensor,
        bar: cute.Tensor,
        rank: Int32,
        barrier_off: Int32,
    ):
        EPN = cutlass.const_expr(self.EPN)
        R = cutlass.const_expr(self.R)
        stages = cutlass.const_expr(self.stages)
        num_sms = cutlass.const_expr(self.num_sms)
        NUM_THREADS = cutlass.const_expr(self.NUM_THREADS)
        meta_stride = cutlass.const_expr(self.meta_stride)
        M_BLOCK = cutlass.const_expr(self.M_BLOCK)
        N_BLOCK = cutlass.const_expr(self.N_BLOCK)
        TILE_ELEMS = cutlass.const_expr(M_BLOCK * N_BLOCK)
        TILE_BYTES = cutlass.const_expr(TILE_ELEMS * self.elem_bytes)

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = utils.SmemAllocator()
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)
        sexp = smem.allocate_tensor(Int32, cute.make_layout((R * EPN,)), byte_alignment=4)
        off = smem.allocate_tensor(Int32, cute.make_layout((EPN + 1,)), byte_alignment=4)
        alist = smem.allocate_tensor(Int32, cute.make_layout((EPN,)), byte_alignment=4)
        acnt = smem.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=4)
        slist = smem.allocate_tensor(Int32, cute.make_layout((R * EPN,)), byte_alignment=4)
        cur = smem.allocate_tensor(Int32, cute.make_layout((EPN + 1,)), byte_alignment=4)
        stage_smem = smem.allocate_tensor(
            self.elem_ty,
            cute.make_ordered_layout(
                (M_BLOCK, N_BLOCK, stages),
                order=(1, 0, 2),
            ),
            byte_alignment=128,
        )

        cta_tiler = (M_BLOCK, N_BLOCK)
        smem_for_tma = cute.group_modes(stage_smem, 0, 2)
        tSsS_list = []
        tSgS_list = []
        tDsD_list = []
        tDgD_list = []
        for tensor_id in cutlass.range_constexpr(self.num_tensors):
            src_tiles = cute.zipped_divide(tma_sources[tensor_id], cta_tiler)
            dst_tiles = cute.zipped_divide(tma_destinations[tensor_id], cta_tiler)
            tSsS, tSgS = cpasync.tma_partition(
                tma_g2s_atoms[tensor_id],
                0,
                cute.make_layout(1),
                smem_for_tma,
                src_tiles,
            )
            tDsD, tDgD = cpasync.tma_partition(
                tma_s2g_atoms[tensor_id],
                0,
                cute.make_layout(1),
                smem_for_tma,
                dst_tiles,
            )
            tSsS_list.append(tSsS)
            tSgS_list.append(tSgS)
            tDsD_list.append(tDsD)
            tDgD_list.append(tDgD)

        # All six store warps share one stage ring. The empty barrier expects
        # one arrival from each store warp, so the producer cannot overwrite a
        # stage until every consumer has observed and released that generation.
        # Keeping one ring preserves the full stage budget across the
        # tensor-major gate/up/down loop, while projection-dedicated store-warp
        # queues let remote write tails overlap across tensor boundaries.
        load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=load_mbar,
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.STORE_WARPS),
            tx_count=TILE_BYTES,
        )

        rank_epn = rank * EPN

        # ---- prescan experts once (parallel): bucket slots per local expert.
        # A single-thread gmem scan of all R*EPN entries is bottlenecked by a
        # serial load latency chain, so it must be cooperative: stage the
        # table to smem with all threads, count with cta atomics, then have
        # warp 0 fill slist chunk-by-chunk with match_any so the final order
        # stays rb-ascending (bitwise-identical to the serial scan).
        for i in cutlass.range(tidx, R * EPN, self.NUM_THREADS, unroll=1):
            sexp[i] = experts[i] - rank_epn
        for i in cutlass.range(tidx, EPN + 1, self.NUM_THREADS, unroll=1):
            off[i] = Int32(0)
        cute.arch.sync_threads()
        for i in cutlass.range(tidx, R * EPN, self.NUM_THREADS, unroll=1):
            e = sexp[i]
            if e >= Int32(0) and e < Int32(EPN):
                cute.arch.atomic_add(off.iterator + (e + Int32(1)), Int32(1), scope="cta")
        cute.arch.sync_threads()
        if tidx == Int32(0):
            running = Int32(0)
            active_count = Int32(0)
            for local_expert in cutlass.range(EPN, unroll=1):
                cur_running = off[local_expert + Int32(1)]
                if cur_running > Int32(0):
                    alist[active_count] = Int32(local_expert)
                    active_count += Int32(1)
                running += cur_running
                off[local_expert + Int32(1)] = running
            acnt[0] = active_count
        cute.arch.sync_threads()
        for i in cutlass.range(tidx, EPN + 1, self.NUM_THREADS, unroll=1):
            cur[i] = off[i]
        cute.arch.sync_threads()
        if warp_idx == Int32(0):
            lane = cute.arch.lane_idx()
            lanes_lt = (Uint32(1) << lane) - Uint32(1)
            NCHUNK = cutlass.const_expr((R * EPN + 31) // 32)
            for chunk in cutlass.range(NCHUNK, unroll=1):
                rb = chunk * Int32(32) + lane
                local_expert = Int32(EPN)
                if rb < Int32(R * EPN):
                    e = sexp[rb]
                    if e >= Int32(0) and e < Int32(EPN):
                        local_expert = e
                peers = match_any_b32(local_expert)
                cell = cur.iterator + local_expert
                base = cute.arch.load(cell, Int32)
                pos = base + Int32(cute.arch.popc(peers & lanes_lt))
                cute.arch.sync_warp()
                if (peers & lanes_lt) == Uint32(0):
                    cute.arch.store(cell, base + Int32(cute.arch.popc(peers)))
                cute.arch.sync_warp()
                if local_expert < Int32(EPN):
                    slist[pos] = Int32(rb)
        cute.arch.sync_threads()
        n_active = acnt[0]

        if warp_idx == self.PRODUCER_WARP:
            load_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages
            )
            for tensor_id in cutlass.range_constexpr(self.num_tensors):
                H, Hp = self.tensor_shapes[tensor_id]
                MTILES = cutlass.const_expr(H // M_BLOCK)
                NTILES = cutlass.const_expr(Hp // N_BLOCK)
                TILES_PER_EXPERT = cutlass.const_expr(MTILES * NTILES)
                total_tiles = n_active * TILES_PER_EXPERT

                for tile in cutlass.range(bidx, total_tiles, num_sms, unroll=1):
                    i = tile // TILES_PER_EXPERT
                    rem = tile % TILES_PER_EXPERT
                    mt = rem // NTILES
                    nt = rem % NTILES

                    load_pipe.producer_acquire(load_state)
                    src_mt = alist[i] * MTILES + mt
                    cute.copy(
                        tma_g2s_atoms[tensor_id],
                        tSgS_list[tensor_id][(None, (src_mt, nt))],
                        tSsS_list[tensor_id][(None, load_state.index)],
                        tma_bar_ptr=load_pipe.producer_get_barrier(load_state),
                    )
                    load_state.advance()

        elif warp_idx >= self.FIRST_STORE_WARP and warp_idx < self.LAST_STORE_WARP:
            store_warp_idx = warp_idx - self.FIRST_STORE_WARP
            # Every store warp advances through every producer tile, including
            # tiles owned by another projection. This keeps all consumer
            # states on the same ring index/phase and avoids phase-skip ABA
            # after the physical stages wrap around.
            use_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages
            )
            for tensor_id in cutlass.range_constexpr(self.num_tensors):
                H, Hp = self.tensor_shapes[tensor_id]
                MTILES = cutlass.const_expr(H // M_BLOCK)
                NTILES = cutlass.const_expr(Hp // N_BLOCK)
                TILES_PER_EXPERT = cutlass.const_expr(MTILES * NTILES)
                total_tiles = n_active * TILES_PER_EXPERT
                projection = tensor_id % 3

                tensor_work = Int32(0)
                for tile in cutlass.range(bidx, total_tiles, num_sms, unroll=1):
                    # Each projection owns two independent store-warp queues.
                    # Alternate CTA-local tiles between them; using the local
                    # ordinal matters because global tile ids advance by
                    # num_sms and may otherwise keep the same parity.
                    owner = (
                        projection * Int32(self.STORE_WARPS_PER_PROJECTION)
                        + tensor_work % Int32(self.STORE_WARPS_PER_PROJECTION)
                    )
                    is_owner = owner == store_warp_idx

                    load_pipe.consumer_wait(use_state)
                    if is_owner:
                        i = tile // TILES_PER_EXPERT
                        rem = tile % TILES_PER_EXPERT
                        mt = rem // NTILES
                        nt = rem % NTILES
                        local_expert = alist[i]
                        beg = off[local_expert]
                        nslot = off[local_expert + Int32(1)] - beg

                        # The unique tile owner issues the complete fan-out,
                        # reusing one local G2S tile for every destination slot.
                        for s in cutlass.range(nslot, unroll=1):
                            rb = slist[beg + s]
                            dst_rank = rb // EPN
                            dst_mt = (rb % EPN) * MTILES + mt
                            cute.copy(
                                tma_s2g_atoms[tensor_id],
                                tDsD_list[tensor_id][(None, use_state.index)],
                                tDgD_list[tensor_id][(None, (dst_mt, nt, dst_rank))],
                            )
                        cute.arch.cp_async_bulk_commit_group()

                    # `.read` waits only until this warp's S2G groups stop
                    # reading the shared source; remote destination writes may
                    # remain in flight. All six consumers then arrive on the
                    # shared ring's empty barrier. Non-owners have no payload
                    # group for this tile, but must still arrive and advance;
                    # the owner's arrival is ordered after its source-read
                    # completion, making producer reuse of the stage safe.
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    load_pipe.consumer_release(use_state)
                    use_state.advance()
                    tensor_work += Int32(1)

            # Unlike the per-stage read wait above, this full wait drains each
            # store warp's remote destination writes before the cross-rank
            # barrier publishes them to the receiving ranks.
            cute.arch.cp_async_bulk_wait_group(0)

        cross_rank_barrier(
            meta, meta_stride, barrier_off, rank, R,
            bar.iterator, Int32(num_sms), NUM_THREADS, tidx,
        )


@functools.lru_cache(maxsize=None)
def _max_smem_per_block_optin(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


@functools.lru_cache(maxsize=None)
def _get_compiled(
    EPN: int,
    R: int,
    tensor_shapes: tuple[tuple[int, int], ...],
    rank_strides: tuple[int, ...],
    has_scales: bool,
    meta_stride: int,
    num_sms: int,
    device_index: int,
    torch_dtype: torch.dtype,
):
    elem_ty, elem_bytes = _ELEM_TYPES[torch_dtype]
    smem_budget = _max_smem_per_block_optin(device_index) - 1024
    kernel = PrefetchKernel(
        EPN=EPN,
        R=R,
        tensor_shapes=tensor_shapes,
        rank_strides=rank_strides,
        has_scales=has_scales,
        meta_stride=meta_stride,
        num_sms=num_sms,
        smem_budget=smem_budget,
        elem_ty=elem_ty,
        elem_bytes=elem_bytes,
    )

    data_ptr = make_ptr(elem_ty, 0, cute.AddressSpace.gmem, assumed_align=16)
    data_ptrs = tuple(data_ptr for _ in range(6 if has_scales else 3))
    i32_experts_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=4)
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    stream_arg = cuda.CUstream(0)

    return cute.compile(
        kernel,
        data_ptrs,
        data_ptrs,
        i32_experts_ptr,
        i32_ptr,
        i32_ptr,
        Int32(0),
        Int32(0),
        stream_arg,
    )


def prefetch_retile_nbytes(per_expert_nbytes: int) -> int:
    """Round a per-expert byte count up to what ``retile_for_prefetch`` needs.
    """
    tile = PrefetchKernel.M_BLOCK * PrefetchKernel.N_BLOCK
    return (per_expert_nbytes + tile - 1) // tile * tile


def retile_for_prefetch(t: torch.Tensor) -> torch.Tensor:
    """View a contiguous ``[N, ...]`` expert tensor as ``[N, 128, X]`` uint8.
    """
    assert t.is_contiguous(), "retile_for_prefetch requires a contiguous tensor"
    n = int(t.shape[0])
    per_expert = t.nbytes // n if n else 0
    tile = PrefetchKernel.M_BLOCK * PrefetchKernel.N_BLOCK
    assert per_expert % tile == 0, (
        f"retile_for_prefetch: per-expert extent {per_expert} bytes is not a "
        f"multiple of {tile}; allocate {prefetch_retile_nbytes(per_expert)} "
        f"bytes per expert instead"
    )
    return t.view(torch.uint8).reshape(n, PrefetchKernel.M_BLOCK, -1)


def _validate_prefetch_pair(
    local: torch.Tensor,
    pool: torch.Tensor,
    EPN: int,
    R: int,
    dtype: torch.dtype,
    device_index: int,
) -> None:
    assert local.dtype == dtype and local.is_contiguous(), \
        f"local tensors must be contiguous with dtype {dtype}"
    assert local.ndim == 3 and int(local.shape[0]) == EPN, \
        f"local tensors must be [EPN={EPN}, H, H']"
    assert pool.dtype == dtype, f"prefetch pools must have dtype {dtype}"
    assert pool.ndim == 4 and tuple(pool.shape[:2]) == (R, EPN), \
        f"prefetch pools must have leading shape [R, EPN]=[{R}, {EPN}]"
    assert tuple(pool.shape[2:]) == tuple(local.shape[1:]), \
        "local tensor and prefetch pool trailing shapes must match"
    assert local.device.index == device_index and pool.device.index == device_index, \
        "all prefetch tensors must be on the same CUDA device"
    H, Hp = (int(x) for x in local.shape[1:])
    assert H % PrefetchKernel.M_BLOCK == 0 and Hp % PrefetchKernel.N_BLOCK == 0, \
        "prefetch tensor trailing dimensions must be multiples of " \
        f"({PrefetchKernel.M_BLOCK}, {PrefetchKernel.N_BLOCK}), got ({H}, {Hp})"


def _validate_prefetch_launch(
    experts_to_copy: torch.Tensor,
    meta_buf: torch.Tensor,
    grid_sync_bar: torch.Tensor,
    R: int,
    EPN: int,
    rank: int,
    num_sms: int,
    device_index: int,
) -> None:
    assert experts_to_copy.dtype == torch.int32 and experts_to_copy.is_contiguous(), \
        "experts_to_copy must be contiguous int32 [R, EPN]"
    assert tuple(experts_to_copy.shape) == (R, EPN), \
        f"experts_to_copy must be [R, EPN]=[{R}, {EPN}]"
    assert experts_to_copy.device.index == device_index, \
        "experts_to_copy must be on the same CUDA device as prefetch pools"
    assert 0 <= int(rank) < R, f"rank must be in [0, {R}), got {rank}"
    assert isinstance(num_sms, int) and num_sms > 0, \
        f"num_sms must be a positive int, got {num_sms}"
    assert meta_buf.dtype == torch.int32 and meta_buf.is_contiguous()
    assert grid_sync_bar.dtype == torch.int32 and grid_sync_bar.is_contiguous() and grid_sync_bar.numel() == 1
    assert meta_buf.device.index == device_index and grid_sync_bar.device.index == device_index, \
        "barrier tensors must be on the same CUDA device as prefetch pools"


def _prepare_prefetch_scales(
    scales: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
    EPN: int,
    R: int,
    device_index: int,
    weight_dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    if scales is None:
        return (), ()

    assert len(scales) == 3, \
        "prefetch scales must contain three local/pool pairs"
    assert weight_dtype == torch.uint8, \
        "weight and re-tiled scale tensors must share uint8 copy dtype"
    scale_locals = []
    scale_pools = []
    for local_scale, scale_pool in scales:
        assert local_scale.is_contiguous() and int(local_scale.shape[0]) == EPN
        assert tuple(scale_pool.shape[:2]) == (R, EPN)
        assert scale_pool.dtype == local_scale.dtype
        assert tuple(scale_pool.shape[2:]) == tuple(local_scale.shape[1:])
        assert local_scale.device.index == device_index and scale_pool.device.index == device_index

        local_tiled = retile_for_prefetch(local_scale)
        # Preserve the writable pool alias and its rank gaps while re-tiling bytes.
        scale_bytes = scale_pool.view(torch.uint8)
        pool_tiled = scale_bytes.as_strided(
            (R, EPN, *local_tiled.shape[1:]),
            (int(scale_pool.stride(0)) * scale_pool.element_size(), *local_tiled.stride()),
        )
        scale_locals.append(local_tiled)
        scale_pools.append(pool_tiled)

    return tuple(scale_locals), tuple(scale_pools)


def launch_prefetch(
    local_weights: tuple[torch.Tensor, ...],
    prefetch_buffers: tuple[torch.Tensor, ...],
    experts_to_copy: torch.Tensor,
    rank: int,
    num_sms: int,
    meta_buf: torch.Tensor,
    meta_stride: int,
    barrier_off: int,
    grid_sync_bar: torch.Tensor,
    scales: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
):
    """Launch one fused owner-push prefetch for all weight/scale tensors.

    ``local_weights`` contains gate/up/down tensors shaped ``[EPN, H, H']``;
    ``prefetch_buffers`` contains their all-rank ``[R, EPN, H, H']`` pools,
    contiguous within each rank with optional padding between ranks.
    ``experts_to_copy[R, EPN]`` stores global expert ids. When ``scales`` is
    present it contains three ``(local_scale, all_rank_scale_pool)`` pairs;
    they are re-tiled internally and participate in the same kernel launch.
    """
    assert len(local_weights) == 3 and len(prefetch_buffers) == 3, \
        "prefetch requires three local weights and three prefetch buffers"

    dtype = local_weights[0].dtype
    assert dtype in _ELEM_TYPES, (
        f"prefetch: unsupported dtype {dtype}; "
        f"supported: {sorted(str(d) for d in _ELEM_TYPES)}"
    )
    assert local_weights[0].ndim == 3 and prefetch_buffers[0].ndim == 4
    EPN = int(local_weights[0].shape[0])
    R = int(prefetch_buffers[0].shape[0])
    device_index = prefetch_buffers[0].device.index
    assert device_index is not None, "prefetch buffers must be CUDA tensors"

    for local_weight, prefetch_buffer in zip(
        local_weights, prefetch_buffers, strict=True
    ):
        _validate_prefetch_pair(
            local_weight, prefetch_buffer, EPN, R, dtype, device_index
        )

    _validate_prefetch_launch(
        experts_to_copy, meta_buf, grid_sync_bar,
        R, EPN, rank, num_sms, device_index,
    )

    has_scales = scales is not None
    scale_locals, scale_pools = _prepare_prefetch_scales(
        scales, EPN, R, device_index, dtype
    )

    all_locals = (*local_weights, *scale_locals)
    all_pools = (*prefetch_buffers, *scale_pools)
    tensor_shapes = tuple(
        (int(local.shape[1]), int(local.shape[2])) for local in all_locals
    )
    rank_strides = tuple(int(pool.stride(0)) for pool in all_pools)

    compiled = _get_compiled(
        EPN, R, tensor_shapes, rank_strides, has_scales,
        int(meta_stride), int(num_sms), int(device_index), dtype,
    )

    elem_ty, _ = _ELEM_TYPES[dtype]
    local_ptrs = tuple(
        make_ptr(
            elem_ty,
            tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        for tensor in all_locals
    )
    pool_ptrs = tuple(
        make_ptr(
            elem_ty,
            tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        for tensor in all_pools
    )
    experts_ptr = make_ptr(
        Int32,
        experts_to_copy.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    meta_ptr = make_ptr(
        Int32,
        meta_buf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    bar_ptr = make_ptr(
        Int32,
        grid_sync_bar.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled(
        local_ptrs, pool_ptrs, experts_ptr, meta_ptr, bar_ptr,
        Int32(int(rank)), Int32(int(barrier_off)), stream,
    )
