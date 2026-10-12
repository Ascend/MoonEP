// Hidden uses UDMA PUT; weights are gathered by MTE only in Final.
// Pipeline owns the shared utilities and the complete combine payload flow.
// 64P: 0..7 relay; 8..39 reduce (8 groups x 4 shards);
//      40..55 QP0 send, 56..63 QP1 send (two workers/core).
// >=128P: 0..7 relay; 8..23 reduce (8 groups x 2 shards);
//       24..39 QP0 send, 40..55 QP1 send (one source lane/core), 56..63 final-only.
// Execution-plan generation belongs to planning.cpp; Pipeline only consumes it.
#include "kernel_operator.h"
#include "ascend_moonep/kernels/types.h"
#include "shmem.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_functions.h"

// Adapted from simt-moonep/ascend-moonep/kernels/moonep_dispatch_udma.cpp:
// MoonEPFillWriteWqeSimt / MoonEPSimtSubmitWqes. One VF thread writes one WQE
// directly to the SQ, followed by V_S and a scalar doorbell. No MTE SQ staging.
namespace CombinePutSimt {
constexpr uint32_t kBatch = 64U;
constexpr uint32_t kEntryWords = 8U;
constexpr uint32_t kContextWords = 16U;
// Do not overlap Wqe [128,136) KiB, relay metadata [120,124) KiB or sender
// descriptors [144,146) KiB. These buffers are only live in relay/send phases.
constexpr uint32_t kEntriesOffset = 136U * 1024U;
constexpr uint32_t kContextOffset = 138U * 1024U;
constexpr uint32_t kLaunchUbBytes = 192U * 1024U;
static_assert(kEntriesOffset + kBatch * kEntryWords * 4U == kContextOffset);
static_assert(kContextOffset + 2U * kContextWords * 4U < 144U * 1024U);

__simt_vf__ LAUNCH_BOUND(kBatch) inline void FillWriteWqes(
    uint32_t count, __ubuf__ uint32_t* entries, __ubuf__ uint32_t* ctx) {
    const uint32_t tid = threadIdx.x;
    if (tid >= count) return;
    __ubuf__ uint32_t* entry = entries + tid * kEntryWords;
    const uint64_t remote = uint64_t(entry[2]) | (uint64_t(entry[3]) << 32U);
    const uint64_t local = uint64_t(entry[4]) | (uint64_t(entry[5]) << 32U);
    const uint64_t sq = uint64_t(ctx[6]) | (uint64_t(ctx[7]) << 32U);
    // Preserve combine's per-batch completion: only the last row is SO+CQE;
    // preceding rows are RO without CQE. ctx[13] contains the last-row DW0.
    const uint32_t dw0 = tid + 1U == count ? ctx[13] : ctx[0];
    __gm__ uint64_t* wqe = reinterpret_cast<__gm__ uint64_t*>(
        sq + uint64_t(entry[7]) * ctx[12]);
    wqe[0] = uint64_t(dw0) | (uint64_t(ctx[1]) << 32U);
    wqe[1] = uint64_t(ctx[2]) | (uint64_t(ctx[3]) << 32U);
    wqe[2] = uint64_t(ctx[8]) | (uint64_t(ctx[9]) << 32U);
    wqe[3] = uint64_t(ctx[10]) | (uint64_t(ctx[11]) << 32U);
    wqe[4] = uint64_t(ctx[4]) | (uint64_t(ctx[5]) << 32U);
    wqe[5] = remote;
    wqe[6] = uint64_t(entry[6]);
    wqe[7] = local;
}
} // namespace CombinePutSimt

using namespace AscendC;
using bf16_t = bfloat16_t;
namespace {
using namespace ascend_moonep;

// ---- Begin combine-only definitions ----
// Combine-only 64-AIV layout; leaves kMoonEPAivNum=32 unchanged for other operators.
// Matches TileXR combine_v2: 8 relay, 32 reducer (8 groups x 4 shards), 24 physical sender cores.
inline constexpr uint32_t kCombinePutRelayCores = 8;
inline constexpr uint32_t kCombinePutSourceLanes = 16;
// 8 reducer groups x 4 shards. Each group owns a source-lane sequence.
inline constexpr uint32_t kCombinePutReducerGroups = 8;
inline constexpr uint32_t kCombinePutReducerShards = 4;
inline constexpr uint32_t kCombinePutReducerBegin = kCombinePutRelayCores;
inline constexpr uint32_t kCombinePutSenderBegin =
    kCombinePutReducerBegin + kCombinePutReducerGroups * kCombinePutReducerShards;
// QP0: 16 physical workers. QP1: 8 physical workers, each owns two peers.
inline constexpr uint32_t kCombinePutSenderEnd =
    kCombinePutSenderBegin + kCombinePutSourceLanes + kCombinePutSourceLanes / 2U;
// All worlds >=128P share moonep-r18's static first-reduce layout. Keep the
// legacy constants above untouched: they remain the exact 64P layout.
// The historical "128" suffix identifies this shared path, not a world-size limit.
inline constexpr uint32_t kCombinePut128ReducerBegin = 8U;
inline constexpr uint32_t kCombinePut128ReducerGroups = 8U;
inline constexpr uint32_t kCombinePut128ReducerShards = 2U;
inline constexpr uint32_t kCombinePut128SenderBegin = 24U;
// >=128P: AIV24..39 own QP0; AIV40..55 own QP1 (one source lane/core).
// AIV56..63 final-only. Keep each peer/QP on one core through all payload passes.
inline constexpr uint32_t kCombinePut128SenderEnd =
    kCombinePut128SenderBegin + 2U * kCombinePutSourceLanes;
static_assert(kCombinePut128SenderEnd == 56U);
static_assert(kCombinePut128SenderEnd <= kCombinePutCores);
static_assert(kCombinePutReducerGroups * 2U == kCombinePutSourceLanes);
static_assert(kCombinePutSenderEnd ==
    kCombinePutSenderBegin + 3U * kCombinePutSourceLanes / 2U);
static_assert(kCombinePutSenderEnd <= kCombinePutCores);

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define MOONEP_PUT_SCHEDULE_INLINE __forceinline__ __aicore__
#else
#define MOONEP_PUT_SCHEDULE_INLINE inline constexpr
#endif

// Shared >=128P mapping matches dispatch's current Staged GetGroupDstRank.
// 128P/256P/512P use 4x4 groups inside 64-rank frames. At 256P, XOR frame
// offsets visit the local 128P half first, then the other half. At 512P, row
// and side spread source groups across target frames. Other supported world
// sizes use dispatch's Legacy cross-half-world mapping, not a partial-frame extension.
// In combine, source is the original rank that ultimately receives the result. This uses
// dispatch's forward destination-rank mapping, not its reverse incoming-peer mapping.
// Relay supplies data in the ingress mapping order; Reduce/Send/planning share this function.
// Do not compact/reorder same-server entries: Reduce/Send skip them; Relay finishes LocalFinal.
MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutSource128(
    uint32_t rank, uint32_t ordinal, uint32_t ranks) {
    const uint32_t frame = rank / 64U;
    const uint32_t groupRow = (rank % 64U) / 32U;
    const uint32_t groupSide = (rank % 8U) / 4U;
    const uint32_t frames = ranks / 64U;
    const bool staged = ranks >= 128U && ranks <= 512U && ranks % 64U == 0U &&
        (frames & (frames - 1U)) == 0U;
    if (staged) {
        const uint32_t phase = ordinal / 16U, lane = ordinal % 16U;
        const uint32_t subround = phase % 4U;
        const uint32_t framePhase = phase / 4U;
        const uint32_t frameSpread = groupSide + groupRow * (frames >= 4U ? frames / 2U : 0U);
        const uint32_t frameOffset = frames == 4U
            ? (framePhase / 2U) * 2U + (groupSide ^ (framePhase % 2U))
            : (framePhase + frameSpread) % frames;
        const uint32_t targetFrame = frames == 4U
            ? frame ^ frameOffset : (frame + frameOffset) % frames;
        const uint32_t targetGroupRow = (groupRow + subround / 2U) % 2U;
        const uint32_t targetGroupSide = groupSide ^ (subround % 2U);
        return targetFrame * 64U + (targetGroupRow * 4U + lane / 4U) * 8U +
               targetGroupSide * 4U + lane % 4U;
    }
    const uint32_t halfWorld = ranks / 2U;
    const uint32_t rankGroup = (rank % halfWorld) / 8U;
    const uint32_t peerGroup = (rankGroup + ordinal / 16U) % (ranks / 16U);
    const uint32_t lane = ordinal % 16U;
    return (lane / 8U) * halfWorld + peerGroup * 8U + lane % 8U;
}

// The host ensures ranks>0. Callers iterate ordinal<ranks; unused lanes in small worlds stay idle.
MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutSource(
    uint32_t rank, uint32_t ordinal, uint32_t ranks) {
    if (ranks <= 8U) return (rank + ordinal) % ranks;
    // Match the current reference source schedule, not the older +8 rotation.
    const uint32_t lane = ordinal % 16U, phase = ordinal / 16U;
    const uint32_t serverBegin = rank / 8U * 8U, local = rank % 8U;
    if (ranks >= 128U) return CombinePutSource128(rank, ordinal, ranks);
    if (ranks == 16U || ranks == 32U || ranks == 64U) {
        const uint32_t remaining = ranks - 8U > phase * 16U ? ranks - 8U - phase * 16U : 0U;
        const uint32_t remote = remaining < 16U ? remaining : 16U;
        if (lane >= remote) return serverBegin + lane - remote;
        uint32_t seen = 0;
        for (uint32_t side = 0; side < 2U; ++side) {
            uint32_t start = side == 0U ? (serverBegin + ranks - (phase + 1U) * 8U + local) % ranks
                : (serverBegin + (phase + 1U) * 8U + (phase == 0U ? 0U : local)) % ranks;
            for (uint32_t i = 0; i < (side == 0U ? 8U : ranks); ++i) {
                uint32_t candidate = (start + i) % ranks;
                if (candidate >= serverBegin && candidate < serverBegin + 8U) continue;
                if (seen++ == lane) return candidate;
            }
        }
    }
    // Preserve the public API's other world sizes (reference does not support them).
    const uint32_t base = ((rank / 8U) * 8U + 8U) % ranks;
    return (base + ordinal) % ranks;
}

// The relay receiver (peer) performs first-reduce: >=128P supplies sources in demand order.
// The producer's rotated order could put the receiver's first awaited source at the end.
// Worlds below 128P retain their original producer order.
MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutRelaySource(
    uint32_t rank, uint32_t peer, uint32_t ordinal, uint32_t ranks) {
    return ranks >= 128U ? CombinePutSource128(peer, ordinal, ranks)
                         : CombinePutSource(rank, ordinal, ranks);
}

// Find a source's scheduling ordinal. The base is only 8-aligned, so source%16 cannot
// identify its sender; preparation and payload must use exactly the same peer/QP ownership.
MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutSourceOrdinal(
    uint32_t rank, uint32_t source, uint32_t ranks) {
    for (uint32_t ordinal = 0; ordinal < ranks; ++ordinal)
        if (CombinePutSource(rank, ordinal, ranks) == source) return ordinal;
    return ranks;
}

MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutSenderCore(
    uint32_t rank, uint32_t source, uint32_t ranks) {
    return (ranks >= 128U ? kCombinePut128SenderBegin : kCombinePutSenderBegin) +
        CombinePutSourceOrdinal(rank, source, ranks) % kCombinePutSourceLanes;
}

MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutSenderLane128(uint32_t core) {
    return core < kCombinePut128SenderBegin + kCombinePutSourceLanes ? 0U : 1U;
}
MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutSenderWorker128(uint32_t core) {
    return CombinePutSenderLane128(core) == 0U ? core - kCombinePut128SenderBegin
        : core - (kCombinePut128SenderBegin + kCombinePutSourceLanes);
}

// 64P has 56 remote sources, so ordinal 48..55 is the final, half-full
// 16-source wave.  Give its hidden payload exclusively to QP0's six-port
// path instead of splitting eight peers over QP0/QP1.  QP1 emits neither
// payload nor final_ready; Final() skips that lane for the same tail peers.
MOONEP_PUT_SCHEDULE_INLINE bool CombinePutSenderTailQp0Only(
    uint32_t rank, uint32_t source, uint32_t ranks) {
    if (ranks != 64U) return false;
    const uint32_t ordinal = CombinePutSourceOrdinal(rank, source, ranks);
    return ordinal >= 3U * kCombinePutSourceLanes && ordinal < ranks - 8U;
}

// QP0 six-port: one worker/core. QP1 two-port: two peers/core.
MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutSenderLane(uint32_t core) {
    return core < kCombinePutSenderBegin + kCombinePutSourceLanes ? 0U : 1U;
}
MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutSenderWorker(uint32_t core, uint32_t index) {
    return core < kCombinePutSenderBegin + kCombinePutSourceLanes
               ? core - kCombinePutSenderBegin
               : core - (kCombinePutSenderBegin + kCombinePutSourceLanes) + index * 8U;
}
// Rotate batch ownership across sources so a short tail does not keep landing on one shard.
MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutReducerFirstBatch(
    uint32_t shard, uint32_t batchOwnerRotation) {
    return (shard + kCombinePutReducerShards - batchOwnerRotation) %
           kCombinePutReducerShards;
}
// TileXR first-reduce scheduling: every four-shard group owns one source
// in each 8-source round: g, g+8, g+16, ... .
MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutReducerOrdinal(uint32_t group, uint32_t round) {
    return round * kCombinePutReducerGroups + group;
}

MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutReducerOrdinal(
    uint32_t ranks, uint32_t group, uint32_t round) {
    (void)ranks;
    return CombinePutReducerOrdinal(group, round);
}
#undef MOONEP_PUT_SCHEDULE_INLINE

// ---- End combine-only definitions ----

constexpr uint32_t kCores = ascend_moonep::kCombinePutCores;
constexpr uint32_t kReduceBegin = kCombinePutReducerBegin;
constexpr uint32_t kSendBegin = kCombinePutSenderBegin;
constexpr uint32_t kSendEnd = kCombinePutSenderEnd;
constexpr uint32_t kBatch = ascend_moonep::kCombinePutBatchRows, kTile = 8192;
// TileXR source-final scheduling constants.  Use the native AscendC vector
// operations for final reduction; this is the stable pre-RegBase path.
constexpr uint32_t kAlignedFinalTile = 4096U;
constexpr uint32_t kAlignedFinalRouteBatch = 4U;

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
inline constexpr aclshmemx_udma_op_config_t kPayload{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kCheckpoint{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
inline constexpr aclshmemx_udma_op_config_t kDataCheckpointRo{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
#endif

template <typename T>
__aicore__ inline LocalTensor<T> Ub(uint32_t bytes, uint32_t count) {
    LocalTensor<T> t;
    t.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    t.address_.bufferAddr = bytes;
    t.address_.dataLen = count * sizeof(T);
    return t;
}
template <typename T>
__aicore__ inline void Load(LocalTensor<T> ub, const __gm__ T* p, uint32_t n) {
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(const_cast<__gm__ T*>(p), n);
    DataCopyExtParams cp{1, n * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(ub, gm, cp, pad);
}
template <typename T>
__aicore__ inline void Store(__gm__ T* p, LocalTensor<T> ub, uint32_t n) {
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(p, n);
    DataCopyExtParams cp{1, n * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(gm, ub, cp);
}

__aicore__ inline void Flush(__gm__ void* p) {
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(p));
}
__aicore__ inline void Publish(__gm__ uint64_t* p, uint64_t magic) {
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    DataSyncBarrier<MemDsbT::DDR>();
    *p = magic;
    Flush(p);
}
__aicore__ inline void Wait(__gm__ uint64_t* p, uint64_t magic) {
    aclshmem_uint64_wait_until(p, ACLSHMEM_CMP_EQ, magic);
}
__aicore__ inline bool Ready(__gm__ uint64_t* p, uint64_t magic) {
    Flush(p);
    return *p == magic;
}
__aicore__ inline uint32_t Raw(int32_t v) {
    return ascend_moonep::IsMoonEPServerForwardedRoute(v)
        ? ascend_moonep::DecodeMoonEPServerForwardedRoute(v)
        : v >= 0 ? static_cast<uint32_t>(v)
                 : static_cast<uint32_t>(-static_cast<int64_t>(v) - 1);
}

struct Pipeline {
    // Each AIV owns a Pipeline; GM pointers refer to rank-shared data, while counters are core-private.
    __gm__ uint8_t* ws;       // Symmetric workspace: reverse plans, relay/send/final hidden, weights, ready slots.
    __gm__ bf16_t* hidden;    // Original input [n, h], indexed by producer-local loff; combine does not overwrite it.
    __gm__ float* weights;   // Symmetric SHMEM input [n]; Final reads rank/loff from dst; private GM is not valid.

    // Read-only planning inputs; source is the token's original rank, producer holds the input row.
    const __gm__ int32_t
        *dst,                // Source-rank route map [n]; Raw(dst[route]) = producer*n + loff.
                             // Nonnegative/forwarded tags mark canonical entries, ~raw duplicates; weights return per route.
        *groups,             // Dedup groups [n, 3]: representative loff, dups offset, duplicate count excluding the representative.
                             // In server mode column 3 packs source_route and duplicate count, decoded by DupCount().
        *dups,               // Rank mode: local loffs, capacity n; server mode: producer*n+loff, capacity 8*n.
        *counts;             // [0] valid groups, [1] valid duplicate members; r extra source-group prefixes in server ABI are unused here.
    __gm__ bf16_t* out;       // Source-rank output [n/k, h]; Final() merges canonical partial sums.
    __gm__ float* outw;       // Source-rank weights [n/k, k], restored per original route without deduplication or summation.
    ascend_moonep::CombinePutLayout p; // Workspace byte offsets relative to ws, batches per peer, etc.

    uint32_t n,              // S*K: per-rank route/input-row capacity, not the actual deduplicated send count.
             h,              // H: hidden elements per row; each BF16 element is 2 bytes.
             k,              // K: top-k routes per original token; the original token count is n/k.
             r,              // R: ranks in the EP communication world, e.g. 64; not the server count.
             rank,           // Global rank; in server mode rank/8 is the server ID and rank%8 the local device ID.
             core;           // AIV ID; see the file header for 64P and >=128P core roles.
    bool server,             // True: 8-rank/server dedup ABI with intra-server relay; false: rank-local dedup ABI.
         with_weights;       // Whether to return weights; false skips the weights/outw path.
    uint64_t magic;           // Unique ready marker per invocation; planning clears once, later calls use distinct magic values.

    // Two UB context records, one for each alternating QP. Reload on peer/QP
    // changes; head and CQ count are always read from the shared SHMEM context
    // because scalar metadata/ready PUTs also advance this same queue.
    uint32_t simt_peer[2] = {0xffffffffU, 0xffffffffU};
    // CQ failure attribution: 1=ReverseRoutes, 2=ReverseRelays, 3=FinishPlan,
    // 4=payload relay/reduce/send, 5=Final. No extra synchronization is added.
    uint32_t phase_id = 0U, simt_batches_posted = 0U;
    uint32_t simt_qp[2] = {0xffffffffU, 0xffffffffU};
    __aicore__ inline __gm__ int32_t* Meta(uint64_t base, uint64_t index) {
        uint64_t offset = 0U, stride = 64U;
        if (base == p.tx_rows) { offset = plan_layout.tx_rows; stride = kCombinePlanTxBytes; }
        else if (base == p.source_counts) offset = plan_layout.source_counts;
        else if (base == p.contributor) { offset = plan_layout.contributor; stride = kCombinePlanContributorBytes; }
        else if (base == p.relay_send) { offset = plan_layout.relay_send; stride = kCombinePlanRelayBytes; }
        else if (base == p.relay_ranges) offset = plan_layout.relay_ranges;
        else aclshmemi_kernel_abort("combine attempted mutable plan access rank=%u\n", rank);
        auto record = reinterpret_cast<__gm__ int32_t*>(execution_plan + offset + index * stride);
        // Planning writes directly into a reusable allocator block. Source
        // counts are read by PIPE_S without a caller-side Flush; invalidate
        // that cache line even when a previous plan occupied the same address.
        if (base == p.source_counts) Flush(record);
        return record;
    }
    // Prepared execution only reads this tensor. PUT targets/flags remain in ws.
    __gm__ uint8_t* execution_plan = nullptr;
    CombineExecutionPlanLayout plan_layout{};
    __aicore__ inline __gm__ uint64_t* Flag(uint64_t base, uint64_t index) {
        return reinterpret_cast<__gm__ uint64_t*>(ws + base + index * 512U);
    }
    __aicore__ inline __gm__ bf16_t* Row(uint64_t base, uint64_t index) {
        return reinterpret_cast<__gm__ bf16_t*>(ws + base) + index * h;
    }
    __aicore__ inline __ubuf__ uint8_t* Wqe() {
        return reinterpret_cast<__ubuf__ uint8_t*>(Ub<uint8_t>(128 * 1024, 8192).GetPhyAddr());
    }
    // Default batched PUT retains scalar SHMEM WQE construction and MTE3 SQ staging.
    // DIRECT, cross-server reduced results, and small-world LocalFinal use this non-SIMT entry.
    __aicore__ inline void SubmitPutBatch(__gm__ bf16_t* target_base,
                                          __gm__ bf16_t* const* inputs,
                                          __gm__ bf16_t* const* targets,
                                          uint32_t count, uint32_t peer,
                                          uint32_t qp) {
        (void)target_base;
        if (!count) return;
        {
            // Exact pre-SIMT aggregate path: defer payload rows, then submit
            // the last row with one completion checkpoint for this batch.
            aclshmemx_submit_state_t state{};
            for (uint32_t j = 0; j < count; ++j) {
                if (j + 1U == count) {
                    aclshmemx_submit_t submit(state);
                    if (r == 128U && peer / 8U != rank / 8U) {
                        aclshmemx_udma_qp_put_nbi<bf16_t, PIPE_MTE3, kDataCheckpointRo>(
                            targets[j], inputs[j], reinterpret_cast<__ubuf__ bf16_t*>(Wqe()),
                            h, peer, qp, EVENT_ID0, submit);
                    } else {
                        aclshmemx_udma_qp_put_nbi<bf16_t, PIPE_MTE3, kCheckpoint>(
                            targets[j], inputs[j], reinterpret_cast<__ubuf__ bf16_t*>(Wqe()),
                            h, peer, qp, EVENT_ID0, submit);
                    }
                } else {
                    aclshmemx_defer_t defer(state);
                    aclshmemx_udma_qp_put_nbi<bf16_t, PIPE_MTE3, kPayload>(
                        targets[j], inputs[j], reinterpret_cast<__ubuf__ bf16_t*>(Wqe()),
                        h, peer, qp, EVENT_ID0, defer);
                }
            }
            return;
        }
    }
    // Intra-server SIMT payload, shared by Relay() and 64P/>=128P LocalFinal(); never for cross-server sends.
    // Metadata, weights, and ready keep their original paths; the last WQE in each batch still requests SO+CQE.
    __aicore__ inline void SubmitIntraPutBatch(__gm__ bf16_t* target_base,
                                               __gm__ bf16_t* const* inputs,
                                               __gm__ bf16_t* const* targets,
                                               uint32_t count, uint32_t peer,
                                               uint32_t qp) {
        (void)target_base;
        if (!count) return;
        if (!server || peer / 8U != rank / 8U) {
            aclshmemi_kernel_abort("combine SIMT intra PUT requires same-server peer\n");
            return;
        }
        auto info = aclshmemi_udma_qp_info_fetch();
        auto ctx = aclshmemi_udma_get_qp_ctx(info, peer, qp);
        auto queue_state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(ctx->state_addr);
        if (count > CombinePutSimt::kBatch || peer == rank || ctx->wqe_size != 64U) {
            aclshmemi_kernel_abort("combine SIMT invalid count/peer/WQE size\n");
            return;
        }
        // Apply the same SQ credit guard as the aggregate SHMEM writer, using
        // our checked Drain so a CQ timeout never permits overwriting live SQEs.
        if (!aclshmemi_udma_sq_has_credit(queue_state, count)) Drain(peer, qp);
        if (!aclshmemi_udma_sq_has_credit(queue_state, count)) {
            aclshmemi_kernel_abort("combine SIMT SQ credit unavailable\n");
            return;
        }
        auto entries = reinterpret_cast<__ubuf__ uint32_t*>(
            Ub<uint32_t>(CombinePutSimt::kEntriesOffset, 64U * 8U).GetPhyAddr());
        const uint32_t cache = qp & 1U;
        auto rec = reinterpret_cast<__ubuf__ uint32_t*>(
            Ub<uint32_t>(CombinePutSimt::kContextOffset + cache * 64U, 16U).GetPhyAddr());
        if (simt_peer[cache] != peer || simt_qp[cache] != qp) {
            auto mem = aclshmemi_udma_get_mem_info(info, peer, qp);
            const uint32_t common = (uint32_t(mem->token_value_valid) << 28U) |
                (ACLSHMEM_UDMA_REMOTE_JETTY_TYPE << 29U);
            rec[0] = common | (aclshmemi_udma_build_flag<
                aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kPayload>() << 16U);
            rec[1] = (mem->target_hint & 0xffU) |
                (uint32_t(aclshmemi_udma_opcode_t::UDMA_OP_WRITE) << 8U);
            rec[2] = (mem->tpn & 0xffffffU) | (1U << 24U);
            rec[3] = mem->tid & 0xfffffU;
            rec[4] = mem->rmt_token_value;
            rec[5] = 0U;
            const uint64_t sq = ctx->buf_addr;
            rec[6] = uint32_t(sq); rec[7] = uint32_t(sq >> 32U);
            auto eid = reinterpret_cast<__gm__ uint64_t*>(mem->eid_addr);
            const uint64_t eid0 = eid[0], eid1 = eid[1];
            rec[8] = uint32_t(eid0); rec[9] = uint32_t(eid0 >> 32U);
            rec[10] = uint32_t(eid1); rec[11] = uint32_t(eid1 >> 32U);
            rec[12] = ctx->wqe_size;
            rec[13] = common | (aclshmemi_udma_build_flag<
                aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kCheckpoint>() << 16U);
            rec[14] = 0U; rec[15] = 0U;
            simt_peer[cache] = peer; simt_qp[cache] = qp;
        }
        const uint32_t head = queue_state->sq_head;
        for (uint32_t j = 0U; j < count; ++j) {
            auto entry = entries + j * CombinePutSimt::kEntryWords;
            const uint64_t remote = reinterpret_cast<uint64_t>(aclshmem_ptr(targets[j], peer));
            const uint64_t local = reinterpret_cast<uint64_t>(inputs[j]);
            entry[2] = uint32_t(remote); entry[3] = uint32_t(remote >> 32U);
            entry[4] = uint32_t(local); entry[5] = uint32_t(local >> 32U);
            entry[6] = h * sizeof(bf16_t);
            entry[7] = (head + j) % shm::UDMA_SQ_BASKBLK_CNT;
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        const uint32_t threads = (count + 31U) / 32U * 32U;
        AscendC::Simt::VF_CALL<CombinePutSimt::FillWriteWqes>(
            AscendC::Simt::Dim3(threads, 1, 1), count, entries, rec);
        // Match dispatch's direct-GM-store protocol: V_S completes VF stores
        // before the doorbell and before S can reuse entries/context UB.
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto db = reinterpret_cast<__gm__ uint32_t*>(ctx->db_addr);
        st_dev(head + count, db, 0);
        queue_state->sq_head = head + count;
        queue_state->cqe_cnt += 1U; // exactly one SO+CQE WQE per nonempty batch
        ++simt_batches_posted;
    }
    __aicore__ inline uint32_t Source(uint32_t ordinal) {
        return CombinePutSource(rank, ordinal, r);
    }
    __aicore__ inline uint32_t Producer(uint32_t address) { return server ? address / n : rank; }
    __aicore__ inline uint32_t Loff(uint32_t address) { return server ? address % n : address; }

    template <typename T>
    __aicore__ inline void Copy(__gm__ T* to, const __gm__ T* from, uint32_t size) {
        auto buf = Ub<T>(0, kTile);
        for (uint32_t off = 0; off < size; off += kTile) {
            uint32_t count = size - off < kTile ? size - off : kTile;
            Load(buf, from + off, count);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            Store(to + off, buf, count);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void Drain(uint32_t peer, uint32_t qp) {
        auto info = aclshmemi_udma_qp_info_fetch();
        auto ctx = aclshmemi_udma_get_qp_ctx(info, peer, qp);
        auto queue_state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(ctx->state_addr);
        uint32_t status = aclshmemi_udma_poll_cq(peer, qp, queue_state->cqe_cnt);
        if (status == 0xffU) {
            // The SDK's one-million-spin budget is not a transport error.
            // Retry the same completion target for at most one second on
            // Ascend950 (GetSystemCycle is 1 GHz); never ignore a CQ error.
            uint64_t start = GetSystemCycle();
            do {
                status = aclshmemi_udma_poll_cq(peer, qp, queue_state->cqe_cnt);
            } while (status == 0xffU && GetSystemCycle() - start < 1000000000ULL);
        }
        if (status != 0U) {
            auto tbl = aclshmemi_udma_active_table(info);
            auto cq = reinterpret_cast<__gm__ aclshmemi_udma_cq_ctx_t*>(tbl->scq_ptr) + peer * info->qp_num + qp;
            AscendC::printf("combine PUT queue head=%u tail=%u expected_cq=%u cq_tail=%u wqe_size=%u cqe_size=%u\n",
                queue_state->sq_head, queue_state->sq_tail, queue_state->cqe_cnt, queue_state->cq_tail, ctx->wqe_size, cq->cqe_size);
            // poll_cq only commits queue_state->cq_tail after reaching its full target.
            // Probe without consuming to distinguish zero progress from a missing
            // last checkpoint; cq_tail=0 alone does NOT mean no CQEs arrived.
            uint32_t observed = queue_state->cq_tail;
            while (observed != queue_state->cqe_cnt && observed - queue_state->cq_tail < shm::UDMA_CQ_DEPTH_DEFAULT) {
                auto entry = reinterpret_cast<__gm__ aclshmemi_jfc_cqe_ctx_t*>(
                    cq->buf_addr + uint64_t(cq->cqe_size) * (observed & (shm::UDMA_CQ_DEPTH_DEFAULT - 1U)));
                dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(entry), sizeof(aclshmemi_jfc_cqe_ctx_t));
                if ((((observed / shm::UDMA_CQ_DEPTH_DEFAULT) & 1U) ^ entry->owner) == 0U) break;
                if (entry->status || entry->substatus) break;
                ++observed;
            }
            aclshmemi_kernel_abort("combine PUT CQ failure rank=%u core=%u peer=%u qp=%u status=%u phase=%u simt_batches=%u observed_cq=%u\n",
                rank, core, peer, qp, status, phase_id, simt_batches_posted, observed);
        }
    }
    template <typename State>
    __aicore__ inline void SendReady(uint64_t base, uint64_t index, uint32_t peer,
                                    uint32_t qp, State& state) {
        aclshmemx_submit_t submit(state);
        aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kCheckpoint>(
            Flag(base, index), Flag(p.magic, core), reinterpret_cast<__ubuf__ uint64_t*>(Wqe()),
            1U, peer, qp, EVENT_ID0, submit);
    }
    // Same public plan ABI and SHMEM per-peer QPs; combine_v2 payload scheduling.
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inputQ;
    TQue<QuePosition::VECOUT, 2> outputQ;
    TBuf<TPosition::VECCALC> accumBuf, tempBuf, descriptorBuf, inputAddressBuf, readyBuf;
    // Stable pre-RegBase final path: MTE2 inputs, MTE3 outputs and native
    // AscendC Cast/Add accumulation in VECCALC.
    TBuf<TPosition::VECIN> finalInputBuf;
    TBuf<TPosition::VECOUT> finalOutputBuf;
    TBuf<TPosition::VECCALC> finalAccumulatorBuf, finalTempBuf, finalRouteBuf;
    static constexpr uint32_t tile = 3584U;

    __aicore__ bool LocalSource(uint32_t source) const {
        return server && source / 8U == rank / 8U;
    }
    __aicore__ uint32_t LocalLane(uint32_t peer) const { return server ? peer % 8U : 0U; }

    __aicore__ __gm__ uint64_t* Descriptors(uint32_t slot) {
        // Two eight-byte fields in one packed 16-byte plan record. word0 is
        // {count:high32,tx index:low32}; count=0 marks an interior slot.
        // word1 is the original global contributor address (producer*N+loff).
        // It lets Reduce() read an ingress-local contributor directly from
        // hidden instead of requiring a self relay_hidden staging copy.
        return reinterpret_cast<__gm__ uint64_t*>(Meta(p.contributor, slot));
    }

    __aicore__ void Init() {
        // Planning initializes ready storage once. Every collective invocation
        // receives a new nonzero magic from NextCombineMagic(); all payload
        // waits compare against that value, so old flags cannot satisfy them.
        // Never clear remote-writable flags here: a faster peer may have already
        // published this invocation's ready. Workspace calls must remain serial.
        // Match TileXR's topology-fused hidden entry: no rank/core barrier.
        // Each core consumes its own PUT-source magic only after this publish;
        // incoming payload is protected by relay/reduce/final-ready instead.
        // These slots remain PUT-source tokens for hidden ready notifications.
        // Weights do not poll them: callers provide ready, immutable inputs.
        Publish(Flag(p.magic, core), magic);
    }

    __aicore__ void BindPlan(__gm__ uint8_t* plan) {
        auto header = reinterpret_cast<__gm__ uint64_t*>(plan);
        Flush(header);
        if (header[0] != kCombineExecutionPlanVersion || header[1] != n ||
            header[2] != h || header[3] != k || header[4] != r ||
            header[5] != rank || header[6] != uint64_t(server))
            aclshmemi_kernel_abort("combine execution plan header mismatch rank=%u\n", rank);
        execution_plan = plan;
        plan_layout = MakeCombineExecutionPlanLayout(n, r);
    }

    __aicore__ void SubmitWeightBatch(LocalTensor<int32_t> routes,
        LocalTensor<float> gathered, uint64_t first, uint32_t size, uint32_t bank) {
        const TEventID event = bank == 0U ? EVENT_ID0 : EVENT_ID1;
        SetFlag<HardEvent::S_MTE2>(event);
        WaitFlag<HardEvent::S_MTE2>(event);
        // Read dst directly: a full batch reads 64*4=256B instead of a separate 1024B weights table.
        Load(routes, dst + first, size);
        SetFlag<HardEvent::MTE2_S>(event);
        WaitFlag<HardEvent::MTE2_S>(event);
        for (uint32_t j = 0U; j < size; ++j) {
            const uint32_t raw = Raw(routes.GetValue(j));
            const uint32_t peer = raw / n, loff = raw % n;
            // Each scalar MTE uses a separate 32B-aligned slot so padding cannot overwrite adjacent weights.
            const uint32_t offset = j * 8U;
            if (peer == rank) Load(gathered[offset], weights + loff, 1U);
            else {
                // Input contract: upstream dispatch/input production has completed; all weights
                // referenced by dst remain unchanged during GET. Do not wait for peers to enter combine.
                aclshmemx_mte_get_nbi<float>(
                    reinterpret_cast<__ubuf__ float*>(gathered[offset].GetPhyAddr()),
                    weights + loff, 1U, peer, event);
            }
        }
        // One event covers all scalar reads for this bank.
        SetFlag<HardEvent::MTE2_V>(event);
    }

    __aicore__ void FinalWeights() {
        if (!with_weights) return;
        const uint64_t firstBatch = uint64_t(core) * 64U;
        if (firstBatch >= n) return;
        // Run after this core finishes WORK, without delaying its pending hidden transfers.
        // Finish outstanding MTE/V instructions and release reducer TPipe resources before reusing low UB.
        PipeBarrier<PIPE_ALL>();
        pipe.Reset();
        LocalTensor<int32_t> routes[2] = {
            Ub<int32_t>(0U, 64U), Ub<int32_t>(4096U, 64U)};
        // Route-independent layout: construct once per core and reuse for all batches, outside planning.
        auto gatherOffsets = Ub<uint32_t>(256U, 64U);
        for (uint32_t j = 0U; j < 64U; ++j) gatherOffsets.SetValue(j, j * 32U);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        LocalTensor<float> gathered[2] = {
            Ub<float>(1024U, 512U), Ub<float>(5120U, 512U)};
        LocalTensor<float> packed[2] = {
            Ub<float>(3072U, 64U), Ub<float>(7168U, 64U)};
        bool used[2] = {false, false};
        const uint32_t firstSize = n - firstBatch < 64U ? n - firstBatch : 64U;
        SubmitWeightBatch(routes[0], gathered[0], firstBatch, firstSize, 0U);
        uint32_t bank = 0U;
        for (uint64_t first = firstBatch; first < n; first += uint64_t(kCores) * 64U) {
            const uint32_t size = n - first < 64U ? n - first : 64U;
            const TEventID event = bank == 0U ? EVENT_ID0 : EVENT_ID1;
            WaitFlag<HardEvent::MTE2_V>(event);
            const uint32_t next = bank ^ 1U;
            const uint64_t nextFirst = first + uint64_t(kCores) * 64U;
            if (nextFirst < n) {
                const TEventID nextEvent = next == 0U ? EVENT_ID0 : EVENT_ID1;
                if (used[next]) {
                    WaitFlag<HardEvent::V_MTE2>(nextEvent);
                    WaitFlag<HardEvent::MTE3_V>(nextEvent);
                    used[next] = false;
                }
                const uint32_t nextSize = n - nextFirst < 64U ? n - nextFirst : 64U;
                SubmitWeightBatch(routes[next], gathered[next], nextFirst, nextSize, next);
            }
            // Compact this batch with vector Gather while the next MTE2 is in flight; avoid per-element Get/SetValue.
            Gather(packed[bank], gathered[bank], gatherOffsets, 0U, size);
            SetFlag<HardEvent::V_MTE2>(event); // protects gathered input reuse
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
            Store(outw + first, packed[bank], size);
            SetFlag<HardEvent::MTE3_V>(event); // wait only on reuse / final drain
            used[bank] = true;
            // Route-load-to-store-issue latency; adjacent batches can overlap.
            bank = next;
        }
        for (uint32_t b = 0U; b < 2U; ++b) {
            if (!used[b]) continue;
            const TEventID event = b == 0U ? EVENT_ID0 : EVENT_ID1;
            WaitFlag<HardEvent::V_MTE2>(event);
            WaitFlag<HardEvent::MTE3_V>(event);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void LocalFinal(uint32_t peer) {
        // Match combine_v2 self-peer behavior: the rows already live in this
        // rank's hidden workspace, so FinalPair reads them directly.
        if (peer == rank) {
            Publish(Flag(p.final_ready, uint64_t(rank) * 2U), magic);
            return;
        }
        auto sc = Meta(p.source_counts, peer);
        LocalTensor<uint64_t> txPages[2] = {
            Ub<uint64_t>(120U * 1024U, 64U * 4U),
            Ub<uint64_t>(122U * 1024U, 64U * 4U),
        };
        uint32_t current = 0U;
        uint32_t count = static_cast<uint32_t>(sc[0]);
        if (count) SubmitTxRowLoad(txPages[current], sc[2], count < 64U ? count : 64U);
        for (uint32_t first = 0; first < count; first += 64U) {
            uint32_t size = count - first < 64U ? count - first : 64U;
            WaitRelayRequestLoad();
            uint32_t next = current ^ 1U, nextFirst = first + 64U;
            if (nextFirst < count) {
                uint32_t nextSize = count - nextFirst < 64U ? count - nextFirst : 64U;
                SubmitTxRowLoad(txPages[next], sc[2] + nextFirst, nextSize);
            }
            __gm__ bf16_t* inputs[64U];
            __gm__ bf16_t* targets[64U];
            for (uint32_t j = 0; j < size; ++j) {
                uint32_t loff = static_cast<uint32_t>(txPages[current].GetValue(j * 4U));
                inputs[j] = hidden + uint64_t(loff) * h;
                targets[j] = Row(p.final_hidden, sc[1] + first + j);
            }
            // Use SIMT for local-server peers in 64P and >=128P, sharing Relay's peer/QP0 queue and context.
            // The destination remains final_hidden; other world sizes retain scalar SHMEM submission.
            if (server && (r == 64U || r >= 128U)) {
                SubmitIntraPutBatch(Row(p.final_hidden, 0U), inputs, targets,
                                    size, peer, 0U);
            } else {
                SubmitPutBatch(Row(p.final_hidden, 0U), inputs, targets,
                               size, peer, 0U);
            }
            current = next;
        }
        aclshmemx_submit_state_t state{};
        SendReady(p.final_ready, uint64_t(rank) * 2U, peer, 0U, state);
    }

    // Phase A: this rank is a producer; each AIV0..7 sends to one fixed local-server peer.
    // For duplicate contributors of remote sources, read this rank's original hidden and
    // PUT into the ingress rank's relay_hidden via intra-server UDMA, then perform server reduce.
    // Same-server sources bypass server reduce. Submit all relay PUT/ready operations before
    // LocalFinal() returns data to final_hidden; do not interleave it with relay for remote sources.
    __aicore__ void Relay() {
        // Server mode uses AIV0..7, one per receiving rank in this server;
        // rank mode uses only AIV0.
        if (core >= (server ? 8U : 1U)) return;

        // The fixed local-server receiver (ingress for cross-server data) of this relay core:
        // for example, rank=36 and core=3 give peer=35.
        uint32_t peer = server ? rank / 8U * 8U + core : rank;
        if (peer >= r) return;

        // The self producer's input already resides in this rank's hidden.
        // The reducer reads hidden directly using original contributor addresses in descriptors,
        // so no self staging copy from hidden to relay_hidden is needed.
        if (peer == rank) {
            for (uint32_t ord = 0; ord < r; ++ord) {
                uint32_t source = Source(ord);

                // Publish ready even if this producer has no contributors for the source:
                // the reducer waits on a fixed producer set and cannot distinguish empty from pending.
                uint64_t readyIndex = uint64_t(LocalLane(rank)) * r + source;
                Publish(Flag(p.relay_ready, readyIndex), magic);

            }
            // Self needs no hidden copy; publish all relay-ready flags before this rank's final-ready.
            if (server) LocalFinal(peer);
            return;
        }

        // Non-self receiver: >=128P supplies data in the receiving peer's Reduce/Send order,
        // not this producer's rotated order; other world sizes retain their original order.
        // LocalFinal must follow the entire loop: the staged schedule may visit local sources early,
        // but source==peer must not trigger a return transfer that interrupts subsequent remote relay.
        for (uint32_t ord = 0; ord < r; ++ord) {
            uint32_t source = CombinePutRelaySource(rank, peer, ord, r);

            // Index relay_ranges by receiver/target source, independent of physical request ordering.
            // The ingress builds this compact relay request range for the producer/source during planning:
            // each entry in [first, first + count) is {producer-local input loff, relay scratch slot}.
            auto range = Meta(p.relay_ranges, uint64_t(core) * r + source);
            Flush(range);
            uint32_t first = range[0], count = range[1];

            // TileXR local-UDMA pattern: each producer/source CSR range is
            // consumed in 64-row batches.  MTE2 gathers the strided metadata
            // into one of two UB pages; while MTE3 posts page k, MTE2 prefetches
            // page k+1.  This replaces 64 scalar Meta()+Flush() operations.
            LocalTensor<uint64_t> requestPages[2] = {
                Ub<uint64_t>(120U * 1024U, 64U * 4U),
                Ub<uint64_t>(122U * 1024U, 64U * 4U),
            };
            uint32_t current = 0U;
            uint32_t size = count < 64U ? count : 64U;
            if (count) {
                SubmitRelayRequestLoad(requestPages[current],
                    uint64_t(core) * n + first, size);
            }
            for (uint32_t off = 0; off < count; off += 64U) {
                size = count - off < 64U ? count - off : 64U;

                // This is the visible metadata dependency for batch k. The
                // corresponding MTE2 read may already have overlapped WQE post
                // for batch k-1; the event therefore reports remaining wait
                // plus UB descriptor decode/address setup.
                WaitRelayRequestLoad();
                const uint32_t nextOff = off + 64U;
                const uint32_t next = current ^ 1U;
                if (nextOff < count) {
                    uint32_t nextSize = count - nextOff < 64U ? count - nextOff : 64U;
                    SubmitRelayRequestLoad(requestPages[next],
                        uint64_t(core) * n + first + nextOff, nextSize);
                }
                __gm__ bf16_t* inputs[64U];
                __gm__ bf16_t* targets[64U];
                for (uint32_t j = 0; j < size; ++j) {
                    uint64_t request = requestPages[current].GetValue(j * 4U);
                    inputs[j] = hidden + uint64_t(static_cast<uint32_t>(request)) * h;
                    targets[j] = Row(p.relay_hidden,
                        static_cast<uint32_t>(request >> 32U));
                }

                SubmitIntraPutBatch(Row(p.relay_hidden, 0U), inputs, targets,
                                   size, peer, 0U);

                current = next;
            }

            // After enqueuing all relay PUTs for this producer/source on the QP,
            // write magic to the ingress rank's corresponding relay_ready slot.
            // Reduce starts only after all producers in the server report ready for this source.
            aclshmemx_submit_state_t state{};
            uint64_t readyIndex = uint64_t(LocalLane(rank)) * r + source;
            SendReady(p.relay_ready, readyIndex, peer, 0U, state);
            // No periodic eight-source Drain; continue submitting subsequent sources.
            // Payload-before-ready ordering on the same QP makes data safe for the receiver to consume.
            // Submission still polls when queue credits run low to avoid overwriting in-flight WQEs.
        }

        // All remote-source relay payload and ready operations are submitted before local returns.
        // No extra Drain between Relay and LocalFinal; preserve 64P behavior.
        // No additional all-core barrier; relay cores progress independently.
        // 64P retains relay -> local final order; local final never interrupts >=128P relay.
        if (server) LocalFinal(peer);

        // Drain this core's exclusive local peer/QP0 at the end, covering both relay and LocalFinal.
        // Each relay core drains independently; no core reclaims another core's queue.
        Drain(peer, 0U);
    }

    __aicore__ void InitVector(bool final) {
        pipe.Reset();
        if (final) {
            pipe.InitBuffer(finalInputBuf,
                            2U * kAlignedFinalRouteBatch * kAlignedFinalTile * sizeof(bf16_t));
            pipe.InitBuffer(finalOutputBuf, 2U * kAlignedFinalTile * sizeof(bf16_t));
            pipe.InitBuffer(finalAccumulatorBuf, kAlignedFinalTile * sizeof(float));
            pipe.InitBuffer(finalTempBuf, kAlignedFinalTile * sizeof(float));
            // Two 128-byte final-route descriptors.  MTE2 fills the next bank
            // while Vector consumes the current token's hidden rows.
            pipe.InitBuffer(finalRouteBuf, 2U * 128U);
            return;
        }
        pipe.InitBuffer(inputQ, 2, 4U * tile * 2U);
        pipe.InitBuffer(outputQ, 2, 2U * tile * 2U);
        pipe.InitBuffer(accumBuf, 2U * tile * 4U);
        pipe.InitBuffer(tempBuf, 2U * tile * 4U);
        pipe.InitBuffer(descriptorBuf, 64U * 32U);
        pipe.InitBuffer(inputAddressBuf, 64U * 32U);
        pipe.InitBuffer(readyBuf, 32U);
        auto token = readyBuf.Get<uint64_t>();
        for (uint32_t i = 0; i < 4U; ++i) token.SetValue(i, i == 0U ? magic : 0U);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    }

    __aicore__ void PublishMte(uint32_t source, uint32_t batch) {
        // FIFO on MTE3 orders result stores before this stable 32-byte token.
        // No per-batch DDR DSB or MTE3->scalar wait is inserted.
        Store(Flag(p.reduce_ready, uint64_t(source) * p.batches + batch), readyBuf.Get<uint64_t>(), 4U);
    }

    __aicore__ void LoadDescriptors(LocalTensor<uint64_t> ub, uint32_t first, uint32_t size) {
        GlobalTensor<uint64_t> gm;
        gm.SetGlobalBuffer(Descriptors(first), uint64_t(size) * 2U);
        DataCopyExtParams cp{static_cast<uint16_t>(size), 8U, kCombinePlanContributorBytes - 8U, 0U, 0U};
        DataCopyPadExtParams<uint64_t> pad{false, 0U, 0U, 0U};
        DataCopyPad(ub, gm, cp, pad); // each 8-byte block is UB padded to 32 bytes
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    __aicore__ void LoadInputAddresses(LocalTensor<uint64_t> ub, uint32_t first, uint32_t size) {
        GlobalTensor<uint64_t> gm;
        gm.SetGlobalBuffer(Descriptors(first) + 1U, uint64_t(size) * 2U - 1U);
        DataCopyExtParams cp{static_cast<uint16_t>(size), 8U, kCombinePlanContributorBytes - 8U, 0U, 0U};
        DataCopyPadExtParams<uint64_t> pad{false, 0U, 0U, 0U};
        DataCopyPad(ub, gm, cp, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    // Packed relay requests: 8 GM bytes {input_loff,scratch} per row. Keep the
    // existing 32-byte UB lanes consumed by SIMT; no payload/WQE order changes.
    // Planner has completed exact-byte MTE3 stores before publishing the plan.
    __aicore__ void SubmitRelayRequestLoad(LocalTensor<uint64_t> ub,
                                           uint64_t requestFirst, uint32_t size) {
        GlobalTensor<uint64_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(
                               Meta(p.relay_send, requestFirst)),
                           uint64_t(size));
        DataCopyExtParams cp{static_cast<uint16_t>(size), 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<uint64_t> pad{false, 0U, 0U, 0U};
        DataCopyPad(ub, gm, cp, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    __aicore__ void WaitRelayRequestLoad() {
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    // Packed 16-byte TX rows; LocalFinal still gathers the first 8 bytes into
    // the unchanged 32-byte UB lanes used by the existing double buffer.
    __aicore__ void SubmitTxRowLoad(LocalTensor<uint64_t> ub,
                                    uint32_t first, uint32_t size) {
        GlobalTensor<uint64_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(Meta(p.tx_rows, first)),
                           uint64_t(size) * 2U);
        DataCopyExtParams cp{static_cast<uint16_t>(size), 8U, kCombinePlanTxBytes - 8U, 0U, 0U};
        DataCopyPadExtParams<uint64_t> pad{false, 0U, 0U, 0U};
        DataCopyPad(ub, gm, cp, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    // >=128P DIRECT: read each full 16B GM tx_rows record to obtain
    // both the input loff in tx[0] and the contributor count in tx[3].
    // Each record is padded to 32B in UB, so fields use a j*8 uint32_t offset.
    // Planning has completed exact-length MTE3 writes and published the plan; tx_rows is immutable during payload.
    __aicore__ void SubmitDirectTxRowLoad128(LocalTensor<uint32_t> ub,
                                            uint32_t first, uint32_t size) {
        // Finish previous scalar reads before reusing the page; use a dedicated EVENT_ID1,
        // separate from scalar PUT's EVENT_ID0 and result-descriptor load events.
        SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
        GlobalTensor<uint32_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(Meta(p.tx_rows, first)),
                           uint64_t(size) * 4U);
        DataCopyExtParams cp{static_cast<uint16_t>(size), 16U, 0U, 0U, 0U};
        DataCopyPadExtParams<uint32_t> pad{false, 0U, 0U, 0U};
        DataCopyPad(ub, gm, cp, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
    }

    __aicore__ uint32_t InputAddress(uint32_t slot, uint32_t scratchBase,
                                     uint32_t loadedAddresses,
                                     const uint32_t* inputAddresses) {
        uint32_t index = slot - scratchBase;
        if (index < loadedAddresses) return inputAddresses[index];
        // A contributor group may straddle the 64-slot reduce batch boundary.
        // Its tail was not gathered into UB, so read its immutable plan record.
        auto descriptor = Descriptors(slot);
        Flush(descriptor);
        return static_cast<uint32_t>(descriptor[1]);
    }

    __aicore__ bool DirectInput(uint32_t slot, uint32_t scratchBase,
                                uint32_t loadedAddresses,
                                const uint32_t* inputAddresses) {
        return Producer(InputAddress(slot, scratchBase, loadedAddresses, inputAddresses)) == rank;
    }

    __aicore__ void LoadInputRow(LocalTensor<bf16_t> ub, uint32_t dstOffset,
                                 uint32_t slot, uint32_t scratchBase,
                                 uint32_t loadedAddresses,
                                 const uint32_t* inputAddresses,
                                 uint32_t off, uint32_t size) {
        GlobalTensor<bf16_t> gm;
        uint32_t address = InputAddress(slot, scratchBase, loadedAddresses, inputAddresses);
        if (Producer(address) == rank)
            gm.SetGlobalBuffer(hidden + uint64_t(Loff(address)) * h + off, size);
        else
            gm.SetGlobalBuffer(Row(p.relay_hidden, slot) + off, size);
        DataCopyExtParams cp{static_cast<uint16_t>(1U), size * 2U, 0U, 0U, 0U};
        DataCopyPadExtParams<bf16_t> pad{false, 0U, 0U, 0U};
        DataCopyPad(ub[dstOffset], gm, cp, pad);
    }

    __aicore__ void LoadGroup(LocalTensor<bf16_t> ub, uint32_t slot, uint32_t count,
                             uint32_t scratchBase, uint32_t loadedAddresses,
                             const uint32_t* inputAddresses,
                             uint32_t off, uint32_t size) {
        bool direct = false;
        for (uint32_t c = 0; c < count; ++c)
            direct |= DirectInput(slot + c, scratchBase, loadedAddresses, inputAddresses);
        if (direct) {
            uint32_t stride = (size + 15U) & ~15U;
            for (uint32_t c = 0; c < count; ++c)
                LoadInputRow(ub, c * stride, slot + c, scratchBase, loadedAddresses, inputAddresses, off, size);
            return;
        }
        GlobalTensor<bf16_t> gm;
        gm.SetGlobalBuffer(Row(p.relay_hidden, slot) + off, uint64_t(count - 1U) * h + size);
        DataCopyExtParams cp{static_cast<uint16_t>(count), size * 2U, (h - size) * 2U, 0U, 0U};
        DataCopyPadExtParams<bf16_t> pad{false, 0U, 0U, 0U};
        DataCopyPad(ub, gm, cp, pad);
    }

    __aicore__ void LoadPair(LocalTensor<bf16_t> ub, uint32_t a, uint32_t b,
                            uint32_t scratchBase, uint32_t loadedAddresses,
                            const uint32_t* inputAddresses,
                            uint32_t off, uint32_t size, uint32_t count) {
        bool direct = false;
        for (uint32_t c = 0; c < 2U; ++c)
            for (uint32_t group = 0; group < count; ++group)
                direct |= DirectInput((group ? b : a) + c, scratchBase, loadedAddresses, inputAddresses);
        if (direct) {
            for (uint32_t c = 0; c < 2U; ++c)
                for (uint32_t group = 0; group < count; ++group)
                    LoadInputRow(ub, c * 2U * size + group * size, (group ? b : a) + c,
                                 scratchBase, loadedAddresses, inputAddresses, off, size);
            return;
        }
        // Contributor-major: [A0,B0,A1,B1], just like combine_v2.
        for (uint32_t c = 0; c < 2U; ++c) {
            GlobalTensor<bf16_t> gm;
            gm.SetGlobalBuffer(Row(p.relay_hidden, a + c) + off,
                              uint64_t(count == 2U ? b - a : 0U) * h + size);
            DataCopyExtParams cp{static_cast<uint16_t>(count), size * 2U,
                count == 2U ? static_cast<uint32_t>((uint64_t(b - a) * h - size) * 2U) : 0U, 0U, 0U};
            DataCopyPadExtParams<bf16_t> pad{false, 0U, 0U, 0U};
            DataCopyPad(ub[c * 2U * size], gm, cp, pad);
        }
    }

    __aicore__ void ReducePairs(const uint32_t* slots, uint32_t count, uint32_t scratchBase,
                                uint32_t loadedAddresses, const uint32_t* inputAddresses,
                                uint32_t off, uint32_t size) {
        if (!count) return;
        auto acc = accumBuf.Get<float>(), tmp = tempBuf.Get<float>();
        auto page = inputQ.template AllocTensor<bf16_t>();
        uint32_t rows = count < 2U ? count : 2U;
        LoadPair(page, slots[0], rows == 2U ? slots[1] : slots[0], scratchBase, loadedAddresses, inputAddresses, off, size, rows);
        inputQ.EnQue(page);
        for (uint32_t i = 0; i < count; i += 2U) {
            rows = count - i < 2U ? count - i : 2U;
            page = inputQ.template DeQue<bf16_t>();
            if (i + 2U < count) {
                auto next = inputQ.template AllocTensor<bf16_t>();
                uint32_t more = count - i - 2U < 2U ? count - i - 2U : 2U;
                LoadPair(next, slots[i + 2U], more == 2U ? slots[i + 3U] : slots[i + 2U], scratchBase, loadedAddresses, inputAddresses, off, size, more);
                inputQ.EnQue(next);
            }
            Cast(acc, page, RoundMode::CAST_NONE, rows * size);
            PipeBarrier<PIPE_V>();
            Cast(tmp, page[2U * size], RoundMode::CAST_NONE, rows * size);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, tmp, rows * size);
            PipeBarrier<PIPE_V>();
            inputQ.FreeTensor(page);
            auto output = outputQ.template AllocTensor<bf16_t>();
            Cast(output, acc, RoundMode::CAST_RINT, rows * size);
            outputQ.EnQue(output);
            output = outputQ.template DeQue<bf16_t>();
            for (uint32_t j = 0; j < rows; ++j)
                Store(Row(p.relay_hidden, slots[i + j]) + off, output[j * size], size);
            outputQ.FreeTensor(output);
        }
    }

    __aicore__ void ReduceGroups(const uint32_t* slots, const uint32_t* lengths,
                                uint32_t count, uint32_t scratchBase, uint32_t loadedAddresses,
                                const uint32_t* inputAddresses,
                                uint32_t off, uint32_t size) {
        if (!count) return;
        uint32_t stride = (size + 15U) & ~15U;
        auto acc = accumBuf.Get<float>(), tmp = tempBuf.Get<float>();
        auto page = inputQ.template AllocTensor<bf16_t>();
        LoadGroup(page, slots[0], lengths[0], scratchBase, loadedAddresses, inputAddresses, off, size);
        inputQ.EnQue(page);
        for (uint32_t i = 0; i < count; ++i) {
            page = inputQ.template DeQue<bf16_t>();
            if (i + 1U < count) {
                auto next = inputQ.template AllocTensor<bf16_t>();
                LoadGroup(next, slots[i + 1U], lengths[i + 1U], scratchBase, loadedAddresses, inputAddresses, off, size);
                inputQ.EnQue(next);
            }
            Cast(acc, page, RoundMode::CAST_NONE, size);
            PipeBarrier<PIPE_V>();
            for (uint32_t c = 1; c < lengths[i]; ++c) {
                Cast(tmp, page[c * stride], RoundMode::CAST_NONE, size);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, tmp, size);
                PipeBarrier<PIPE_V>();
            }
            inputQ.FreeTensor(page);
            auto output = outputQ.template AllocTensor<bf16_t>();
            Cast(output, acc, RoundMode::CAST_RINT, size);
            outputQ.EnQue(output);
            output = outputQ.template DeQue<bf16_t>();
            Store(Row(p.relay_hidden, slots[i]) + off, output, size);
            outputQ.FreeTensor(output);
        }
    }

    __aicore__ void ReduceLarge(uint32_t slot, uint32_t count, uint32_t scratchBase,
                                uint32_t loadedAddresses, const uint32_t* inputAddresses,
                                uint32_t off, uint32_t size) {
        // >4 contributors: bounded four-route input pages, prefetch within group.
        auto acc = accumBuf.Get<float>(), tmp = tempBuf.Get<float>();
        uint32_t stride = (size + 15U) & ~15U;
        auto page = inputQ.template AllocTensor<bf16_t>();
        LoadGroup(page, slot, count < 4U ? count : 4U, scratchBase, loadedAddresses, inputAddresses, off, size);
        inputQ.EnQue(page);
        for (uint32_t first = 0; first < count; first += 4U) {
            page = inputQ.template DeQue<bf16_t>();
            if (first + 4U < count) {
                auto next = inputQ.template AllocTensor<bf16_t>();
                uint32_t more = count - first - 4U;
                LoadGroup(next, slot + first + 4U, more < 4U ? more : 4U, scratchBase, loadedAddresses, inputAddresses, off, size);
                inputQ.EnQue(next);
            }
            uint32_t rows = count - first < 4U ? count - first : 4U;
            for (uint32_t j = 0; j < rows; ++j) {
                if (first == 0U && j == 0U) Cast(acc, page, RoundMode::CAST_NONE, size);
                else {
                    Cast(tmp, page[j * stride], RoundMode::CAST_NONE, size);
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, tmp, size);
                }
                PipeBarrier<PIPE_V>();
            }
            inputQ.FreeTensor(page);
        }
        auto output = outputQ.template AllocTensor<bf16_t>();
        Cast(output, acc, RoundMode::CAST_RINT, size);
        outputQ.EnQue(output);
        output = outputQ.template DeQue<bf16_t>();
        Store(Row(p.relay_hidden, slot) + off, output, size);
        outputQ.FreeTensor(output);
    }

    __aicore__ void Reduce() {
        InitVector(false);
        uint32_t group = (core - kCombinePutReducerBegin) /
                         kCombinePutReducerShards;
        uint32_t shard = (core - kCombinePutReducerBegin) %
                         kCombinePutReducerShards;
        uint32_t batchOwnerRotation = 0U;
        for (uint32_t round = 0; ; ++round) {
            uint32_t ord = CombinePutReducerOrdinal(r, group, round);
            if (ord >= r) break;
            uint32_t source = Source(ord);
            auto sc = Meta(p.source_counts, source);
            if (LocalSource(source) || sc[5] == 0) continue;
            if (shard + 1U == kCombinePutReducerShards) {
                uint32_t producers = server ? (r < 8U ? r : 8U) : 1U;
                for (uint32_t local = 0; local < producers; ++local)
                    Wait(Flag(p.relay_ready, uint64_t(local) * r + source), magic);
                Publish(Flag(p.source_input_ready, source), magic);
            } else Wait(Flag(p.source_input_ready, source), magic);
            uint32_t batchCount =
                (sc[5] + kCombinePutBatchRows - 1U) / kCombinePutBatchRows;
            uint32_t firstBatch =
                CombinePutReducerFirstBatch(shard, batchOwnerRotation);
            for (uint32_t first = firstBatch * kCombinePutBatchRows;
                 first < static_cast<uint32_t>(sc[5]);
                 first += kCombinePutReducerShards * kCombinePutBatchRows) {
                uint32_t size = sc[5] - first < kCombinePutBatchRows
                                    ? sc[5] - first
                                    : kCombinePutBatchRows;
                uint32_t slots[64], lengths[64], pairs[64], generic[64], sizes[64];
                uint32_t valid = 0, pairCount = 0;
                auto desc = descriptorBuf.Get<uint64_t>();
                auto inputAddressDesc = inputAddressBuf.Get<uint64_t>();
                uint32_t inputAddresses[64];
                LoadDescriptors(desc, sc[4] + first, size);
                LoadInputAddresses(inputAddressDesc, sc[4] + first, size);
                for (uint32_t i = 0; i < size; ++i) {
                    inputAddresses[i] = static_cast<uint32_t>(inputAddressDesc.GetValue(i * 4U));
                    uint64_t value = desc.GetValue(i * 4U);
                    uint32_t count = value >> 32U;
                    if (!count) continue;
                    slots[valid] = sc[4] + first + i; lengths[valid++] = count;
                    if (count == 2U) pairs[pairCount++] = sc[4] + first + i;
                }
                for (uint32_t off = 0; off < h; off += tile) {
                    uint32_t elems = h - off < tile ? h - off : tile;
                    bool paired = elems % 16U == 0U;
                    if (paired) ReducePairs(pairs, pairCount, sc[4] + first, size, inputAddresses, off, elems);
                    uint32_t genericCount = 0;
                    for (uint32_t i = 0; i < valid; ++i) {
                        if (lengths[i] <= 4U && !(paired && lengths[i] == 2U)) {
                            generic[genericCount] = slots[i]; sizes[genericCount++] = lengths[i];
                        }
                    }
                    ReduceGroups(generic, sizes, genericCount, sc[4] + first, size, inputAddresses, off, elems);
                    for (uint32_t i = 0; i < valid; ++i)
                        if (lengths[i] > 4U) ReduceLarge(slots[i], lengths[i], sc[4] + first, size, inputAddresses, off, elems);
                }
                PublishMte(source, first / kCombinePutBatchRows);
            }
            batchOwnerRotation =
                (batchOwnerRotation + batchCount) % kCombinePutReducerShards;
        }
        PipeBarrier<PIPE_ALL>(); // retire queued result/ready stores before role transition
    }

    // Shared >=128P first-reduce: AIV8..23 form eight groups with two
    // shards each.  Shard 0/1 owns even/odd 64-row batches respectively.
    __aicore__ void Reduce128() {
        InitVector(false);
        const uint32_t group = (core - kCombinePut128ReducerBegin) /
                               kCombinePut128ReducerShards;
        const uint32_t shard = (core - kCombinePut128ReducerBegin) %
                               kCombinePut128ReducerShards;
        for (uint32_t round = 0U; ; ++round) {
            const uint32_t ord = CombinePutReducerOrdinal(r, group, round);
            if (ord >= r) break;
            const uint32_t source = Source(ord);
            auto sc = Meta(p.source_counts, source);
            if (LocalSource(source) || sc[5] == 0) continue;
            if (shard + 1U == kCombinePut128ReducerShards) {
                const uint32_t producers = server ? 8U : 1U;
                for (uint32_t local = 0U; local < producers; ++local)
                    Wait(Flag(p.relay_ready, uint64_t(local) * r + source), magic);
                Publish(Flag(p.source_input_ready, source), magic);
            } else {
                Wait(Flag(p.source_input_ready, source), magic);
            }

            for (uint32_t first = shard * kCombinePutBatchRows;
                 first < static_cast<uint32_t>(sc[5]);
                 first += kCombinePut128ReducerShards * kCombinePutBatchRows) {
                const uint32_t size = sc[5] - first < kCombinePutBatchRows
                    ? sc[5] - first : kCombinePutBatchRows;
                uint32_t slots[64], lengths[64], pairs[64], generic[64], sizes[64];
                uint32_t valid = 0U, pairCount = 0U;
                auto desc = descriptorBuf.Get<uint64_t>();
                auto inputAddressDesc = inputAddressBuf.Get<uint64_t>();
                uint32_t inputAddresses[64];
                LoadDescriptors(desc, sc[4] + first, size);
                LoadInputAddresses(inputAddressDesc, sc[4] + first, size);
                for (uint32_t i = 0U; i < size; ++i) {
                    inputAddresses[i] = static_cast<uint32_t>(
                        inputAddressDesc.GetValue(i * 4U));
                    const uint64_t value = desc.GetValue(i * 4U);
                    const uint32_t count = value >> 32U;
                    if (!count) continue;
                    slots[valid] = sc[4] + first + i;
                    lengths[valid++] = count;
                    if (count == 2U) pairs[pairCount++] = sc[4] + first + i;
                }
                for (uint32_t off = 0U; off < h; off += tile) {
                    const uint32_t elems = h - off < tile ? h - off : tile;
                    const bool paired = elems % 16U == 0U;
                    if (paired)
                        ReducePairs(pairs, pairCount, sc[4] + first, size,
                                    inputAddresses, off, elems);
                    uint32_t genericCount = 0U;
                    for (uint32_t i = 0U; i < valid; ++i) {
                        if (lengths[i] <= 4U && !(paired && lengths[i] == 2U)) {
                            generic[genericCount] = slots[i];
                            sizes[genericCount++] = lengths[i];
                        }
                    }
                    ReduceGroups(generic, sizes, genericCount, sc[4] + first,
                                 size, inputAddresses, off, elems);
                    for (uint32_t i = 0U; i < valid; ++i)
                        if (lengths[i] > 4U)
                            ReduceLarge(slots[i], lengths[i], sc[4] + first,
                                        size, inputAddresses, off, elems);
                }
                PublishMte(source, first / kCombinePutBatchRows);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // source: destination source rank of this chunk; PUT writes into that rank's final_hidden.
    // lane: sending lane and SHMEM QP ID (0=QP0, 1=QP1).
    // ordinal: chunk index within this source; chunks with ordinal%4==0 belong to QP1,
    //          all others to QP0. Non-owner lanes return immediately so each chunk is sent exactly once.
    // inputs: up to 64 local input-row indices into hidden if direct=true, otherwise into relay_hidden.
    // targets: corresponding destination-row indices in the source rank's final_hidden.
    // count: valid rows in this chunk, 1..64; the last chunk may contain fewer than 64 rows.
    // direct: true for unreduced DIRECT rows, false for group-head rows after server reduce.
    // qp0_only: the last 64P group has only eight remote sources; send all its payload through QP0.
    __aicore__ void PostChunk(uint32_t source, uint32_t lane, uint32_t ordinal,
                             const uint32_t* inputs, const uint32_t* targets,
                             uint32_t count, bool direct, bool qp0_only) {
        if (!count || (qp0_only ? lane != 0U
                                : ((ordinal & 3U) == 0U ? 1U : 0U) != lane)) {
            return;
        }
        __gm__ bf16_t* batch_inputs[64U];
        __gm__ bf16_t* batch_targets[64U];
        for (uint32_t j = 0; j < count; ++j) {
            batch_inputs[j] = direct ? hidden + uint64_t(inputs[j]) * h
                                     : Row(p.relay_hidden, inputs[j]);
            batch_targets[j] = Row(p.final_hidden, targets[j]);
        }
        if (source == rank) {
            for (uint32_t j = 0; j < count; ++j)
                Copy(batch_targets[j], batch_inputs[j], h);
        } else {
            SubmitPutBatch(Row(p.final_hidden, 0U), batch_inputs, batch_targets,
                               count, source, lane);
        }
    }

    /** Source: target source rank for PUT; lane: QP ID; pass: DIRECT rows (0) or reduced rows (1). */
    __aicore__ void SendSource(uint32_t source, uint32_t lane, uint32_t pass) {
        if (LocalSource(source)) {
            return; // Phase-A core already returns ALL local routes.
        }

        auto sc = Meta(p.source_counts, source);
        // In the 64P tail wave there are only eight remote peers.  QP0 owns
        // all chunks.  QP1 owns no payload, ready WQE or CQE for this source;
        // return before pass=1 can wait on reducer batches that it will never
        // submit.  Final() already skips this absent QP1 ready flag.
        const bool qp0_only = CombinePutSenderTailQp0Only(rank, source, r);
        if (qp0_only && lane == 1U) {
            return;
        }
        uint32_t inputs[64], targets[64], selected = 0, chunk = 0;
        if (pass == 0U) {
            // Restore the original direct path: scan tx_rows in source order.
            for (uint32_t i = 0; i < static_cast<uint32_t>(sc[0]); ++i) {
                auto tx = Meta(p.tx_rows, sc[2] + i);
                Flush(tx);
                if (tx[3] != 1U) continue;
                inputs[selected] = tx[0];
                targets[selected++] = sc[1] + i;
                if (selected == 64U) {
                    PostChunk(source, lane, chunk++, inputs, targets,
                              selected, true, qp0_only);
                    selected = 0;
                }
            }
        } else {
            // The sender scans the SAME 64 scratch slots as the reducer. Only
            // group heads become payload rows; contributor holes are not work.
            auto desc = Ub<uint64_t>(144U * 1024U, 64U * 4U);
            // sc[0]: total result rows to return.
            // sc[5]: staged contributor rows used to produce the duplicate-group results.
            for (uint32_t first = 0; first < static_cast<uint32_t>(sc[5]); first += 64U) {
                uint32_t size = sc[5] - first < 64U ? sc[5] - first : 64U;
                Wait(Flag(p.reduce_ready, uint64_t(source) * p.batches + first / 64U), magic);
                LoadDescriptors(desc, sc[4] + first, size);
                for (uint32_t j = 0; j < size; ++j) {
                    uint64_t value = desc.GetValue(j * 4U);
                    if (!(value >> 32U)) {
                        continue;
                    }
                    uint32_t txIndex = static_cast<uint32_t>(value);
                    inputs[selected] = sc[4] + first + j;
                    targets[selected++] = sc[1] + txIndex - sc[2];
                    if (selected == 64U) {
                        // chunk 0,4,8,... -> lane=1 -> QP1
                        // chunk 1,2,3,5,6,7,... -> lane=0 -> QP0
                        PostChunk(source, lane, chunk++, inputs, targets, selected, false,
                                  qp0_only);
                        selected = 0;
                    }
                }
            }
        }

        if (selected) {
            PostChunk(source, lane, chunk, inputs, targets, selected, pass == 0U,
                      qp0_only);
        }

        if (pass == 1U) {
            // Each QP's SO completion closes BOTH direct and reduced streams.
            // The 64P half-full tail is QP0-only: lane 1 owns neither payload
            // nor completion there, so it must not create a ready WQE/CQE.
            aclshmemx_submit_state_t state{};
            if (source == rank) {
                Publish(Flag(p.final_ready, uint64_t(rank) * 2U + lane), magic);
            } else if (!(qp0_only && lane == 1U)) {
                SendReady(p.final_ready, uint64_t(rank) * 2U + lane, source, lane, state);
            }

            // After submitting each source's result/ready, immediately drain only this core's exclusive QP.
            // This also completes its earlier DIRECT submissions; skip the empty QP1 in the 64P tail group.
            if (source != rank && !(qp0_only && lane == 1U)) {
                Drain(source, lane);
            }
        }
    }

    // >=128P DIRECT: two-page MTE2 prefetch; scalar reads use UB instead of per-record GM access/Flush.
    // Metadata pages are not send batches: retain selected across pages and collect 64 rows
    // in original order before PostChunk. Per-page tail batches would change chunk/QP assignment.
    __aicore__ void PostDirect128(uint32_t source, uint32_t count,
                                  uint32_t txBase, uint32_t targetBase,
                                  uint32_t& payloadChunkOrdinal, uint32_t ownerQp) {
        LocalTensor<uint32_t> txPages[2] = {
            Ub<uint32_t>(120U * 1024U, 64U * 8U),
            Ub<uint32_t>(122U * 1024U, 64U * 8U),
        }; // 2KiB per page; does not overlap the WQE buffer starting at 128KiB.
        uint32_t inputs[64], targets[64], selected = 0U, current = 0U;
        if (count) SubmitDirectTxRowLoad128(txPages[current], txBase,
                                          count < 64U ? count : 64U);
        for (uint32_t first = 0U; first < count; first += 64U) {
            const uint32_t size = count - first < 64U ? count - first : 64U;
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            const uint32_t next = current ^ 1U, nextFirst = first + 64U;
            if (nextFirst < count) {
                const uint32_t nextSize = count - nextFirst < 64U ? count - nextFirst : 64U;
                SubmitDirectTxRowLoad128(txPages[next], txBase + nextFirst, nextSize);
            }
            // Prefetch the next page before filtering this page to overlap MTE2 with filtering/scalar WQE submission.
            for (uint32_t j = 0U; j < size; ++j) {
                if (txPages[current].GetValue(j * 8U + 3U) != 1U) continue;
                inputs[selected] = txPages[current].GetValue(j * 8U);
                targets[selected++] = targetBase + first + j;
                if (selected == 64U) {
                    // Both QP cores enumerate the same chunks; PostChunk first excludes batches owned by the other QP.
                    PostChunk(source, ownerQp, payloadChunkOrdinal++, inputs, targets,
                              selected, true, false);
                    selected = 0U;
                }
            }
            current = next;
        }
        // The last page's MTE2_S is consumed; empty sources/DIRECT streams leave no pending events.
        if (selected) {
            PostChunk(source, ownerQp, payloadChunkOrdinal++, inputs, targets,
                      selected, true, false);
        }
    }

    // >=128P: cores 24..39 own QP0, 40..55 own QP1; no cross-core SQ/CQ handoff.
    // Each logical source lane retains its cumulative DIRECT+result chunk ordinal,
    // unchanged from before the core split; core24+j and core40+j must enumerate identical counts.
    __aicore__ void SendSource128(uint32_t source, uint32_t pass, uint32_t ownerQp,
                                 uint32_t& payloadChunkOrdinal) {
        if (LocalSource(source)) return;

        auto sc = Meta(p.source_counts, source);
        uint32_t inputs[64], targets[64], selected = 0U;
        if (pass == 0U) {
            PostDirect128(source, static_cast<uint32_t>(sc[0]),
                          static_cast<uint32_t>(sc[2]), static_cast<uint32_t>(sc[1]),
                          payloadChunkOrdinal, ownerQp);
        } else {
            auto desc = Ub<uint64_t>(144U * 1024U, 64U * 4U);
            for (uint32_t first = 0; first < static_cast<uint32_t>(sc[5]);
                 first += kCombinePutBatchRows) {
                const uint32_t size = sc[5] - first < kCombinePutBatchRows
                    ? sc[5] - first : kCombinePutBatchRows;
                // Each QP core independently waits for/reads descriptors and submits only its own chunks.
                Wait(Flag(p.reduce_ready,
                           uint64_t(source) * p.batches + first / kCombinePutBatchRows), magic);
                LoadDescriptors(desc, sc[4] + first, size);
                for (uint32_t j = 0; j < size; ++j) {
                    const uint64_t value = desc.GetValue(j * 4U);
                    if (!(value >> 32U)) continue;
                    const uint32_t txIndex = static_cast<uint32_t>(value);
                    inputs[selected] = sc[4] + first + j;
                    targets[selected++] = sc[1] + txIndex - sc[2];
                    if (selected == 64U) {
                        PostChunk(source, ownerQp, payloadChunkOrdinal++, inputs, targets,
                                  selected, false, false);
                        selected = 0U;
                    }
                }
            }
        }

        if (selected) {
            PostChunk(source, ownerQp, payloadChunkOrdinal++, inputs, targets,
                      selected, pass == 0U, false);
        }

        if (pass == 0U) {
            // At 128P, submit this peer's DIRECT, result, and SO-ready before
            // reclaiming its CQ. Protective polling for SQ space stays enabled.
            // Keep the existing DIRECT drain policy at all other world sizes.
            if (r != 128U && (r == 256U || ownerQp == 1U)) {
                Drain(source, ownerQp);
            }
        } else {
            // Publish ready only for this QP, even if it has no payload.
            // Final waits for both QPs; never publish early on behalf of the other core.
            aclshmemx_submit_state_t state{};
            SendReady(p.final_ready, uint64_t(rank) * 2U + ownerQp,
                      source, ownerQp, state);
            // Ready is ordered after this QP's DIRECT and reduced payload.
            // Each sender core exclusively owns this peer/QP's completion queue.
            if (r == 128U) {
                Drain(source, ownerQp);
            }
        }
    }

    __aicore__ void Send128() {
        const uint32_t ownerQp = CombinePutSenderLane128(core);
        const uint32_t worker = CombinePutSenderWorker128(core);
        uint32_t payloadChunkOrdinal = 0U;
        for (uint32_t pass = 0U; pass < 2U; ++pass) {
            for (uint32_t phase = 0U; phase * kCombinePutSourceLanes < r; ++phase) {
                const uint32_t ord = phase * kCombinePutSourceLanes + worker;
                const uint32_t source = Source(ord);
                SendSource128(source, pass, ownerQp, payloadChunkOrdinal);
            }
        }
        // At 128P all nonlocal peer/QP queues were drained after their ready WQE.
        if (r == 128U) return;
        // Complete result/ready here for both QPs. Outside 256P, this also
        // drains QP0 DIRECT; at 256P both QPs already drain DIRECT per peer.
        for (uint32_t phase = 0U; phase * kCombinePutSourceLanes < r; ++phase) {
            const uint32_t source = Source(phase * kCombinePutSourceLanes + worker);
            if (LocalSource(source)) continue;
            Drain(source, ownerQp);
        }
    }

    __aicore__ void Send() {
        uint32_t lane = CombinePutSenderLane(core); // QP assignment: cores 40..55 use QP0, cores 56..63 use QP1.
        uint32_t workers = lane == 0U ? 1U : 2U; // Source workers per core: one for QP0, two for QP1.
        for (uint32_t pass = 0; pass < 2U; ++pass) {
            // pass=0: DIRECT rows that do not need server reduce.
            // pass=1: duplicate-group results after server reduce.
            for (uint32_t phase = 0; phase * 16U < r; ++phase) {
                // Each phase covers a group of 16 sources.
                for (uint32_t i = 0; i < workers; ++i) {
                    // One iteration for QP0, two for QP1.
                    uint32_t ord = phase * 16U + CombinePutSenderWorker(core, i);
                    if (ord < r) {
                        SendSource(Source(ord), lane, pass);
                    }
                }
            }
        }
    }

    __aicore__ __gm__ uint32_t* FinalRoutes(uint32_t token) {
        return reinterpret_cast<__gm__ uint32_t*>(
            (execution_plan + plan_layout.final_routes) + uint64_t(token) * 128U);
    }

    __aicore__ __gm__ bf16_t* FinalInput(uint32_t route) {
        return (route & 0x80000000U)
            ? hidden + uint64_t(route & 0x7fffffffU) * h
            : Row(p.final_hidden, route);
    }

    __aicore__ void PrefetchFinalRoutes(
        uint32_t token, uint32_t bank, LocalTensor<uint32_t> routeEntries,
        bool routeReusable[2]) {
        const TEventID event = bank == 0U ? EVENT_ID2 : EVENT_ID3;
        if (routeReusable[bank]) {
            // Scalar has finished reading this UB bank; only now may MTE2
            // overwrite it with the descriptor for a later token.
            SetFlag<HardEvent::S_MTE2>(event);
            WaitFlag<HardEvent::S_MTE2>(event);
            routeReusable[bank] = false;
        }
        Load(routeEntries[bank * 32U], FinalRoutes(token), 32U);
        SetFlag<HardEvent::MTE2_S>(event);
    }

    // A singleton token needs no BF16->FP32 cast, accumulator, add, or
    // FP32->BF16 cast.  Copy its only contributor directly through one input
    // UB page.  The copy is completed before that page is reused.
    __aicore__ void FinalCopyToken(
        uint32_t token, uint32_t route, LocalTensor<bf16_t> inputRows,
        bool inputPending[2]) {
        if (inputPending[0]) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            inputPending[0] = false;
        }
        auto input = inputRows[0];
        auto source = FinalInput(route);
        for (uint32_t off = 0U; off < h; off += kAlignedFinalTile) {
            const uint32_t size = h - off < kAlignedFinalTile ?
                h - off : kAlignedFinalTile;
            Load(input, source + off, size);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            Store(out + uint64_t(token) * h + off, input, size);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    // TileXR source-final pipeline for one token.  `routes` has already been
    // decoded once from final_map, so the fallback never re-reads route
    // metadata.  Two MTE2 banks prefetch the next four contributors while V
    // reduces the current four; two MTE3 slots let the next token proceed while
    // the preceding output store retires.
    __aicore__ void FinalReduceToken(
        uint32_t token, LocalTensor<uint32_t> routes, uint32_t routeCount,
        LocalTensor<bf16_t> inputRows, LocalTensor<bf16_t> outputRows,
        LocalTensor<float> accumulator, LocalTensor<float> temporary,
        bool inputPending[2],
        bool outputPending[2], uint32_t& outputSlot) {
        for (uint32_t off = 0U; off < h; off += kAlignedFinalTile) {
            const uint32_t size = h - off < kAlignedFinalTile ? h - off : kAlignedFinalTile;
            const uint32_t slot = outputSlot;
            const TEventID outputEvent = slot == 0U ? EVENT_ID2 : EVENT_ID3;
            if (outputPending[slot]) {
                WaitFlag<HardEvent::MTE3_V>(outputEvent);
                outputPending[slot] = false;
            }

            uint32_t begin = 0U, bank = 0U;
            uint32_t batchCount = routeCount < kAlignedFinalRouteBatch ?
                routeCount : kAlignedFinalRouteBatch;
            TEventID inputEvent = EVENT_ID0;
            if (inputPending[bank]) {
                WaitFlag<HardEvent::V_MTE2>(inputEvent);
                inputPending[bank] = false;
            }
            for (uint32_t row = 0U; row < batchCount; ++row)
                Load(inputRows[row * kAlignedFinalTile],
                     FinalInput(routes.GetValue(row)) + off, size);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            bool accumulatorValid = false;
            while (true) {
                const uint32_t nextBegin = begin + batchCount;
                const bool finalBatch = nextBegin >= routeCount;
                if (!finalBatch) {
                    const uint32_t nextBank = bank ^ 1U;
                    const TEventID nextInputEvent = nextBank == 0U ? EVENT_ID0 : EVENT_ID1;
                    if (inputPending[nextBank]) {
                        WaitFlag<HardEvent::V_MTE2>(nextInputEvent);
                        inputPending[nextBank] = false;
                    }
                    const uint32_t nextCount = routeCount - nextBegin < kAlignedFinalRouteBatch ?
                        routeCount - nextBegin : kAlignedFinalRouteBatch;
                    for (uint32_t row = 0U; row < nextCount; ++row)
                        Load(inputRows[(nextBank * kAlignedFinalRouteBatch + row) *
                                           kAlignedFinalTile],
                             FinalInput(routes.GetValue(nextBegin + row)) + off, size);
                    SetFlag<HardEvent::MTE2_V>(nextInputEvent);
                }

                WaitFlag<HardEvent::MTE2_V>(inputEvent);
                for (uint32_t row = 0U; row < batchCount; ++row) {
                    auto input = inputRows[(bank * kAlignedFinalRouteBatch + row) *
                                           kAlignedFinalTile];
                    if (!accumulatorValid && row == 0U) {
                        Cast(accumulator, input, RoundMode::CAST_NONE, size);
                        accumulatorValid = true;
                    } else {
                        Cast(temporary, input, RoundMode::CAST_NONE, size);
                        PipeBarrier<PIPE_V>();
                        Add(accumulator, accumulator, temporary, size);
                        PipeBarrier<PIPE_V>();
                    }
                }
                if (finalBatch)
                    Cast(outputRows[slot * kAlignedFinalTile], accumulator,
                         RoundMode::CAST_RINT, size);
                SetFlag<HardEvent::V_MTE2>(inputEvent);
                inputPending[bank] = true;
                if (finalBatch) {
                    SetFlag<HardEvent::V_MTE3>(outputEvent);
                    WaitFlag<HardEvent::V_MTE3>(outputEvent);
                    Store(out + uint64_t(token) * h + off,
                          outputRows[slot * kAlignedFinalTile], size);
                    SetFlag<HardEvent::MTE3_V>(outputEvent);
                    outputPending[slot] = true;
                    outputSlot ^= 1U;
                    break;
                }
                begin = nextBegin;
                bank ^= 1U;
                batchCount = routeCount - begin < kAlignedFinalRouteBatch ?
                    routeCount - begin : kAlignedFinalRouteBatch;
                inputEvent = bank == 0U ? EVENT_ID0 : EVENT_ID1;
            }
        }
    }

    // Shared >=128P per-token computation matches ReduceOneToken in moonep-r18-128p-2454us.
    // Preserve this planner's compact valid routes in their original order and zero-copy self-input addressing.
    // Queues manage MTE2/V/MTE3 dependencies instead of 64P's fixed 4096-row stride and manual bank events.
    __aicore__ void FinalReduceToken128(
        uint32_t token, LocalTensor<uint32_t> routes, uint32_t routeCount) {
        auto accumulator = finalAccumulatorBuf.Get<float>();
        auto row = finalTempBuf.Get<float>();
        for (uint32_t off = 0U; off < h; off += kAlignedFinalTile) {
            const uint32_t size = h - off < kAlignedFinalTile ? h - off : kAlignedFinalTile;
            // Match the reference MoonEpCombineV2ReduceInputStrideElements:
            // align this tile to 32B; H3584 uses stride 3584 instead of a fixed 4096.
            const uint32_t stride = (size + 15U) / 16U * 16U;
            if (routeCount == 1U) {
                // CopyOneFinalRoute: a single input goes directly MTE2->MTE3 without floating-point conversion.
                auto input = inputQ.AllocTensor<bf16_t>();
                Load(input, FinalInput(routes.GetValue(0U)) + off, size);
                inputQ.EnQue(input);
                input = inputQ.DeQue<bf16_t>();
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                Store(out + uint64_t(token) * h + off, input, size);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                inputQ.FreeTensor(input);
                continue;
            }
            Duplicate(accumulator, 0.0f, size);
            PipeBarrier<PIPE_V>();
            for (uint32_t begin = 0U; begin < routeCount; begin += kAlignedFinalRouteBatch) {
                const uint32_t count = routeCount - begin < kAlignedFinalRouteBatch
                    ? routeCount - begin : kAlignedFinalRouteBatch;
                auto input = inputQ.AllocTensor<bf16_t>();
                for (uint32_t j = 0U; j < count; ++j)
                    Load(input[j * stride], FinalInput(routes.GetValue(begin + j)) + off, size);
                inputQ.EnQue(input);
                input = inputQ.DeQue<bf16_t>();
                for (uint32_t j = 0U; j < count; ++j) {
                    Cast(row, input[j * stride], RoundMode::CAST_NONE, size);
                    PipeBarrier<PIPE_V>();
                    Add(accumulator, accumulator, row, size);
                    PipeBarrier<PIPE_V>();
                }
                inputQ.FreeTensor(input);
            }
            // routeCount==0 also takes this path, producing zeros instead of retaining the previous output.
            auto output = outputQ.AllocTensor<bf16_t>();
            Cast(output, accumulator, RoundMode::CAST_RINT, size);
            outputQ.EnQue(output);
            output = outputQ.DeQue<bf16_t>();
            Store(out + uint64_t(token) * h + off, output, size);
            outputQ.FreeTensor(output);
        }
    }

    __aicore__ void FinalReference128() {
        // Double queues match the reference InitReduceBuffers sizes; Final has already completed fan-in.
        pipe.Reset();
        pipe.InitBuffer(inputQ, 2U, kAlignedFinalRouteBatch * kAlignedFinalTile * sizeof(bf16_t));
        pipe.InitBuffer(outputQ, 2U, kAlignedFinalTile * sizeof(bf16_t));
        pipe.InitBuffer(finalAccumulatorBuf, kAlignedFinalTile * sizeof(float));
        pipe.InitBuffer(finalTempBuf, kAlignedFinalTile * sizeof(float));
        pipe.InitBuffer(finalRouteBuf, 2U * 128U);
        auto routeEntries = finalRouteBuf.Get<uint32_t>();
        bool routeReusable[2] = {false, false};
        const uint32_t tokenCount = n / k;
        uint32_t bank = 0U;
        if (core < tokenCount) PrefetchFinalRoutes(core, bank, routeEntries, routeReusable);
        for (uint32_t token = core; token < tokenCount; token += kCores) {
            WaitFlag<HardEvent::MTE2_S>(bank == 0U ? EVENT_ID2 : EVENT_ID3);
            const uint32_t count = routeEntries.GetValue(bank * 32U);
            const uint32_t nextBank = bank ^ 1U;
            if (token + kCores < tokenCount)
                PrefetchFinalRoutes(token + kCores, nextBank, routeEntries, routeReusable);
            FinalReduceToken128(token, routeEntries[bank * 32U + 1U], count);
            routeReusable[bank] = true;
            bank = nextBank;
        }
        PipeBarrier<PIPE_ALL>();
        pipe.Reset();
    }

    __aicore__ void Final() {
        // Parallel final fan-in.  At 64P, every core owns exactly one source
        // peer: 56 cores wait one remote peer and the remaining eight wait the
        // local-server peers.  This replaces core0's serial 120-flag scan.
        // For larger worlds a core advances by kCores, preserving correctness.
        for (uint32_t ord = core; ord < r; ord += kCores) {
            const uint32_t peer = Source(ord);
            Wait(Flag(p.final_ready, uint64_t(peer) * 2U), magic);
            // This is the reverse direction: decide using the producer peer's
            // schedule.  A 64P tail wave has no QP1 payload, ready WQE or CQE.
            if (!LocalSource(peer) &&
                !CombinePutSenderTailQp0Only(peer, rank, r)) {
                Wait(Flag(p.final_ready, uint64_t(peer) * 2U + 1U), magic);
            }
        }
        // A rank-local core barrier is the new fan-in gate: nobody reads
        // final_hidden until every peer's completion has been observed.
        aclshmemi_sync_core<true>();
        if (r >= 128U) {
            FinalReference128();
            return;
        }
        InitVector(true);
        auto inputRows = finalInputBuf.Get<bf16_t>();
        auto outputRows = finalOutputBuf.Get<bf16_t>();
        auto accumulator = finalAccumulatorBuf.Get<float>();
        auto temporary = finalTempBuf.Get<float>();
        auto routeEntries = finalRouteBuf.Get<uint32_t>();
        bool inputPending[2] = {false, false};
        bool outputPending[2] = {false, false};
        bool routeReusable[2] = {false, false};
        uint32_t outputSlot = 0U;

        // Exact TileXR ownership: one token per active core, token-strided.
        // Unlike the previous fallback, routes are collected once and fed
        // directly into the 4-route double-buffered source-final pipeline.
        const uint32_t tokenCount = n / k;
        uint32_t routeBank = 0U;
        if (core < tokenCount)
            PrefetchFinalRoutes(core, routeBank, routeEntries, routeReusable);
        for (uint32_t token = core; token < tokenCount; token += kCores) {
            const TEventID routeEvent = routeBank == 0U ? EVENT_ID2 : EVENT_ID3;
            WaitFlag<HardEvent::MTE2_S>(routeEvent);
            const uint32_t count = routeEntries.GetValue(routeBank * 32U);
            const uint32_t nextToken = token + kCores;
            const uint32_t nextBank = routeBank ^ 1U;
            if (nextToken < tokenCount)
                PrefetchFinalRoutes(nextToken, nextBank, routeEntries, routeReusable);
            auto routes = routeEntries[routeBank * 32U + 1U];
            if (count > 1U) {
                FinalReduceToken(token, routes, count, inputRows, outputRows,
                                 accumulator, temporary, inputPending,
                                 outputPending, outputSlot);
            } else if (count == 1U) {
                FinalCopyToken(token, routes.GetValue(0), inputRows, inputPending);
            }
            else {
                // Match TileXR's zero-route source-final path tile by tile;
                // this also remains correct when H exceeds one 4096-element
                // final buffer.
                for (uint32_t off = 0U; off < h; off += kAlignedFinalTile) {
                    const uint32_t size = h - off < kAlignedFinalTile ?
                        h - off : kAlignedFinalTile;
                    const uint32_t slot = outputSlot;
                    const TEventID event = slot == 0U ? EVENT_ID2 : EVENT_ID3;
                    if (outputPending[slot]) {
                        WaitFlag<HardEvent::MTE3_V>(event);
                        outputPending[slot] = false;
                    }
                    Duplicate(outputRows[slot * kAlignedFinalTile], bf16_t(0), size);
                    SetFlag<HardEvent::V_MTE3>(event);
                    WaitFlag<HardEvent::V_MTE3>(event);
                    Store(out + uint64_t(token) * h + off,
                          outputRows[slot * kAlignedFinalTile], size);
                    SetFlag<HardEvent::MTE3_V>(event);
                    outputPending[slot] = true;
                    outputSlot ^= 1U;
                }
            }
            routeReusable[routeBank] = true;
            routeBank = nextBank;
        }
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            if (outputPending[slot])
                WaitFlag<HardEvent::MTE3_V>(slot == 0U ? EVENT_ID2 : EVENT_ID3);
        }
        for (uint32_t bank = 0U; bank < 2U; ++bank) {
            if (inputPending[bank])
                WaitFlag<HardEvent::V_MTE2>(bank == 0U ? EVENT_ID0 : EVENT_ID1);
        }
        PipeBarrier<PIPE_ALL>();
        pipe.Reset();
    }
};

__aicore__ void Run(GM_ADDR hidden, GM_ADDR weights, GM_ADDR dst, GM_ADDR groups,
    GM_ADDR dups, GM_ADDR counts, GM_ADDR out, GM_ADDR outw, GM_ADDR workspace,
    ascend_moonep::MoonEPUdmaCombineTilingData t, uint64_t magic,
    GM_ADDR execution_plan) {
    uint32_t core = get_block_idx(), qp_num = aclshmemi_udma_qp_info_fetch()->qp_num;
    uint32_t active = qp_num == 6U ? 4U : qp_num;
    if (active == 0U || active > 4U || qp_num == 4U) return;
    if (t.server_dedup != 0U && active < 2U) {
        aclshmemi_kernel_abort("server PUT combine requires at least 2 QPs\n");
        return;
    }
    Pipeline x{reinterpret_cast<__gm__ uint8_t*>(workspace),
        reinterpret_cast<__gm__ bf16_t*>(hidden), reinterpret_cast<__gm__ float*>(weights),
        reinterpret_cast<const __gm__ int32_t*>(dst), reinterpret_cast<const __gm__ int32_t*>(groups),
        reinterpret_cast<const __gm__ int32_t*>(dups), reinterpret_cast<const __gm__ int32_t*>(counts),
        reinterpret_cast<__gm__ bf16_t*>(out), reinterpret_cast<__gm__ float*>(outw),
        ascend_moonep::MakeCombinePutLayout(t.S * t.K, t.H, t.R),
        t.S * t.K, t.H, t.K, t.R, t.rank, core, t.server_dedup != 0, t.with_weights != 0, magic};

    x.Init();
    x.BindPlan(reinterpret_cast<__gm__ uint8_t*>(execution_plan));
    x.phase_id = 4U;
    if (t.R >= 128U) {
        if (core < kCombinePut128ReducerBegin) x.Relay();
        else if (core < kCombinePut128SenderBegin) x.Reduce128();
        else if (core < kCombinePut128SenderEnd) x.Send128();
    } else {
        // Preserve the existing 64P and smaller-world core ownership exactly.
        if (core < kReduceBegin) x.Relay();
        else if (core < kSendBegin) x.Reduce();
        else if (core < kSendEnd) x.Send();
    }
    // All >=128P worlds reuse the parallel final fan-in, then the same final reducer.
    // Keep the token-ready/out-of-order experiment out of this path.
    x.phase_id = 5U;
    x.FinalWeights(); // Fetch weights after this core's WORK, before waiting for hidden ready, to hide part of the wait.
    x.Final();
    aclshmemx_sync_vec_all();

}
} // namespace
// Payload-only entry: its immutable plan is produced by planning.cpp.
[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
moonep_udma_combine_prepared_kernel(GM_ADDR hidden, GM_ADDR weights, GM_ADDR dst,
    GM_ADDR groups, GM_ADDR dups, GM_ADDR counts, GM_ADDR out, GM_ADDR outw,
    GM_ADDR workspace, GM_ADDR plan,
    ascend_moonep::MoonEPUdmaCombineTilingData t, uint64_t magic) {
    Run(hidden, weights, dst, groups, dups, counts, out, outw,
        workspace, t, magic, plan);
}
extern "C" __attribute__((visibility("default"))) void moonep_udma_combine_plan_kernel_do(
    void* stream, uint8_t* hidden, uint8_t* weights, uint8_t* dst,
    uint8_t* groups, uint8_t* dups, uint8_t* counts, uint8_t* out, uint8_t* outw,
    uint8_t* workspace, uint8_t* plan, uint8_t* tiling,
    uint64_t magic) {
    auto t = *reinterpret_cast<ascend_moonep::MoonEPUdmaCombineTilingData*>(tiling);
    moonep_udma_combine_prepared_kernel<<<kCores, CombinePutSimt::kLaunchUbBytes, stream>>>(
        hidden, weights, dst, groups, dups, counts, out, outw, workspace, plan, t, magic);
}
