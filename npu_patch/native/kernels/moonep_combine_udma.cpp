// Hidden uses UDMA PUT; Final returns weights with independent SIMT GET lanes.
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
#include "simt_api/device_warp_functions.h"

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

// Relay-only fast path: consume the already-loaded padded UB metadata directly.
// No intermediate inputs/targets or scalar entries[] packing is needed.
__simt_vf__ LAUNCH_BOUND(kBatch) inline void FillRelayWriteWqes(
    uint32_t count, __ubuf__ uint64_t* requests, __ubuf__ uint64_t* maps,
    uint32_t offset, uint64_t hidden_base, uint64_t remote_page_base,
    uint32_t row_bytes, uint32_t head, __ubuf__ uint32_t* ctx) {
    const uint32_t tid = threadIdx.x;
    if (tid >= count) return;
    const uint32_t index = (offset + tid) * 4U; // 8B record padded to 32B in UB
    const uint32_t loff = static_cast<uint32_t>(requests[index]);
    const uint32_t row = static_cast<uint32_t>(maps[index] >> 32U);
    const uint64_t remote = remote_page_base + uint64_t(row) * row_bytes;
    const uint64_t local = hidden_base + uint64_t(loff) * row_bytes;
    const uint64_t sq = uint64_t(ctx[6]) | (uint64_t(ctx[7]) << 32U);
    const uint32_t dw0 = tid + 1U == count ? ctx[13] : ctx[0];
    __gm__ uint64_t* wqe = reinterpret_cast<__gm__ uint64_t*>(
        sq + uint64_t((head + tid) % shm::UDMA_SQ_BASKBLK_CNT) * ctx[12]);
    wqe[0] = uint64_t(dw0) | (uint64_t(ctx[1]) << 32U);
    wqe[1] = uint64_t(ctx[2]) | (uint64_t(ctx[3]) << 32U);
    wqe[2] = uint64_t(ctx[8]) | (uint64_t(ctx[9]) << 32U);
    wqe[3] = uint64_t(ctx[10]) | (uint64_t(ctx[11]) << 32U);
    wqe[4] = uint64_t(ctx[4]) | (uint64_t(ctx[5]) << 32U);
    wqe[5] = remote;
    wqe[6] = uint64_t(row_bytes);
    wqe[7] = local;
}
// Same WQE protocol, with bounds validation fused into VF metadata reads.
__simt_vf__ LAUNCH_BOUND(kBatch) inline void FillRelayWriteWqesChecked(
    uint32_t count, __ubuf__ uint64_t* requests, __ubuf__ uint64_t* maps,
    uint32_t offset, uint64_t hidden_base, uint64_t remote_page_base,
    uint32_t row_bytes, uint32_t head, uint32_t expected_page,
    uint32_t page_rows, uint32_t input_rows, uint32_t tail_dw0, __ubuf__ uint32_t* ctx) {
    const uint32_t tid = threadIdx.x;
    // All lanes in a launched warp participate, including padded tail lanes.
    const bool active = tid < count;
    const uint32_t index = (offset + tid) * 4U;
    const uint64_t mapping = active ? maps[index] : 0U;
    const uint32_t loff = active ? static_cast<uint32_t>(requests[index]) : 0U;
    const uint32_t row = static_cast<uint32_t>(mapping >> 32U);
    const bool invalid = active && (static_cast<uint32_t>(mapping) != expected_page ||
                                   row >= page_rows || loff >= input_rows);
    const uint32_t invalid_lanes = asc_ballot(invalid);
    if ((tid & 31U) == 0U) ctx[14U + tid / 32U] = invalid_lanes;
    // Invalid entries never produce payload WQEs. S checks both warp masks
    // after V_S and aborts before posting ANY WQE via the doorbell.
    if (!active || invalid) return;
    const uint64_t remote = remote_page_base + uint64_t(row) * row_bytes;
    const uint64_t local = hidden_base + uint64_t(loff) * row_bytes;
    const uint64_t sq = uint64_t(ctx[6]) | (uint64_t(ctx[7]) << 32U);
    // Payload is RO throughout the page. Keep one CQE per batch for SQ
    // credit accounting; only the subsequent page-ready flag supplies SO.
    const uint32_t dw0 = tid + 1U == count ? tail_dw0 : ctx[0];
    __gm__ uint64_t* wqe = reinterpret_cast<__gm__ uint64_t*>(
        sq + uint64_t((head + tid) % shm::UDMA_SQ_BASKBLK_CNT) * ctx[12]);
    wqe[0] = uint64_t(dw0) | (uint64_t(ctx[1]) << 32U);
    wqe[1] = uint64_t(ctx[2]) | (uint64_t(ctx[3]) << 32U);
    wqe[2] = uint64_t(ctx[8]) | (uint64_t(ctx[9]) << 32U);
    wqe[3] = uint64_t(ctx[10]) | (uint64_t(ctx[11]) << 32U);
    wqe[4] = uint64_t(ctx[4]) | (uint64_t(ctx[5]) << 32U);
    wqe[5] = remote;
    wqe[6] = uint64_t(row_bytes);
    wqe[7] = local;
}
// LocalFinal-only: packed tx input offsets and contiguous final target rows.
__simt_vf__ LAUNCH_BOUND(kBatch) inline void FillLocalFinalWriteWqes(
    uint32_t count, __ubuf__ uint64_t* tx, uint64_t hidden_base,
    uint64_t remote_first, uint32_t row_bytes, uint32_t head,
    uint32_t input_rows, uint32_t tail_dw0, __ubuf__ uint32_t* ctx) {
    const uint32_t tid = threadIdx.x;
    // All lanes in a launched warp participate, including padded tail lanes.
    const bool active = tid < count;
    const uint32_t loff = active ? static_cast<uint32_t>(tx[tid * 4U]) : 0U;
    const bool invalid = active && loff >= input_rows;
    const uint32_t invalid_lanes = asc_ballot(invalid);
    if ((tid & 31U) == 0U) ctx[14U + tid / 32U] = invalid_lanes;
    // Invalid entries never produce payload WQEs. S checks both warp masks
    // after V_S and aborts before posting ANY WQE via the doorbell.
    if (!active || invalid) return;
    const uint64_t remote = remote_first + uint64_t(tid) * row_bytes;
    const uint64_t local = hidden_base + uint64_t(loff) * row_bytes;
    const uint64_t sq = uint64_t(ctx[6]) | (uint64_t(ctx[7]) << 32U);
    // LocalFinal payload uses RO with one CQE per batch for SQ accounting.
    // The final-ready SO orders all preceding payload writes.
    const uint32_t dw0 = tid + 1U == count ? tail_dw0 : ctx[0];
    __gm__ uint64_t* wqe = reinterpret_cast<__gm__ uint64_t*>(
        sq + uint64_t((head + tid) % shm::UDMA_SQ_BASKBLK_CNT) * ctx[12]);
    wqe[0] = uint64_t(dw0) | (uint64_t(ctx[1]) << 32U);
    wqe[1] = uint64_t(ctx[2]) | (uint64_t(ctx[3]) << 32U);
    wqe[2] = uint64_t(ctx[8]) | (uint64_t(ctx[9]) << 32U);
    wqe[3] = uint64_t(ctx[10]) | (uint64_t(ctx[11]) << 32U);
    wqe[4] = uint64_t(ctx[4]) | (uint64_t(ctx[5]) << 32U);
    wqe[5] = remote;
    wqe[6] = uint64_t(row_bytes);
    wqe[7] = local;
}
// Each core owns whole 512B output windows. This keeps different AIVs from
// writing the same GM conflict window and coalesces contiguous route outputs.
// Inputs are immutable for the call; weights are returned per route, not reduced.
constexpr uint32_t kWeightThreads = 128U;
__simt_vf__ LAUNCH_BOUND(kWeightThreads) inline void GatherWeights(
    __gm__ const int32_t* routes, __gm__ float* weights, __gm__ float* output,
    uint32_t n, uint32_t first_group, uint32_t end_group) {
    for (uint32_t group = first_group; group < end_group; ++group) {
        const uint32_t route = group * kWeightThreads + threadIdx.x;
        if (route >= n) continue;
        const uint32_t encoded = static_cast<uint32_t>(routes[route]);
        // Preserve all three public route encodings, including server-forwarded
        // canonical routes (10xxxxxx), not only normal/~raw duplicate routes.
        const uint32_t raw = (encoded & 0xc0000000U) ==
                ascend_moonep::kMoonEPServerForwardedRouteTag
            ? encoded & ascend_moonep::kMoonEPServerRoutePayloadMask
            : encoded ^ (0U - (encoded >> 31U));
        const uint32_t peer = raw / n;
        const uint32_t loff = raw % n;
        const float weight = ::simt::aclshmem_float_g(weights + loff, int32_t(peer));
        asc_stwt(output + route, weight);
    }
    asc_threadfence();
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

// Find a source's scheduling ordinal. The base is only 8-aligned, so source%16 cannot
// identify its sender; preparation and payload must use exactly the same peer/QP ownership.
MOONEP_PUT_SCHEDULE_INLINE uint32_t CombinePutSourceOrdinal(
    uint32_t rank, uint32_t source, uint32_t ranks) {
    for (uint32_t ordinal = 0; ordinal < ranks; ++ordinal)
        if (CombinePutSource(rank, ordinal, ranks) == source) return ordinal;
    return ranks;
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
inline constexpr aclshmemx_udma_op_config_t kRingPayloadCqe{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
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

struct Pipeline {
    // Each AIV owns a Pipeline; GM pointers refer to rank-shared data, while counters are core-private.
    __gm__ uint8_t* ws;       // Symmetric workspace: ring input, reduced results, final hidden and ready slots.
    __gm__ bf16_t* hidden;    // Original input [n, h], indexed by producer-local loff; combine does not overwrite it.
    __gm__ float* weights;   // Symmetric SHMEM input [n]; Final reads rank/loff from dst; private GM is not valid.

    // Source-rank route map [n]; weights return per route without deduplication.
    // Payload planning metadata comes from execution_plan.
    const __gm__ int32_t* dst;
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
    // CQ failure attribution: 4=payload relay/reduce/send, 5=Final.
    // Execution-plan generation runs separately; no extra synchronization is added.
    uint32_t phase_id = 0U, simt_batches_posted = 0U;
    uint32_t simt_qp[2] = {0xffffffffU, 0xffffffffU};
    // Only reducer AIVs use these mappings. Immutable descriptors retain the
    // logical 8*N scratch offsets; the physical input is a bounded ring page.
    uint32_t ring_logical_begin = 0U, ring_physical_begin = 0U;
    uint32_t ring_batch_begin = 0U, ring_batch_size = 0U;
    uint32_t ring_tx_indices[64U]{};
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
    __aicore__ inline __gm__ bf16_t* RelayInputRow(uint32_t logical) {
        if (logical < ring_logical_begin ||
            logical - ring_logical_begin >= p.ring_page_rows)
            aclshmemi_kernel_abort("combine ring input row out of page rank=%u row=%u\n", rank, logical);
        return Row(p.relay_hidden, uint64_t(ring_physical_begin) + logical - ring_logical_begin);
    }
    __aicore__ inline __gm__ bf16_t* ReducedRow(uint32_t logical) {
        const uint32_t offset = logical - ring_batch_begin;
        if (logical < ring_batch_begin || offset >= ring_batch_size ||
            ring_tx_indices[offset] >= n)
            aclshmemi_kernel_abort("combine ring result index mismatch rank=%u row=%u\n", rank, logical);
        return Row(p.relay_result, ring_tx_indices[offset]);
    }
    __aicore__ inline __gm__ uint32_t* RingPage(uint32_t index) {
        if (index >= p.ring_page_capacity)
            aclshmemi_kernel_abort("combine ring page index overflow rank=%u page=%u\n", rank, index);
        auto page = reinterpret_cast<__gm__ uint32_t*>(
            execution_plan + plan_layout.ring_pages + uint64_t(index) * 64U);
        Flush(page);
        return page;
    }
    __aicore__ inline __ubuf__ uint8_t* Wqe() {
        return reinterpret_cast<__ubuf__ uint8_t*>(Ub<uint8_t>(128 * 1024, 8192).GetPhyAddr());
    }
    // Default batched PUT retains scalar SHMEM WQE construction and MTE3 SQ staging.
    // DIRECT, cross-server reduced results, and small-world LocalFinal use this non-SIMT entry.
    __aicore__ inline void SubmitPutBatch(__gm__ bf16_t* const* inputs,
                                          __gm__ bf16_t* const* targets,
                                          uint32_t count, uint32_t peer,
                                          uint32_t qp) {
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
    // Intra-server SIMT payload for 64P/>=128P LocalFinal(); never for cross-server sends.
    // Metadata, weights, and ready keep their original paths; the last WQE in each batch still requests SO+CQE.
    __aicore__ inline void SubmitIntraPutBatch(__gm__ bf16_t* const* inputs,
                                               __gm__ bf16_t* const* targets,
                                               uint32_t count, uint32_t peer,
                                               uint32_t qp) {
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
    // Same peer/QP ownership and completion protocol as SubmitIntraPutBatch.
    // Current metadata bank stays immutable until V_S retires the VF reads.
    template <bool Checked = false>
    __aicore__ inline void SubmitRingPutBatch(LocalTensor<uint64_t> requests,
                                             LocalTensor<uint64_t> maps,
                                             uint32_t offset, uint32_t count,
                                             uint32_t slotIndex, uint32_t peer,
                                             uint32_t qp, uint32_t expected_page = 0U) {
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
        const uint64_t remote_page = reinterpret_cast<uint64_t>(aclshmem_ptr(
            Row(p.relay_hidden, uint64_t(slotIndex) * p.ring_page_rows), peer));
        auto request_ptr = reinterpret_cast<__ubuf__ uint64_t*>(requests.GetPhyAddr());
        auto map_ptr = reinterpret_cast<__ubuf__ uint64_t*>(maps.GetPhyAddr());
        if constexpr (Checked) { rec[14] = 0U; rec[15] = 0U; }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        const uint32_t threads = (count + 31U) / 32U * 32U;
        // Both flags request one CQE. Their XOR changes only the order
        // bits and cannot touch the cached peer token/jetty fields. Do not
        // mutate rec[13]: legacy paths reuse this cached SO context.
        constexpr uint32_t tail_delta = (aclshmemi_udma_build_flag<
            aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kCheckpoint>() ^
            aclshmemi_udma_build_flag<aclshmemi_udma_opcode_t::UDMA_OP_WRITE,
                                      kRingPayloadCqe>()) << 16U;
        static_assert((tail_delta & 0xf0000000U) == 0U,
                      "ring order transform must not overlap token/jetty");
        const uint32_t ring_tail_dw0 = rec[13] ^ tail_delta;
        if constexpr (Checked) {
            AscendC::Simt::VF_CALL<CombinePutSimt::FillRelayWriteWqesChecked>(
                AscendC::Simt::Dim3(threads, 1, 1), count, request_ptr, map_ptr,
                offset, reinterpret_cast<uint64_t>(hidden), remote_page,
                h * sizeof(bf16_t), head, expected_page, p.ring_page_rows, n, ring_tail_dw0, rec);
        } else {
            AscendC::Simt::VF_CALL<CombinePutSimt::FillRelayWriteWqes>(
                AscendC::Simt::Dim3(threads, 1, 1), count, request_ptr, map_ptr,
                offset, reinterpret_cast<uint64_t>(hidden), remote_page,
                h * sizeof(bf16_t), head, rec);
        }
        // Match dispatch's direct-GM-store protocol: V_S completes VF stores
        // before the doorbell and before S can reuse entries/context UB.
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if constexpr (Checked) {
            if (rec[14] | rec[15]) {
                aclshmemi_kernel_abort("combine ring relay map invalid rank=%u page=%u\n", rank, expected_page);
                return;
            }
        }
        auto db = reinterpret_cast<__gm__ uint32_t*>(ctx->db_addr);
        st_dev(head + count, db, 0);
        queue_state->sq_head = head + count;
        queue_state->cqe_cnt += 1U; // one CQE per nonempty batch; checked 16/32P payload is RO, page ready remains SO
        ++simt_batches_posted;
    }
    // Same peer/QP owner, SQ guard, CQE accounting and completion as Relay.
    __aicore__ inline void SubmitLocalFinalPutBatch(LocalTensor<uint64_t> tx,
                                                    uint64_t first, uint32_t count,
                                                    uint32_t peer, uint32_t qp) {
        if (!count) return;
        if (first > n || count > uint64_t(n) - first) {
            aclshmemi_kernel_abort("combine LocalFinal target outside final arena\n");
            return;
        }
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
        const uint64_t remote_first = reinterpret_cast<uint64_t>(
            aclshmem_ptr(Row(p.final_hidden, first), peer));
        auto tx_ptr = reinterpret_cast<__ubuf__ uint64_t*>(tx.GetPhyAddr());
        rec[14] = 0U; rec[15] = 0U;
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        const uint32_t threads = (count + 31U) / 32U * 32U;
        // Both flags request one CQE. Their XOR changes only the order
        // bits and cannot touch the cached peer token/jetty fields. Do not
        // mutate rec[13]: legacy 64P+ calls reuse its original SO tail.
        constexpr uint32_t tail_delta = (aclshmemi_udma_build_flag<
            aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kCheckpoint>() ^
            aclshmemi_udma_build_flag<aclshmemi_udma_opcode_t::UDMA_OP_WRITE,
                                      kRingPayloadCqe>()) << 16U;
        static_assert((tail_delta & 0xf0000000U) == 0U,
                      "local order transform must not overlap token/jetty");
        const uint32_t ring_tail_dw0 = rec[13] ^ tail_delta;
        AscendC::Simt::VF_CALL<CombinePutSimt::FillLocalFinalWriteWqes>(
            AscendC::Simt::Dim3(threads, 1, 1), count, tx_ptr,
            reinterpret_cast<uint64_t>(hidden), remote_first,
            h * sizeof(bf16_t), head, n, ring_tail_dw0, rec);
        // Match dispatch's direct-GM-store protocol: V_S completes VF stores
        // before the doorbell and before S can reuse entries/context UB.
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (rec[14] | rec[15]) {
            aclshmemi_kernel_abort("combine LocalFinal input offset invalid rank=%u\n", rank);
            return;
        }
        auto db = reinterpret_cast<__gm__ uint32_t*>(ctx->db_addr);
        st_dev(head + count, db, 0);
        queue_state->sq_head = head + count;
        queue_state->cqe_cnt += 1U; // one CQE per nonempty batch; 16/32P local payload is RO, final ready remains SO
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
    TBuf<TPosition::VECCALC> accumBuf, tempBuf, descriptorBuf, inputAddressBuf;
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
        // It lets ReduceRing() read an ingress-local contributor directly from
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
            header[5] != rank || header[6] != uint64_t(server) ||
            header[7] != CombineRingLayoutFingerprint(k))
            aclshmemi_kernel_abort("combine execution plan header mismatch rank=%u\n", rank);
        execution_plan = plan;
        plan_layout = MakeCombineExecutionPlanLayout(n, r);
    }

    __aicore__ void FinalWeights() {
        if (!with_weights) return;
        // Hidden WORK has finished on this core. Release its live UB/events
        // before VF reuses the vector pipeline; hidden scheduling is unchanged.
        PipeBarrier<PIPE_ALL>();
        pipe.Reset();
        const uint32_t groups = (n + CombinePutSimt::kWeightThreads - 1U) /
                                CombinePutSimt::kWeightThreads;
        const uint32_t first = uint64_t(groups) * core / kCores;
        const uint32_t end = uint64_t(groups) * (core + 1U) / kCores;
        if (first == end) return;
        AscendC::Simt::VF_CALL<CombinePutSimt::GatherWeights>(
            AscendC::Simt::Dim3(CombinePutSimt::kWeightThreads, 1, 1),
            dst, weights, outw, n, first, end);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
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
            if (server && (r == 16U || r == 32U)) {
                // tx bank remains immutable through V_S. Its next bank was
                // prefetched above; no scalar pointer/entry packing is needed.
                SubmitLocalFinalPutBatch(txPages[current], uint64_t(sc[1]) + first,
                                         size, peer, 0U);
            } else {
                __gm__ bf16_t* inputs[64U];
                __gm__ bf16_t* targets[64U];
                for (uint32_t j = 0; j < size; ++j) {
                    uint32_t loff = static_cast<uint32_t>(txPages[current].GetValue(j * 4U));
                    inputs[j] = hidden + uint64_t(loff) * h;
                    targets[j] = Row(p.final_hidden, sc[1] + first + j);
                }
                // Preserve the legacy generic paths for the remaining world sizes.
                // The destination remains final_hidden; other world sizes retain scalar SHMEM submission.
                if (server && (r == 64U || r >= 128U)) {
                    SubmitIntraPutBatch(inputs, targets,
                                        size, peer, 0U);
                } else {
                    SubmitPutBatch(inputs, targets,
                                   size, peer, 0U);
                }
            }
            current = next;
        }
        aclshmemx_submit_state_t state{};
        SendReady(p.final_ready, uint64_t(rank) * 2U, peer, 0U, state);
    }

    // Relay control reads use local-server MTE mapping. No receiver reducer
    // submits UDMA work: the producer Relay AIV remains the sole peer/QP0 owner.
    __aicore__ uint64_t ReadRingControl(uint64_t base, uint32_t index, uint32_t peer) {
        if (peer == rank) {
            auto word = Flag(base, index);
            Flush(word);
            return *word;
        }
        auto ub = Ub<uint64_t>(124U * 1024U, 4U);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
        auto remote = reinterpret_cast<__gm__ uint64_t*>(aclshmem_ptr(Flag(base, index), peer));
        Load(ub, remote, 1U);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
        return ub.GetValue(0U);
    }

    __aicore__ void SubmitRingRequestLoad(LocalTensor<uint64_t> requests,
                                     LocalTensor<uint64_t> maps,
                                     uint64_t first, uint32_t size) {
        SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
        GlobalTensor<uint64_t> requestGm, mapGm;
        requestGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(Meta(p.relay_send, first)), size);
        mapGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(
            execution_plan + plan_layout.relay_page_map + first * 8U), size);
        DataCopyExtParams cp{static_cast<uint16_t>(size), 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<uint64_t> pad{false, 0U, 0U, 0U};
        DataCopyPad(requests, requestGm, cp, pad);
        DataCopyPad(maps, mapGm, cp, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
    }

    __aicore__ void WaitRingRequestLoad() {
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    }

    __aicore__ void NextRingSource(uint32_t peer, uint32_t& ordinal,
                                    uint32_t& first, uint32_t& count,
                                    uint32_t& pages, uint32_t& cursor,
                                    uint32_t& page) {
        cursor = 0U;
        page = 0U;
        for (; ordinal < r; ordinal += kCombineRingStreams) {
            // Every producer follows its receiver's demand order, including
            // small worlds. Producer-relative rotations can deadlock a ring.
            const uint32_t source = CombinePutSource(peer, ordinal, r);
            auto range = Meta(p.relay_ranges, uint64_t(core) * r + source);
            Flush(range);
            if (range[0] < 0 || range[1] < 0 || range[2] < 0 ||
                uint64_t(range[0]) + uint32_t(range[1]) > n)
                aclshmemi_kernel_abort("combine ring relay range invalid rank=%u source=%u\n", rank, source);
            first = static_cast<uint32_t>(range[0]);
            count = static_cast<uint32_t>(range[1]);
            pages = static_cast<uint32_t>(range[2]);
            if (pages) return;
            if (count)
                aclshmemi_kernel_abort("combine ring requests without pages rank=%u source=%u\n", rank, source);
        }
    }

    __aicore__ void PublishRingReady(uint32_t peer, uint32_t stream,
                                     uint32_t slot, uint64_t ticket) {
        const uint32_t producer = LocalLane(rank);
        const uint64_t target = (uint64_t(stream) * kCombineRingDepth + slot) * 8U + producer;
        if (peer == rank) {
            Publish(Flag(p.ring_ready, target), ticket);
            return;
        }
        // One immutable source word per live destination slot. Reusing a word
        // is legal only after that slot's returned credit proves the old SO
        // and every reader of its payload have completed (no source-token ABA).
        const uint64_t token = (uint64_t(LocalLane(peer)) * kCombineRingStreams + stream) *
            kCombineRingDepth + slot;
        Publish(Flag(p.ring_ready_source, token), ticket);
        aclshmemx_submit_state_t state{};
        aclshmemx_submit_t submit(state);
        aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kCheckpoint>(
            Flag(p.ring_ready, target), Flag(p.ring_ready_source, token),
            reinterpret_cast<__ubuf__ uint64_t*>(Wqe()), 1U, peer, 0U, EVENT_ID0, submit);
    }

    template <bool FastBounds = false>
    __aicore__ void RelayRing() {
        if (core >= (server ? 8U : 1U)) return;
        const uint32_t peer = server ? rank / 8U * 8U + core : rank;
        if (peer >= r) return;
        uint32_t ordinal[8], first[8], count[8], pages[8], cursor[8], page[8];
        uint64_t ticket[8];
        bool epoch_seen[8];
        for (uint32_t stream = 0U; stream < kCombineRingStreams; ++stream) {
            ordinal[stream] = stream;
            ticket[stream] = 1U;
            epoch_seen[stream] = false;
            NextRingSource(peer, ordinal[stream], first[stream], count[stream],
                           pages[stream], cursor[stream], page[stream]);
        }
        // Two 2KiB banks per stream for each padded metadata table. Retain a
        // partially consumed bank across page switches; prefetch the next bank
        // while the current batch prepares/submits its WQEs. Relay-only UB
        // [48,112)KiB does not overlap control [124,...), WQE or SIMT buffers.
        uint32_t bank[8] = {}, loadedBegin[8] = {}, loadedSize[8] = {};
        bool cached[8] = {};
        uint32_t stream = 0U;
        for (;;) {
            bool active = false;
            for (uint32_t g = 0U; g < kCombineRingStreams; ++g) active |= ordinal[g] < r;
            if (!active) break;
            const uint32_t g = stream;
            stream = (stream + 1U) % kCombineRingStreams;
            if (ordinal[g] >= r) continue;
            if (!epoch_seen[g]) {
                if (ReadRingControl(p.ring_epoch, g, peer) != magic) continue;
                epoch_seen[g] = true;
            }
            const uint32_t slot = static_cast<uint32_t>((ticket[g] - 1U) % kCombineRingDepth);
            const uint32_t slotIndex = g * kCombineRingDepth + slot;
            // Never block all streams behind one full stream. This round-robin
            // credit check is essential when receiver/source orders differ.
            if (ReadRingControl(p.ring_credit, slotIndex, peer) != ticket[g]) continue;
            for (;;) {
                if (cursor[g] == count[g]) break;
                auto requests = Ub<uint64_t>((48U + g * 4U + bank[g] * 2U) * 1024U, 64U * 4U);
                auto maps = Ub<uint64_t>((80U + g * 4U + bank[g] * 2U) * 1024U, 64U * 4U);
                if (!cached[g]) {
                    loadedBegin[g] = cursor[g];
                    loadedSize[g] = count[g] - cursor[g] < 64U ? count[g] - cursor[g] : 64U;
                    SubmitRingRequestLoad(requests, maps,
                        uint64_t(core) * n + first[g] + cursor[g], loadedSize[g]);
                    WaitRingRequestLoad(); cached[g] = true;
                }
                const uint32_t offset = cursor[g] - loadedBegin[g];
                const uint32_t size = loadedSize[g] - offset;
                uint32_t consumed = 0U;
                if constexpr (FastBounds) {
                    if (peer != rank) {
                        // Planner emits each immutable producer CSR in nondecreasing
                        // page order. Find the <=current-page prefix in <=7 scalar
                        // probes instead of reading every request/map twice.
                        uint32_t low = 0U, high = size - 1U;
                        // The common case is a whole bank within this page.
                        if (static_cast<uint32_t>(maps.GetValue((offset + high) * 4U)) <= page[g])
                            low = high = size;
                        while (low < high) {
                            const uint32_t mid = low + (high - low) / 2U;
                            const uint32_t seq = static_cast<uint32_t>(maps.GetValue((offset + mid) * 4U));
                            if (seq <= page[g]) low = mid + 1U;
                            else high = mid;
                        }
                        consumed = low;
                        // Full bounds/page validation is retained in the VF before
                        // its doorbell. Self producers retain the scalar checks.
                    } else {
                        for (uint32_t j = 0U; j < size; ++j) {
                            const uint64_t mapping = maps.GetValue((offset + j) * 4U);
                            const uint32_t pageSeq = static_cast<uint32_t>(mapping);
                            if (pageSeq > page[g]) break;
                            const uint32_t row = static_cast<uint32_t>(mapping >> 32U);
                            const uint32_t loff = static_cast<uint32_t>(requests.GetValue((offset + j) * 4U));
                            if (pageSeq != page[g] || row >= p.ring_page_rows || loff >= n)
                                aclshmemi_kernel_abort("combine ring relay map invalid rank=%u page=%u\n", rank, page[g]);
                            ++consumed;
                        }
                    }
                } else {
                    for (uint32_t j = 0U; j < size; ++j) {
                        const uint64_t mapping = maps.GetValue((offset + j) * 4U);
                        const uint32_t pageSeq = static_cast<uint32_t>(mapping);
                        if (pageSeq > page[g]) break;
                        const uint32_t row = static_cast<uint32_t>(mapping >> 32U);
                        const uint32_t loff = static_cast<uint32_t>(requests.GetValue((offset + j) * 4U));
                        if (pageSeq != page[g] || row >= p.ring_page_rows || loff >= n)
                            aclshmemi_kernel_abort("combine ring relay map invalid rank=%u page=%u\n", rank, page[g]);
                        ++consumed;
                    }
                }
                const uint32_t nextCursor = cursor[g] + consumed;
                const uint32_t next = bank[g] ^ 1U;
                const bool prefetch = consumed == size && nextCursor < count[g];
                const uint32_t nextSize = count[g] - nextCursor < 64U ? count[g] - nextCursor : 64U;
                if (prefetch) {
                    SubmitRingRequestLoad(
                        Ub<uint64_t>((48U + g * 4U + next * 2U) * 1024U, 64U * 4U),
                        Ub<uint64_t>((80U + g * 4U + next * 2U) * 1024U, 64U * 4U),
                        uint64_t(core) * n + first[g] + nextCursor, nextSize);
                }
                if (peer != rank && consumed) {
                    if constexpr (FastBounds) SubmitRingPutBatch<true>(
                        requests, maps, offset, consumed, slotIndex, peer, 0U, page[g]);
                    else SubmitRingPutBatch(
                        requests, maps, offset, consumed, slotIndex, peer, 0U);
                }
                if (prefetch) WaitRingRequestLoad();
                cursor[g] = nextCursor;
                if (consumed == size) {
                    cached[g] = prefetch;
                    bank[g] = next; loadedBegin[g] = nextCursor; loadedSize[g] = nextSize;
                }
                if (consumed < size) break;
            }
            // Self and zero-contributor producers also signal every nonempty
            // receiver page, since the consumer waits on the fixed producer set.
            PublishRingReady(peer, g, slot, ticket[g]);
            ++ticket[g];
            if (++page[g] == pages[g]) {
                if (cursor[g] != count[g])
                    aclshmemi_kernel_abort("combine ring unconsumed relay requests rank=%u\n", rank);
                ordinal[g] += kCombineRingStreams;
                cached[g] = false;
                NextRingSource(peer, ordinal[g], first[g], count[g], pages[g], cursor[g], page[g]);
            }
        }
        if (server) LocalFinal(peer);
        if (peer != rank) Drain(peer, 0U);
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
            gm.SetGlobalBuffer(RelayInputRow(slot) + off, size);
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
        gm.SetGlobalBuffer(RelayInputRow(slot) + off, uint64_t(count - 1U) * h + size);
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
            gm.SetGlobalBuffer(RelayInputRow(a + c) + off,
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
                Store(ReducedRow(slots[i + j]) + off, output[j * size], size);
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
            Store(ReducedRow(slots[i]) + off, output, size);
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
        Store(ReducedRow(slot) + off, output, size);
        outputQ.FreeTensor(output);
    }

    __aicore__ void InitRingStream(uint32_t stream, uint32_t shard) {
        if (shard == 0U) {
            for (uint32_t slot = 0U; slot < kCombineRingDepth; ++slot) {
                const uint32_t index = stream * kCombineRingDepth + slot;
                for (uint32_t producer = 0U; producer < 8U; ++producer)
                    Publish(Flag(p.ring_ready, uint64_t(index) * 8U + producer), 0U);
                for (uint32_t worker = 0U; worker < kCombineRingMaxShards; ++worker)
                    Publish(Flag(p.ring_done, uint64_t(index) * kCombineRingMaxShards + worker), 0U);
                Publish(Flag(p.ring_credit, index), uint64_t(slot) + 1U);
            }
            // The full invocation magic is published only after every reset.
            // No producer can race a consumer clearing its freshly sent ready.
            Publish(Flag(p.ring_epoch, stream), magic);
        } else {
            Wait(Flag(p.ring_epoch, stream), magic);
        }
    }

    __aicore__ void ReduceRing() {
        InitVector(false);
        const uint32_t shards = r >= 128U ? kCombinePut128ReducerShards : kCombinePutReducerShards;
        const uint32_t group = (core - kCombinePutReducerBegin) / shards;
        const uint32_t shard = (core - kCombinePutReducerBegin) % shards;
        InitRingStream(group, shard);
        const uint32_t producers = server ? (r < 8U ? r : 8U) : 1U;
        uint64_t ticket = 1U;
        uint32_t batchOwnerRotation = 0U;
        for (uint32_t ord = group; ord < r; ord += kCombineRingStreams) {
            const uint32_t source = Source(ord);
            auto sc = Meta(p.source_counts, source);
            if (sc[4] < 0 || sc[5] < 0 || sc[6] < 0 || sc[7] < 0 ||
                uint64_t(sc[4]) + uint32_t(sc[5]) > uint64_t(n) * 8U ||
                uint64_t(sc[7]) + uint32_t(sc[6]) > p.ring_page_capacity ||
                (LocalSource(source) && sc[6] != 0))
                aclshmemi_kernel_abort("combine ring source page range invalid rank=%u source=%u\n", rank, source);
            for (uint32_t pageSeq = 0U; pageSeq < static_cast<uint32_t>(sc[6]); ++pageSeq) {
                const uint32_t slot = static_cast<uint32_t>((ticket - 1U) % kCombineRingDepth);
                const uint32_t slotIndex = group * kCombineRingDepth + slot;
                auto page = RingPage(static_cast<uint32_t>(sc[7]) + pageSeq);
                const uint32_t firstSlot = page[0], rows = page[1], groupsInPage = page[2];
                if (page[3] != source || !rows || rows > p.ring_page_rows ||
                    !groupsInPage || groupsInPage > rows / 2U ||
                    firstSlot < static_cast<uint32_t>(sc[4]) ||
                    uint64_t(firstSlot) + rows > uint64_t(sc[4]) + uint32_t(sc[5]))
                    aclshmemi_kernel_abort("combine ring page metadata invalid rank=%u source=%u page=%u\n", rank, source, pageSeq);
                for (uint32_t producer = 0U; producer < producers; ++producer)
                    Wait(Flag(p.ring_ready, uint64_t(slotIndex) * 8U + producer), ticket);
                ring_logical_begin = firstSlot;
                ring_physical_begin = slotIndex * p.ring_page_rows;
                // A whole duplicate group is confined to one input page. Its
                // head is owned by exactly one 64-slot microbatch/shard, even
                // when the group's tail extends past that microbatch boundary.
                const uint32_t firstBatch = (shard + shards - batchOwnerRotation) % shards;
                for (uint32_t first = firstBatch * kCombinePutBatchRows; first < rows;
                     first += shards * kCombinePutBatchRows) {
                    const uint32_t size = rows - first < kCombinePutBatchRows ? rows - first : kCombinePutBatchRows;
                    ring_batch_begin = firstSlot + first;
                    ring_batch_size = size;
                    uint32_t slots[64U], lengths[64U], pairs[64U], generic[64U], sizes[64U];
                    uint32_t inputAddresses[64U], valid = 0U, pairCount = 0U;
                    auto desc = descriptorBuf.Get<uint64_t>();
                    auto addresses = inputAddressBuf.Get<uint64_t>();
                    LoadDescriptors(desc, ring_batch_begin, size);
                    LoadInputAddresses(addresses, ring_batch_begin, size);
                    for (uint32_t i = 0U; i < size; ++i) {
                        inputAddresses[i] = static_cast<uint32_t>(addresses.GetValue(i * 4U));
                        const uint64_t value = desc.GetValue(i * 4U);
                        const uint32_t count = static_cast<uint32_t>(value >> 32U);
                        ring_tx_indices[i] = static_cast<uint32_t>(value);
                        if (!count) continue;
                        if (count > k || uint64_t(first) + i + count > rows || ring_tx_indices[i] >= n)
                            aclshmemi_kernel_abort("combine ring group crosses page rank=%u source=%u\n", rank, source);
                        slots[valid] = ring_batch_begin + i;
                        lengths[valid++] = count;
                        if (count == 2U) pairs[pairCount++] = ring_batch_begin + i;
                    }
                    for (uint32_t off = 0U; off < h; off += tile) {
                        const uint32_t elems = h - off < tile ? h - off : tile;
                        const bool paired = elems % 16U == 0U;
                        if (paired) ReducePairs(pairs, pairCount, ring_batch_begin, size,
                                                inputAddresses, off, elems);
                        uint32_t genericCount = 0U;
                        for (uint32_t i = 0U; i < valid; ++i) {
                            if (lengths[i] <= 4U && !(paired && lengths[i] == 2U)) {
                                generic[genericCount] = slots[i];
                                sizes[genericCount++] = lengths[i];
                            }
                        }
                        ReduceGroups(generic, sizes, genericCount, ring_batch_begin,
                                     size, inputAddresses, off, elems);
                        for (uint32_t i = 0U; i < valid; ++i)
                            if (lengths[i] > 4U)
                                ReduceLarge(slots[i], lengths[i], ring_batch_begin,
                                            size, inputAddresses, off, elems);
                    }
                }
                // Input pages cannot be returned while any shard still has
                // an outstanding MTE2 read. Result stores must also retire
                // before the persistent sender-ready flag becomes visible.
                PipeBarrier<PIPE_ALL>();
                SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
                Publish(Flag(p.ring_done, uint64_t(slotIndex) * kCombineRingMaxShards + shard), ticket);
                if (shard == 0U) {
                    for (uint32_t worker = 0U; worker < shards; ++worker)
                        Wait(Flag(p.ring_done, uint64_t(slotIndex) * kCombineRingMaxShards + worker), ticket);
                    Publish(Flag(p.ring_result_ready, uint32_t(sc[7]) + pageSeq), magic);
                    Publish(Flag(p.ring_credit, slotIndex), ticket + kCombineRingDepth);
                }
                batchOwnerRotation = (batchOwnerRotation +
                    (rows + kCombinePutBatchRows - 1U) / kCombinePutBatchRows) % shards;
                ++ticket;
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // source: destination source rank of this chunk; PUT writes into that rank's final_hidden.
    // lane: sending lane and SHMEM QP ID (0=QP0, 1=QP1).
    // ordinal: chunk index within this source; chunks with ordinal%4==0 belong to QP1,
    //          all others to QP0. Non-owner lanes return immediately so each chunk is sent exactly once.
    // inputs: up to 64 local indices into hidden if direct=true, otherwise stable relay_result TX indices.
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
                                     : Row(p.relay_result, inputs[j]);
            batch_targets[j] = Row(p.final_hidden, targets[j]);
        }
        if (source == rank) {
            for (uint32_t j = 0; j < count; ++j)
                Copy(batch_targets[j], batch_inputs[j], h);
        } else {
            SubmitPutBatch(batch_inputs, batch_targets,
                               count, source, lane);
        }
    }

    // Page boundaries follow contributor rows, independently of the 64-result
    // send chunks. Wait once at each page's first group before allowing any of
    // its results into a chunk; a chunk may span several completed pages.
    __aicore__ void WaitRingResultRow(uint32_t source, __gm__ int32_t* sc,
                                     uint32_t logical, uint32_t& pageSeq,
                                     uint32_t& pageEnd) {
        if (logical < pageEnd) return;
        if (pageSeq >= static_cast<uint32_t>(sc[6]))
            aclshmemi_kernel_abort("combine ring sender page overflow rank=%u source=%u\n", rank, source);
        auto page = RingPage(uint32_t(sc[7]) + pageSeq);
        if (page[0] != logical || !page[1] || page[1] > p.ring_page_rows ||
            page[3] != source || uint64_t(page[0]) + page[1] > uint64_t(sc[4]) + uint32_t(sc[5]))
            aclshmemi_kernel_abort("combine ring sender page boundary rank=%u source=%u\n", rank, source);
        Wait(Flag(p.ring_result_ready, uint32_t(sc[7]) + pageSeq), magic);
        pageEnd = page[0] + page[1];
        ++pageSeq;
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
            uint32_t pageSeq = 0U, pageEnd = 0U;
            // sc[0]: total result rows to return.
            // sc[5]: staged contributor rows used to produce the duplicate-group results.
            for (uint32_t first = 0; first < static_cast<uint32_t>(sc[5]); first += 64U) {
                uint32_t size = sc[5] - first < 64U ? sc[5] - first : 64U;
                LoadDescriptors(desc, sc[4] + first, size);
                for (uint32_t j = 0; j < size; ++j) {
                    uint64_t value = desc.GetValue(j * 4U);
                    if (!(value >> 32U)) {
                        continue;
                    }
                    uint32_t txIndex = static_cast<uint32_t>(value);
                    WaitRingResultRow(source, sc, uint32_t(sc[4]) + first + j,
                                      pageSeq, pageEnd);
                    inputs[selected] = txIndex;
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
            if (pageSeq != static_cast<uint32_t>(sc[6]))
                aclshmemi_kernel_abort("combine ring sender page count mismatch rank=%u source=%u\n", rank, source);
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
            } else {
                SendReady(p.final_ready, uint64_t(rank) * 2U + lane, source, lane, state);
            }

            // After submitting each source's result/ready, immediately drain only this core's exclusive QP.
            // This also completes its earlier DIRECT submissions; skip the empty QP1 in the 64P tail group.
            if (source != rank) {
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
            uint32_t pageSeq = 0U, pageEnd = 0U;
            for (uint32_t first = 0; first < static_cast<uint32_t>(sc[5]);
                 first += kCombinePutBatchRows) {
                const uint32_t size = sc[5] - first < kCombinePutBatchRows
                    ? sc[5] - first : kCombinePutBatchRows;
                // Each QP core enumerates the original immutable group order;
                // page-ready waits do not alter cumulative chunk/QP ownership.
                LoadDescriptors(desc, sc[4] + first, size);
                for (uint32_t j = 0; j < size; ++j) {
                    const uint64_t value = desc.GetValue(j * 4U);
                    if (!(value >> 32U)) continue;
                    const uint32_t txIndex = static_cast<uint32_t>(value);
                    WaitRingResultRow(source, sc, uint32_t(sc[4]) + first + j,
                                      pageSeq, pageEnd);
                    inputs[selected] = txIndex;
                    targets[selected++] = sc[1] + txIndex - sc[2];
                    if (selected == 64U) {
                        PostChunk(source, ownerQp, payloadChunkOrdinal++, inputs, targets,
                                  selected, false, false);
                        selected = 0U;
                    }
                }
            }
            if (pageSeq != static_cast<uint32_t>(sc[6]))
                aclshmemi_kernel_abort("combine ring sender page count mismatch rank=%u source=%u\n", rank, source);
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

__aicore__ void Run(GM_ADDR hidden, GM_ADDR weights, GM_ADDR dst,
    GM_ADDR out, GM_ADDR outw, GM_ADDR workspace,
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
        reinterpret_cast<const __gm__ int32_t*>(dst),
        reinterpret_cast<__gm__ bf16_t*>(out), reinterpret_cast<__gm__ float*>(outw),
        ascend_moonep::MakeCombinePutLayout(t.S * t.K, t.H, t.R, t.K),
        t.S * t.K, t.H, t.K, t.R, t.rank, core, t.server_dedup != 0, t.with_weights != 0, magic};

    x.BindPlan(reinterpret_cast<__gm__ uint8_t*>(execution_plan));
    x.Init();
    x.phase_id = 4U;
    if (t.R >= 128U) {
        if (core < kCombinePut128ReducerBegin) x.RelayRing();
        else if (core < kCombinePut128SenderBegin) x.ReduceRing();
        else if (core < kCombinePut128SenderEnd) x.Send128();
    } else {
        // Preserve the existing 64P and smaller-world core ownership exactly.
        if (core < kReduceBegin) {
            if (t.R == 16U || t.R == 32U) x.RelayRing<true>();
            else x.RelayRing();
        }
        else if (core < kSendBegin) x.ReduceRing();
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
// Legacy planning-input argument slots are retained for launch ABI compatibility.
[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
moonep_udma_combine_prepared_kernel(GM_ADDR hidden, GM_ADDR weights, GM_ADDR dst,
    GM_ADDR /* groups */, GM_ADDR /* dups */, GM_ADDR /* counts */, GM_ADDR out, GM_ADDR outw,
    GM_ADDR workspace, GM_ADDR plan,
    ascend_moonep::MoonEPUdmaCombineTilingData t, uint64_t magic) {
    Run(hidden, weights, dst, out, outw,
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
