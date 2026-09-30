// MoonEP dispatch UDMA kernel with SIMT weight scatter — migrated from shmem_dev_moonep dispatch_moonep.
// Multi-QP UDMA path: scatter hidden via aclshmemx_udma_qp_put_nbi directly
// into hidden_buf; QP signals use a dedicated sync_buffer.
// By default Weight is aggregated per peer and sent as whole segments; server duplicates wait on the source SO and
// expand within the board.
//
// Token keeps group flow control.
// The AIV split between Weight and Duplicate is defined by kSerialWeightWorldSizeLimit.

#include "kernel_operator.h"
#include "ascend_moonep/kernels/types.h"
#include "ascend_moonep/kernels/launch.h"
#include "shmem.h"
#include "device/shmem_def.h"
#include "simt_api/device_sync_functions.h"

namespace DispatchMoonep {

// Weight SIMT fallback, expansion, and direct-write loops share one unroll factor;
// 1 disables unrolling, the default is 4, and SIMD aggregation decoding is unaffected.
#ifndef ASCEND_MOONEP_WEIGHT_SIMT_UNROLL
#define ASCEND_MOONEP_WEIGHT_SIMT_UNROLL 4
#endif
static_assert(ASCEND_MOONEP_WEIGHT_SIMT_UNROLL > 0, "weight SIMT unroll factor must be positive");

constexpr uint32_t kUdmaWqeBatchSize = 128;
constexpr uint32_t kUdmaWqeBytes = 64;
// Perf path uses QP0/1/2/4 by runtime count; transport QP3/QP5 stay idle, slot 5 is the peer credit.
constexpr uint32_t kMaxCreatedUdmaQpCount = 6;
constexpr uint32_t kMaxActiveUdmaQpCount = 4;
constexpr uint32_t kActiveUdmaQpIndices[kMaxActiveUdmaQpCount] = {0, 1, 2, 4};
// QP3 does no transfers; reuse its reserved cacheline to hold the SO send value so it does not share storage with the
// self completion flag.
constexpr uint32_t kPeerCompletionValueSlot = 3;
constexpr uint32_t kPeerCreditFlagSlot = 5;
static_assert(
    kPeerCompletionValueSlot != kActiveUdmaQpIndices[0] &&
    kPeerCompletionValueSlot != kActiveUdmaQpIndices[1] &&
    kPeerCompletionValueSlot != kActiveUdmaQpIndices[2] &&
    kPeerCompletionValueSlot != kActiveUdmaQpIndices[3]);
static_assert(kPeerCompletionValueSlot != kPeerCreditFlagSlot);
static_assert(kActiveUdmaQpIndices[kMaxActiveUdmaQpCount - 1] < kMaxCreatedUdmaQpCount);
static_assert(kPeerCreditFlagSlot < kMaxCreatedUdmaQpCount);
static_assert(kPeerCreditFlagSlot > kActiveUdmaQpIndices[kMaxActiveUdmaQpCount - 1]);

// Data puts always use Relax Order: defer produces no cqe, submit/immediate does.
// CQEs from submit/immediate are reclaimed together at the current peer completion phase.
// Flag puts alone use Strong Order (see PublishPeerCompletionSignal).
// aclshmemx_udma_op_config_t is visible only in the device compile pass (shmem.h
// guards it with __CCE_AICORE__); these constants are used only inside __aicore__
// functions, so guard them identically to avoid host-pass instantiation failures.
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
inline constexpr aclshmemx_udma_op_config_t kUdmaDataRoNoCqeConfig{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kUdmaDataRoCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kUdmaSyncSoCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
#endif

// Profiling is controlled by the bool template parameter:
// MoonEPDispatchUdmaKernelImpl<Profile> instantiates both kernels at once —
//   moonep_dispatch_udma_kernel          (Profile=false, zero timing overhead)
//   moonep_dispatch_udma_profile_kernel  (Profile=true, per-phase cycles + bytes)
// One build gives both launchers, selected at runtime; no rebuild to flip a switch.
// With Profile=true results go to GM profile_cycles_gm
// (int64 [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields]), read and printed by
// host; no AscendC::printf in the kernel, avoiding 32 AIV x R ranks of log spam.

// UB allocation
constexpr uint32_t kUdmaWqeScratchBytes = kUdmaWqeBatchSize * kUdmaWqeBytes;
// One SIMT thread block maps to one 512B weight GM conflict window.
constexpr uint32_t kWeightGmConflictGroupBytes = 512;
constexpr uint32_t kWeightsPerGmConflictGroup = kWeightGmConflictGroupBytes / sizeof(float);
constexpr uint32_t kWeightSimtThreadCount = kWeightsPerGmConflictGroup;
// Epilogue metadata tile stays a fixed 16KiB, independent of Weight slot size.
constexpr uint32_t kMteUbBytes          = 16 * 1024;
constexpr uint32_t kLocalCopyTileBytes  = 16 * 1024;
// Token / intra-board copies still chunk at 16KiB; the Weight stage splits 32KiB into two GM2GM scratch lanes.
constexpr uint32_t kLocalCopyBufferBytes = 32 * 1024;
constexpr uint32_t slotChunkCount = 8192;
constexpr uint32_t kSlotTensorBytes = slotChunkCount * sizeof(int32_t);
constexpr uint32_t kSlotCompareMaskBytes = ((slotChunkCount / 8 + 31) / 32) * 32;
// Selection scratch requirement of Token; the actual allocation below takes the max with the Weight batch layout.
constexpr uint32_t kSlotSelectTokenUbBytes = 4 * kSlotTensorBytes + kSlotCompareMaskBytes;
constexpr uint32_t kDuplicateHiddenStageBytes = 65504;
// Each hidden UB block aggregates at most 4 primary rows, amortizing event overhead and bounding the queued fan-out per
// issue.
constexpr uint32_t kDuplicateMaxPrimaryRows = 4;
constexpr uint32_t kDuplicateHiddenStagingBytes = 2 * kDuplicateHiddenStageBytes;
constexpr uint32_t kDuplicateGroupFields = 3;
constexpr uint32_t kDuplicateGroupStagingBytes = kMteUbBytes;
constexpr uint32_t kDuplicateLoffStagingBytes = kLocalCopyTileBytes;
constexpr uint32_t kDuplicateGroupsPerTile =
    kDuplicateGroupStagingBytes / (kDuplicateGroupFields * sizeof(int32_t));
constexpr uint32_t kAivUbCapacityBytes = 200 * 1024;
// The dynamic UB size of the kernel launch must be declared explicitly: the SIMT path reads UB inside the VF, and the
// runtime reserves based on this.
// 200KiB covers all TBufs; per the original budget this leaves 256-200-8 = 48KiB for the DataCache (requirement >=
// 32KiB).
constexpr uint32_t kMoonEPDispatchUdmaUbSizeBytes = 200 * 1024;
constexpr uint32_t kMaxDispatchGroupWidth = 16;
// 128P production topology: two 64P frames, each laid out as 8 direct-link
// rows x 8 columns.  A red-box dispatch group is one 4-row x 4-column tile,
// so each frame contains two row blocks x two left/right sides = four groups.
// 256P/512P keep the four-group-per-frame layout and extend it with cross-frame rounds.
constexpr uint32_t kDispatchRanksPerFrame = 64;
constexpr uint32_t kDispatchRanksPerDirectRow = 8;
constexpr uint32_t kDispatchRowsPerFrame = 8;
constexpr uint32_t kDispatchGroupRows = 4;
constexpr uint32_t kDispatchGroupColumns = 4;
constexpr uint32_t kDispatchGroupRowBlocks =
    kDispatchRowsPerFrame / kDispatchGroupRows;
constexpr uint32_t kDispatchGroupSides =
    kDispatchRanksPerDirectRow / kDispatchGroupColumns;
constexpr uint32_t kDispatchGroupsPerFrame =
    kDispatchGroupRowBlocks * kDispatchGroupSides;
constexpr uint32_t kMinStagedDispatchWorldSize = 2 * kDispatchRanksPerFrame;
constexpr uint32_t kMaxStagedDispatchFrameCount = 8;
constexpr uint32_t kMaxStagedDispatchWorldSize =
    kMaxStagedDispatchFrameCount * kDispatchRanksPerFrame;
constexpr uint32_t kAlternatingDispatchWorldSize = 2 * kDispatchRanksPerFrame;

enum class DispatchGroupSchedule : uint32_t {
    // Use the cross-half-world mapping for every supported world size.
    Legacy = 0,
    // Four rounds to one frame, then four rounds to the other frame.
    Staged = 1,
    // Alternate local/remote frame on every round while retaining a bijective
    // source-group -> destination-group mapping.
    Alternating = 2,
};

// Compile-time schedule selection; changing it requires rebuilding the kernel.
// Staged supports 128P, 256P, and 512P; Alternating applies only to 128P.
// Other supported sizes retain the Legacy mapping. Staged is the default.
constexpr DispatchGroupSchedule kDispatchGroupSchedule =
    DispatchGroupSchedule::Staged;
constexpr int64_t kPeerSyncFlagBytes = 512;
constexpr int64_t kPeerSyncSlotBytes = kMaxCreatedUdmaQpCount * kPeerSyncFlagBytes;
constexpr int64_t kRankSyncSlotBytes = kMaxDispatchGroupWidth * kPeerSyncSlotBytes;
constexpr uint32_t kMaxDispatchAivCount = 32;
constexpr bool kEnableParallelTokenWeight = true;
// Weight aggregate-send switch: true takes aggregate + whole-segment MTE + local expansion; false keeps the SIMT
// per-route remote direct write for A/B.
constexpr bool kEnableWeightAggregate = true;
constexpr int64_t kGroupSyncFlagBytes = 512;
constexpr uint32_t kGroupSyncRoundCount = 4;
constexpr uint32_t kGroupSyncPhasesPerGroup = 2;
constexpr uint32_t kGroupSyncCompletionPhase = 0;
constexpr uint32_t kGroupSyncDonePhase = 1;
constexpr int32_t kStrictGroupSyncMaxWorldSize = 64;
// At 64P and below, Token AIVs continue into Weight and Duplicate is owned by the last 16 AIVs;
// above 64P, keep the parallel layout with Token owning 16 cores and Weight and Duplicate sharing the rest.
constexpr int32_t kSerialWeightWorldSizeLimit = 64;
constexpr uint32_t kSerialWeightDuplicateAivCount = 16;
// The serial layout gives the 16 consecutive AIVs after the Token group to Duplicate and must not exceed the total AIV
// count.
static_assert(kMaxDispatchGroupWidth + kSerialWeightDuplicateAivCount <= kMaxDispatchAivCount);
static_assert(kPeerSyncFlagBytes >= sizeof(uint64_t));
static_assert(kPeerSyncFlagBytes % sizeof(uint64_t) == 0);
static_assert(kGroupSyncFlagBytes >= sizeof(uint64_t));
static_assert(kGroupSyncFlagBytes % sizeof(uint64_t) == 0);
static_assert(kMaxDispatchGroupWidth % 2 == 0);
static_assert(kMaxDispatchAivCount > kMaxDispatchGroupWidth);
static_assert((1U << kGroupSyncRoundCount) >= kMaxDispatchGroupWidth);
static_assert(kDispatchGroupRows * kDispatchGroupColumns == kMaxDispatchGroupWidth);
static_assert(
    kDispatchRowsPerFrame * kDispatchRanksPerDirectRow == kDispatchRanksPerFrame);
static_assert(
    kDispatchGroupsPerFrame * kMaxDispatchGroupWidth == kDispatchRanksPerFrame);
static_assert(kDispatchGroupRowBlocks == 2);
static_assert(kDispatchGroupSides == 2);
static_assert(kDispatchGroupsPerFrame == 4);
static_assert(kMinStagedDispatchWorldSize == 128);
static_assert(kAlternatingDispatchWorldSize == 128);
static_assert(kMaxStagedDispatchWorldSize == 512);
static_assert(kWeightGmConflictGroupBytes % sizeof(float) == 0);
static_assert(kWeightSimtThreadCount == 128);

// Alignment of the routes processed per chunk.
constexpr uint32_t kVectorSlotSelectCountAlign = 64;

// ============================================================
// Weight aggregate send: gather per peer into a contiguous segment -> whole-segment MTE send -> peer expands locally by
// slot.
// Replace per-route 4B remote stores with contiguous MTE transfers and local scatter,
// reducing the number of remote transactions for the NvS*4B weight payload per rank.
// Region layout, publication order, and capacity constraints are defined below.
// Switch kEnableWeightAggregate: turning it off falls back to the SIMT per-route remote direct write for A/B.
// ============================================================

// Each source rank owns a fixed 256KiB inbox at the end of the peer dispatch_sync_buffer,
// matching WEIGHT_AGG_REGION_BYTES in npu_patch/_common.py.
// Full results first land in each Weight AIV private GM staging; UB keeps only the current 16K batch.
constexpr uint32_t kWeightAggRegionBytes = 256 * 1024;
constexpr uint32_t kWeightAggStagingAivCount = kMaxDispatchGroupWidth;
constexpr uint32_t kWeightAggStagingBytes = kWeightAggRegionBytes;
// Region header 32B: magic(8B) | segment count(4B) | total entries(4B) | status(4B) | reserved(12B).
constexpr uint32_t kWeightAggRegionHeaderBytes = 32;
// Segment header 32B: count(4B) | reserved(28B); payload and slot follow, each 32B-aligned.
constexpr uint32_t kWeightAggSegmentHeaderBytes = 32;
// Each record is 8B: 4B weight value + 4B slot.
constexpr uint32_t kWeightAggEntryBytes = 2 * sizeof(float);
// Per-segment capacity: the entry count that fits after removing the region and segment headers.
// 32760 entries cover the S*K/R = 8192 entries of a balanced 16P layout and leave extra headroom for skew.
constexpr uint32_t kWeightAggCapacity =
    (kWeightAggRegionBytes - kWeightAggRegionHeaderBytes - kWeightAggSegmentHeaderBytes) /
    kWeightAggEntryBytes;
// The payload is 4B per weight; the per-segment capacity stays aligned to 8 entries (32B).
constexpr uint32_t kWeightAggAlignEntries = 32 / sizeof(float);
// Largest on-wire size of one segment: segment header + payload + slot.
constexpr uint32_t kWeightAggBlockBytes =
    kWeightAggSegmentHeaderBytes + kWeightAggCapacity * kWeightAggEntryBytes;
// Max peers per Weight core: at 512P there are 32 Group rounds.
constexpr uint32_t kWeightAggMaxPeersPerCore = kMaxStagedDispatchWorldSize / kMaxDispatchGroupWidth;
// Routes per decode chunk; rank scratch is reused per chunk while slot and mask are concatenated within the 16K batch.
constexpr uint32_t kWeightAggChunkRoutes = 4096;
constexpr uint32_t kWeightAggChunkBytes = kWeightAggChunkRoutes * sizeof(int32_t);
// Tune only this constant for 1/2/4-chunk A/B; UB is always allocated for four chunks so the cache budget does not
// change between runs.
constexpr uint32_t kWeightGatherBatchChunks = 4;
constexpr uint32_t kWeightGatherBatchRoutes = kWeightAggChunkRoutes * kWeightGatherBatchChunks;
constexpr uint32_t kWeightGatherMaxRoutes = 4 * kWeightAggChunkRoutes;
constexpr uint32_t kWeightGatherBatchBytes = kWeightGatherMaxRoutes * sizeof(int32_t);
constexpr uint32_t kWeightGatherMaskBytes = ((kWeightGatherMaxRoutes / 8 + 31) / 32) * 32;
static_assert(kWeightGatherBatchChunks == 1 || kWeightGatherBatchChunks == 2 || kWeightGatherBatchChunks == 4,
    "gather batch must contain 1, 2 or 4 chunks");
// Entries per local scatter in the expansion stage: 4KiB each of payload and slot, sharing the common MTE scratch.
constexpr uint32_t kWeightAggExpandRoutes = 1024;
constexpr uint32_t kWeightAggExpandBytes = kWeightAggExpandRoutes * sizeof(int32_t);
// Borrows slot_select_buffer: four chunks of compare masks | four chunks of slot/weight input | four chunks of compact
// output.
constexpr uint32_t kWeightAggSlotTileOffset = kWeightGatherMaskBytes;
constexpr uint32_t kWeightAggCompactTileOffset = kWeightAggSlotTileOffset + kWeightGatherBatchBytes;
constexpr uint32_t kWeightAggUbBytes = kWeightAggCompactTileOffset + kWeightGatherBatchBytes;
// The two GM2GM copies each use half of local_copy_buffer,
// so two nbi calls never reuse the same still-in-flight MTE3 source.
constexpr uint32_t kWeightAggGm2GmScratchBytes = kLocalCopyBufferBytes / 2;
static_assert(kWeightAggMaxPeersPerCore * kWeightAggRegionHeaderBytes <= kWeightAggGm2GmScratchBytes,
    "weight aggregate headers must fit in one GM2GM scratch half");
// The same TBuf is time-shared and sized for the larger of the two stages; offsets must never extend past the TBuf.
constexpr uint32_t kSlotSelectUbBytes =
    kSlotSelectTokenUbBytes > kWeightAggUbBytes ? kSlotSelectTokenUbBytes : kWeightAggUbBytes;
static_assert(kDuplicateHiddenStagingBytes <= kSlotSelectUbBytes);
static_assert(2 * kWeightAggGm2GmScratchBytes <= kLocalCopyBufferBytes);
// SIMT VF also needs the framework-reserved 8KiB immediate data UB.
static_assert(kUdmaWqeScratchBytes + kMteUbBytes + kLocalCopyBufferBytes + kSlotSelectUbBytes <=
    kAivUbCapacityBytes - 8 * 1024);

// The region header is read as 4-byte words: the first two words form the uint64 magic.
constexpr uint32_t kWeightAggSegCountWord = 2;
constexpr uint32_t kWeightAggEntryCountWord = 3;
constexpr uint32_t kWeightAggStatusWord = 4;
constexpr uint32_t kWeightAggRegionHeaderWords = kWeightAggRegionHeaderBytes / sizeof(uint32_t);
constexpr uint32_t kWeightAggStatusOk = 0;
// Aggregation out of capacity: the sender writes no out-of-bounds data, it only sets the status so the receiver can see
// it.
constexpr uint32_t kWeightAggStatusOverflow = 1;
// The current implementation sends one segment per peer (region capacity equals segment capacity; more segments only
// add 32B each).
constexpr uint32_t kWeightAggMaxSegments = 1;
// On validation failure, a sentinel written into the first element of weights_buf makes the weight comparison fail (no
// trap primitive is available).
constexpr uint32_t kWeightAggErrorSentinelBits = 0x7FC00000U;
// By default no sentinel is written: it would overwrite a real data slot and mask the true misalignment pattern as only
// (0,) being NaN.
// Failure details go to the diagnostics region (kWeightAggDebug*) for host inspection;
// the sentinel is only used when explicitly enabled.
constexpr bool kWeightAggFailHard = false;
// Diagnostics region: 32B per AIV, read by the host from the tail of dispatch_sync_buffer.
constexpr uint32_t kWeightAggDebugBytes = 1024;
constexpr uint32_t kWeightAggDebugRecordBytes = 32;
constexpr uint32_t kWeightAggDebugWords = kWeightAggDebugRecordBytes / sizeof(uint32_t);
// Region and segment headers are two independent transfers; on count mismatch retry a bounded number of times before
// treating a visibility glitch as a data error.
constexpr uint32_t kWeightAggCountRetry = 64;
// Diagnostic error codes: 0 ok; 1 out of capacity; 2 invalid region header; 3 segment/region count mismatch; 4 total
// mismatch across sources.
constexpr uint32_t kWeightAggCodeOk = 0;
constexpr uint32_t kWeightAggCodeOverflow = 1;
constexpr uint32_t kWeightAggCodeBadHeader = 2;
constexpr uint32_t kWeightAggCodeCountMismatch = 3;
constexpr uint32_t kWeightAggCodeTotalMismatch = 4;
static_assert(kWeightAggDebugBytes >= kMaxDispatchAivCount * kWeightAggDebugRecordBytes,
    "the diagnostics region must hold the records of all AIVs");
static_assert(kWeightAggRegionHeaderBytes + kWeightAggBlockBytes <= kWeightAggRegionBytes,
    "the region header plus one segment must fit in the weight inbox");
static_assert(kWeightAggCapacity * kWeightAggEntryBytes <= kWeightAggStagingBytes,
    "each Weight AIV GM staging must hold payload and slot");
static_assert(kWeightAggCapacity % kWeightAggAlignEntries == 0,
    "the aggregation capacity must be 32B-aligned so the in-segment slot start stays aligned");
static_assert(kWeightAggChunkRoutes % kVectorSlotSelectCountAlign == 0,
    "the chunk length must match the alignment granularity of Compares");
static_assert(kWeightAggChunkBytes <= kMteUbBytes, "the weight aggregation rank result chunk must fit in the shared MTE scratch");
static_assert(2 * kWeightAggExpandBytes <= kMteUbBytes,
    "the expansion-stage payload/slot scratch must fit in the shared MTE scratch");
static_assert(kWeightAggUbBytes <= kSlotSelectUbBytes, "the weight aggregation block must fit in the borrowed slot select UB");

// hidden is moved as raw BF16 bytes; row width and nominal byte count derive from DataType.
using DataType = uint16_t;
// Timing helpers gated by the Profile template parameter compile to no-op
// with zero hot-path cost when Profile=false; call GetSystemCycle and accumulate
// cycles only when Profile=true.
template <bool Profile>
__forceinline__ __aicore__ uint64_t UdmaProfileStart() {
    if constexpr (Profile) {
        return AscendC::GetSystemCycle();
    }
    return 0;
}

template <bool Profile>
__forceinline__ __aicore__ void UdmaProfileAccumulate(
    uint64_t start,
    uint64_t& cycles) {
    if constexpr (Profile) {
        cycles += AscendC::GetSystemCycle() - start;
    }
}

// Inner detailed timing switch (GetSystemCycle for wqe_submit / sync).
// Default false: outer brackets (token_scatter / weight_scatter / duplicate / total)
// avoid inner GetSystemCycle overhead. Set true and rebuild for per-core wqe_submit/sync
// breakdowns (outer brackets then include that overhead and read slightly high —
// cannot have both). Affects only GetSystemCycle calls; byte/packet counters stay
// gated by the outer Profile, so the data behind the two bw figures is unchanged.
constexpr bool kEnableDetailedProfile = false;

// Inner-detailed-only helpers: gate on Profile && kEnableDetailedProfile.
// Called only for wqe_submit / sync timing inside DispatchScatterToken.
template <bool DetailedProfile>
__forceinline__ __aicore__ uint64_t UdmaDetailedProfileStart() {
    if constexpr (DetailedProfile) {
        return AscendC::GetSystemCycle();
    }
    return 0;
}

template <bool DetailedProfile>
__forceinline__ __aicore__ void UdmaDetailedProfileAccumulate(
    uint64_t start,
    uint64_t& cycles) {
    if constexpr (DetailedProfile) {
        cycles += AscendC::GetSystemCycle() - start;
    }
}

// 2-QP perf experiment: per group of four chunks, the first three go to QP0 and the last to QP1.
__aicore__ inline uint32_t GetChunkQpSlot(int64_t chunk_index, uint32_t active_qp_count)
{
    // 3:1 weighted split only in 2-QP mode; other modes keep the uniform round-robin.
    if (active_qp_count == 2) {
        return chunk_index % 4 == 1 ? 1U : 0U;
    }
    return static_cast<uint32_t>(chunk_index % active_qp_count);
}

// Each (active AIV, QP) completion flag owns a dedicated 512B cacheline.
__aicore__ inline __gm__ uint64_t *GetPeerCompletionFlag(
    int32_t src_rank, uint32_t peer_rank_index, uint32_t qp_idx, __gm__ uint8_t *sync_buffer)
{
    return reinterpret_cast<__gm__ uint64_t *>(
        sync_buffer +
        static_cast<int64_t>(src_rank) * kRankSyncSlotBytes +
        static_cast<int64_t>(peer_rank_index) * kPeerSyncSlotBytes +
        qp_idx * kPeerSyncFlagBytes);
}

// Each active AIV's peer credit owns a dedicated 512B cacheline (slot 5).
__aicore__ inline __gm__ uint64_t *GetPeerCreditFlag(
    int32_t src_rank, uint32_t peer_rank_index, __gm__ uint8_t *sync_buffer)
{
    return reinterpret_cast<__gm__ uint64_t *>(
        sync_buffer +
        static_cast<int64_t>(src_rank) * kRankSyncSlotBytes +
        static_cast<int64_t>(peer_rank_index) * kPeerSyncSlotBytes +
        kPeerCreditFlagSlot * kPeerSyncFlagBytes);
}

// The credit encodes both kernel round and group generation, so async peers never consume a stale value.
__aicore__ inline uint64_t GetPeerCreditToken(uint64_t magic, uint32_t group, uint32_t group_count)
{
    return magic * (static_cast<uint64_t>(group_count) + 1U) + group + 1U;
}

// Each (active AIV, round) barrier flag owns a dedicated 512B region.
__aicore__ inline __gm__ uint64_t *GetGroupSyncFlag(
    uint32_t aiv_index, uint32_t round, __gm__ uint8_t *group_sync_buffer)
{
    const int64_t flag_index = static_cast<int64_t>(aiv_index) * kGroupSyncRoundCount + round;
    return reinterpret_cast<__gm__ uint64_t *>(group_sync_buffer + flag_index * kGroupSyncFlagBytes);
}

// Group sync tokens increase monotonically: init, then per-group completion, then per-group done.
__aicore__ inline uint64_t GetGroupSyncInitToken(uint64_t magic, uint32_t group_count)
{
    const uint64_t token_count = static_cast<uint64_t>(group_count) * kGroupSyncPhasesPerGroup + 1U;
    return magic * token_count + 1U;
}

// Returns the sync token of the given stage within a group; after the init token it increments in completion, then done
// order.
__aicore__ inline uint64_t GetGroupSyncPhaseToken(
    uint64_t magic, uint32_t group, uint32_t group_count, uint32_t phase)
{
    const uint64_t token_count = static_cast<uint64_t>(group_count) * kGroupSyncPhasesPerGroup + 1U;
    const uint64_t group_token_offset =
        static_cast<uint64_t>(group) * kGroupSyncPhasesPerGroup + phase + 2U;
    return magic * token_count + group_token_offset;
}

// Active AIVs use a four-round dissemination barrier; every flag always has one writer and one reader.
__aicore__ inline void SyncActiveAivs(uint32_t aiv_index, uint32_t active_aiv_count,
    uint64_t sync_token, __gm__ uint8_t *group_sync_buffer)
{
    uint32_t distance = 1U;
    uint32_t round = 0U;
    // Reach doubles each round; 16 active AIVs finish strict sync in at most four rounds.
    while (distance < active_aiv_count) {
        const uint32_t next_aiv = (aiv_index + distance) % active_aiv_count;
        __gm__ uint64_t *next_flag = GetGroupSyncFlag(next_aiv, round, group_sync_buffer);
        *next_flag = sync_token;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(next_flag));

        // Each AIV polls only its own dedicated flag; GE lets it safely consume a token already advanced to the next generation.
        aclshmem_uint64_wait_until(
            GetGroupSyncFlag(aiv_index, round, group_sync_buffer), ACLSHMEM_CMP_GE, sync_token);
        distance <<= 1U;
        ++round;
    }
}

// Below 16P all ranks form one group; larger sizes keep splitting into 16-rank groups.
__aicore__ inline uint32_t GetDispatchGroupWidth(int32_t world_size)
{
    return world_size < static_cast<int32_t>(kMaxDispatchGroupWidth) ?
        static_cast<uint32_t>(world_size) : kMaxDispatchGroupWidth;
}

// Group flow control requires an even rank count and an exact partition into equal-width
// groups; the kernel always launches 32 AIVs: the first group_width run Hidden, the
// rest run Weight in parallel with Hidden.
__aicore__ inline bool CanUseGroupFlowControl(int32_t world_size, int64_t aiv_count)
{
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    return world_size >= 2 && world_size % 2 == 0 &&
        static_cast<uint32_t>(world_size) % group_width == 0 &&
        aiv_count == static_cast<int64_t>(kMaxDispatchAivCount) &&
        static_cast<int64_t>(group_width) < aiv_count;
}

// Staged only takes over power-of-two sizes made of at least two complete 64P frames; Alternating keeps the original
// 128P experimental boundary.
__aicore__ inline bool UseFrameDispatchSchedule(int32_t world_size)
{
    if constexpr (kDispatchGroupSchedule == DispatchGroupSchedule::Legacy) {
        return false;
    }
    if constexpr (kDispatchGroupSchedule == DispatchGroupSchedule::Alternating) {
        return world_size == static_cast<int32_t>(kAlternatingDispatchWorldSize);
    }
    if (world_size < static_cast<int32_t>(kMinStagedDispatchWorldSize) ||
        world_size > static_cast<int32_t>(kMaxStagedDispatchWorldSize) ||
        world_size % static_cast<int32_t>(kDispatchRanksPerFrame) != 0) {
        return false;
    }
    const uint32_t frame_count = static_cast<uint32_t>(world_size) / kDispatchRanksPerFrame;
    return (frame_count & (frame_count - 1U)) == 0;
}

// Splits a logical send round into a target frame phase and the four in-frame red-box sub-rounds.
// Staged completes one frame phase every four rounds; Alternating still switches frames every round per the original
// 128P semantics.
__aicore__ inline uint32_t GetDispatchFramePhase(uint32_t group, uint32_t frame_count)
{
    if constexpr (kDispatchGroupSchedule == DispatchGroupSchedule::Alternating) {
        return group % frame_count;
    }
    return group / kDispatchGroupsPerFrame;
}

__aicore__ inline uint32_t GetDispatchFrameSubround(uint32_t group, uint32_t frame_count)
{
    if constexpr (kDispatchGroupSchedule == DispatchGroupSchedule::Alternating) {
        return group / frame_count;
    }
    return group % kDispatchGroupsPerFrame;
}

// 512P spreads the four red boxes of one source frame across distinct target frames; at 128P the left/right side alone
// decides same/opposite frame.
__aicore__ inline uint32_t GetDispatchFrameSpread(
    uint32_t source_group_row, uint32_t source_group_side, uint32_t frame_count)
{
    if (frame_count == 1U) {
        return 0U;
    }
    const uint32_t row_spread = frame_count >= 4U ? frame_count / 2U : 0U;
    return source_group_side + source_group_row * row_spread;
}

// The frame offset depends on both the current phase and the source red-box position; 256P visits the two 128P halves
// phase by phase.
__aicore__ inline uint32_t GetDispatchFrameOffset(uint32_t source_group_row,
    uint32_t source_group_side, uint32_t frame_phase, uint32_t frame_count)
{
    if constexpr (kDispatchGroupSchedule == DispatchGroupSchedule::Alternating) {
        return source_group_side ^ frame_phase;
    }
    // 256P first sends within its own 128P half in the original 128P left/right order, then switches to the other 128P
    // half.
    if (frame_count == 4U) {
        return (frame_phase / 2U) * 2U + (source_group_side ^ (frame_phase % 2U));
    }
    const uint32_t frame_spread =
        GetDispatchFrameSpread(source_group_row, source_group_side, frame_count);
    return (frame_phase + frame_spread) % frame_count;
}

// Rebuilds the global rank from frame, group row/column position, and in-group index; this is the forward mapping of
// the in-frame layout.
__aicore__ inline int32_t BuildFrameDispatchRank(
    uint32_t frame, uint32_t group_row, uint32_t group_side, uint32_t group_rank_index)
{
    const uint32_t group_rank_row = group_rank_index / kDispatchGroupColumns;
    const uint32_t group_rank_column = group_rank_index % kDispatchGroupColumns;
    const uint32_t row = group_row * kDispatchGroupRows + group_rank_row;
    const uint32_t column = group_side * kDispatchGroupColumns + group_rank_column;
    return static_cast<int32_t>(
        frame * kDispatchRanksPerFrame + row * kDispatchRanksPerDirectRow + column);
}

// Return the rank's index within its 16-rank group. Frame schedules use the in-frame
// 4x4 layout; other sizes use the cross-half-world group mapping.
__aicore__ inline uint32_t GetGroupRankIndex(int32_t rank, int32_t world_size)
{
    if (UseFrameDispatchSchedule(world_size)) {
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t column = local_rank % kDispatchRanksPerDirectRow;
        // Row-major flatten the rank's (row, column) offset inside the 4x4 red box to
        // 0..15; this index locates peer sync slots and matches the indexing used by
        // BuildTwoFrameDispatchRank.
        return (row % kDispatchGroupRows) * kDispatchGroupColumns +
            column % kDispatchGroupColumns;
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t half_group_width = GetDispatchGroupWidth(world_size) / 2;
    return static_cast<uint32_t>(rank / half_world_size) * half_group_width +
        static_cast<uint32_t>(rank % static_cast<int32_t>(half_group_width));
}

// Returns the global target rank this core actually operates on, from the logical round and in-group index.
// Every per-round transform is a bijection, so each target red box receives exactly one source red box.
__aicore__ inline int32_t GetGroupDstRank(
    int32_t rank, int32_t world_size, uint32_t group, uint32_t group_rank_index)
{
    if (UseFrameDispatchSchedule(world_size)) {
        const uint32_t frame_count = static_cast<uint32_t>(world_size) / kDispatchRanksPerFrame;
        const uint32_t source_frame = static_cast<uint32_t>(rank) / kDispatchRanksPerFrame;
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t source_row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t source_column = local_rank % kDispatchRanksPerDirectRow;
        const uint32_t source_group_row = source_row / kDispatchGroupRows;
        const uint32_t source_group_side = source_column / kDispatchGroupColumns;

        // frame_phase selects the current frame phase; subround enumerates the four red-box groups in the target frame.
        const uint32_t frame_phase = GetDispatchFramePhase(group, frame_count);
        const uint32_t subround = GetDispatchFrameSubround(group, frame_count);

        // Treat subround as a two-bit binary number: the high bit is row_delta, the low bit is side_delta.
        //
        //   subround (binary)   row_delta  side_delta  relative to target group
        //          0 (00)           0           0       same row block, same side
        //          1 (01)           0           1       same row block, other side
        //          2 (10)           1           0       other row block, same side
        //          3 (11)           1           1       other row block, other side
        const uint32_t group_row_delta = subround / 2U;
        const uint32_t group_side_delta = subround % 2U;

        // 256P visits only two target frames per round; 512P still spreads the four source red boxes over four target
        // frames.
        const uint32_t frame_offset = GetDispatchFrameOffset(
            source_group_row, source_group_side, frame_phase, frame_count);
        // 256P switches the 128P half with XOR; other sizes keep the original modular-add mapping.
        const uint32_t destination_frame = frame_count == 4U ?
            source_frame ^ frame_offset : (source_frame + frame_offset) % frame_count;
        // row_delta=0 keeps the upper/lower row block; row_delta=1 switches between them.
        const uint32_t destination_group_row =
            (source_group_row + group_row_delta) % kDispatchGroupRowBlocks;
        // side_delta=0 keeps the side; side_delta=1 swaps between left and right groups.
        const uint32_t destination_group_side = source_group_side ^ group_side_delta;

        return BuildFrameDispatchRank(
            destination_frame, destination_group_row, destination_group_side, group_rank_index);
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    const uint32_t half_group_width = group_width / 2;
    const uint32_t group_count = static_cast<uint32_t>(world_size) / group_width;
    const uint32_t rank_group = static_cast<uint32_t>(rank % half_world_size) /
        half_group_width;
    const uint32_t peer_group = (rank_group + group) % group_count;
    return static_cast<int32_t>(group_rank_index / half_group_width) * half_world_size +
        static_cast<int32_t>(peer_group * half_group_width + group_rank_index % half_group_width);
}

// Return the real source rank that sends to this rank in the current group.
// The 128P path is the exact inverse of GetGroupDstRank, keeping completion/credit slots 1:1.
__aicore__ inline int32_t GetGroupIncomingPeer(
    int32_t rank, int32_t world_size, uint32_t group, uint32_t group_rank_index)
{
    if (UseFrameDispatchSchedule(world_size)) {
        const uint32_t frame_count = static_cast<uint32_t>(world_size) / kDispatchRanksPerFrame;
        const uint32_t destination_frame = static_cast<uint32_t>(rank) / kDispatchRanksPerFrame;
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t destination_row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t destination_column = local_rank % kDispatchRanksPerDirectRow;
        const uint32_t destination_group_row = destination_row / kDispatchGroupRows;
        const uint32_t destination_group_side = destination_column / kDispatchGroupColumns;

        // Use the same round decoding as the forward map, then undo side, row, frame in order.
        const uint32_t frame_phase = GetDispatchFramePhase(group, frame_count);
        const uint32_t subround = GetDispatchFrameSubround(group, frame_count);
        const uint32_t group_row_delta = subround / 2U;
        const uint32_t group_side_delta = subround % 2U;
        // XOR is its own inverse: x ^ b ^ b == x. Recover the source group side first,
        // then apply the same formula backwards to derive the source frame.
        const uint32_t source_group_side = destination_group_side ^ group_side_delta;
        const uint32_t source_group_row =
            (destination_group_row + kDispatchGroupRowBlocks - group_row_delta) %
            kDispatchGroupRowBlocks;
        // After recovering the source group, compute the same frame offset: XOR is self-inverse at 256P, other sizes
        // still use modular subtraction.
        const uint32_t frame_offset = GetDispatchFrameOffset(
            source_group_row, source_group_side, frame_phase, frame_count);
        const uint32_t source_frame = frame_count == 4U ?
            destination_frame ^ frame_offset :
            (destination_frame + frame_count - frame_offset) % frame_count;

        return BuildFrameDispatchRank(
            source_frame, source_group_row, source_group_side, group_rank_index);
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    const uint32_t half_group_width = group_width / 2;
    const uint32_t group_count = static_cast<uint32_t>(world_size) / group_width;

    const uint32_t rank_group = static_cast<uint32_t>(rank % half_world_size) / half_group_width;
    const uint32_t incoming_group = (rank_group + group_count - group % group_count) % group_count;

    return static_cast<int32_t>(group_rank_index / half_group_width) * half_world_size +
        static_cast<int32_t>(incoming_group * half_group_width + group_rank_index % half_group_width);
}

// After the current group completes, send credit to the source rank that will send to this rank in the next relative group.
__aicore__ inline int32_t GetNextGroupCreditPeer(
    int32_t rank, int32_t world_size, uint32_t group, uint32_t group_rank_index)
{
    return GetGroupIncomingPeer(rank, world_size, group + 1U, group_rank_index);
}

// Local GM-to-GM copy that goes through UB and is done tile by tile.
__aicore__ inline void CopyLocalBytes(GM_ADDR destination_gm, GM_ADDR source_gm, int64_t bytes,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &copy_buffer)
{
    AscendC::LocalTensor<uint8_t> local_tensor = copy_buffer.Get<uint8_t>();
    AscendC::GlobalTensor<uint8_t> source_tensor;
    AscendC::GlobalTensor<uint8_t> destination_tensor;
    source_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(source_gm), bytes);
    destination_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(destination_gm), bytes);

    for (int64_t copied_bytes = 0; copied_bytes < bytes; copied_bytes += kLocalCopyTileBytes) {
        int64_t current_tile_bytes = bytes - copied_bytes;
        if (current_tile_bytes > static_cast<int64_t>(kLocalCopyTileBytes)) {
            current_tile_bytes = kLocalCopyTileBytes;
        }
        AscendC::DataCopyPadExtParams<uint8_t> pad_parameters {false, 0, 0, 0};
        AscendC::DataCopyExtParams copy_parameters {
            1, static_cast<uint32_t>(current_tile_bytes), 0, 0, 0};
        AscendC::DataCopyPad(local_tensor, source_tensor[copied_bytes], copy_parameters, pad_parameters);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(destination_tensor[copied_bytes], local_tensor, copy_parameters);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// Splits total evenly into part_count parts and returns the range of part_index (the first remainder parts get one
// extra).
__aicore__ inline void DispatchSplitRange(int64_t total, int64_t part_count, int64_t part_index,
    int64_t &range_begin, int64_t &range_end)
{
    if (total <= 0 || part_count <= 0 || part_index < 0 || part_index >= part_count) {
        range_begin = 0;
        range_end = 0;
        return;
    }
    const int64_t base_count = total / part_count;
    const int64_t remainder = total - base_count * part_count;
    range_begin = part_index * base_count + (part_index < remainder ? part_index : remainder);
    range_end = range_begin + base_count + (part_index < remainder ? 1 : 0);
}

// Split weights per 512B GM conflict window so adjacent AIVs never share one 128-float route group.
__aicore__ inline void DispatchSplitWeightRange(int64_t total, int64_t part_count, int64_t part_index,
    int64_t &range_begin, int64_t &range_end)
{
    const int64_t group_size = static_cast<int64_t>(kWeightsPerGmConflictGroup);
    const int64_t group_count = total > 0 ? total / group_size + (total % group_size != 0 ? 1 : 0) : 0;
    int64_t group_begin = 0;
    int64_t group_end = 0;
    DispatchSplitRange(group_count, part_count, part_index, group_begin, group_end);
    range_begin = group_begin * group_size;
    range_end = group_end * group_size;
    if (range_begin > total) {
        range_begin = total;
    }
    if (range_end > total) {
        range_end = total;
    }
}

// Copies a GM int32 array to UB and returns only after MTE2 completes.
__aicore__ inline void CopyGmInt32ToUb(AscendC::LocalTensor<int32_t> destination_ub,
    __gm__ int32_t *source_gm, uint32_t element_count)
{
    AscendC::GlobalTensor<int32_t> source_tensor;
    source_tensor.SetGlobalBuffer(source_gm, element_count);
    AscendC::DataCopyParams copy_parameters {
        1, static_cast<uint16_t>(element_count * sizeof(int32_t)), 0, 0};
    AscendC::DataCopyPad(
        destination_ub, source_tensor, copy_parameters, AscendC::DataCopyPadParams {});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
}

// Submits the MTE2 read of one primary row only; the caller hands the whole tile over to MTE3 after it is loaded.
__aicore__ inline void IssueGmBytesToUb(AscendC::LocalTensor<uint8_t> destination_ub,
    GM_ADDR source_gm, uint32_t bytes)
{
    AscendC::GlobalTensor<uint8_t> source_tensor;
    source_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(source_gm), bytes);
    AscendC::DataCopyPadExtParams<uint8_t> pad_parameters {false, 0, 0, 0};
    AscendC::DataCopyExtParams copy_parameters {1, bytes, 0, 0, 0};
    AscendC::DataCopyPad(destination_ub, source_tensor, copy_parameters, pad_parameters);
}

// Submits the UB-to-GM MTE3 write only; the caller waits for completion centrally.
__aicore__ inline void IssueUbBytesToGm(GM_ADDR destination_gm,
    AscendC::LocalTensor<uint8_t> source_ub, uint32_t bytes)
{
    AscendC::GlobalTensor<uint8_t> destination_tensor;
    destination_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(destination_gm), bytes);
    AscendC::DataCopyExtParams copy_parameters {1, bytes, 0, 0, 0};
    AscendC::DataCopyPad(destination_tensor, source_ub, copy_parameters);
}

// Each Token AIV initializes its own completion/credit cachelines, then does one local strict sync.
__aicore__ inline void PrepareTokenScatterPeerSignals(int32_t current_rank,
    uint32_t group_rank_index, uint64_t magic, uint32_t active_qp_count,
    uint32_t active_aiv_count, uint32_t group_count,
    __gm__ uint8_t *sync_buffer, __gm__ uint8_t *group_sync_buffer)
{
    // Each active QP gets its own completion cacheline.
    // The self completion flag is cleared first; magic can be published only after all local primary copies finish.
    for (uint32_t i = 0; i < active_qp_count; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        __gm__ uint64_t *signal_flag = GetPeerCompletionFlag(current_rank, group_rank_index, qp_idx, sync_buffer);
        *signal_flag = 0;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(signal_flag));
    }
    // The SO send value is read-only and constant for the whole round, so one AIV visiting several peers never triggers
    // self forwarding early.
    __gm__ uint64_t *completion_value =
        GetPeerCompletionFlag(current_rank, group_rank_index, kPeerCompletionValueSlot, sync_buffer);
    *completion_value = magic;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(completion_value));
    __gm__ uint64_t *credit_flag = GetPeerCreditFlag(current_rank, group_rank_index, sync_buffer);
    *credit_flag = 0;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(credit_flag));

    SyncActiveAivs(group_rank_index, active_aiv_count,
        GetGroupSyncInitToken(magic, group_count), group_sync_buffer);
}

// Each active AIV polls only the credit its current target peer published for this source rank.
__aicore__ inline void WaitPeerCredit(uint32_t group, int32_t dst_rank,
    int32_t current_rank, uint32_t current_rank_index, uint64_t magic,
    uint32_t group_count, __gm__ uint8_t *sync_buffer)
{
    // self consumes no remote UDMA receive resources, so no remote credit wait is needed.
    if (dst_rank == current_rank) {
        return;
    }
    const uint64_t credit_token = GetPeerCreditToken(magic, group, group_count);
    aclshmem_uint64_wait_until(
        GetPeerCreditFlag(dst_rank, current_rank_index, sync_buffer), ACLSHMEM_CMP_EQ, credit_token);
}

// The receiver polls only the dedicated completion cachelines of the current incoming peer.
__aicore__ inline void WaitPeerCompletionSignal(int32_t incoming_peer,
    uint32_t current_rank_index, uint64_t magic, uint32_t active_qp_count,
    __gm__ uint8_t *sync_buffer)
{
    for (uint32_t i = 0; i < active_qp_count; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        aclshmem_uint64_wait_until(
            GetPeerCompletionFlag(incoming_peer, current_rank_index, qp_idx, sync_buffer),
            ACLSHMEM_CMP_EQ, magic);
    }
}

// Non-blocking check of all active QPs of a source; returns as soon as any SO is missing so the forwarding core can
// scan other sources.
__aicore__ inline bool IsSourceReadyForForwarding(int32_t incoming_peer,
    uint32_t current_rank_index, uint64_t magic, uint32_t active_qp_count,
    __gm__ uint8_t *sync_buffer)
{
    AscendC::GlobalTensor<uint64_t> flag_tensor;
    // Reuses the wait_until cacheline refresh and scalar read procedure; each QP is checked once.
    for (uint32_t i = 0; i < active_qp_count; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        __gm__ uint64_t *flag = GetPeerCompletionFlag(incoming_peer, current_rank_index, qp_idx, sync_buffer);
        flag_tensor.SetGlobalBuffer(flag);
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(flag));
        // If any QP has not reached this round completion value, the source primary must not be read.
        if (flag_tensor.GetValue(0) != magic) {
            return false;
        }
    }
    return true;
}

// This AIV appends SO on every active QP of the current dst using its own completion cachelines.
__aicore__ inline void PublishPeerCompletionSignal(int32_t dst_rank,
    int32_t current_rank, uint32_t group_rank_index, uint32_t active_qp_count,
    __gm__ uint8_t *sync_buffer, __ubuf__ uint8_t *udma_wqe_buffer)
{
    __gm__ uint64_t *completion_value =
        GetPeerCompletionFlag(current_rank, group_rank_index, kPeerCompletionValueSlot, sync_buffer);
    // self has no SO WQE: first wait for this core own MTE3 self-copy, then publish the same set of completion flags.
    if (dst_rank == current_rank) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
    // Append one SO per active QP so that QP's RO puts become visible to the destination peer.
    for (uint32_t i = 0; i < active_qp_count; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        __gm__ uint64_t *signal_flag = GetPeerCompletionFlag(
            current_rank, group_rank_index, qp_idx, sync_buffer);

        // UDMA has no self put; local data is already done by this AIV.
        // self publishes its local completion value to the completion flag for server forwarding cores to read per
        // source.
        if (dst_rank == current_rank) {
            *signal_flag = *completion_value;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t *>(signal_flag));
            continue;
        }
        aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kUdmaSyncSoCqeConfig>(
            signal_flag, completion_value, reinterpret_cast<__ubuf__ uint64_t *>(udma_wqe_buffer),
            1, dst_rank, qp_idx, EVENT_ID0);
    }
}

// After seeing completion, this AIV grants credit to the next incoming peer via its own credit cacheline.
__aicore__ inline void PublishNextPeerCredit(uint32_t group,
    int32_t current_rank, int32_t world_size, uint32_t group_rank_index,
    uint64_t magic, uint32_t group_count, __gm__ uint8_t *sync_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &local_copy_buffer)
{
    const int32_t next_peer = GetNextGroupCreditPeer(
        current_rank, world_size, group, group_rank_index);
    // self's local copy uses no remote receive resources; no cross-rank credit publish is needed.
    if (next_peer == current_rank) {
        return;
    }

    constexpr uint32_t credit_word_count = kPeerSyncFlagBytes / sizeof(uint64_t);
    const uint64_t next_credit_token = GetPeerCreditToken(magic, group + 1U, group_count);
    // Reuse the local copy UB to build the credit cacheline, avoiding a dedicated buffer.
    AscendC::LocalTensor<uint64_t> credit_local = local_copy_buffer.Get<uint64_t>();
    // Send the full 512B cacheline to avoid read/write conflicts with neighboring peers' sync bits during remote polling.
    for (uint32_t i = 0; i < credit_word_count; ++i) {
        credit_local.SetValue(i, i == 0U ? next_credit_token : 0U);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    __gm__ uint64_t *local_credit = GetPeerCreditFlag(current_rank, group_rank_index, sync_buffer);
    __gm__ uint64_t *remote_credit = reinterpret_cast<__gm__ uint64_t *>(
        aclshmem_ptr(local_credit, next_peer));
    AscendC::GlobalTensor<uint64_t> remote_credit_tensor;
    remote_credit_tensor.SetGlobalBuffer(remote_credit, credit_word_count);
    AscendC::DataCopy(remote_credit_tensor, credit_local, credit_word_count);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

// Each active AIV completes its current peer; at <=64P all lock-step at completion and group end.
__aicore__ inline void CompleteTokenScatterPeerWithSignal(uint32_t group, int32_t dst_rank,
    int32_t current_rank, int32_t world_size, uint32_t group_rank_index,
    uint32_t current_rank_index, uint64_t magic, uint32_t active_qp_count, bool enable_strict_group_sync,
    __gm__ uint8_t *sync_buffer, __gm__ uint8_t *group_sync_buffer,
    __ubuf__ uint8_t *udma_wqe_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &local_copy_buffer)
{
    const uint32_t active_aiv_count = GetDispatchGroupWidth(world_size);
    const uint32_t group_count = static_cast<uint32_t>(world_size) / active_aiv_count;
    PublishPeerCompletionSignal(
        dst_rank, current_rank, group_rank_index, active_qp_count, sync_buffer, udma_wqe_buffer);

    // This AIV's index determines the incoming source peer; the current rank's in-group
    // index determines the completion slot the remote sender uses.
    const int32_t incoming_peer = GetGroupIncomingPeer(current_rank, world_size, group, group_rank_index);
    WaitPeerCompletionSignal(
        incoming_peer, current_rank_index, magic, active_qp_count, sync_buffer);

    // At <=64P, wait for all of this rank's Token AIVs before any may publish the next-round credit.
    if (enable_strict_group_sync) {
        SyncActiveAivs(group_rank_index, active_aiv_count,
            GetGroupSyncPhaseToken(magic, group, group_count, kGroupSyncCompletionPhase),
            group_sync_buffer);
    }

    // The last group starts no next round, so only the current incoming SO acknowledgement is needed.
    if (group + 1U < group_count) {
        PublishNextPeerCredit(
            group, current_rank, world_size, group_rank_index, magic, group_count,
            sync_buffer, local_copy_buffer);
    }

    // The self path has no UDMA WQE and never touches self QP context.
    if (dst_rank != current_rank) {
        // Incoming completion is done and the next peer released when needed; now reclaim
        // all CQEs of the current outgoing peer at once.
        for (uint32_t qp_i = 0; qp_i < active_qp_count; ++qp_i) {
            aclshmemx_udma_qp_quiet(dst_rank, kActiveUdmaQpIndices[qp_i]);
        }
    }

    // At <=64P, align credit publish and CQE reclaim at the group tail; larger sizes keep AIVs advancing async.
    if (enable_strict_group_sync) {
        SyncActiveAivs(group_rank_index, active_aiv_count,
            GetGroupSyncPhaseToken(magic, group, group_count, kGroupSyncDonePhase),
            group_sync_buffer);
    }
}

// Enable chunked vector rank select only when the encoded slot count satisfies the shift-decode constraint.
__aicore__ inline bool CanUseVectorSlotSelect(int64_t route_count)
{
    if (route_count <= 0 || (route_count & (route_count - 1)) != 0) {
        return false;
    }
    uint32_t slot_shift = 0;
    for (int64_t value = route_count; value > 1; value >>= 1) {
        ++slot_shift;
    }
    return slot_shift <= 31;
}

// Phase 1: vector-select the primary dst and batch-issue token WQEs.
// template <bool Profile>, matching the UBMEM version, replaces the constexpr switch;
// with Profile=false the GetSystemCycle timing compiles to no-op, counters come via reference params.
template <bool Profile>
__aicore__ inline uint64_t DispatchScatterToken(
    GM_ADDR hidden_gm,
    GM_ADDR dst_gm,
    GM_ADDR hidden_buf_gm,
    GM_ADDR sync_buffer_gm,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_topk, uint64_t magic,
    uint32_t active_qp_count,
    uint64_t &wqe_submit_cycles,
    uint64_t &wqe_submit_packets,
    uint64_t &sync_cycles,
    __ubuf__ uint8_t *udma_wqe_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &local_copy_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &slot_select_buffer)
{
    constexpr bool detailed_profile = Profile && kEnableDetailedProfile;
    __gm__ uint8_t *hidden = reinterpret_cast<__gm__ uint8_t *>(hidden_gm);
    __gm__ int32_t *encoded_destinations = reinterpret_cast<__gm__ int32_t *>(dst_gm);
    const int64_t route_count = num_tokens * num_topk;
    const int64_t hidden_row_bytes = hidden_dim * sizeof(DataType);
    __gm__ uint8_t *hidden_window = reinterpret_cast<__gm__ uint8_t *>(hidden_buf_gm);
    __gm__ uint8_t *sync_buffer = reinterpret_cast<__gm__ uint8_t *>(sync_buffer_gm);
    const int64_t aiv_index = AscendC::GetBlockIdx();
    const int32_t current_rank = aclshmem_my_pe();
    const int32_t world_size = aclshmem_n_pes();
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    const uint32_t group_count = static_cast<uint32_t>(world_size) / group_width;
    // The first 32KiB after the Token control region holds this rank's local sync flags.
    __gm__ uint8_t *group_sync_buffer =
        sync_buffer + static_cast<uint64_t>(world_size) * kRankSyncSlotBytes;
    const uint32_t group_rank_index = static_cast<uint32_t>(aiv_index) % group_width;
    const uint32_t current_rank_index = GetGroupRankIndex(current_rank, world_size);
    // Rank count is fixed for the kernel run, so compute the strict group sync switch once.
    const bool enable_strict_group_sync = world_size <= kStrictGroupSyncMaxWorldSize;
    uint64_t transferred_bytes = 0;

    // Step 0: Token AIVs init their dedicated signals and start issuing after the local strict sync.
    PrepareTokenScatterPeerSignals(
        current_rank, group_rank_index, magic, active_qp_count,
        group_width, group_count, sync_buffer, group_sync_buffer);

    if (CanUseVectorSlotSelect(route_count)) {
        // Step 1: load dst into UB in chunks, keeping a separate index region for select results so raw dst is not overwritten.
        const uint32_t tensor_bytes = slotChunkCount * sizeof(int32_t);
        const uint32_t compare_mask_bytes = ((slotChunkCount / 8 + 31) / 32) * 32;
        AscendC::LocalTensor<int32_t> dst_local =
            slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, 0);
        AscendC::LocalTensor<int32_t> dst_rank_tensor =
            slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes);
        AscendC::LocalTensor<int32_t> route_indices =
            slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes * 2);
        AscendC::LocalTensor<int32_t> selected_route_indices =
            slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes * 3);
        AscendC::LocalTensor<uint8_t> compare_mask =
            slot_select_buffer.GetWithOffset<uint8_t>(compare_mask_bytes, tensor_bytes * 4);

        AscendC::GlobalTensor<int32_t> dst_global;
        dst_global.SetGlobalBuffer(encoded_destinations, route_count);
        uint32_t slot_shift = 0;
        for (int64_t value = route_count; value > 1; value >>= 1) {
            ++slot_shift;
        }

        // Step 2: each active AIV walks the peer sequence independently; group 0 sends directly, later peers wait for the dedicated credit.
        for (uint32_t group = 0; group < group_count; ++group) {
            const int32_t dst_rank = GetGroupDstRank(current_rank, world_size, group, group_rank_index);
            // Each group records the true boundaries of its credit, issue, and sync sections.

            // Later peers consume only the current-generation credit the destination published for this source rank.
            if (group > 0U) {
                WaitPeerCredit(
                    group, dst_rank, current_rank, current_rank_index, magic, group_count, sync_buffer);
            }

            // This AIV finishes all chunks of the current peer before moving to its next peer.
            for (int64_t chunk_start = 0; chunk_start < route_count; chunk_start += slotChunkCount) {
                const int64_t chunk_index = chunk_start / static_cast<int64_t>(slotChunkCount);
                const uint32_t qp_slot = GetChunkQpSlot(chunk_index, active_qp_count);
                const uint32_t qp_idx = kActiveUdmaQpIndices[qp_slot];

                const int64_t remaining_count = route_count - chunk_start;
                const uint32_t chunk_count = static_cast<uint32_t>(
                    remaining_count > static_cast<int64_t>(slotChunkCount) ? slotChunkCount : remaining_count);
                const uint32_t padded_count =
                    ((chunk_count + kVectorSlotSelectCountAlign - 1) /
                    kVectorSlotSelectCountAlign) * kVectorSlotSelectCountAlign;

                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
                AscendC::DataCopyExtParams dst_copy_parameters {
                    1, static_cast<uint32_t>(chunk_count * sizeof(int32_t)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<int32_t> dst_pad_parameters {false, 0, 0, 0};
                AscendC::DataCopyPad(
                    dst_local, dst_global[chunk_start], dst_copy_parameters, dst_pad_parameters);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

                // A negative dst cannot shift-match a non-negative rank, so the list contains primary routes only.
                AscendC::ShiftRight(
                    dst_rank_tensor, dst_local, static_cast<int32_t>(slot_shift), padded_count);
                AscendC::CreateVecIndex(
                    route_indices, static_cast<int32_t>(chunk_start), padded_count);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);

                AscendC::Compares(compare_mask, dst_rank_tensor, dst_rank, AscendC::CMPMODE::EQ, padded_count);
                AscendC::PipeBarrier<PIPE_V>();
                uint64_t selected_route_count = 0;
                AscendC::GatherMask(selected_route_indices, route_indices,
                    compare_mask.ReinterpretCast<uint32_t>(), true, chunk_count,
                    {1U, 1U, 0U, 0U}, selected_route_count);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);

                // One fixed sender handles the full select list per peer, preserving the global QP traffic ratio and full WQE batches.
                const uint64_t local_selected_count = selected_route_count;
                // This core's byte count accumulates the primary routes it actually owns, covering both local copies and remote sends.
                transferred_bytes += local_selected_count * static_cast<uint64_t>(hidden_row_bytes);

                // Step 4: one QP for the whole chunk; the select list only generates and batch-submits WQEs.
                const bool profile_wqe_submit = dst_rank != current_rank && local_selected_count > 0;
                uint64_t wqe_submit_start_cycle = UdmaDetailedProfileStart<detailed_profile>();
                if constexpr (detailed_profile) {
                    // Count only remote WQE generation/submission; local copies are not on the send path.
                    if (profile_wqe_submit) {
                        wqe_submit_start_cycle = AscendC::GetSystemCycle();
                    }
                }

                // dst_rank is constant across the select list; branch once instead of per route.
                if (dst_rank == current_rank) {
                    // Local copy. Could be folded into the duplicate phase, but kept separate:
                    // single-core UB copies there would drag bandwidth down.
                    for (uint64_t selected = 0; selected < local_selected_count; ++selected) {
                        const int64_t route_index = selected_route_indices.GetValue(selected);
                        const int64_t token_index = route_index / num_topk;
                        // Output position is always decoded from the current route's real dst.
                        const int64_t output_offset =
                            dst_local.GetValue(route_index - chunk_start) % route_count;
                        __gm__ uint8_t *hidden_destination =
                            hidden_window + output_offset * hidden_row_bytes;
                        // The hidden memory type depends on host allocation; the kernel always reads the real token row.
                        __gm__ uint8_t *hidden_source = hidden + token_index * hidden_row_bytes;
                        CopyLocalBytes(reinterpret_cast<GM_ADDR>(hidden_destination),
                            reinterpret_cast<GM_ADDR>(hidden_source), hidden_row_bytes, local_copy_buffer);
                    }
                } else {
                    aclshmemx_submit_state_t udma_submit_state{};
                    aclshmemx_defer_t defer_action(udma_submit_state);
                    aclshmemx_submit_t submit_action(udma_submit_state);

                    // The first N-1 entries of each batch only fill WQEs; the last always submits, removing the per-WQE submit check.
                    for (uint64_t batch_start = 0; batch_start < local_selected_count; batch_start += kUdmaWqeBatchSize) {
                        const uint64_t batch_end = local_selected_count - batch_start > kUdmaWqeBatchSize ?
                            batch_start + kUdmaWqeBatchSize : local_selected_count;

                        // Defer every entry of the current batch except the last.
                        for (uint64_t selected = batch_start; selected + 1 < batch_end; ++selected) {
                            const int64_t route_index = selected_route_indices.GetValue(selected);
                            const int64_t token_index = route_index / num_topk;
                            const int64_t output_offset =
                                dst_local.GetValue(route_index - chunk_start) % route_count;
                            __gm__ uint8_t *hidden_destination =
                                hidden_window + output_offset * hidden_row_bytes;
                            // The hidden memory type depends on host allocation; the kernel always reads the real token row.
                            __gm__ uint8_t *hidden_source = hidden + token_index * hidden_row_bytes;
                            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kUdmaDataRoNoCqeConfig>(hidden_destination, hidden_source,
                                udma_wqe_buffer, static_cast<uint32_t>(hidden_row_bytes), dst_rank, qp_idx,
                                EVENT_ID0, defer_action);
                        }

                        // Once the last entry of the current batch is filled, submit the whole batch immediately.
                        const uint64_t selected = batch_end - 1;
                        const int64_t route_index = selected_route_indices.GetValue(selected);
                        const int64_t token_index = route_index / num_topk;
                        const int64_t output_offset =
                            dst_local.GetValue(route_index - chunk_start) % route_count;
                        __gm__ uint8_t *hidden_destination =
                            hidden_window + output_offset * hidden_row_bytes;
                        // The hidden memory type depends on host allocation; the kernel always reads the real token row.
                        __gm__ uint8_t *hidden_source = hidden + token_index * hidden_row_bytes;
                        aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kUdmaDataRoCqeConfig>(hidden_destination, hidden_source,
                            udma_wqe_buffer, static_cast<uint32_t>(hidden_row_bytes), dst_rank, qp_idx,
                            EVENT_ID0, submit_action);
                    }
                }
                if constexpr (Profile) {
                    // selected_route_count is also this round's remote WQE count.
                    if (profile_wqe_submit) {
                        wqe_submit_packets += local_selected_count;
                    }
                }
                if constexpr (detailed_profile) {
                    if (profile_wqe_submit) {
                        wqe_submit_cycles += AscendC::GetSystemCycle() - wqe_submit_start_cycle;
                    }
                }
            }

            // Step 5: append completion SO on the same dst/QP; after waiting the incoming
            // completion and forwarding the next-round credit, reclaim CQEs together.
            uint64_t sync_start_cycle = UdmaDetailedProfileStart<detailed_profile>();
            CompleteTokenScatterPeerWithSignal(group, dst_rank,
                current_rank, world_size, group_rank_index, current_rank_index,
                magic, active_qp_count, enable_strict_group_sync, sync_buffer, group_sync_buffer,
                udma_wqe_buffer, local_copy_buffer);
            UdmaDetailedProfileAccumulate<detailed_profile>(sync_start_cycle, sync_cycles);
        }
        return transferred_bytes;
    }

    // Non-power-of-two route_count cannot extract rank by shift; the scalar fallback keeps the same peer flow control.
    // The scalar path likewise has each active AIV walk its target peers independently.
    for (uint32_t group = 0; group < group_count; ++group) {
        const int32_t dst_rank = GetGroupDstRank(current_rank, world_size, group, group_rank_index);
        // Later peers must first consume the dedicated credit the destination forwarded after finishing the previous incoming round.
        if (group > 0U) {
            WaitPeerCredit(
                group, dst_rank, current_rank, current_rank_index, magic, group_count, sync_buffer);
        }

        for (int64_t route_index = 0; route_index < route_count; ++route_index) {
            const int64_t encoded_destination = encoded_destinations[route_index];
            if (encoded_destination < 0) {
                continue;
            }
            const int64_t flat_destination = encoded_destination;
            const int32_t route_dst_rank = static_cast<int32_t>(flat_destination / route_count);
            if (route_dst_rank != dst_rank) {
                continue;
            }

            const int64_t destination_local_offset = flat_destination % route_count;
            const int64_t token_index = route_index / num_topk;
            const uint32_t qp_slot = GetChunkQpSlot(route_index, active_qp_count);
            const uint32_t qp_idx = kActiveUdmaQpIndices[qp_slot];
            // Scalar path: once the target rank owned by this core is hit, accumulate the bytes this core actually handles.
            transferred_bytes += static_cast<uint64_t>(hidden_row_bytes);

            __gm__ uint8_t *hidden_destination =
                hidden_window + destination_local_offset * hidden_row_bytes;
            __gm__ uint8_t *hidden_source = hidden + token_index * hidden_row_bytes;
            if (route_dst_rank == current_rank) {
                CopyLocalBytes(reinterpret_cast<GM_ADDR>(hidden_destination),
                    reinterpret_cast<GM_ADDR>(hidden_source), hidden_row_bytes, local_copy_buffer);
            } else {
                uint64_t wqe_submit_start_cycle = UdmaDetailedProfileStart<detailed_profile>();
                aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kUdmaDataRoCqeConfig>(hidden_destination, hidden_source,
                    udma_wqe_buffer, static_cast<uint32_t>(hidden_row_bytes), route_dst_rank, qp_idx, EVENT_ID0);
                if constexpr (Profile) {
                    ++wqe_submit_packets;
                }
                if constexpr (detailed_profile) {
                    wqe_submit_cycles += AscendC::GetSystemCycle() - wqe_submit_start_cycle;
                }
            }
        }

        // The scalar path keeps the same two-phase sync: completion SO plus dedicated credit.
        uint64_t sync_start_cycle = UdmaDetailedProfileStart<detailed_profile>();
        CompleteTokenScatterPeerWithSignal(group, dst_rank,
            current_rank, world_size, group_rank_index, current_rank_index,
            magic, active_qp_count, enable_strict_group_sync, sync_buffer, group_sync_buffer,
            udma_wqe_buffer, local_copy_buffer);
        UdmaDetailedProfileAccumulate<detailed_profile>(sync_start_cycle, sync_cycles);
    }
    return transferred_bytes;
}

// ============================================================
// Weight aggregate send (source-side aggregation + whole-segment MTE + target-side local expansion)
// ============================================================

// Inbox region base = token control area + the local sync area of the 16 Token AIVs,
// matching the region order of the host-side dispatch_sync_buffer_nbytes (the first two offsets must not change).
__aicore__ inline __gm__ uint8_t *GetWeightAggRegion(__gm__ uint8_t *sync_buffer,
    int32_t world_size, int32_t source_rank)
{
    const int64_t base = static_cast<int64_t>(world_size) * kRankSyncSlotBytes +
        static_cast<int64_t>(kMaxDispatchGroupWidth) * kGroupSyncRoundCount * kGroupSyncFlagBytes;
    return sync_buffer + base + static_cast<int64_t>(source_rank) * kWeightAggRegionBytes;
}

// Each Weight AIV uses its own local GM staging so parallel scans of different dsts never overwrite each other.
__aicore__ inline __gm__ uint8_t *GetWeightAggStaging(__gm__ uint8_t *sync_buffer,
    int32_t world_size, uint32_t weight_aiv_index)
{
    return GetWeightAggRegion(sync_buffer, world_size, world_size) +
        static_cast<uint64_t>(weight_aiv_index) * kWeightAggStagingBytes;
}

// Diagnostics region base = after inbox and local staging, matching the tail of the host-side
// dispatch_sync_buffer_nbytes.
__aicore__ inline __gm__ uint8_t *GetWeightAggDebug(__gm__ uint8_t *sync_buffer, int32_t world_size)
{
    return GetWeightAggStaging(sync_buffer, world_size, kWeightAggStagingAivCount);
}

// Diagnostic record for host inspection: code / observed value / expected value / offending source.
__aicore__ inline void PublishWeightAggDebug(GM_ADDR sync_buffer_gm, uint32_t debug_aiv_index,
    int32_t world_size, AscendC::TBuf<AscendC::QuePosition::VECCALC> &mte_buffer,
    uint32_t code, int32_t source_rank, uint32_t observed, uint32_t expected)
{
    __gm__ uint8_t *debug_base = GetWeightAggDebug(
        reinterpret_cast<__gm__ uint8_t *>(sync_buffer_gm), world_size);
    AscendC::LocalTensor<uint32_t> record =
        mte_buffer.GetWithOffset<uint32_t>(kWeightAggDebugWords, 0);
    record.SetValue(0, code);
    record.SetValue(1, debug_aiv_index);
    record.SetValue(2, observed);
    record.SetValue(3, expected);
    record.SetValue(4, static_cast<uint32_t>(source_rank));
    record.SetValue(5, 0);
    record.SetValue(6, 0);
    record.SetValue(7, 0);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID3);
    // Send with MTE3: scalar GM writes are not guaranteed to reach memory and the host could read stale values.
    IssueUbBytesToGm(debug_base + static_cast<uint64_t>(debug_aiv_index) * kWeightAggDebugRecordBytes,
        record.ReinterpretCast<uint8_t>(), kWeightAggDebugRecordBytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
}

// The in-segment length is rounded up to 32B so the slot start after the payload stays 32B-aligned.
__aicore__ inline uint32_t AlignUp32(uint32_t bytes)
{
    return (bytes + 31U) / 32U * 32U;
}

// Fallback for non-power-of-two route_count: decode dst from UB, keeping integer division and bitwise-not-on-negative
// semantics.
__simt_vf__ __launch_bounds__(kWeightSimtThreadCount) inline void
WeightAggregateDecodeSimtVf(__ubuf__ int32_t *rank_tile, __ubuf__ int32_t *slot_tile,
    uint32_t chunk_routes, uint32_t route_count)
{
    const uint32_t thread_index = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count = static_cast<uint32_t>(blockDim.x);
    // Only non-power-of-two sizes decode per thread; weight was already moved by MTE2 and is not re-staged through
    // SIMT.
    #pragma unroll ASCEND_MOONEP_WEIGHT_SIMT_UNROLL
    for (uint32_t index = thread_index; index < chunk_routes; index += thread_count) {
        const uint32_t encoded_bits = static_cast<uint32_t>(rank_tile[index]);
        const uint32_t sign_mask = 0U - (encoded_bits >> 31U);
        const uint32_t flat_destination = encoded_bits ^ sign_mask;
        const uint32_t destination_rank = flat_destination / route_count;
        const uint32_t destination_local_offset = flat_destination - destination_rank * route_count;
        rank_tile[index] = static_cast<int32_t>(destination_rank);
        slot_tile[index] = static_cast<int32_t>(destination_local_offset);
    }
    asc_threadfence();
}

// Target-side expansion VF: scatters weight into this rank weights_buf following the slot list.
// Local 4B writes need no remote address translation, QP, or completion. Invalid slots are skipped;
// header checks validate entry counts, while output correctness checks detect missing or misplaced values.
__simt_vf__ __launch_bounds__(kWeightSimtThreadCount) inline void
WeightAggExpandSimtVf(__ubuf__ float *weight_tile, __ubuf__ int32_t *slot_tile,
    __gm__ float *weights_buf, uint32_t tile_entries, uint32_t route_count)
{
    const uint32_t thread_index = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count = static_cast<uint32_t>(blockDim.x);
    // The loop unroll factor is controlled by a compile parameter; preserve UB input, slot holes, and tail handling.
    #pragma unroll ASCEND_MOONEP_WEIGHT_SIMT_UNROLL
    for (uint32_t index = thread_index; index < tile_entries; index += thread_count) {
        const int32_t slot = slot_tile[index];
        // Skip invalid slots to avoid out-of-bounds writes; output validation must still check value coverage.
        if (slot < 0 || static_cast<uint32_t>(slot) >= route_count) {
            continue;
        }
        weights_buf[slot] = weight_tile[index];
    }
    asc_threadfence();
}

// Parsed result of one source region header: segment count, total entries, and usability.
struct WeightAggRegionInfo {
    uint32_t segment_count;
    uint32_t entry_count;
    bool valid;
};

// Reads the true entry count from the segment header (segment_base points at the segment header); segment and region
// headers are independent transfers, so callers retry as needed.
__aicore__ inline uint32_t ReadWeightAggSegmentCount(__gm__ uint8_t *segment_base)
{
    dcci_cacheline(segment_base);
    AscendC::GlobalTensor<uint32_t> segment_header;
    segment_header.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(segment_base),
        kWeightAggSegmentHeaderBytes / sizeof(uint32_t));
    return segment_header.GetValue(0);
}

// Reads one source region header and validates its shape: read segment count and totals only after magic is ready.
__aicore__ inline WeightAggRegionInfo ReadWeightAggHeader(__gm__ uint8_t *source_region, uint64_t magic)
{
    WeightAggRegionInfo info {0, 0, false};
    // magic is published by the sender after all data MTEs complete and quiet; it is the sole readiness criterion.
    aclshmem_uint64_wait_until(
        reinterpret_cast<__gm__ uint64_t *>(source_region), ACLSHMEM_CMP_EQ, magic);
    dcci_cacheline(source_region);
    AscendC::GlobalTensor<uint32_t> header_words;
    header_words.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t *>(source_region), kWeightAggRegionHeaderWords);
    info.segment_count = header_words.GetValue(kWeightAggSegCountWord);
    info.entry_count = header_words.GetValue(kWeightAggEntryCountWord);
    const uint32_t status = header_words.GetValue(kWeightAggStatusWord);
    // Out of capacity or out-of-range segment/entry counts only set an error: no out-of-bounds reads and no dirty data
    // written into weights_buf.
    info.valid = status == kWeightAggStatusOk &&
        info.segment_count <= kWeightAggMaxSegments &&
        info.entry_count <= kWeightAggCapacity;
    return info;
}

// Segment data of one source record: MTE2 moves it into UB in chunks, then SIMT scatters it locally by slot.
__aicore__ inline uint64_t ExpandWeightAggRegion(__gm__ uint8_t *source_region, uint32_t segment_count,
    uint32_t entry_count, __gm__ float *weights_buf, uint32_t route_count,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &mte_buffer, bool &failed)
{
    __gm__ uint8_t *segment_base = source_region + kWeightAggRegionHeaderBytes;
    uint32_t walked_entries = 0;
    uint64_t expanded_bytes = 0;
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        const uint32_t segment_entries = ReadWeightAggSegmentCount(segment_base);
        const uint32_t payload_bytes = AlignUp32(segment_entries * sizeof(float));
        // GM staging is packed contiguously; the segment count must equal the region-header truth and stay within the
        // per-segment capacity.
        if (segment_entries > kWeightAggCapacity || segment_entries != entry_count) {
            failed = true;
            return expanded_bytes;
        }
        __gm__ uint8_t *payload_gm = segment_base + kWeightAggSegmentHeaderBytes;
        __gm__ uint8_t *slot_gm = payload_gm + payload_bytes;
        AscendC::LocalTensor<uint8_t> payload_stage = mte_buffer.GetWithOffset<uint8_t>(kWeightAggExpandBytes, 0);
        AscendC::LocalTensor<uint8_t> slot_stage =
            mte_buffer.GetWithOffset<uint8_t>(kWeightAggExpandBytes, kWeightAggExpandBytes);
        for (uint32_t tile_begin = 0; tile_begin < segment_entries; tile_begin += kWeightAggExpandRoutes) {
            uint32_t tile_entries = segment_entries - tile_begin;
            // When the tail chunk is shorter than one tile, move only the remaining entries.
            if (tile_entries > kWeightAggExpandRoutes) {
                tile_entries = kWeightAggExpandRoutes;
            }
            IssueGmBytesToUb(payload_stage,
                reinterpret_cast<GM_ADDR>(payload_gm + static_cast<uint64_t>(tile_begin) * sizeof(float)),
                tile_entries * sizeof(float));
            IssueGmBytesToUb(slot_stage,
                reinterpret_cast<GM_ADDR>(slot_gm + static_cast<uint64_t>(tile_begin) * sizeof(int32_t)),
                tile_entries * sizeof(int32_t));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<WeightAggExpandSimtVf>(
                dim3(kWeightSimtThreadCount),
                reinterpret_cast<__ubuf__ float *>(payload_stage.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t *>(slot_stage.GetPhyAddr()),
                weights_buf, tile_entries, route_count);
            // The next chunk MTE2 may overwrite the scratch only after SIMT has finished reading this chunk.
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            expanded_bytes += static_cast<uint64_t>(tile_entries) * sizeof(float);
        }
        walked_entries += segment_entries;
        segment_base += kWeightAggSegmentHeaderBytes + 2U * payload_bytes;
    }
    // The contiguous single-segment layout must exactly cover the true entry count declared in the region header.
    if (walked_entries != entry_count) {
        failed = true;
    }
    return expanded_bytes;
}

// On aggregation validation failure, write an NaN sentinel into the first element of weights_buf: there is no trap
// primitive,
// so a visible data error replaces missing weights that look fine. Must be called after the exit barrier,
// when every AIV has finished its local scatter and the read-modify-write cannot race another core for the same 32B.
__aicore__ inline void MarkWeightAggregateFailure(GM_ADDR weights_buf_gm,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &mte_buffer)
{
    AscendC::GlobalTensor<float> head_tensor;
    head_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weights_buf_gm), 8);
    AscendC::LocalTensor<float> head_tensor_ub = mte_buffer.Get<float>();
    // Read-modify-write via MTE3: a direct __gm__ scalar write is not guaranteed to reach memory and the host could
    // read stale values.
    AscendC::DataCopy(head_tensor_ub, head_tensor, 8);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    head_tensor_ub.ReinterpretCast<uint32_t>().SetValue(0, kWeightAggErrorSentinelBits);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopy(head_tensor, head_tensor_ub, 8);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

// Weight aggregate-send main flow: the source side scans the full route range peer by peer, following the core-to-peer
// assignment of tokens,
// aggregates that peer (weight, slot) pairs into one whole-segment send; the same batch of cores then waits for their
// sources and expands locally.
// Returns the weight byte count this core handled, for Profile statistics.
template <bool Profile>
__aicore__ inline uint64_t DispatchScatterWeightsAggregate(
    GM_ADDR weights_gm, GM_ADDR dst_gm, GM_ADDR weights_buf_gm, GM_ADDR sync_buffer_gm,
    int64_t route_count, int32_t world_size, int32_t weight_aiv_index, int32_t weight_aiv_count,
    uint64_t magic,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &mte_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &slot_select_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &local_copy_buffer,
    bool &aggregate_failed)
{
    // Invalid AIV indices and empty inputs skip the aggregation flow.
    if (route_count <= 0 || weight_aiv_count <= 0 || weight_aiv_count > kWeightAggStagingAivCount ||
        weight_aiv_index < 0 || weight_aiv_index >= weight_aiv_count) {
        return 0;
    }

    const uint32_t route_count_u32 = static_cast<uint32_t>(route_count);
    const uint32_t route_count_is_power_of_two =
        (route_count_u32 & (route_count_u32 - 1U)) == 0U ? 1U : 0U;
    uint32_t route_shift = 0;
    // Precompute the shift amount for power-of-two route_count so each chunk does not redo it.
    if (route_count_is_power_of_two != 0U) {
        for (uint32_t value = route_count_u32; value > 1U; value >>= 1U) {
            ++route_shift;
        }
    }

    __gm__ int32_t *encoded_destinations = reinterpret_cast<__gm__ int32_t *>(dst_gm);
    __gm__ float *weights = reinterpret_cast<__gm__ float *>(weights_gm);
    __gm__ float *weights_buf = reinterpret_cast<__gm__ float *>(weights_buf_gm);
    __gm__ uint8_t *sync_buffer = reinterpret_cast<__gm__ uint8_t *>(sync_buffer_gm);
    const int32_t current_rank = aclshmem_my_pe();
    const uint32_t group_rank_index = static_cast<uint32_t>(weight_aiv_index);
    const uint32_t group_count =
        static_cast<uint32_t>(world_size) / static_cast<uint32_t>(weight_aiv_count);
    // Peers per core is decided by the Group round count, at most 32 at 512P; exceeding the reserved array skips
    // aggregation.
    if (group_count == 0U || group_count > kWeightAggMaxPeersPerCore) {
        return 0;
    }
    // This rank inbox is expressed as a local symmetric address; each Weight AIV has its own staging, so parallel runs
    // cannot trample each other.
    __gm__ uint8_t *local_region = GetWeightAggRegion(sync_buffer, world_size, current_rank);
    __gm__ uint8_t *local_staging = GetWeightAggStaging(
        sync_buffer, world_size, static_cast<uint32_t>(weight_aiv_index));
    __gm__ uint8_t *staging_payload = local_staging;
    __gm__ uint8_t *staging_slot = local_staging + kWeightAggCapacity * sizeof(float);

    AscendC::LocalTensor<uint8_t> rank_chunk_stage =
        mte_buffer.GetWithOffset<uint8_t>(kWeightAggChunkBytes, 0);
    AscendC::LocalTensor<uint8_t> compare_mask =
        slot_select_buffer.GetWithOffset<uint8_t>(kWeightGatherMaskBytes, 0);
    AscendC::LocalTensor<int32_t> slot_batch =
        slot_select_buffer.GetWithOffset<int32_t>(kWeightGatherMaxRoutes, kWeightAggSlotTileOffset);
    // The compact output is reused after landing in local GM; slot_batch becomes the whole-batch weight input after the
    // compares.
    AscendC::LocalTensor<int32_t> compact_batch =
        slot_select_buffer.GetWithOffset<int32_t>(kWeightGatherMaxRoutes, kWeightAggCompactTileOffset);
    AscendC::LocalTensor<int32_t> rank_tile = rank_chunk_stage.ReinterpretCast<int32_t>();
    __ubuf__ uint8_t *payload_copy_ub = reinterpret_cast<__ubuf__ uint8_t *>(
        local_copy_buffer.GetWithOffset<uint8_t>(kWeightAggGm2GmScratchBytes, 0).GetPhyAddr());
    __ubuf__ uint8_t *slot_copy_ub = reinterpret_cast<__ubuf__ uint8_t *>(
        local_copy_buffer.GetWithOffset<uint8_t>(
            kWeightAggGm2GmScratchBytes, kWeightAggGm2GmScratchBytes).GetPhyAddr());
    // Diagnostics keep only the first error so later similar errors do not overwrite the true root cause.
    uint32_t agg_code = kWeightAggCodeOk;
    int32_t agg_source_rank = -1;
    uint32_t agg_observed = 0;
    uint32_t agg_expected = 0;


    // ---------------- Source side: aggregate per peer into local GM staging, then whole-segment GM2GM send
    // ----------------
    uint32_t peer_entry_counts[kWeightAggMaxPeersPerCore];
    uint32_t peer_status[kWeightAggMaxPeersPerCore];
    // Each peer segment-header info is held in scalars first; the region flag is published once after all data is sent.
    for (uint32_t peer_index = 0; peer_index < kWeightAggMaxPeersPerCore; ++peer_index) {
        peer_entry_counts[peer_index] = 0;
        peer_status[peer_index] = kWeightAggStatusOk;
    }

    for (uint32_t group = 0; group < group_count; ++group) {
        const int32_t dst_rank = GetGroupDstRank(current_rank, world_size, group, group_rank_index);
        uint32_t block_count = 0;
        uint32_t block_observed = 0;
        uint32_t block_status = kWeightAggStatusOk;
        const int32_t peer_slot_base = static_cast<int32_t>(static_cast<int64_t>(dst_rank) * route_count);
        // Scan this peer in batches: decode 4096 entries per chunk, then run the two-lane GatherMask after four chunks.
        for (uint32_t batch_begin = 0; batch_begin < route_count_u32; batch_begin += kWeightGatherBatchRoutes) {
            const uint32_t remaining = route_count_u32 - batch_begin;
            const uint32_t batch_routes = remaining < kWeightGatherBatchRoutes ? remaining : kWeightGatherBatchRoutes;
            // rank scratch is reused per chunk; slot and bitmap write to their own offsets within this batch.
            for (uint32_t part = 0; part < batch_routes; part += kWeightAggChunkRoutes) {
                const uint32_t left = batch_routes - part;
                const uint32_t chunk_routes = left < kWeightAggChunkRoutes ? left : kWeightAggChunkRoutes;
                const uint32_t padded_routes = (chunk_routes + kVectorSlotSelectCountAlign - 1U) /
                    kVectorSlotSelectCountAlign * kVectorSlotSelectCountAlign;
                AscendC::LocalTensor<int32_t> slot_tile = slot_batch[part];
                IssueGmBytesToUb(rank_chunk_stage,
                    reinterpret_cast<GM_ADDR>(encoded_destinations + batch_begin + part),
                    chunk_routes * sizeof(int32_t));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                // Keep the original SIMD decode and non-power-of-two fallback; only the slot output offset advances per
                // chunk.
                if (route_count_is_power_of_two != 0U) {
                    AscendC::Not(slot_tile, rank_tile, padded_routes);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Max(slot_tile, rank_tile, slot_tile, padded_routes);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::ShiftRight(rank_tile, slot_tile, static_cast<int32_t>(route_shift), padded_routes);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(slot_tile, slot_tile, -peer_slot_base, padded_routes);
                } else {
                    asc_vf_call<WeightAggregateDecodeSimtVf>(dim3(kWeightSimtThreadCount),
                        reinterpret_cast<__ubuf__ int32_t *>(rank_tile.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ int32_t *>(slot_tile.GetPhyAddr()), chunk_routes, route_count_u32);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                // The next MTE2 may overwrite only after the compares read rank; the four chunk masks concatenate in
                // input order.
                AscendC::Compares(compare_mask[part / 8U], rank_tile, dst_rank, AscendC::CMPMODE::EQ, padded_routes);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            }
            // The first lane compacts slot first, then checks local GM staging capacity once the true hit count is
            // known.
            uint64_t selected_count = 0;
            AscendC::GatherMask(compact_batch, slot_batch, compare_mask.ReinterpretCast<uint32_t>(),
                true, batch_routes, {1U, 1U, 0U, 0U}, selected_count);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            // GM staging appends contiguously, with no inter-batch holes from GatherMask UB destination alignment.
            if (selected_count > kWeightAggCapacity - block_count) {
                block_observed = block_count + static_cast<uint32_t>(selected_count);
                block_status = kWeightAggStatusOverflow;
                break;
            }
            if (selected_count != 0U) {
                IssueUbBytesToGm(staging_slot + static_cast<uint64_t>(block_count) * sizeof(int32_t),
                    compact_batch.ReinterpretCast<uint8_t>(),
                    static_cast<uint32_t>(selected_count) * sizeof(int32_t));
                // The compact output can be overwritten only after slot lands in GM; slot_batch then becomes the weight
                // input.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
            // The compare input of slot_batch is fully consumed; the whole batch is reused as the weight input.
            IssueGmBytesToUb(slot_batch.ReinterpretCast<uint8_t>(),
                reinterpret_cast<GM_ADDR>(weights + batch_begin), batch_routes * sizeof(int32_t));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // The second lane compacts weight with the same masks and appends contiguously to local GM staging.
            AscendC::GatherMask(compact_batch, slot_batch, compare_mask.ReinterpretCast<uint32_t>(),
                true, batch_routes, {1U, 1U, 0U, 0U}, selected_count);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            if (selected_count != 0U) {
                IssueUbBytesToGm(staging_payload + static_cast<uint64_t>(block_count) * sizeof(float),
                    compact_batch.ReinterpretCast<uint8_t>(),
                    static_cast<uint32_t>(selected_count) * sizeof(float));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
            block_count += static_cast<uint32_t>(selected_count);
        }
        // On capacity shortfall no partial data is sent: the count is recorded as 0 so the receiver only sees the
        // overflow status.
        if (block_status != kWeightAggStatusOk) {
            block_count = 0;
        }
        peer_entry_counts[group] = block_count;
        peer_status[group] = block_status;
        // Read staging only after every UB-to-GM write of this batch completes; GM2GM must never read data that has not
        // landed.
        aclshmemx_mte_quiet();
        // The segment header carries count: send even when count == 0, or the peer waits forever for its share of this
        // source.
        AscendC::LocalTensor<uint8_t> segment_header =
            mte_buffer.GetWithOffset<uint8_t>(kWeightAggSegmentHeaderBytes, 0);
        segment_header.ReinterpretCast<uint32_t>().SetValue(0, block_count);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        const uint32_t payload_bytes = AlignUp32(block_count * sizeof(float));
        __gm__ uint8_t *segment_payload = local_region + kWeightAggRegionHeaderBytes +
            kWeightAggSegmentHeaderBytes;
        aclshmemx_mte_put_nbi<uint8_t>(local_region + kWeightAggRegionHeaderBytes,
            reinterpret_cast<__ubuf__ uint8_t *>(segment_header.GetPhyAddr()),
            kWeightAggSegmentHeaderBytes, dst_rank, EVENT_ID0);
        // The two GM2GM lanes use separate 16KiB scratch; the same staging is reusable for the next peer only after
        // quiet.
        if (payload_bytes > 0U) {
            aclshmemx_mte_put_nbi<uint8_t>(segment_payload, staging_payload, payload_copy_ub,
                kWeightAggGm2GmScratchBytes, payload_bytes, dst_rank, EVENT_ID1);
            aclshmemx_mte_put_nbi<uint8_t>(segment_payload + payload_bytes, staging_slot, slot_copy_ub,
                kWeightAggGm2GmScratchBytes, payload_bytes, dst_rank, EVENT_ID2);
        }
        // Staging may be overwritten or region magic published only after data, segment headers, and GM2GM scratch all
        // complete.
        aclshmemx_mte_quiet();
        if (block_status != kWeightAggStatusOk && agg_code == kWeightAggCodeOk) {
            agg_code = kWeightAggCodeOverflow;
            agg_source_rank = dst_rank;
            agg_observed = block_observed;
            agg_expected = kWeightAggCapacity;
        }
    }

    // All peer data is done; reuse the GM2GM scratch to hold the region headers and publish magic in one go.
    AscendC::LocalTensor<uint32_t> header_stage = local_copy_buffer.GetWithOffset<uint32_t>(
        kWeightAggMaxPeersPerCore * kWeightAggRegionHeaderWords, 0);
    // Write every peer region header in this core first, then hand them all to MTE3, avoiding a per-peer wait for the
    // UB read.
    for (uint32_t group = 0; group < group_count; ++group) {
        const uint32_t header_base = group * kWeightAggRegionHeaderWords;
        const uint32_t segment_count = peer_status[group] == kWeightAggStatusOk ? 1U : 0U;
        header_stage.SetValue(header_base, static_cast<uint32_t>(magic));
        header_stage.SetValue(header_base + 1, static_cast<uint32_t>(magic >> 32U));
        header_stage.SetValue(header_base + kWeightAggSegCountWord, segment_count);
        header_stage.SetValue(header_base + kWeightAggEntryCountWord, peer_entry_counts[group]);
        header_stage.SetValue(header_base + kWeightAggStatusWord, peer_status[group]);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    for (uint32_t group = 0; group < group_count; ++group) {
        const int32_t dst_rank = GetGroupDstRank(current_rank, world_size, group, group_rank_index);
        const uint32_t header_base = group * kWeightAggRegionHeaderWords;
        // One region-header scratch per peer, avoiding a per-peer wait for the previous UB read.
        AscendC::LocalTensor<uint8_t> peer_header = header_stage[header_base].ReinterpretCast<uint8_t>();
        if (dst_rank != current_rank) {
            aclshmemx_mte_put_nbi<uint8_t>(local_region, reinterpret_cast<__ubuf__ uint8_t *>(
                peer_header.GetPhyAddr()), kWeightAggRegionHeaderBytes, dst_rank, EVENT_ID2);
        } else {
            IssueUbBytesToGm(local_region, peer_header, kWeightAggRegionHeaderBytes);
        }
    }
    // Enter the receive phase only after all region magics land; this also lets later stages reuse local_copy_buffer.
    aclshmemx_mte_quiet();

    // ---------------- Target side: wait for sources under the same peer mapping and expand locally ----------------
    uint64_t transferred_bytes = 0;
    for (uint32_t group = 0; group < group_count; ++group) {
        const int32_t incoming_peer = GetGroupIncomingPeer(
            current_rank, world_size, group, group_rank_index);
        __gm__ uint8_t *source_region = GetWeightAggRegion(sync_buffer, world_size, incoming_peer);
        // The self source is written by this core itself; wait for this core MTE3 to land before reading it.
        if (incoming_peer == current_rank) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        }
        const WeightAggRegionInfo source_info = ReadWeightAggHeader(source_region, magic);
        // On a bad status the region header has no valid segment; skip the read to stay in bounds.
        if (!source_info.valid) {
            aggregate_failed = true;
            if (agg_code == kWeightAggCodeOk) {
                agg_code = kWeightAggCodeBadHeader;
                agg_source_rank = incoming_peer;
                agg_observed = source_info.entry_count;
                agg_expected = source_info.segment_count;
            }
            continue;
        }
        transferred_bytes += ExpandWeightAggRegion(source_region, source_info.segment_count,
            source_info.entry_count, weights_buf, route_count_u32, mte_buffer, aggregate_failed);
    }

    // Coverage check: every target rank should receive exactly route_count weights.
    // Weight core 0 sums the region-header counts of all sources; any missing source or short count errors immediately
    // instead of silently dropping weights.
    if (weight_aiv_index == 0) {
        uint64_t total_entries = 0;
        for (int32_t source_rank = 0; source_rank < world_size; ++source_rank) {
            __gm__ uint8_t *source_region = GetWeightAggRegion(sync_buffer, world_size, source_rank);
            const WeightAggRegionInfo source_info = ReadWeightAggHeader(source_region, magic);
            if (!source_info.valid) {
                aggregate_failed = true;
            }
            total_entries += source_info.entry_count;
        }
        if (total_entries != route_count_u32) {
            aggregate_failed = true;
            if (agg_code == kWeightAggCodeOk) {
                agg_code = kWeightAggCodeTotalMismatch;
                agg_source_rank = current_rank;
                agg_observed = static_cast<uint32_t>(total_entries);
                agg_expected = route_count_u32;
            }
        }
    }
    // Record success or failure at the sync_buffer tail for host inspection.
    PublishWeightAggDebug(sync_buffer_gm, static_cast<uint32_t>(AscendC::GetBlockIdx()),
        world_size, mte_buffer, agg_code, agg_source_rank, agg_observed, agg_expected);
    if (agg_code != kWeightAggCodeOk) {
        aggregate_failed = true;
    }
    // Local scatters must be visible to other cores before the exit barrier.
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    // The normal kernel generates no Profile byte-count statistics code.
    if constexpr (Profile) {
        return transferred_bytes;
    }
    return 0;
}

// Each SIMT thread owns one weight in the conflict window and writes it directly to the destination rank's final output.
__simt_vf__ __launch_bounds__(kWeightSimtThreadCount) inline void
DispatchScatterWeightsSimtVf(__ubuf__ float *weight_tile, __ubuf__ int32_t *dst_tile,
    __gm__ float *weight_window, uint32_t local_route_count, uint32_t route_count,
    uint32_t route_count_is_power_of_two, uint32_t route_shift, uint32_t first_group)
{
    const uint32_t thread_index = static_cast<uint32_t>(threadIdx.x);
    const uint32_t local_group_count =
        (local_route_count + kWeightsPerGmConflictGroup - 1U) / kWeightsPerGmConflictGroup;
    // Threads round-robin over 512B groups, staggering how ranks/AIVs hit destination GM conflict windows.
    #pragma unroll ASCEND_MOONEP_WEIGHT_SIMT_UNROLL
    for (uint32_t group_order = 0; group_order < local_group_count; ++group_order) {
        uint32_t group_index = first_group + group_order;
        // Wrap back to the first group once the index passes the local range.
        if (group_index >= local_group_count) {
            group_index -= local_group_count;
        }
        const uint32_t local_route_index = group_index * kWeightsPerGmConflictGroup + thread_index;
        // When the tail has fewer than 128 routes, idle threads touch neither input nor output.
        if (local_route_index >= local_route_count) {
            continue;
        }

        // This chunk dst and weight were pre-staged into UB; SIMT reads UB only and no longer touches GM per route.
        const uint32_t encoded_bits = static_cast<uint32_t>(dst_tile[local_route_index]);
        const uint32_t sign_mask = 0U - (encoded_bits >> 31U);
        const uint32_t flat_destination = encoded_bits ^ sign_mask;
        uint32_t destination_rank = 0;
        uint32_t destination_local_offset = 0;
        // Power-of-two route_count decodes with shifts; other sizes keep division semantics.
        if (route_count_is_power_of_two != 0U) {
            destination_rank = flat_destination >> route_shift;
            destination_local_offset = flat_destination & (route_count - 1U);
        } else {
            destination_rank = flat_destination / route_count;
            destination_local_offset = flat_destination - destination_rank * route_count;
        }

        // dst is globally unique, so no atomic is needed; self rank also goes through the same SIMT aclshmem_float_p path.
        ::simt::aclshmem_float_p(weight_window + destination_local_offset,
            weight_tile[local_route_index], static_cast<int32_t>(destination_rank));
    }
    // Every thread completes its remote store fence before the VF returns.
    asc_threadfence();
}

// Each AIV launches the VF once; 128 threads cover this core's full route range with group stride.
template <bool Profile>
__aicore__ inline uint64_t DispatchScatterWeightsSimt(GM_ADDR weights_gm, GM_ADDR dst_gm,
    GM_ADDR weights_buf_gm, int64_t route_count, int64_t aiv_index, int64_t aiv_count,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &weight_stage_buffer
)
{
    // Invalid AIV index and empty input skip the split and modulo logic.
    if (route_count <= 0 || aiv_count <= 0 || aiv_index < 0 || aiv_index >= aiv_count) {
        return 0;
    }

    int64_t route_begin = 0;
    int64_t route_end = 0;
    DispatchSplitWeightRange(route_count, aiv_count, aiv_index, route_begin, route_end);
    // When there are fewer route groups than AIVs, some AIVs have no work.
    if (route_begin >= route_end) {
        return 0;
    }

    const uint32_t route_count_u32 = static_cast<uint32_t>(route_count);
    const uint32_t route_count_is_power_of_two =
        (route_count_u32 & (route_count_u32 - 1U)) == 0U ? 1U : 0U;
    uint32_t route_shift = 0;
    // Precompute the shift for power-of-two route_count, avoiding recomputation per SIMT thread.
    if (route_count_is_power_of_two != 0U) {
        for (uint32_t value = route_count_u32; value > 1U; value >>= 1U) {
            ++route_shift;
        }
    }

    __gm__ float *weights = reinterpret_cast<__gm__ float *>(weights_gm);
    __gm__ int32_t *encoded_destinations = reinterpret_cast<__gm__ int32_t *>(dst_gm);
    __gm__ float *weight_window = reinterpret_cast<__gm__ float *>(weights_buf_gm);
    const int32_t current_rank = aclshmem_my_pe();
    const uint32_t route_begin_u32 = static_cast<uint32_t>(route_begin);
    const uint32_t local_route_count = static_cast<uint32_t>(route_end - route_begin);

    // Each chunk first pre-stages the contiguous dst and weight into UB with MTE2; SIMT reads UB only, avoiding
    // per-route GM scalar reads.
    constexpr uint32_t kWeightStageRoutes = 1024;
    constexpr uint32_t kWeightStageBytes = kWeightStageRoutes * sizeof(int32_t);
    static_assert(2 * kWeightStageBytes <= kMteUbBytes, "weight staging tiles must fit the shared MTE buffer");
    AscendC::LocalTensor<uint8_t> dst_tile_ub =
        weight_stage_buffer.GetWithOffset<uint8_t>(kWeightStageBytes, 0);
    AscendC::LocalTensor<uint8_t> weight_tile_ub =
        weight_stage_buffer.GetWithOffset<uint8_t>(kWeightStageBytes, kWeightStageBytes);
    for (uint32_t chunk_begin = 0; chunk_begin < local_route_count; chunk_begin += kWeightStageRoutes) {
        uint32_t chunk_routes = local_route_count - chunk_begin;
        // The tail chunk may hold fewer than 1024 routes.
        if (chunk_routes > kWeightStageRoutes) {
            chunk_routes = kWeightStageRoutes;
        }
        const uint32_t chunk_bytes = chunk_routes * sizeof(int32_t);
        const uint32_t chunk_group_base = chunk_begin / kWeightsPerGmConflictGroup;
        // Submit MTE2 only, then hand over to SIMT with MTE2_V; a SIMT UB read cannot use MTE2_S.
        IssueGmBytesToUb(dst_tile_ub,
            reinterpret_cast<GM_ADDR>(encoded_destinations + route_begin_u32 + chunk_begin),
            chunk_bytes);
        IssueGmBytesToUb(weight_tile_ub,
            reinterpret_cast<GM_ADDR>(weights + route_begin_u32 + chunk_begin), chunk_bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        const uint32_t chunk_group_count =
            (chunk_routes + kWeightsPerGmConflictGroup - 1U) / kWeightsPerGmConflictGroup;
        // After chunking, the staggered start offset follows the in-chunk group count, preserving the original rank/AIV
        // rotation semantics.
        const uint32_t chunk_first_group = (static_cast<uint32_t>(current_rank) +
            static_cast<uint32_t>(aiv_index) + chunk_group_base) % chunk_group_count;
        asc_vf_call<DispatchScatterWeightsSimtVf>(
            dim3(kWeightSimtThreadCount),
            reinterpret_cast<__ubuf__ float *>(weight_tile_ub.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t *>(dst_tile_ub.GetPhyAddr()), weight_window,
            chunk_routes, route_count_u32, route_count_is_power_of_two, route_shift,
            chunk_first_group);
        // The next chunk MTE2 may overwrite the same UB only after SIMT finishes reading this chunk.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }

    // The scalar side waits for VF completion and ensures SIMT GM stores are visible to the later cross-PE barrier.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    // The non-profile kernel emits no byte-counter code.
    if constexpr (Profile) {
        return static_cast<uint64_t>(local_route_count) * sizeof(float);
    }
    return 0;
}

// Phase 3: expand duplicate hidden per plan directly on this rank's symmetric hidden output.
// Primary rows are bulk-loaded into the double-buffered hidden_stage and written back to
// GM once per duplicate, replacing the per-tile CopyLocalBytes round trip.
// With ServerDedup=true, dup_loffs holds full addresses; legacy rank plans still use local offsets.
template <bool ServerDedup>
__aicore__ inline void DispatchEpilogue(GM_ADDR hidden_buf_gm, GM_ADDR dup_groups_gm,
    GM_ADDR dup_loffs_gm, GM_ADDR dup_counts_gm, int64_t route_count, int64_t hidden_dim,
    int64_t group_begin, int64_t group_end, uint32_t &hidden_stage_index,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &mte_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &local_copy_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &slot_select_buffer)
{
    const int64_t hidden_row_bytes = hidden_dim * sizeof(DataType);
    const int32_t current_rank = aclshmem_my_pe();
    const int64_t dup_capacity = ServerDedup ? ascend_moonep::kMoonEPDispatchRanksPerServer * route_count : route_count;
    __gm__ int32_t *duplicate_groups = reinterpret_cast<__gm__ int32_t *>(dup_groups_gm);
    __gm__ int32_t *duplicate_local_offsets = reinterpret_cast<__gm__ int32_t *>(dup_loffs_gm);
    __gm__ int32_t *duplicate_counts = reinterpret_cast<__gm__ int32_t *>(dup_counts_gm);
    int64_t number_of_groups = duplicate_counts[0];
    int64_t number_of_duplicate_offsets = duplicate_counts[1];
    if (number_of_groups <= 0 || number_of_duplicate_offsets <= 0) {
        return;
    }
    if (number_of_groups > route_count) {
        number_of_groups = route_count;
    }
    // A valid server plan has D <= 8*NvS; keep the legacy read bound, since truncation cannot guarantee accuracy for
    // invalid plans.
    if (number_of_duplicate_offsets > dup_capacity) {
        number_of_duplicate_offsets = dup_capacity;
    }
    // The caller splits ranges by source and worker core; the transfer upper bound is still limited by the actual group
    // count.
    if (group_end > number_of_groups) {
        group_end = number_of_groups;
    }
    if (group_begin >= group_end) {
        return;
    }

    // Rows are loaded into UB with a 32B stride; large hidden rows fall back to single-row chunking, and the legacy
    // rank path still goes row by row.
    const int64_t hidden_row_stride = (hidden_row_bytes + 31) / 32 * 32;
    uint32_t primary_rows_per_tile = 1;
    // Only the server path enables multi-row aggregation; the row count is bounded by both capacity and a fixed cap.
    if constexpr (ServerDedup) {
        primary_rows_per_tile = static_cast<uint32_t>(kDuplicateHiddenStageBytes / hidden_row_stride);
        // With room for less than a full row, keep single-row chunking; with enough room, aggregate at most 4 rows.
        if (primary_rows_per_tile == 0) {
            primary_rows_per_tile = 1;
        } else if (primary_rows_per_tile > kDuplicateMaxPrimaryRows) {
            primary_rows_per_tile = kDuplicateMaxPrimaryRows;
        }
    }
    AscendC::LocalTensor<int32_t> group_stage = mte_buffer.Get<int32_t>();
    AscendC::LocalTensor<int32_t> loff_stage = local_copy_buffer.Get<int32_t>();
    // The server path holds double buffers in the outer scope; the legacy rank path still initializes and drains them
    // within this epilogue.
    if constexpr (!ServerDedup) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        hidden_stage_index = 0;
    }

    // Step 1: load group metadata in chunks and decide from the actual span whether to cache the secondary address
    // table.
    for (int64_t tile_begin = group_begin; tile_begin < group_end;) {
        int64_t tile_count = group_end - tile_begin;
        if (tile_count > static_cast<int64_t>(kDuplicateGroupsPerTile)) {
            tile_count = kDuplicateGroupsPerTile;
        }
        CopyGmInt32ToUb(group_stage,
            duplicate_groups + tile_begin * kDuplicateGroupFields,
            static_cast<uint32_t>(tile_count * kDuplicateGroupFields));

        int64_t loff_span_begin = number_of_duplicate_offsets;
        int64_t loff_span_end = 0;
        // Compute this tile address-table span first; larger lists stay in GM and the UB capacity is unchanged.
        for (int64_t local_group_index = 0; local_group_index < tile_count; ++local_group_index) {
            const uint32_t group_offset = static_cast<uint32_t>(local_group_index * kDuplicateGroupFields);
            const int32_t primary_local_offset = group_stage.GetValue(group_offset);
            const int32_t duplicate_list_begin = group_stage.GetValue(group_offset + 1);
            int64_t duplicate_count = group_stage.GetValue(group_offset + 2);
            if (primary_local_offset < 0 || duplicate_list_begin < 0 || duplicate_count <= 0 ||
                primary_local_offset >= route_count || duplicate_list_begin >= number_of_duplicate_offsets) {
                continue;
            }
            const int64_t available_duplicates = number_of_duplicate_offsets - duplicate_list_begin;
            if (duplicate_count > available_duplicates) {
                duplicate_count = available_duplicates;
            }
            if (duplicate_list_begin < loff_span_begin) {
                loff_span_begin = duplicate_list_begin;
            }
            const int64_t duplicate_list_end = duplicate_list_begin + duplicate_count;
            if (duplicate_list_end > loff_span_end) {
                loff_span_end = duplicate_list_end;
            }
        }

        const int64_t loff_span = loff_span_end > loff_span_begin ?
            loff_span_end - loff_span_begin : 0;
        const bool stage_duplicate_offsets = loff_span > 0 &&
            loff_span <= static_cast<int64_t>(kDuplicateLoffStagingBytes / sizeof(int32_t));
        if (stage_duplicate_offsets) {
            CopyGmInt32ToUb(loff_stage, duplicate_local_offsets + loff_span_begin,
                static_cast<uint32_t>(loff_span));
        }

        // Step 2: collect the few valid primaries; addresses may stay scattered, and the plan format and per-row
        // fan-out targets are unchanged.
        constexpr uint32_t kPrimaryRowCapacity = ServerDedup ? kDuplicateMaxPrimaryRows : 1;
        int32_t primary_offsets[kPrimaryRowCapacity];
        int32_t row_dup_begins[kPrimaryRowCapacity];
        int32_t row_dup_counts[kPrimaryRowCapacity];
        // Aggregate in the current source group order and fetch the next batch once a tile fills.
        for (int64_t group = 0; group < tile_count;) {
            uint32_t row_count = 0;
            // No cross-source or metadata tile collection; invalid groups keep the original skip-and-truncate rules.
            while (group < tile_count && row_count < primary_rows_per_tile) {
                const uint32_t group_offset = static_cast<uint32_t>(group * kDuplicateGroupFields);
                ++group;
                const int32_t primary_local_offset = group_stage.GetValue(group_offset);
                const int32_t duplicate_list_begin = group_stage.GetValue(group_offset + 1);
                int64_t duplicate_count = group_stage.GetValue(group_offset + 2);
                // Only groups with valid address and count occupy a UB row.
                if (primary_local_offset < 0 || duplicate_list_begin < 0 || duplicate_count <= 0 ||
                    primary_local_offset >= route_count || duplicate_list_begin >= number_of_duplicate_offsets) {
                    continue;
                }
                const int64_t available_duplicates = number_of_duplicate_offsets - duplicate_list_begin;
                // Keep the legacy bound handling to avoid reading past the end of the address table.
                if (duplicate_count > available_duplicates) {
                    duplicate_count = available_duplicates;
                }
                primary_offsets[row_count] = primary_local_offset;
                row_dup_begins[row_count] = duplicate_list_begin;
                row_dup_counts[row_count] = static_cast<int32_t>(duplicate_count);
                ++row_count;
            }
            // If every candidate group is invalid, issue no hidden reads/writes and consume no double-buffer events.
            if (row_count == 0) {
                continue;
            }

            // Hidden rows larger than one UB chunk are split; row_count is then always 1.
            for (int64_t offset = 0; offset < hidden_row_bytes; offset += kDuplicateHiddenStageBytes) {
                int64_t current_hidden_bytes = hidden_row_bytes - offset;
                // The last hidden chunk submits only the remaining bytes and must not pass the row end.
                if (current_hidden_bytes > static_cast<int64_t>(kDuplicateHiddenStageBytes)) {
                    current_hidden_bytes = kDuplicateHiddenStageBytes;
                }
                const AscendC::TEventID event_id = hidden_stage_index == 0 ? EVENT_ID0 : EVENT_ID1;
                AscendC::LocalTensor<uint8_t> hidden_stage = slot_select_buffer.GetWithOffset<uint8_t>(
                    kDuplicateHiddenStageBytes, hidden_stage_index * kDuplicateHiddenStageBytes);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                // Step 3: submit scattered primary reads row by row; the whole tile shares one MTE2-to-MTE3 handover.
                for (uint32_t row = 0; row < row_count; ++row) {
                    GM_ADDR primary_hidden = hidden_buf_gm +
                        static_cast<int64_t>(primary_offsets[row]) * hidden_row_bytes + offset;
                    IssueGmBytesToUb(hidden_stage[row * hidden_row_stride], primary_hidden,
                        static_cast<uint32_t>(current_hidden_bytes));
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);

                // Step 4: each row still writes to its own secondaries; scattered targets are not merged and the total
                // moved bytes are unchanged.
                for (uint32_t row = 0; row < row_count; ++row) {
                    AscendC::LocalTensor<uint8_t> primary_stage = hidden_stage[row * hidden_row_stride];
                    // The same primary row is reused in UB and MTE3 is submitted per target following the original
                    // address table.
                    for (int64_t dup = 0; dup < row_dup_counts[row]; ++dup) {
                        const int64_t loff_index = row_dup_begins[row] + dup;
                        const int32_t duplicate_address = stage_duplicate_offsets ?
                            loff_stage.GetValue(static_cast<uint32_t>(loff_index - loff_span_begin)) :
                            duplicate_local_offsets[loff_index];
                        int64_t duplicate_local_offset = duplicate_address;
                        GM_ADDR target_hidden = hidden_buf_gm;
                        // Decode the target card from the full address first, then use the SHMEM mapping to get its
                        // MTE-reachable address.
                        if constexpr (ServerDedup) {
                            const int32_t target_rank = static_cast<int32_t>(duplicate_address / route_count);
                            // Intra-board forwarding accepts only valid secondaries on the same server as the primary.
                            if (duplicate_address < 0 || target_rank / ascend_moonep::kMoonEPDispatchRanksPerServer !=
                                current_rank / ascend_moonep::kMoonEPDispatchRanksPerServer) {
                                continue;
                            }
                            duplicate_local_offset = duplicate_address % route_count;
                            target_hidden = reinterpret_cast<GM_ADDR>(aclshmem_ptr(hidden_buf_gm, target_rank));
                        }
                        // The legacy rank plan local offset must likewise fall inside the current output window.
                        if (duplicate_local_offset < 0 || duplicate_local_offset >= route_count) {
                            continue;
                        }
                        GM_ADDR duplicate_hidden_chunk = target_hidden +
                            duplicate_local_offset * hidden_row_bytes + offset;
                        IssueUbBytesToGm(duplicate_hidden_chunk, primary_stage,
                            static_cast<uint32_t>(current_hidden_bytes));
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                hidden_stage_index ^= 1U;
            }
        }

        tile_begin += tile_count;
    }

    // The legacy rank path drains here; the server path lets the previous source MTE3 overlap the next source MTE2.
    if constexpr (!ServerDedup) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// Scans repeatedly by src_rank, skipping sources that are not ready; every source still shares its groups across all
// forwarding cores.
__aicore__ inline void DispatchForwardBySource(GM_ADDR hidden_buf_gm, GM_ADDR dup_groups_gm,
    GM_ADDR dup_loffs_gm, GM_ADDR dup_counts_gm, GM_ADDR sync_buffer_gm,
    int64_t route_count, int64_t hidden_dim, int64_t worker_index, int64_t worker_count,
    uint64_t magic, uint32_t active_qp_count, AscendC::LocalTensor<uint32_t> source_submitted,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &mte_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &local_copy_buffer,
    AscendC::TBuf<AscendC::QuePosition::VECCALC> &slot_select_buffer
)
{
    const int32_t current_rank = aclshmem_my_pe();
    const int32_t world_size = aclshmem_n_pes();
    const uint32_t current_rank_index = GetGroupRankIndex(current_rank, world_size);
    __gm__ int32_t *counts = reinterpret_cast<__gm__ int32_t *>(dup_counts_gm);
    __gm__ uint8_t *sync_buffer = reinterpret_cast<__gm__ uint8_t *>(sync_buffer_gm);
    // Each uint32_t records 32 sources; the idle WQE UB of this core is reused to cover every rank SHMEM supports.
    constexpr uint32_t kSourcesPerWord = sizeof(uint32_t) * 8U;
    static_assert((ACLSHMEM_MAX_PES + kSourcesPerWord - 1U) / kSourcesPerWord <=
        kUdmaWqeScratchBytes / sizeof(uint32_t));
    const uint32_t word_count = (world_size + kSourcesPerWord - 1U) / kSourcesPerWord;
    // Every dispatch round clears the submitted state so the source bitmap from the previous call is never reused.
    for (uint32_t word = 0; word < word_count; ++word) {
        source_submitted.SetValue(word, 0U);
    }
    // Step 1: the whole source scan shares the double buffers; on source switch the buffer index and in-flight MTE3 are
    // preserved.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t hidden_stage_index = 0;
    int32_t remaining_sources = world_size;
    // Step 2: exit only after every source has been submitted; sources that are not yet ready wait for the next scan
    // round.
    while (remaining_sources > 0) {
        // Scan by src_rank and give idle transfer capacity to whichever source is currently ready.
        for (int32_t src = 0; src < world_size; ++src) {
            const uint32_t word = src / kSourcesPerWord;
            const uint32_t mask = 1U << (src % kSourcesPerWord);
            const uint32_t submitted = source_submitted.GetValue(word);
            // Submitted sources skip the SO check and never forward twice.
            if ((submitted & mask) != 0U) {
                continue;
            }
            // P[src] is the group start; the last source uses G as its end.
            const int64_t source_begin = counts[2 + src];
            const int64_t source_end = src + 1 < world_size ? counts[3 + src] : counts[0];
            int64_t group_begin = 0;
            int64_t group_end = 0;
            DispatchSplitRange(source_end - source_begin, worker_count, worker_index, group_begin, group_end);
            // Empty ranges are marked handled directly; self and remote sources uniformly check all active QP
            // completion flags.
            if (group_begin < group_end) {
                // A single check never waits; if any QP is not there yet, continue with other sources.
                if (!IsSourceReadyForForwarding(src, current_rank_index, magic, active_qp_count, sync_buffer)) {
                    continue;
                }
                // Forward this source's group range only after all of its primary writes are ready.
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
                DispatchEpilogue<true>(hidden_buf_gm, dup_groups_gm, dup_loffs_gm, dup_counts_gm,
                    route_count, hidden_dim, source_begin + group_begin, source_begin + group_end,
                    hidden_stage_index, mte_buffer, local_copy_buffer, slot_select_buffer);
            }
            source_submitted.SetValue(word, submitted | mask);
            --remaining_sources;
        }
    }
    // Step 3: drain uniformly after all sources are submitted, ensuring both UB MTE3s complete before the exit sync.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace DispatchMoonep

// ============================================================
// Dispatch kernel body: the bool Profile template parameter controls timing.
// One build produces both extern entries:
//   moonep_dispatch_udma_kernel         (Profile=false)
//   moonep_dispatch_udma_profile_kernel (Profile=true)
// ============================================================
template <bool Profile>
__forceinline__ __aicore__ void MoonEPDispatchUdmaKernelImpl(
    GM_ADDR hidden_gm,
    GM_ADDR weights_gm,
    GM_ADDR dst_gm,
    GM_ADDR dup_groups_gm,
    GM_ADDR dup_loffs_gm,
    GM_ADDR dup_counts_gm,
    GM_ADDR hidden_buf_gm,
    GM_ADDR weights_buf_gm,
    GM_ADDR sync_buffer_gm,
    GM_ADDR profile_cycles_gm,   // int64 [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields], used only by the profile kernel
    ascend_moonep::MoonEPDispatchTilingData tiling,
    uint64_t magic)
{
    if constexpr (g_coreType == AscendC::AIV) {
        if (hidden_gm == nullptr || weights_gm == nullptr || dst_gm == nullptr ||
            dup_groups_gm == nullptr || dup_loffs_gm == nullptr || dup_counts_gm == nullptr ||
            hidden_buf_gm == nullptr || weights_buf_gm == nullptr ||
            sync_buffer_gm == nullptr) {
            return;
        }

        const int64_t num_tokens = static_cast<int64_t>(tiling.S);
        const int64_t hidden_dim = static_cast<int64_t>(tiling.H);
        const int64_t num_topk = static_cast<int64_t>(tiling.K);
        const uint32_t with_weights = tiling.with_weights;
        const bool server_dedup = tiling.server_dedup != 0;

        // route_count = num_tokens * num_topk is used in later divide/modulo; require it positive up front to avoid divide-by-zero.
        if (num_tokens <= 0 || hidden_dim <= 0 || num_topk <= 0) {
            return;
        }

        __gm__ aclshmemi_aiv_udma_info_t* udma_info = aclshmemi_udma_qp_info_fetch();
        const uint32_t qp_num = udma_info->qp_num;
        const uint32_t active_qp_count =
            (qp_num >= 1U && qp_num < DispatchMoonep::kMaxActiveUdmaQpCount) ? qp_num :
            (qp_num == DispatchMoonep::kMaxCreatedUdmaQpCount) ? DispatchMoonep::kMaxActiveUdmaQpCount : 0U;

        const int32_t world_size = aclshmem_n_pes();
        const int64_t aiv_count = AscendC::GetBlockNum();
        const int64_t aiv_index = AscendC::GetBlockIdx();
        // Below 16P all ranks use one group; larger sizes split into 16-rank groups.
        if (!DispatchMoonep::CanUseGroupFlowControl(world_size, aiv_count) ||
            active_qp_count == 0U || active_qp_count > DispatchMoonep::kMaxActiveUdmaQpCount) {
            return;
        }
        const int64_t token_aiv_count = DispatchMoonep::GetDispatchGroupWidth(world_size);
        // At 64P and below use the serial layout: Token AIVs continue into Weight after Token, and Duplicate owns the
        // last 16 cores;
        // above 64P use the parallel layout: Token owns the first 16 cores, and Weight and Duplicate share the
        // remaining AIVs.
        const bool serial_weight_layout =
            world_size <= DispatchMoonep::kSerialWeightWorldSizeLimit;
        const bool token_worker = aiv_index < token_aiv_count;
        // In the serial layout Weight shares the Token AIV set; the index is the AIV number and the shard width equals
        // the Token core count.
        const bool weight_worker = serial_weight_layout ? token_worker : !token_worker;
        const int64_t weight_aiv_index = serial_weight_layout ? aiv_index : aiv_index - token_aiv_count;
        const int64_t weight_aiv_count = serial_weight_layout ? token_aiv_count : aiv_count - token_aiv_count;
        // In the serial layout Duplicate is handled by the fixed 16 AIVs after the Token group; other AIVs take no part
        // in intra-board transfer.
        const bool duplicate_worker = serial_weight_layout ?
            (aiv_index >= token_aiv_count && aiv_index < token_aiv_count +
                static_cast<int64_t>(DispatchMoonep::kSerialWeightDuplicateAivCount)) : !token_worker;
        const int64_t duplicate_aiv_index = aiv_index - token_aiv_count;
        const int64_t duplicate_aiv_count = serial_weight_layout ?
            static_cast<int64_t>(DispatchMoonep::kSerialWeightDuplicateAivCount) : aiv_count - token_aiv_count;

        // With Profile=true, per-phase cycles and byte counts go to the GM profile buffer;
        // with Profile=false UdmaProfileStart<Profile>() compiles to constant 0, zero cost.
        const uint64_t operator_start_cycle = DispatchMoonep::UdmaProfileStart<Profile>();
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECOUT> udma_wqe_buffer;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> mte_buffer;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> local_copy_buffer;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> slot_select_buffer;
        pipe.InitBuffer(udma_wqe_buffer, DispatchMoonep::kUdmaWqeScratchBytes);
        pipe.InitBuffer(mte_buffer, DispatchMoonep::kMteUbBytes);
        pipe.InitBuffer(local_copy_buffer, DispatchMoonep::kLocalCopyBufferBytes);
        pipe.InitBuffer(slot_select_buffer, DispatchMoonep::kSlotSelectUbBytes);
        __ubuf__ uint8_t *udma_wqe =
            reinterpret_cast<__ubuf__ uint8_t *>(udma_wqe_buffer.Get<uint8_t>().GetPhyAddr());
        const int64_t route_count = num_tokens * num_topk;

        // Wait until all ranks have prepared normal GM inputs and local symmetric windows.
        aclshmemx_barrier_all_vec();

        // Scatter step 1: the Token group sends tokens with aggregate send; in the serial layout these AIVs then take
        // on Weight.
        uint64_t token_wqe_submit_cycles = 0;
        uint64_t token_wqe_submit_packets = 0;
        uint64_t token_sync_cycles = 0;
        const uint64_t token_scatter_start_cycle = DispatchMoonep::UdmaProfileStart<Profile>();
        uint64_t token_scatter_bytes = 0;
        if (token_worker) {
            token_scatter_bytes = DispatchMoonep::DispatchScatterToken<Profile>(
                hidden_gm, dst_gm, hidden_buf_gm, sync_buffer_gm,
                num_tokens, hidden_dim, num_topk, magic, active_qp_count,
                token_wqe_submit_cycles, token_wqe_submit_packets, token_sync_cycles,
                udma_wqe, local_copy_buffer, slot_select_buffer);
        }
        const uint64_t token_scatter_end_cycle = token_worker ?
            DispatchMoonep::UdmaProfileStart<Profile>() : token_scatter_start_cycle;

        // Serial mode joins Token first; the default parallel mode lets the last 16 AIVs enter Weight directly.
        if constexpr (!DispatchMoonep::kEnableParallelTokenWeight) {
            aclshmemx_barrier_all_vec();
        }

        // Scatter step 2: in the parallel layout non-Token AIVs run Weight; in the serial layout the Token AIVs
        // continue into it.
        const uint64_t weight_scatter_start_cycle = DispatchMoonep::UdmaProfileStart<Profile>();
        uint64_t weight_scatter_bytes = 0;
        // Only weight workers set this flag; optional sentinel writes occur after the exit barrier.
        bool weight_aggregate_failed = false;
        if (weight_worker && with_weights != 0) {
            if constexpr (DispatchMoonep::kEnableWeightAggregate) {
                weight_scatter_bytes = DispatchMoonep::DispatchScatterWeightsAggregate<Profile>(
                    weights_gm, dst_gm, weights_buf_gm, sync_buffer_gm,
                    route_count, world_size,
                    static_cast<int32_t>(weight_aiv_index), static_cast<int32_t>(weight_aiv_count),
                    magic, mte_buffer, slot_select_buffer, local_copy_buffer,
                    weight_aggregate_failed);
            } else {
                weight_scatter_bytes = DispatchMoonep::DispatchScatterWeightsSimt<Profile>(
                    weights_gm, dst_gm, weights_buf_gm,
                    route_count, weight_aiv_index, weight_aiv_count,
                    mte_buffer);
            }
        }
        const uint64_t weight_scatter_end_cycle = weight_worker ?
            DispatchMoonep::UdmaProfileStart<Profile>() : weight_scatter_start_cycle;

        // Legacy rank plans keep full stage syncs; server plans judge readability from the per-source SO.
        if (!server_dedup) {
            aclshmemx_barrier_all_vec();
        }

        // Scatter step 3: in the parallel layout non-Token cores forward after finishing Weight; in the serial layout
        // the dedicated cores forward per source directly.
        const uint64_t duplicate_start_cycle = DispatchMoonep::UdmaProfileStart<Profile>();
        if (server_dedup) {
            // Reuse the idle WQE UB of the forwarding cores to hold the source-submitted bitmap; Token cores take no
            // part in the transfer.
            if (duplicate_worker) {
                DispatchMoonep::DispatchForwardBySource(
                    hidden_buf_gm, dup_groups_gm, dup_loffs_gm, dup_counts_gm, sync_buffer_gm,
                    route_count, hidden_dim, duplicate_aiv_index, duplicate_aiv_count, magic, active_qp_count,
                    udma_wqe_buffer.Get<uint32_t>(), mte_buffer, local_copy_buffer, slot_select_buffer);
            }
        } else {
            // Legacy rank plans still share local duplicates across all AIVs, keeping the original single-operator
            // baseline entry.
            int64_t group_begin = 0;
            int64_t group_end = 0;
            const int64_t group_count = reinterpret_cast<__gm__ int32_t *>(dup_counts_gm)[0];
            DispatchMoonep::DispatchSplitRange(group_count, aiv_count, aiv_index, group_begin, group_end);
            uint32_t hidden_stage_index = 0;
            DispatchMoonep::DispatchEpilogue<false>(
                hidden_buf_gm, dup_groups_gm, dup_loffs_gm, dup_counts_gm,
                route_count, hidden_dim, group_begin, group_end, hidden_stage_index,
                mte_buffer, local_copy_buffer, slot_select_buffer);
        }
        const uint64_t duplicate_end_cycle = DispatchMoonep::UdmaProfileStart<Profile>();

        // All ranks exit together after normal GM outputs complete; the host does no phase synchronization.
        aclshmemx_barrier_all_vec();
        const uint64_t operator_end_cycle = DispatchMoonep::UdmaProfileStart<Profile>();

        // Weight aggregation failures are recorded in the diagnostics region by default, without overwriting
        // weights_buf;
        // only with kWeightAggFailHard=true does it fall back to the NaN sentinel behavior.
        // Placed after the exit barrier: all AIV local scatters are done and the read-modify-write cannot overwrite one
        // another.
        if (weight_aggregate_failed && DispatchMoonep::kWeightAggFailHard) {
            DispatchMoonep::MarkWeightAggregateFailure(weights_buf_gm, mte_buffer);
        }

        if constexpr (Profile) {
            // Per-phase cycles and bytes are first written to UB, then pushed to GM via
            // MTE3 DataCopy, with a final PipeBarrier so the host sees them. Direct
            // __gm__ scalar writes bypass the pipe; PipeBarrier/dcci cannot guarantee
            // they land, and the host would read back the -1 sentinel.
            AscendC::LocalTensor<int64_t> profile_ub = local_copy_buffer.Get<int64_t>();
            profile_ub.SetValue(0, static_cast<int64_t>(operator_end_cycle - operator_start_cycle));
            profile_ub.SetValue(1, static_cast<int64_t>(token_scatter_end_cycle - token_scatter_start_cycle));
            profile_ub.SetValue(2, static_cast<int64_t>(token_wqe_submit_cycles));
            profile_ub.SetValue(3, static_cast<int64_t>(token_sync_cycles));
            profile_ub.SetValue(4, static_cast<int64_t>(token_scatter_bytes));
            profile_ub.SetValue(5, static_cast<int64_t>(token_wqe_submit_packets));
            profile_ub.SetValue(
                6, static_cast<int64_t>(weight_scatter_end_cycle - weight_scatter_start_cycle));
            profile_ub.SetValue(7, static_cast<int64_t>(weight_scatter_bytes));
            profile_ub.SetValue(8, static_cast<int64_t>(duplicate_end_cycle - duplicate_start_cycle));

            // One row is 9 int64 = 72 bytes; plain DataCopy's 32B granularity would
            // drop the last field. Reuse IssueUbBytesToGm's byte-granular DataCopyPad,
            // then PipeBarrier to wait for MTE3, so the host reads all 9 fields.
            DispatchMoonep::IssueUbBytesToGm(
                profile_cycles_gm +
                    static_cast<uint64_t>(AscendC::GetBlockIdx()) *
                        ascend_moonep::kMoonEPDispatchUdmaProfileFields * sizeof(int64_t),
                profile_ub.ReinterpretCast<uint8_t>(),
                static_cast<uint32_t>(
                    ascend_moonep::kMoonEPDispatchUdmaProfileFields * sizeof(int64_t)));
            AscendC::PipeBarrier<PIPE_ALL>();
        }

    }
}

// ============================================================
// AscendC kernel entry points — the two template<Profile> instantiations
// ============================================================

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void moonep_dispatch_udma_kernel(
    GM_ADDR hidden_gm,
    GM_ADDR weights_gm,
    GM_ADDR dst_gm,
    GM_ADDR dup_groups_gm,
    GM_ADDR dup_loffs_gm,
    GM_ADDR dup_counts_gm,
    GM_ADDR hidden_buf_gm,
    GM_ADDR weights_buf_gm,
    GM_ADDR sync_buffer_gm,
    ascend_moonep::MoonEPDispatchTilingData tiling,
    uint64_t magic)
{
    MoonEPDispatchUdmaKernelImpl<false>(
        hidden_gm, weights_gm, dst_gm,
        dup_groups_gm, dup_loffs_gm, dup_counts_gm,
        hidden_buf_gm, weights_buf_gm, sync_buffer_gm,
        hidden_buf_gm,   // dummy profile_cycles_gm: never dereferenced when Profile=false
        tiling, magic);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void moonep_dispatch_udma_profile_kernel(
    GM_ADDR hidden_gm,
    GM_ADDR weights_gm,
    GM_ADDR dst_gm,
    GM_ADDR dup_groups_gm,
    GM_ADDR dup_loffs_gm,
    GM_ADDR dup_counts_gm,
    GM_ADDR hidden_buf_gm,
    GM_ADDR weights_buf_gm,
    GM_ADDR sync_buffer_gm,
    GM_ADDR profile_cycles_gm,   // int64 [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields]
    ascend_moonep::MoonEPDispatchTilingData tiling,
    uint64_t magic)
{
    MoonEPDispatchUdmaKernelImpl<true>(
        hidden_gm, weights_gm, dst_gm,
        dup_groups_gm, dup_loffs_gm, dup_counts_gm,
        hidden_buf_gm, weights_buf_gm, sync_buffer_gm,
        profile_cycles_gm,
        tiling, magic);
}

// ============================================================
// Host-side launcher (C-linkage, matches launch.h ABI)
// ============================================================

extern "C" __attribute__((visibility("default")))
void moonep_dispatch_udma_kernel_do(
    void* stream,
    uint8_t* hidden_sh,
    uint8_t* route_weights,
    uint8_t* dst,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* hidden_buf,
    uint8_t* weights_buf,
    uint8_t* sync_buffer,
    uint8_t* tiling_host,
    uint64_t magic)
{
    auto* td = reinterpret_cast<ascend_moonep::MoonEPDispatchTilingData*>(tiling_host);

    // Always launch 32 AIVs: the first group_width run Token, the rest run Weight independently.
    const uint32_t block_dim = ascend_moonep::kMoonEPAivNum;
    moonep_dispatch_udma_kernel<<<block_dim, DispatchMoonep::kMoonEPDispatchUdmaUbSizeBytes, stream>>>(
        hidden_sh,
        route_weights,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        hidden_buf,
        weights_buf,
        sync_buffer,
        *td,
        magic);
}

// Profile kernel launcher — differs from the regular version only by the extra
// profile_cycles output buffer (int64 [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields]),
// from which the host reads per-AIV phase cycles and byte counts.
extern "C" __attribute__((visibility("default")))
void moonep_dispatch_udma_profile_kernel_do(
    void* stream,
    uint8_t* hidden_sh,
    uint8_t* route_weights,
    uint8_t* dst,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* hidden_buf,
    uint8_t* weights_buf,
    uint8_t* sync_buffer,
    uint8_t* profile_cycles,
    uint8_t* tiling_host,
    uint64_t magic)
{
    auto* td = reinterpret_cast<ascend_moonep::MoonEPDispatchTilingData*>(tiling_host);

    const uint32_t block_dim = ascend_moonep::kMoonEPAivNum;
    moonep_dispatch_udma_profile_kernel<<<block_dim, DispatchMoonep::kMoonEPDispatchUdmaUbSizeBytes, stream>>>(
        hidden_sh,
        route_weights,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        hidden_buf,
        weights_buf,
        sync_buffer,
        profile_cycles,
        *td,
        magic);
}
