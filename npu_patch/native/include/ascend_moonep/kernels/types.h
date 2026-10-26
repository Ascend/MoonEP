#pragma once

#include <cstdint>

namespace ascend_moonep {

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define MOONEP_DEVICE_INLINE __forceinline__ __aicore__
#else
#define MOONEP_DEVICE_INLINE static inline constexpr
#endif

// Default host/device launch width for the other operators. PUT UDMA combine
// uses its own 64-AIV schedule in moonep_combine_udma.cpp; do not change this
// shared default when tuning combine.
inline constexpr uint32_t kMoonEPAivNum = 32;
// Server dedup groups 8 globally consecutive ranks into one board.
inline constexpr uint32_t kMoonEPDispatchRanksPerServer = 8;
inline constexpr uint32_t kMoonEPUdmaCombineQpCount = 2;

// Server-dedup planning keeps the public dup_groups tensor at [NvS, 3].  The
// third word packs the source-rank-local route index above the duplicate count.
// K is capped at 16 by combine, so five low bits are sufficient for dup_n and
// leave 26 bits for a route index (well above the advertised 8M-token limit).
inline constexpr uint32_t kMoonEPServerDupCountBits = 5;
inline constexpr uint32_t kMoonEPServerDupCountMask =
    (1U << kMoonEPServerDupCountBits) - 1U;
inline constexpr uint32_t kMoonEPServerDupRouteLimit =
    1U << (31U - kMoonEPServerDupCountBits);

MOONEP_DEVICE_INLINE int32_t PackMoonEPServerDupRoute(
    uint32_t source_route, uint32_t duplicate_count) {
    return static_cast<int32_t>(
        (source_route << kMoonEPServerDupCountBits) |
        (duplicate_count & kMoonEPServerDupCountMask));
}

MOONEP_DEVICE_INLINE uint32_t UnpackMoonEPServerDupCount(int32_t packed) {
    return static_cast<uint32_t>(packed) & kMoonEPServerDupCountMask;
}

MOONEP_DEVICE_INLINE uint32_t UnpackMoonEPServerDupSourceRoute(int32_t packed) {
    return static_cast<uint32_t>(packed) >> kMoonEPServerDupCountBits;
}

// dst uses three states in server mode.  Existing positive values are normal
// canonical routes, ~raw remains a secondary duplicate, and 0x80000000|raw is
// a canonical route whose reduced payload is pushed into source-rank staging.
// The latter two are disjoint while R*NvS < 2^30.
inline constexpr uint32_t kMoonEPServerForwardedRouteTag = 0x80000000U;
inline constexpr uint32_t kMoonEPServerRoutePayloadMask = 0x3fffffffU;

MOONEP_DEVICE_INLINE int32_t EncodeMoonEPServerForwardedRoute(uint32_t raw) {
    return static_cast<int32_t>(kMoonEPServerForwardedRouteTag | raw);
}

MOONEP_DEVICE_INLINE bool IsMoonEPServerForwardedRoute(int32_t encoded) {
    return (static_cast<uint32_t>(encoded) & 0xc0000000U) ==
        kMoonEPServerForwardedRouteTag;
}

MOONEP_DEVICE_INLINE uint32_t DecodeMoonEPServerForwardedRoute(int32_t encoded) {
    return static_cast<uint32_t>(encoded) & kMoonEPServerRoutePayloadMask;
}

// Legacy GET scratch constants retained for compatibility. The current PUT
// workspace ABI is defined by MakeCombinePutLayout in types.h.
inline constexpr uint32_t kMoonEPUdmaCombineScratchTileElements = 8192;
inline constexpr uint32_t kMoonEPUdmaCombineScratchTilesPerAiv = 2;

#undef MOONEP_DEVICE_INLINE

struct alignas(8) MoonEPCombineTilingData {
    uint32_t rank;              // current rank
    uint32_t world_size;        // EP size
    uint32_t num_tokens;        // S
    uint32_t hidden;            // H
    uint32_t num_topk;          // K
    uint32_t block_dim;         // fixed 32 AIV (host always writes 32)
    uint32_t with_weight;       // 1: gather weights → [S,K]; 0: skip
    uint32_t reserved;
};

struct alignas(8) MoonEPUdmaCombineTilingData {
    uint32_t S;
    uint32_t H;
    uint32_t K;
    uint32_t R;
    uint32_t rank;
    uint32_t with_weights;
    uint32_t server_dedup;
    uint32_t reserved; // Preserve the 32-byte host/device tiling structure size.
};

struct alignas(8) MoonEPDispatchTilingData {
    uint32_t S;
    uint32_t H;
    uint32_t K;
    uint32_t R;
    uint32_t rank;
    uint32_t with_weights;
    uint32_t server_dedup;
};

// Udma dispatch profile ABI (mirrors the UBMEM template-<bool Profile> pattern
// from the timing branch): moonep_dispatch_udma_profile_kernel writes each AIV's
// metrics as an int64 row into a [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields]
// GM buffer (profile_cycles_gm); the host reads the buffer back instead of
// kernel printf. Plain (non-profile) kernel passes a dummy pointer that is
// never dereferenced when Profile=false.
inline constexpr uint32_t kMoonEPDispatchUdmaProfileFields = 9;

enum MoonEPDispatchUdmaProfileField : uint32_t {
    kDispatchUdmaProfileTotal = 0,      // full operator wall cycles (incl. barriers)
    kDispatchUdmaProfileTokenScatter,   // token scatter stage cycles
    kDispatchUdmaProfileTokenWqeSubmit, // accumulated WQE fill/submit cycles
    kDispatchUdmaProfileTokenSync,      // token scatter completion sync cycles
    kDispatchUdmaProfileTokenBytes,     // bytes actually handled by this AIV (token)
    kDispatchUdmaProfileWqePackets,     // WQE packets issued by this AIV
    kDispatchUdmaProfileWeightScatter,  // weight scatter stage cycles
    kDispatchUdmaProfileWeightBytes,    // weight bytes scattered by this AIV
    kDispatchUdmaProfileDuplicate,      // duplicate expansion stage cycles
};

struct alignas(8) PlanningTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t num_tokens;
    uint32_t num_topk;
    uint32_t num_experts;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_prefetch_slots;
    uint32_t block_dim;
    uint32_t symmetric_tpe_offset;
    uint32_t symmetric_dup_rank_counts_offset;
    uint32_t source_dup_tokens_per_tile;
    uint32_t owner_reduce_tile_elems;
    uint32_t server_dedup;
    uint32_t combine_mode; // 0: base only; 1: fused tail; 2: precomputed-dst only.
    uint32_t combine_h;
    uint32_t combine_reserved;
    uint64_t combine_workspace;
    uint64_t combine_plan;
};

static_assert(sizeof(PlanningTilingData) == 80,
              "PlanningTilingData fused ABI must remain 80 bytes");

}  // namespace ascend_moonep

// Shared combine ABI: launch/batch sizes, workspace/plan layouts and trace capacity.
// Used by host launchers, planning.cpp and moonep_combine_udma.cpp.

// Shared launch/workspace capacities, also consumed by host and planning.
namespace ascend_moonep {
inline constexpr uint32_t kCombinePutCores = 64;
inline constexpr uint32_t kCombinePutBatchRows = 64;
// Capacity factor only: runtime pages are packed by actual contributor rows.
inline constexpr uint32_t kCombineRingGroupsPerPage = 64;
inline constexpr uint32_t kCombineRingDepth = 4;
inline constexpr uint32_t kCombineRingStreams = 8;
inline constexpr uint32_t kCombineRingMaxShards = 4;
} // namespace ascend_moonep

// ---- Communication workspace layout ----
namespace ascend_moonep {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define MOONEP_PUT_INLINE __forceinline__ __aicore__
#else
#define MOONEP_PUT_INLINE inline constexpr
#endif

// Aligned PUT workspace ABI. All uint64 members except bytes are byte offsets
// relative to symmetric ws, NOT pointers or lengths. Host and Python mirror it.
// N=S*K, H=hidden, R=ranks, B=ceil(N/64), A(x)=ceil(x/512)*512.
// Meta(base,i) uses independent 64-byte records; Flag(base,i) uses 512-byte slots.
// Only the first uint64 of a flag carries magic; reducer MTE3 writes 32 bytes.
//
// Source=original token owner/final destination; producer=expert-output holder;
// ingress=server group reducer. The same rank can have all three roles.
// Workspace offsets are shared by Planning and Combine. Host, Python and
// device kernels MUST be rebuilt and installed together after layout changes.
struct CombinePutLayout {
    uint64_t final_hidden; // BF16[N,H], compact producer segments at each source.
                           // Local-server routes include suppressed members too.
    uint64_t relay_hidden; // Phase pool: Planning staging, then BF16 input ring.
                           // 8 streams * 4 pages * (64*K) rows; groups never cross pages.
    uint64_t relay_result; // BF16[N,H], stable sums indexed by the original TX index.
    uint64_t weights;      // float[N], every original source route, no weight dedup.

    uint64_t reverse;     // Retired reverse-route records; offset reserved for ABI stability.
    uint64_t reverse_send;// Reserved N*64B arena; planner publishes contiguous dst[N]
                           // in its first N*4B. Peers read it, never PUT reverse requests.
    uint64_t final_map;    // Retired scalar final mapping arena, offset reserved.
                           // Planner now builds final descriptors by partition prefixes.
    uint64_t group_of_loff;// Retired group lookup arena; offset reserved.
    uint64_t tx_rows;      // N records in source/ordinal order:
                           // {loff,group,scratchHead(-1 for DIRECT),contributorCount}.
                           // DIRECT count=1; send original hidden without server reduction.
    uint64_t source_counts;// R records:
                           // [0]=active result count, [1]=remote final segment base,
                           // [2]=local tx base, [3]=DIRECT count,
                           // [4]=scratch range start, [5]=scratch slot count,
                            // [6]=ring page count, [7]=first logical ring page.
                           // Scratch slots INCLUDE contributor holes; <=N/source.
    uint64_t counts_send;  // Planner-local R records: {groupCount,groupPrefix} per source.
    uint64_t contributor; // 8*N records: word0 is {count:high32,txIndex:low32},
                           // word1 is original global contributor address.
                           // Only a group head has count>0; interior slots are zero.
                           // 64-byte spacing prevents concurrent planner cacheline
                           // writeback races. Reducer gathers 64 entries into UB.
    uint64_t relay_send;   // 8*N requests at producer: [ingressLocal*N+ordinal]
                           // -> {input loff,remote scratch slot}. Includes ingress
                           // self contributors for planning; payload skips them because
                           // Reduce reads their original input directly.
    uint64_t relay_publish;// Retired relay request PUT arena; offset reserved.
    uint64_t relay_counts, relay_counts_send; // Legacy reserved 8-record tables.

    uint64_t relay_ready; // 8*R slots, producerLocal*R+source, source-level SO ready.
                           // Empty producer streams MUST publish as well.
    uint64_t reduce_ready;// R*B slots, source*B+scratchBatch; MTE3-ordered after sum.
                           // B uses N, NOT N/R: nonuniform input must fit safely.
    uint64_t final_ready; // 2*R slots, producer*2+QP lane. Both remote lanes close
                           // DIRECT+RELAY even when empty. Local Phase A uses lane0.
    uint64_t weight_ready;// R slots, producer -> complete all original route weights.
    uint64_t magic;       // 64 stable per-core SHMEM PUT-source flag slots.
    uint64_t relay_ranges;// 8*R records at producer: ingressLocal*R+source
                            // -> {requestBegin,requestCount,receiverSourcePageCount}.
    uint64_t relay_ranges_send; // Planner-local 8*R stats records:
                           // {scratchCount,requestCount,scratchPrefix,requestPrefix}.
                           // After metadata completion, reused as R*512B final
                           // histograms: 64 partition counts/prefixes + peer total.
    uint64_t source_input_ready;// R slots, shard1 waits all producers, wakes shard0.
    uint64_t final_gate;        // One local slot: core0 publishes after it has
                                // observed every final_ready; all other cores wait here.
    uint64_t final_routes;      // N independent 128-byte source-final descriptors.
                                // Each output token owns {count, route[16]}; route is either
                                // local hidden loff (bit31 set) or final_hidden row. N records
                                // reserve for K=1 too; the active prefix is N/K records.
    uint64_t ring_epoch;        // 8 receiver/peer invocation epochs.
    uint64_t ring_ready;        // [stream,slot,producerLocal], 8*4*8 flags.
    uint64_t ring_credit;       // [stream,slot], 8*4 reusable input-page credits.
    uint64_t ring_done;         // [stream,slot,shard], 8*4*4 reducer completion flags.
    uint64_t ring_ready_source; // [ingressLocal,stream,slot], 8*8*4 SO source flags.
    uint64_t ring_result_ready; // Persistent logical-page ready flags, never ring-indexed.
    uint64_t bytes;       // Total bytes including all alignment/reserved arenas.
    uint32_t batches;     // ceil(N/64), worst per-source scratch batch capacity.
    uint32_t ring_page_rows, ring_page_capacity;
};

MOONEP_PUT_INLINE uint64_t CombinePutAlign(uint64_t bytes) {
    return (bytes + 511U) & ~uint64_t(511U);
}
MOONEP_PUT_INLINE uint32_t CombineRingPageCapacity(uint32_t n, uint32_t ranks) {
    // Each nonempty source can contribute a separate partial final page.
    return static_cast<uint32_t>((uint64_t(n) + kCombineRingGroupsPerPage - 1U) /
                                 kCombineRingGroupsPerPage + ranks);
}
MOONEP_PUT_INLINE uint64_t CombineRingLayoutFingerprint(uint32_t k) {
    // Bit 40 distinguishes row-packed plans from legacy 64-group pages.
    return (uint64_t(1U) << 40U) | (uint64_t(kCombineRingStreams) << 56U) |
           (uint64_t(kCombineRingDepth) << 48U) |
           (uint64_t(kCombineRingGroupsPerPage) << 32U) |
           uint64_t(kCombineRingGroupsPerPage) * k;
}
MOONEP_PUT_INLINE uint64_t CombinePlanningStageBytes(
    uint32_t n, uint32_t h, uint32_t ranks) {
    // Preserve the original rank/server staged eligibility. Buffer allocation
    // precedes the plan's mode, so reserve the largest eligible mode's staging.
    if (ranks > 128U) return 0U;
    const uint64_t server = 32U + 8U * 8U, rank = 32U + 8U;
    if (uint64_t(ranks) * server <= uint64_t(h) * 16U)
        return uint64_t(ranks) * n * server;
    if (uint64_t(ranks) * rank <= uint64_t(h) * 16U)
        return uint64_t(ranks) * n * rank;
    return 0U;
}
MOONEP_PUT_INLINE uint64_t CombineRingPublishBytes(uint32_t n, uint32_t ranks) {
    const uint64_t legacy = uint64_t(n) * 8U * 64U;
    const uint64_t mirror = CombinePutAlign(uint64_t(ranks) * 64U) +
        CombinePutAlign(uint64_t(CombineRingPageCapacity(n, ranks)) * 64U);
    return CombinePutAlign(legacy > mirror ? legacy : mirror);
}
MOONEP_PUT_INLINE CombinePutLayout MakeCombinePutLayout(
    uint32_t n, uint32_t h, uint32_t ranks, uint32_t k = 16U) {
    CombinePutLayout p{};
    const uint64_t row_bytes = uint64_t(n) * h * 2U; // all N rows, not one row
    const uint64_t meta = uint64_t(n) * 64U;
    p.batches = (uint64_t(n) + 63U) / 64U;
    p.ring_page_rows = kCombineRingGroupsPerPage * k;
    p.ring_page_capacity = CombineRingPageCapacity(n, ranks);
    p.final_hidden = 0;
    p.relay_hidden = CombinePutAlign(row_bytes);
    const uint64_t ring_bytes = CombinePutAlign(uint64_t(kCombineRingStreams) *
        kCombineRingDepth * p.ring_page_rows * h * 2U);
    p.relay_result = p.relay_hidden + ring_bytes;
    const uint64_t runtime_bytes = ring_bytes + CombinePutAlign(row_bytes);
    const uint64_t stage_bytes = CombinePutAlign(CombinePlanningStageBytes(n, h, ranks));
    p.weights = p.relay_hidden + (stage_bytes > runtime_bytes ? stage_bytes : runtime_bytes);
    p.reverse = p.weights + CombinePutAlign(uint64_t(n) * 4U);
    p.reverse_send = p.reverse + CombinePutAlign(meta);
    p.final_map = p.reverse_send + CombinePutAlign(meta);
    p.group_of_loff = p.final_map + CombinePutAlign(meta);
    p.tx_rows = p.group_of_loff + CombinePutAlign(meta);
    p.source_counts = p.tx_rows + CombinePutAlign(meta);
    p.counts_send = p.source_counts + CombinePutAlign(uint64_t(ranks) * 64U);
    p.contributor = p.counts_send + CombinePutAlign(uint64_t(ranks) * 64U);
    p.relay_send = p.contributor + CombinePutAlign(8U * meta);
    p.relay_publish = p.relay_send + CombinePutAlign(8U * meta);
    p.relay_counts = p.relay_publish + CombineRingPublishBytes(n, ranks);
    p.relay_counts_send = p.relay_counts + 512U;
    p.relay_ready = p.relay_counts_send + 512U;
    p.reduce_ready = p.relay_ready + uint64_t(8U) * ranks * 512U;
    p.final_ready = p.reduce_ready + uint64_t(ranks) * p.batches * 512U;
    p.weight_ready = p.final_ready + uint64_t(ranks) * 2U * 512U;
    p.magic = p.weight_ready + uint64_t(ranks) * 512U;
    p.relay_ranges = p.magic + uint64_t(kCombinePutCores) * 512U;
    p.relay_ranges_send = p.relay_ranges + CombinePutAlign(uint64_t(8U) * ranks * 64U);
    p.source_input_ready = p.relay_ranges_send + CombinePutAlign(uint64_t(8U) * ranks * 64U);
    p.final_gate = p.source_input_ready + uint64_t(ranks) * 512U;
    p.final_routes = p.final_gate + 512U;
    p.ring_epoch = p.final_routes + CombinePutAlign(uint64_t(n) * 128U);
    p.ring_ready = p.ring_epoch + uint64_t(8U) * 512U;
    p.ring_credit = p.ring_ready + uint64_t(kCombineRingStreams) * kCombineRingDepth * 8U * 512U;
    p.ring_done = p.ring_credit + uint64_t(kCombineRingStreams) * kCombineRingDepth * 512U;
    p.ring_ready_source = p.ring_done + uint64_t(kCombineRingStreams) *
        kCombineRingDepth * kCombineRingMaxShards * 512U;
    p.ring_result_ready = p.ring_ready_source + uint64_t(8U) *
        kCombineRingStreams * kCombineRingDepth * 512U;
    p.bytes = p.ring_result_ready + uint64_t(p.ring_page_capacity) * 512U;
    return p;
}
#undef MOONEP_PUT_INLINE
} // namespace ascend_moonep

// ---- Immutable execution-plan layout ----
namespace ascend_moonep {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define MOONEP_PLAN_INLINE __forceinline__ __aicore__
#else
#define MOONEP_PLAN_INLINE inline constexpr
#endif
// Rank-local immutable snapshot. No payload, ready flags, magic or QP state.
// Offsets are relative to this tensor, not the symmetric communication workspace.
// v6: append immutable whole-group input pages and producer request mappings.
// Existing TX/contributor/relay record strides are unchanged. Ready flags,
// invocation epochs and ring credits remain in the communication workspace.
// Regenerate plans and deploy host/device together; older snapshots are incompatible.
inline constexpr uint64_t kCombineExecutionPlanVersion = 0x4d45504300000006ULL;
inline constexpr uint32_t kCombinePlanTxBytes = 16U;
inline constexpr uint32_t kCombinePlanContributorBytes = 16U;
inline constexpr uint32_t kCombinePlanRelayBytes = 8U;
struct CombineExecutionPlanLayout {
    uint64_t tx_rows, source_counts, contributor, relay_send, relay_ranges;
    uint64_t final_routes, ring_pages, relay_page_map, bytes;
};
MOONEP_PLAN_INLINE CombineExecutionPlanLayout MakeCombineExecutionPlanLayout(
    uint32_t n, uint32_t ranks) {
    CombineExecutionPlanLayout q{};
    q.tx_rows = 512U; // Header: version,N,H,K,R,rank,server_dedup,ring fingerprint.
    q.source_counts = q.tx_rows + CombinePutAlign(uint64_t(n) * kCombinePlanTxBytes);
    q.contributor = q.source_counts + CombinePutAlign(uint64_t(ranks) * 64U);
    q.relay_send = q.contributor + CombinePutAlign(uint64_t(n) * 8U * kCombinePlanContributorBytes);
    q.relay_ranges = q.relay_send + CombinePutAlign(uint64_t(n) * 8U * kCombinePlanRelayBytes);
    q.final_routes = q.relay_ranges + CombinePutAlign(uint64_t(ranks) * 8U * 64U);
    // Weights read the existing dst directly; no dedicated weights table.
    q.ring_pages = q.final_routes + CombinePutAlign(uint64_t(n) * 128U);
    // Each 64B page begins {scratchBegin,scratchCount,groupCount,source}.
    q.relay_page_map = q.ring_pages +
        CombinePutAlign(uint64_t(CombineRingPageCapacity(n, ranks)) * 64U);
    // Parallel to relay_send[8*N], each packed 8B entry is {sourcePageSeq,rowOffset}.
    q.bytes = q.relay_page_map + CombinePutAlign(uint64_t(n) * 8U * 8U);
    return q;
}
#undef MOONEP_PLAN_INLINE
} // namespace ascend_moonep
