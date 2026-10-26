#include "acl/acl.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"
#include "shmem.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"

#include "ascend_moonep/kernels/launch.h"
#include "ascend_moonep/kernels/types.h"

using namespace AscendC;

namespace {

constexpr int32_t kFillTileElems = 1024; // must be 32B aligned
constexpr int32_t kInt32Bytes = 4;
constexpr int32_t kMaxPlanningCores = 56;
constexpr int32_t kMaxPlanningRanks = 1024;
constexpr int32_t kPlanningRanksPerServer = ascend_moonep::kMoonEPDispatchRanksPerServer;
constexpr int32_t kAscend950MixedUbSizeBytes = 216 * 1024;
constexpr int32_t kPlanningUbSizeBytes = 216 * 1024;
constexpr int32_t kCoreCursorSimtMaxThreads = 256;
constexpr int32_t kCoreCursorThreadPrefixBytes =
    kCoreCursorSimtMaxThreads * kInt32Bytes;
constexpr int32_t kCoreCursorCarryBytes = 32;
constexpr int32_t kCursorWorkerScratchRowElems = 32 / kInt32Bytes;
constexpr int32_t kDupSrcTileElems = 1024;
constexpr int32_t kSourceDupSimtMaxThreads = 1024;
constexpr int32_t kSourceDupInboxSlotElems = 8;
constexpr int32_t kPrefetchSimtMaxThreads = 1024;
constexpr int32_t kPrefetchSimtItemsPerThread = 32;
constexpr int32_t kRouteHistogramTileElems = 4096;
constexpr int32_t kTpeReduceExpertsPerCore = 64;
constexpr int32_t kOwnerReduceTotalBytes = 32;
constexpr int32_t kTpeReduceRegRowBytes =
    kTpeReduceExpertsPerCore * kInt32Bytes;
constexpr int32_t kSourceDupPackTinyChunkGroups = 16;
constexpr int32_t kSourceDupPackSmallChunkGroups = 32;
constexpr int32_t kSourceDupPackMediumChunkGroups = 64;
constexpr int32_t kSourceDupPackLargeChunkGroups = 128;
constexpr int32_t kSourceDupPackXLargeChunkGroups = 256;
constexpr int32_t kDstSimtMaxThreads = 1024;
constexpr int32_t kDstSimtMinChunkElems = 32;
constexpr int32_t kDstSimtMaxChunkElems = 128;
constexpr int32_t kDstSimtContextElems = 16;
constexpr int32_t kDstSimtContextBytes = kDstSimtContextElems * kInt32Bytes;
constexpr int32_t kMaxDuplicateInt32Elems =
    kPlanningUbSizeBytes / kInt32Bytes;
static_assert(kPlanningUbSizeBytes <= kAscend950MixedUbSizeBytes);
static_assert((kDupSrcTileElems * kInt32Bytes) % 32 == 0);
static_assert(
    (kPrefetchSimtMaxThreads & (kPrefetchSimtMaxThreads - 1)) == 0);
static_assert(kDstSimtContextBytes % 32 == 0);
static_assert((kRouteHistogramTileElems * kInt32Bytes) % 32 == 0);
static_assert(
    kRouteHistogramTileElems * kInt32Bytes < kPlanningUbSizeBytes);
static_assert(kOwnerReduceTotalBytes % 32 == 0);
static_assert(kTpeReduceRegRowBytes == 256);
static_assert(
    kTpeReduceRegRowBytes + kMaxPlanningRanks * 32 <=
    kPlanningUbSizeBytes);
static_assert(
    (kMaxPlanningCores + 1) * 32 + kCoreCursorThreadPrefixBytes +
            kCoreCursorCarryBytes <=
        kPlanningUbSizeBytes);
static_assert(kCoreCursorThreadPrefixBytes % 32 == 0);
static_assert(kCoreCursorCarryBytes % 32 == 0);
static_assert(kCursorWorkerScratchRowElems * kInt32Bytes == 32);
static_assert(kSourceDupInboxSlotElems * kInt32Bytes == 32);
static_assert(
    kMaxPlanningRanks * kSourceDupInboxSlotElems * kInt32Bytes +
        ((kMaxPlanningRanks + 2 + 7) / 8) * 32 <=
    kPlanningUbSizeBytes);

// Pointer fields are int32 element offsets from the single UB workspace base.
enum DstSimtContextIndex : int32_t {
    kDstSimtTopkOffset = 0,
    kDstSimtHistOffset,
    kDstSimtOrdinalOffset,
    kDstSimtAllocCumsumOffset,
    kDstSimtExpertOffsetsOffset,
    kDstSimtDstOffset,
    kDstSimtChunkSize = 9,
    kDstSimtHistStride,
    kDstSimtRankCount,
    kDstSimtExpertCount,
    kDstSimtMaxTokensPerRank,
};

struct DstUbLayout {
    int32_t context_offset;
    int32_t dst_tile_offset;
    int32_t lookup_offset;
    int32_t gm_simt_topk_offset;
    int32_t lookup_table_elems;
    int32_t lookup_table_bytes;
    int32_t simt_topk_offset;
    int32_t simt_hist_offset;
    int32_t simt_hist_stride;
    int32_t simt_chunk_size;
    bool use_simt;
};

struct PlanningAscend950 : Catlass::Arch::Ascend950 {
    static constexpr uint32_t UB_SIZE = kPlanningUbSizeBytes;
};

using PlanningResource = Catlass::Arch::Resource<PlanningAscend950>;

struct SourceDupConfig {
    int32_t rank;
    int32_t rank_count;
    int32_t token_count;
    int32_t topk;
    int32_t max_tokens_per_rank;
    int32_t core_idx;
    int32_t core_num;
    int32_t tokens_per_tile;
    int32_t server_dedup;
};

struct SourceDupPartition {
    int32_t token_begin;
    int32_t token_end;
    int32_t max_tokens_per_core;
    int32_t max_tiles_per_core;
};

struct SourceDupUbLayout {
    int32_t route_tile_elems;
    int32_t count_bytes;
    int32_t staging_bytes;
    int32_t prefix_offset;
    int32_t meta_offset;
    int32_t raw_group_offset;
    int32_t raw_loff_offset;
    int32_t group_send_offset;
    int32_t tile_count_offset;
    int32_t tile_count_row_elems;
    int32_t token_group_count_offset;
    int32_t token_loff_count_offset;
    int32_t token_group_prefix_offset;
    int32_t token_loff_prefix_offset;
    int32_t compact_loff_offset;
    int32_t pack_hist_offset;
    int32_t pack_hist_row_elems;
    int32_t pack_chunk_groups;
    int32_t total_bytes;
};

struct SourceDupWorkspaceLayout {
    int32_t tile_count_offset;
    int32_t group_stride;
    int32_t loff_stride;
    int32_t group_offset;
    int32_t loff_offset;
    int32_t prebuilt_group_stride;
    int32_t prebuilt_loff_stride;
    int32_t prebuilt_group_offset;
    int32_t prebuilt_loff_offset;
};

struct SourceDupCountTileLayout {
    int32_t max_ranks_per_core;
    int32_t row_elems;
    int32_t core_counts_offset;
    int32_t slots_offset;
    int32_t result_offset;
    int32_t total_bytes;
};

struct PrefetchUbLayout {
    int32_t group_counts_offset;
    int32_t group_counts_bytes;
    int32_t selected_experts_offset;
    int32_t selected_experts_bytes;
    int32_t scratch_counts_offset;
    int32_t scratch_experts_offset;
    int32_t scratch_row_elems;
    int32_t scratch_row_bytes;
    int32_t total_bytes;
};

__simd_vf__ inline void ReduceTpeRowsReg(
    __ubuf__ int32_t* dst,
    __ubuf__ int32_t* src,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t row_stride_elems) {
    AscendC::Reg::RegTensor<int32_t> accumulator;
    AscendC::Reg::RegTensor<int32_t> row_value;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<int32_t>(column_count);

    AscendC::Reg::LoadAlign(accumulator, src);
    for (uint32_t row = 1; row < row_count; ++row) {
        AscendC::Reg::LoadAlign(row_value, src + row * row_stride_elems);
        AscendC::Reg::Add(accumulator, accumulator, row_value, mask);
    }
    AscendC::Reg::StoreAlign(dst, accumulator, mask);
}

__simd_vf__ inline void ReduceTpeOwnerRowsReg(
    __ubuf__ int32_t* dst,
    __ubuf__ int32_t* total_dst,
    __ubuf__ int32_t* src,
    uint32_t row_count,
    uint32_t column_count,
    uint32_t row_stride_elems) {
    AscendC::Reg::RegTensor<int32_t> accumulator;
    AscendC::Reg::RegTensor<int32_t> row_value;
    AscendC::Reg::RegTensor<int32_t> total_value;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<int32_t>(column_count);

    AscendC::Reg::LoadAlign(accumulator, src);
    for (uint32_t row = 1; row < row_count; ++row) {
        AscendC::Reg::LoadAlign(row_value, src + row * row_stride_elems);
        AscendC::Reg::Add(accumulator, accumulator, row_value, mask);
    }
    AscendC::Reg::StoreAlign(dst, accumulator, mask);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(
        total_value, accumulator, mask);
    uint32_t total_count = 1;
    AscendC::Reg::MaskReg total_mask = AscendC::Reg::UpdateMask<int32_t>(total_count);
    AscendC::Reg::StoreAlign(total_dst, total_value, total_mask);
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void BuildRouteHistogramSimt(
    __ubuf__ int32_t* routes,
    __ubuf__ int32_t* expert_counts,
    int32_t route_count) {
    for (int32_t route = static_cast<int32_t>(threadIdx.x);
         route < route_count;
         route += kSourceDupSimtMaxThreads) {
        const int32_t expert = routes[route];
        asc_atomic_add(expert_counts + expert, 1);
    }
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void BuildAllocCumsumSimt(
    __ubuf__ int32_t* alloc,
    __gm__ int32_t* alloc_cumsum,
    int32_t rank_count,
    int32_t experts_per_rank,
    int32_t alloc_row_elems,
    int32_t expert_base,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);

    for (int32_t local_expert = thread;
         local_expert < experts_per_rank;
         local_expert += thread_count) {
        int32_t prefix = 0;
        for (int32_t dest = 0; dest < rank_count; ++dest) {
            const int32_t offset = dest * alloc_row_elems + local_expert;
            prefix += alloc[offset];
            alloc[offset] = prefix;
        }
    }
    asc_syncthreads();

    const int32_t total_elems = rank_count * experts_per_rank;
    for (int32_t index = thread; index < total_elems; index += thread_count) {
        const int32_t local_expert = index / rank_count;
        const int32_t dest = index - local_expert * rank_count;
        alloc_cumsum[(expert_base + local_expert) * rank_count + dest] =
            alloc[dest * alloc_row_elems + local_expert];
    }
}

__simt_vf__ __launch_bounds__(kPrefetchSimtMaxThreads) inline void SelectPrefetchExpertsSimt(
    __ubuf__ int32_t* group_counts,
    __ubuf__ int32_t* selected_experts,
    __ubuf__ int32_t* candidate_counts,
    __ubuf__ int32_t* candidate_experts,
    int32_t expert_count,
    int32_t prefetch_count,
    int32_t local_expert_begin,
    int32_t local_expert_end,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);

    for (int32_t slot = thread; slot < prefetch_count; slot += thread_count) {
        selected_experts[slot] = -1;
        group_counts[expert_count + slot] = 0;
    }
    asc_syncthreads();

    for (int32_t slot = 0; slot < prefetch_count; ++slot) {
        int32_t best_count = 0;
        int32_t best_expert = -1;
        for (int32_t expert = thread;
             expert < expert_count;
             expert += thread_count) {
            if (expert >= local_expert_begin && expert < local_expert_end) {
                continue;
            }
            const int32_t count = group_counts[expert];
            if (count > best_count ||
                (count == best_count && count > 0 && expert > best_expert)) {
                best_count = count;
                best_expert = expert;
            }
        }
        candidate_counts[thread] = best_count;
        candidate_experts[thread] = best_expert;
        asc_syncthreads();

        for (int32_t stride = thread_count >> 1; stride > 0; stride >>= 1) {
            if (thread < stride) {
                const int32_t other_count = candidate_counts[thread + stride];
                const int32_t other_expert = candidate_experts[thread + stride];
                const int32_t current_count = candidate_counts[thread];
                const int32_t current_expert = candidate_experts[thread];
                if (other_count > current_count ||
                    (other_count == current_count && other_count > 0 &&
                     other_expert > current_expert)) {
                    candidate_counts[thread] = other_count;
                    candidate_experts[thread] = other_expert;
                }
            }
            asc_syncthreads();
        }

        if (thread == 0) {
            const int32_t selected_count = candidate_counts[0];
            const int32_t selected_expert = candidate_experts[0];
            if (selected_count > 0 && selected_expert >= 0) {
                selected_experts[slot] = selected_expert;
                group_counts[expert_count + slot] = selected_count;
                group_counts[selected_expert] = 0;
            }
        }
        asc_syncthreads();
        if (selected_experts[slot] < 0) {
            break;
        }
    }
}

__simt_vf__ __launch_bounds__(kPrefetchSimtMaxThreads) inline void BuildGroupPrefixSimt(
    __ubuf__ int32_t* group_counts,
    __ubuf__ int32_t* thread_prefix,
    int32_t group_count,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t elems_per_thread =
        (group_count + thread_count - 1) / thread_count;
    const int32_t begin = thread * elems_per_thread;
    int32_t end = begin + elems_per_thread;
    if (end > group_count) {
        end = group_count;
    }

    int32_t local_prefix = 0;
    for (int32_t group = begin; group < end; ++group) {
        local_prefix += group_counts[group];
        group_counts[group] = local_prefix;
    }
    thread_prefix[thread] = local_prefix;
    asc_syncthreads();

    if (thread == 0) {
        int32_t block_prefix = 0;
        for (int32_t source_thread = 0;
             source_thread < thread_count;
             ++source_thread) {
            const int32_t thread_total = thread_prefix[source_thread];
            thread_prefix[source_thread] = block_prefix;
            block_prefix += thread_total;
        }
    }
    asc_syncthreads();

    const int32_t block_prefix = thread_prefix[thread];
    for (int32_t group = begin; group < end; ++group) {
        group_counts[group] += block_prefix;
    }
    asc_syncthreads();
}

__simt_vf__ __launch_bounds__(kCoreCursorSimtMaxThreads) inline void ReduceExpertTileTotalSimt(
    __ubuf__ int32_t* expert_counts,
    __ubuf__ int32_t* thread_totals,
    __ubuf__ int32_t* tile_total,
    int32_t expert_count,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    int32_t local_total = 0;
    for (int32_t expert = thread; expert < expert_count;
         expert += thread_count) {
        local_total += expert_counts[expert];
    }
    thread_totals[thread] = local_total;
    asc_syncthreads();

    if (thread == 0) {
        int32_t total = 0;
        for (int32_t source_thread = 0; source_thread < thread_count;
             ++source_thread) {
            total += thread_totals[source_thread];
        }
        tile_total[0] = total;
    }
}

__simt_vf__ __launch_bounds__(kCoreCursorSimtMaxThreads) inline void BuildCoreExpertCursorsSimt(
    __ubuf__ int32_t* core_counts,
    __ubuf__ int32_t* tpe_counts,
    __ubuf__ int32_t* thread_prefix,
    __ubuf__ int32_t* expert_carry,
    int32_t core_count,
    int32_t expert_count,
    int32_t row_stride,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t experts_per_thread =
        (expert_count + thread_count - 1) / thread_count;
    const int32_t begin = thread * experts_per_thread;
    int32_t end = begin + experts_per_thread;
    if (end > expert_count) {
        end = expert_count;
    }

    int32_t local_total = 0;
    for (int32_t expert = begin; expert < end; ++expert) {
        local_total += tpe_counts[expert];
        tpe_counts[expert] = local_total;
    }
    thread_prefix[thread] = local_total;
    asc_syncthreads();

    if (thread == 0) {
        int32_t prefix = expert_carry[0];
        for (int32_t source_thread = 0;
             source_thread < thread_count;
             ++source_thread) {
            const int32_t thread_total = thread_prefix[source_thread];
            thread_prefix[source_thread] = prefix;
            prefix += thread_total;
        }
        expert_carry[0] = prefix;
    }
    asc_syncthreads();

    const int32_t thread_base = thread_prefix[thread];
    for (int32_t expert = begin; expert < end; ++expert) {
        const int32_t expert_base =
            thread_base + (expert == begin ? 0 : tpe_counts[expert - 1]);
        int32_t cursor = expert_base;
        for (int32_t source_core = 0;
             source_core < core_count;
             ++source_core) {
            const int32_t offset = source_core * row_stride + expert;
            const int32_t count = core_counts[offset];
            core_counts[offset] = cursor;
            cursor += count;
        }
    }
    asc_syncthreads();
}

__simt_vf__ __launch_bounds__(kPrefetchSimtMaxThreads) inline void
ConvertGroupPrefixToExpertOffsetsSimt(
    __ubuf__ int32_t* group_prefix,
    __ubuf__ int32_t* selected_experts,
    __ubuf__ int32_t* selected_offsets,
    int32_t expert_count,
    int32_t prefetch_count,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);

    for (int32_t slot = thread; slot < prefetch_count; slot += thread_count) {
        const int32_t group = expert_count + slot;
        const int32_t begin = group == 0 ? 0 : group_prefix[group - 1];
        const int32_t end = group_prefix[group];
        selected_offsets[slot] = end > begin ? begin : -1;
    }

    const int32_t experts_per_thread =
        (expert_count + thread_count - 1) / thread_count;
    const int32_t begin = thread * experts_per_thread;
    int32_t end = begin + experts_per_thread;
    if (end > expert_count) {
        end = expert_count;
    }
    const int32_t preceding_prefix =
        begin > 0 && begin < expert_count ? group_prefix[begin - 1] : 0;
    asc_syncthreads();

    for (int32_t expert = end - 1; expert >= begin; --expert) {
        const int32_t group_begin =
            expert == begin ? preceding_prefix : group_prefix[expert - 1];
        const int32_t group_end = group_prefix[expert];
        group_prefix[expert] = group_end > group_begin ? group_begin : -1;
    }
    asc_syncthreads();

    for (int32_t slot = thread; slot < prefetch_count; slot += thread_count) {
        const int32_t expert = selected_experts[slot];
        const int32_t offset = selected_offsets[slot];
        if (expert >= 0 && offset >= 0) {
            group_prefix[expert] = offset;
        }
    }
    asc_syncthreads();
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void DedupRouteTileSimt(
    __ubuf__ int32_t* dst,
    __ubuf__ int32_t* token_groups,
    __ubuf__ int32_t* token_loffs,
    __ubuf__ int32_t* token_group_counts,
    __ubuf__ int32_t* token_loff_counts,
    __ubuf__ int32_t* tile_counts,
    int32_t tile_tokens,
    int32_t topk,
    int32_t max_tokens_per_rank,
    int32_t server_dedup) {
    const int32_t token = static_cast<int32_t>(threadIdx.x);
    // Tail-padding SIMT threads process no token, so the next tile is never overwritten.
    if (token >= tile_tokens) {
        return;
    }

    const int32_t route_base = token * topk;
    const int32_t group_base = token * (topk / 2) * 4;
    const int32_t loff_base = token * topk;
    int32_t group_count = 0;
    int32_t loff_count = 0;

    // The first actual route in K order of each dedup domain is the primary; a rank domain is 1 card, a server domain
    // is 8 cards.
    for (int32_t kidx = 0; kidx < topk; ++kidx) {
        const int32_t raw = dst[route_base + kidx];
        // Negatively marked routes already belong to an earlier primary and must not form a new group.
        if (raw < 0) {
            continue;
        }

        const int32_t primary_rank = raw / max_tokens_per_rank;
        const int32_t primary_base = primary_rank * max_tokens_per_rank;
        const int32_t ranks_per_group = server_dedup ? kPlanningRanksPerServer : 1;
        const int32_t rank_begin = (primary_rank / ranks_per_group) * ranks_per_group;
        const int32_t address_begin = rank_begin * max_tokens_per_rank;
        const int32_t address_end = address_begin + ranks_per_group * max_tokens_per_rank;
        const int32_t group_loff_start = loff_count;
        int32_t duplicate_count = 0;
        // Rank mode stores the in-card offset and server mode the full destination encoding; the swap destination is
        // still the actual primary card.
        for (int32_t next_kidx = kidx + 1; next_kidx < topk; ++next_kidx) {
            const int32_t next_raw = dst[route_base + next_kidx];
            // Negative values and addresses from other dedup domains never belong to the current group.
            if (next_raw < address_begin || next_raw >= address_end) {
                continue;
            }
            token_loffs[loff_base + loff_count] = server_dedup ? next_raw : next_raw - primary_base;
            ++loff_count;
            ++duplicate_count;
            dst[route_base + next_kidx] = -next_raw - 1;
        }

        // Counts are still organized over R primary cards and must not collapse into R/8 servers.
        if (duplicate_count > 0) {
            const int32_t group_pos = group_base + group_count * 4;
            token_groups[group_pos] = primary_rank;
            token_groups[group_pos + 1] = raw - primary_base;
            token_groups[group_pos + 2] = loff_base + group_loff_start;
            token_groups[group_pos + 3] = duplicate_count;
            asc_atomic_add(tile_counts + primary_rank * 2, 1);
            asc_atomic_add(tile_counts + primary_rank * 2 + 1, duplicate_count);
            ++group_count;
        }
    }

    token_group_counts[token] = group_count;
    token_loff_counts[token] = loff_count;
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void CompactRouteTileSimt(
    __ubuf__ int32_t* token_groups,
    __ubuf__ int32_t* token_loffs,
    __ubuf__ int32_t* token_group_counts,
    __ubuf__ int32_t* token_loff_counts,
    __ubuf__ int32_t* token_group_prefix,
    __ubuf__ int32_t* token_loff_prefix,
    __ubuf__ int32_t* compact_groups,
    __ubuf__ int32_t* compact_loffs,
    int32_t tile_tokens,
    int32_t topk) {
    const int32_t token = static_cast<int32_t>(threadIdx.x);
    if (token >= tile_tokens) {
        return;
    }

    const int32_t group_count = token_group_counts[token];
    const int32_t loff_count = token_loff_counts[token];
    const int32_t source_group_base = token * (topk / 2) * 4;
    const int32_t source_loff_base = token * topk;
    const int32_t compact_group_base = token_group_prefix[token] * 3;
    const int32_t compact_loff_base = token_loff_prefix[token];

    for (int32_t group = 0; group < group_count; ++group) {
        const int32_t source_group_pos = source_group_base + group * 4;
        const int32_t compact_group_pos = compact_group_base + group * 3;
        const int32_t source_loff_start = token_groups[source_group_pos + 2];
        compact_groups[compact_group_pos] = token_groups[source_group_pos];
        compact_groups[compact_group_pos + 1] = token_groups[source_group_pos + 1];
        compact_groups[compact_group_pos + 2] =
            compact_loff_base + source_loff_start - source_loff_base;
    }
    for (int32_t i = 0; i < loff_count; ++i) {
        compact_loffs[compact_loff_base + i] = token_loffs[source_loff_base + i];
    }
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void
PackSourceDupByDestSimt(
    __ubuf__ int32_t* compact_groups,
    __ubuf__ int32_t* compact_loffs,
    __ubuf__ int32_t* dest_meta,
    __ubuf__ int32_t* packed_groups,
    __ubuf__ int32_t* packed_loffs,
    __ubuf__ int32_t* chunk_hist,
    int32_t group_count,
    int32_t loff_count,
    int32_t rank_count,
    int32_t groups_per_chunk,
    int32_t hist_row_elems,
    int32_t thread_count) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t chunk_count =
        (group_count + groups_per_chunk - 1) / groups_per_chunk;

    // One thread owns each chunk, so its histogram and the later cursor
    // updates are deterministic without atomics.
    for (int32_t chunk = thread; chunk < chunk_count; chunk += thread_count) {
        __ubuf__ int32_t* hist_row = chunk_hist + chunk * hist_row_elems;
        const int32_t group_begin = chunk * groups_per_chunk;
        const int32_t group_end =
            group_begin + groups_per_chunk < group_count
                ? group_begin + groups_per_chunk
                : group_count;
        for (int32_t group = group_begin; group < group_end; ++group) {
            const int32_t compact_pos = group * 3;
            const int32_t dest = compact_groups[compact_pos];
            const int32_t raw_loff_start = compact_groups[compact_pos + 2];
            const int32_t next_raw_loff_start =
                group + 1 < group_count
                    ? compact_groups[compact_pos + 5]
                    : loff_count;
            const int32_t hist_pos = dest * 2;
            hist_row[hist_pos] = hist_row[hist_pos] + 1;
            hist_row[hist_pos + 1] = hist_row[hist_pos + 1] +
                next_raw_loff_start - raw_loff_start;
        }
    }
    asc_syncthreads();

    // Convert every destination's chunk histogram to exclusive chunk bases.
    for (int32_t dest = thread; dest < rank_count; dest += thread_count) {
        int32_t group_prefix = 0;
        int32_t loff_prefix = 0;
        const int32_t hist_pos = dest * 2;
        for (int32_t chunk = 0; chunk < chunk_count; ++chunk) {
            __ubuf__ int32_t* hist_row = chunk_hist + chunk * hist_row_elems;
            const int32_t chunk_groups = hist_row[hist_pos];
            const int32_t chunk_loffs = hist_row[hist_pos + 1];
            hist_row[hist_pos] = group_prefix;
            hist_row[hist_pos + 1] = loff_prefix;
            group_prefix += chunk_groups;
            loff_prefix += chunk_loffs;
        }
    }
    asc_syncthreads();

    // Re-scan each chunk in source order. The exclusive bases above place
    // chunks in order, while this serial per-chunk scan preserves group order.
    for (int32_t chunk = thread; chunk < chunk_count; chunk += thread_count) {
        __ubuf__ int32_t* cursor_row = chunk_hist + chunk * hist_row_elems;
        const int32_t group_begin = chunk * groups_per_chunk;
        const int32_t group_end =
            group_begin + groups_per_chunk < group_count
                ? group_begin + groups_per_chunk
                : group_count;
        for (int32_t group = group_begin; group < group_end; ++group) {
            const int32_t compact_pos = group * 3;
            const int32_t dest = compact_groups[compact_pos];
            const int32_t primary_loff = compact_groups[compact_pos + 1];
            const int32_t raw_loff_start = compact_groups[compact_pos + 2];
            const int32_t next_raw_loff_start =
                group + 1 < group_count
                    ? compact_groups[compact_pos + 5]
                    : loff_count;
            const int32_t duplicate_count =
                next_raw_loff_start - raw_loff_start;
            const int32_t hist_pos = dest * 2;
            const int32_t group_ordinal = cursor_row[hist_pos];
            cursor_row[hist_pos] = group_ordinal + 1;
            const int32_t loff_ordinal = cursor_row[hist_pos + 1];
            cursor_row[hist_pos + 1] = loff_ordinal + duplicate_count;

            const int32_t meta_pos = dest * 5;
            const int32_t group_pos =
                dest_meta[meta_pos] + group_ordinal * 3;
            const int32_t loff_base = dest_meta[meta_pos + 3];
            const int32_t loff_pos = loff_base + loff_ordinal;
            packed_groups[group_pos] = primary_loff;
            packed_groups[group_pos + 1] = loff_ordinal;
            packed_groups[group_pos + 2] = duplicate_count;
            // Only the full destination encoding is packed here; the card number and slot the secondary points to are
            // untouched.
            for (int32_t i = 0; i < duplicate_count; ++i) {
                packed_loffs[loff_pos + i] =
                    compact_loffs[raw_loff_start + i];
            }
        }
    }
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void
AdjustSourceDupGroupLoffsSimt(
    __ubuf__ int32_t* groups,
    __ubuf__ int32_t* tile_counts,
    __ubuf__ int32_t* dest_meta,
    int32_t rank_count,
    int32_t threads_per_dest) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int32_t logical_thread_count = rank_count * threads_per_dest;
    for (int32_t logical_thread = thread;
         logical_thread < logical_thread_count;
         logical_thread += kSourceDupSimtMaxThreads) {
        const int32_t dest = logical_thread / threads_per_dest;
        const int32_t dest_lane =
            logical_thread - dest * threads_per_dest;
        const int32_t group_count = tile_counts[dest * 2];
        const int32_t meta_pos = dest * 5;
        const int32_t group_start = dest_meta[meta_pos];
        const int32_t global_loff_base = dest_meta[meta_pos + 1];
        for (int32_t group = dest_lane;
             group < group_count;
             group += threads_per_dest) {
            const int32_t loff_field = group_start + group * 3 + 1;
            // Only the group list start is corrected; the secondary full destination encoding is left as is.
            groups[loff_field] = global_loff_base + groups[loff_field];
        }
    }
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void
PackSourceDupCountSlotsSimt(
    __ubuf__ int32_t* core_counts,
    __ubuf__ int32_t* count_slots,
    int32_t core_count,
    int32_t dest_count,
    int32_t core_row_elems) {
    const int32_t slot_element = static_cast<int32_t>(threadIdx.x);
    const int32_t slot_elems = dest_count * kSourceDupInboxSlotElems;
    if (slot_element >= slot_elems) {
        return;
    }

    const int32_t slot_lane = slot_element % kSourceDupInboxSlotElems;
    if (slot_lane >= 2) {
        count_slots[slot_element] = 0;
        return;
    }

    const int32_t local_dest = slot_element / kSourceDupInboxSlotElems;
    const int32_t count_element = local_dest * 2 + slot_lane;
    int32_t total = 0;
    for (int32_t source_core = 0; source_core < core_count; ++source_core) {
        total += core_counts[source_core * core_row_elems + count_element];
    }
    count_slots[slot_element] = total;
}

__simt_vf__ __launch_bounds__(32) inline void
BuildSourceDupDestinationPrefixesSimt(
    __ubuf__ int32_t* count_slots,
    __ubuf__ int32_t* totals,
    int32_t rank_count) {
    const int32_t count_kind = static_cast<int32_t>(threadIdx.x);
    // Two threads handle the group and secondary prefixes separately; other threads write no output.
    if (count_kind >= 2) {
        return;
    }

    int32_t prefix = 0;
    // Receive intervals are assigned contiguously in original-source order, and the group starts dispatch needs are
    // exported.
    for (int32_t source_rank = 0; source_rank < rank_count; ++source_rank) {
        const int32_t offset =
            source_rank * kSourceDupInboxSlotElems + count_kind;
        const int32_t count = count_slots[offset];
        count_slots[offset] = prefix;
        // Empty sources also write the current start; the end boundary of the last source is given by the total G.
        if (count_kind == 0) {
            totals[2 + source_rank] = prefix;
        }
        prefix += count;
    }
    totals[count_kind] = prefix;
}

__simt_vf__ __launch_bounds__(kSourceDupSimtMaxThreads) inline void
BuildSourceDupCorePrefixesSimt(
    __ubuf__ int32_t* core_counts,
    __ubuf__ int32_t* rank_prefix_slots,
    int32_t core_count,
    int32_t dest_count,
    int32_t core_row_elems) {
    const int32_t count_element = static_cast<int32_t>(threadIdx.x);
    const int32_t count_elems = dest_count * 2;
    if (count_element >= count_elems) {
        return;
    }

    const int32_t local_dest = count_element / 2;
    const int32_t count_lane = count_element - local_dest * 2;
    int32_t prefix = rank_prefix_slots[
        local_dest * kSourceDupInboxSlotElems + count_lane];
    for (int32_t source_core = 0; source_core < core_count; ++source_core) {
        const int32_t offset =
            source_core * core_row_elems + count_element;
        const int32_t count = core_counts[offset];
        core_counts[offset] = prefix;
        prefix += count;
    }
}

__simt_vf__ __launch_bounds__(kDstSimtMaxThreads) inline void BuildDstTileSimt(
    __ubuf__ int32_t* workspace,
    int32_t context_offset,
    int32_t tile_base,
    int32_t tile_len) {
    (void)tile_base;
    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    __ubuf__ int32_t* context = workspace + context_offset;
    __ubuf__ int32_t* topk_tile = workspace + context[kDstSimtTopkOffset];
    __ubuf__ int32_t* chunk_prefix = workspace + context[kDstSimtHistOffset];
    __ubuf__ int32_t* ordinal_base = workspace + context[kDstSimtOrdinalOffset];
    __ubuf__ int32_t* alloc_cumsum = workspace + context[kDstSimtAllocCumsumOffset];
    __ubuf__ int32_t* expert_offsets = workspace + context[kDstSimtExpertOffsetsOffset];
    __ubuf__ int32_t* dst_tile = workspace + context[kDstSimtDstOffset];
    const int32_t chunk_size = context[kDstSimtChunkSize];
    const int32_t hist_stride = context[kDstSimtHistStride];
    const int32_t rank_count = context[kDstSimtRankCount];
    const int32_t expert_count = context[kDstSimtExpertCount];
    const int32_t max_tokens_per_rank = context[kDstSimtMaxTokensPerRank];
    const int32_t chunk_count = (tile_len + chunk_size - 1) / chunk_size;

    if (thread < chunk_count) {
        const int32_t route_begin = thread * chunk_size;
        const int32_t route_end =
            route_begin + chunk_size < tile_len ? route_begin + chunk_size : tile_len;
        __ubuf__ int32_t* hist_row = chunk_prefix + thread * hist_stride;
        for (int32_t route = route_begin; route < route_end; ++route) {
            const int32_t expert = topk_tile[route];
            // Reuse the output tile for the stable ordinal inside this chunk.
            const int32_t rank_in_chunk = hist_row[expert];
            hist_row[expert] = rank_in_chunk + 1;
            dst_tile[route] = rank_in_chunk;
        }
    }
    asc_syncthreads();

    for (int32_t expert = thread; expert < expert_count;
         expert += kDstSimtMaxThreads) {
        int32_t prefix = 0;
        for (int32_t chunk = 0; chunk < chunk_count; ++chunk) {
            const int32_t offset = chunk * hist_stride + expert;
            prefix += chunk_prefix[offset];
            chunk_prefix[offset] = prefix;
        }
    }
    asc_syncthreads();

    if (thread < tile_len) {
        const int32_t route = thread;
        const int32_t expert = topk_tile[route];
        const int32_t chunk = route / chunk_size;
        const int32_t rank_in_chunk = dst_tile[route];

        const int32_t prior_chunk_count =
            chunk == 0 ? 0 : chunk_prefix[(chunk - 1) * hist_stride + expert];
        const int32_t global_ordinal =
            ordinal_base[expert] + prior_chunk_count + rank_in_chunk;

        int32_t dst_rank = rank_count - 1;
        int32_t previous_boundary = 0;
        for (int32_t dst = 0; dst < rank_count; ++dst) {
            const int32_t current_boundary = alloc_cumsum[expert * rank_count + dst];
            if (global_ordinal < current_boundary) {
                dst_rank = dst;
                break;
            }
            previous_boundary = current_boundary;
        }
        const int32_t dst_base = expert_offsets[dst_rank * expert_count + expert];
        dst_tile[route] = dst_rank * max_tokens_per_rank + dst_base +
            (global_ordinal - previous_boundary);
    }
    asc_syncthreads();
}

__simt_vf__ __launch_bounds__(kDstSimtMaxThreads) inline void BuildDstTileFromGmSimt(
    __ubuf__ int32_t* topk_tile,
    __ubuf__ int32_t* expert_ordinal,
    __ubuf__ int32_t* dst_tile,
    __gm__ int32_t* alloc_cumsum,
    __gm__ int32_t* expert_offsets,
    int32_t tile_len,
    int32_t rank_count,
    int32_t expert_count,
    int32_t max_tokens_per_rank) {
    const int32_t thread = static_cast<int32_t>(threadIdx.x);

    // A single thread assigns stable ordinals in input-route order. The
    // expensive rank lookup below remains parallel across the whole tile.
    if (thread == 0) {
        for (int32_t route = 0; route < tile_len; ++route) {
            const int32_t expert = topk_tile[route];
            const int32_t ordinal = expert_ordinal[expert];
            expert_ordinal[expert] = ordinal + 1;
            dst_tile[route] = ordinal;
        }
    }
    asc_syncthreads();

    for (int32_t route = thread; route < tile_len;
         route += kDstSimtMaxThreads) {
        const int32_t expert = topk_tile[route];
        const int32_t global_ordinal = dst_tile[route];
        const int32_t cumsum_base = expert * rank_count;

        int32_t low = 0;
        int32_t high = rank_count;
        while (low < high) {
            const int32_t mid = low + (high - low) / 2;
            if (global_ordinal < alloc_cumsum[cumsum_base + mid]) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        const int32_t dst_rank = low < rank_count ? low : rank_count - 1;
        const int32_t previous_boundary = dst_rank == 0
            ? 0
            : alloc_cumsum[cumsum_base + dst_rank - 1];
        const int32_t dst_base =
            expert_offsets[dst_rank * expert_count + expert];
        dst_tile[route] = dst_rank * max_tokens_per_rank + dst_base +
            (global_ordinal - previous_boundary);
    }
    asc_syncthreads();
}

__aicore__ inline int32_t IntMin(int32_t a, int32_t b) {
    return a < b ? a : b;
}

__aicore__ inline int32_t AlignUpBytes(int32_t value, int32_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__aicore__ inline int32_t AlignUpInt32(int32_t value) {
    return ((value + 7) / 8) * 8;
}

__aicore__ inline void InitDstUbLayout(
    int32_t rank_count,
    int32_t expert_count,
    int32_t expert_row_bytes,
    DstUbLayout& layout) {
    layout.context_offset = expert_row_bytes;
    const int32_t dst_tile_bytes =
        AlignUpBytes(kFillTileElems * kInt32Bytes, 32);
    layout.dst_tile_offset =
        layout.context_offset + kDstSimtContextBytes;
    layout.lookup_offset = layout.dst_tile_offset + dst_tile_bytes;
    layout.gm_simt_topk_offset = layout.lookup_offset;

    const int64_t lookup_table_elems =
        static_cast<int64_t>(rank_count) * expert_count;
    const int64_t lookup_table_data_bytes =
        lookup_table_elems * kInt32Bytes;
    const int64_t lookup_table_bytes =
        ((lookup_table_data_bytes + 31) / 32) * 32;
    const int64_t lookup_end_offset =
        static_cast<int64_t>(layout.lookup_offset) +
        lookup_table_bytes * 2;
    const bool lookup_fits = lookup_end_offset <= kPlanningUbSizeBytes;
    layout.lookup_table_elems = lookup_fits
        ? static_cast<int32_t>(lookup_table_elems)
        : 0;
    layout.lookup_table_bytes = lookup_fits
        ? static_cast<int32_t>(lookup_table_bytes)
        : 0;

    const int32_t simt_topk_bytes =
        AlignUpBytes(kFillTileElems * kInt32Bytes, 32);
    layout.simt_topk_offset = lookup_fits
        ? AlignUpBytes(static_cast<int32_t>(lookup_end_offset), 32)
        : 0;
    layout.simt_hist_offset = lookup_fits
        ? layout.simt_topk_offset + simt_topk_bytes
        : 0;
    layout.simt_hist_stride = AlignUpInt32(expert_count);
    layout.simt_chunk_size = kDstSimtMinChunkElems;
    if (lookup_fits) {
        int64_t simt_end_offset = 0;
        while (true) {
            const int32_t chunk_count =
                (kFillTileElems + layout.simt_chunk_size - 1) /
                layout.simt_chunk_size;
            simt_end_offset =
                static_cast<int64_t>(layout.simt_hist_offset) +
                static_cast<int64_t>(chunk_count) *
                    layout.simt_hist_stride * kInt32Bytes;
            if (simt_end_offset <= kPlanningUbSizeBytes ||
                layout.simt_chunk_size >= kDstSimtMaxChunkElems) {
                break;
            }
            layout.simt_chunk_size *= 2;
        }
        layout.use_simt = simt_end_offset <= kPlanningUbSizeBytes;
    } else {
        layout.use_simt = false;
    }
}

__aicore__ inline int32_t SelectPrefetchSimtThreadCount(int32_t expert_count) {
    int32_t required_threads =
        (expert_count + kPrefetchSimtItemsPerThread - 1) /
        kPrefetchSimtItemsPerThread;
    int32_t thread_count = 32;
    while (thread_count < required_threads &&
           thread_count < kPrefetchSimtMaxThreads) {
        thread_count <<= 1;
    }
    return thread_count;
}

__aicore__ inline void InitPrefetchUbLayout(
    int32_t expert_count,
    int32_t prefetch_count,
    PrefetchUbLayout& layout) {
    layout.group_counts_offset = 0;
    layout.group_counts_bytes =
        AlignUpBytes((expert_count + prefetch_count) * kInt32Bytes, 32);
    layout.selected_experts_offset = layout.group_counts_bytes;
    layout.selected_experts_bytes =
        AlignUpBytes(prefetch_count * kInt32Bytes, 32);
    layout.scratch_counts_offset =
        layout.selected_experts_offset + layout.selected_experts_bytes;
    const int32_t scratch_elems =
        prefetch_count > kPrefetchSimtMaxThreads
            ? prefetch_count
            : kPrefetchSimtMaxThreads;
    layout.scratch_row_elems = AlignUpInt32(scratch_elems);
    layout.scratch_row_bytes = layout.scratch_row_elems * kInt32Bytes;
    layout.scratch_experts_offset =
        layout.scratch_counts_offset + layout.scratch_row_bytes;
    layout.total_bytes =
        layout.scratch_experts_offset + layout.scratch_row_bytes;
}

__aicore__ inline void InitSourceDupCountTileLayout(
    const SourceDupConfig& config,
    SourceDupCountTileLayout& layout) {
    layout.max_ranks_per_core =
        (config.rank_count + config.core_num - 1) / config.core_num;
    layout.row_elems = AlignUpInt32(layout.max_ranks_per_core * 2);
    layout.core_counts_offset = 0;
    layout.slots_offset =
        config.core_num * layout.row_elems * kInt32Bytes;
    layout.result_offset = layout.slots_offset + AlignUpBytes(
        layout.max_ranks_per_core * kSourceDupInboxSlotElems * kInt32Bytes,
        32);
    layout.total_bytes = layout.result_offset + 32;
}

__aicore__ inline void GetSourceDupRankPartition(
    const SourceDupConfig& config,
    int32_t& rank_begin,
    int32_t& rank_count) {
    const int32_t ranks_per_core = config.rank_count / config.core_num;
    const int32_t tail_cores =
        config.rank_count - ranks_per_core * config.core_num;
    rank_begin =
        config.core_idx * ranks_per_core + IntMin(config.core_idx, tail_cores);
    rank_count = ranks_per_core + (config.core_idx < tail_cores ? 1 : 0);
}

__aicore__ inline void LoadSourceDupCoreCountTile(
    const SourceDupConfig& config,
    const SourceDupCountTileLayout& layout,
    int32_t dest_begin,
    int32_t dest_count,
    AscendC::LocalTensor<int32_t>& core_counts_ub,
    AscendC::GlobalTensor<int32_t>& core_counts_gm) {
    const int32_t active_elems = dest_count * 2;
    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
    AscendC::DataCopyExtParams copy_params(
        1,
        static_cast<uint32_t>(active_elems * kInt32Bytes),
        0,
        0,
        0);
    for (int32_t source_core = 0;
         source_core < config.core_num;
         ++source_core) {
        AscendC::DataCopyPad(
            core_counts_ub[source_core * layout.row_elems],
            core_counts_gm[
                (source_core * config.rank_count + dest_begin) * 2],
            copy_params,
            pad_params);
    }
}

__aicore__ inline void StoreSourceDupCoreCountTile(
    const SourceDupConfig& config,
    const SourceDupCountTileLayout& layout,
    int32_t dest_begin,
    int32_t dest_count,
    AscendC::GlobalTensor<int32_t>& core_counts_gm,
    AscendC::LocalTensor<int32_t>& core_counts_ub) {
    const int32_t active_elems = dest_count * 2;
    AscendC::DataCopyExtParams copy_params(
        1,
        static_cast<uint32_t>(active_elems * kInt32Bytes),
        0,
        0,
        0);
    for (int32_t source_core = 0;
         source_core < config.core_num;
         ++source_core) {
        AscendC::DataCopyPad(
            core_counts_gm[
                (source_core * config.rank_count + dest_begin) * 2],
            core_counts_ub[source_core * layout.row_elems],
            copy_params);
    }
}

__aicore__ inline int32_t SelectSourceDupPackChunkGroups(
    int32_t rank_count,
    int32_t topk) {
    if (rank_count <= 32 && topk <= 8) {
        return kSourceDupPackTinyChunkGroups;
    }
    if (rank_count <= 32) {
        return kSourceDupPackSmallChunkGroups;
    }
    if (rank_count <= 128) {
        return kSourceDupPackMediumChunkGroups;
    }
    if (rank_count <= 256) {
        return kSourceDupPackLargeChunkGroups;
    }
    return kSourceDupPackXLargeChunkGroups;
}

__aicore__ inline void InitSourceDupPartition(
    const SourceDupConfig& config,
    SourceDupPartition& partition) {
    const int32_t tokens_per_core = config.token_count / config.core_num;
    const int32_t token_tail_cores =
        config.token_count - tokens_per_core * config.core_num;
    partition.token_begin =
        config.core_idx * tokens_per_core + IntMin(config.core_idx, token_tail_cores);
    const int32_t token_assigned =
        tokens_per_core + (config.core_idx < token_tail_cores ? 1 : 0);
    partition.token_end = partition.token_begin + token_assigned;
    partition.max_tokens_per_core =
        (config.token_count + config.core_num - 1) / config.core_num;
    partition.max_tiles_per_core =
        (partition.max_tokens_per_core + config.tokens_per_tile - 1) /
        config.tokens_per_tile;

}

__aicore__ inline void InitSourceDupUbLayout(
    const SourceDupConfig& config,
    SourceDupUbLayout& layout) {
    const int32_t rank_count = config.rank_count;
    const int32_t tokens_per_tile = config.tokens_per_tile;
    layout.route_tile_elems = tokens_per_tile * config.topk;
    layout.count_bytes = AlignUpBytes(rank_count * 2 * kInt32Bytes, 32);

    const int32_t staging_elems =
        layout.route_tile_elems + rank_count * 7 > rank_count * 2
            ? layout.route_tile_elems + rank_count * 7
            : rank_count * 2;
    layout.staging_bytes = AlignUpBytes(staging_elems * kInt32Bytes, 32);
    layout.prefix_offset = layout.count_bytes + layout.staging_bytes;

    const int32_t prefix_bytes =
        AlignUpBytes(rank_count * 2 * kInt32Bytes, 32);
    layout.meta_offset = layout.prefix_offset + prefix_bytes;
    const int32_t meta_bytes =
        AlignUpBytes(rank_count * 5 * kInt32Bytes, 32);
    layout.raw_group_offset = layout.meta_offset + meta_bytes;

    const int32_t max_group_count = layout.route_tile_elems / 2;
    const int32_t group_send_elems =
        max_group_count * 3 + rank_count * 7 > rank_count * 2
            ? max_group_count * 3 + rank_count * 7
            : rank_count * 2;
    const int32_t raw_group_elems =
        max_group_count * 4 > group_send_elems
            ? max_group_count * 4
            : group_send_elems;
    const int32_t raw_group_bytes =
        AlignUpBytes(raw_group_elems * kInt32Bytes, 32);
    layout.raw_loff_offset = layout.raw_group_offset + raw_group_bytes;
    const int32_t raw_loff_elems = layout.route_tile_elems + rank_count * 7;
    const int32_t raw_loff_bytes =
        AlignUpBytes(raw_loff_elems * kInt32Bytes, 32);
    layout.group_send_offset = layout.raw_loff_offset + raw_loff_bytes;
    const int32_t group_send_bytes =
        AlignUpBytes(group_send_elems * kInt32Bytes, 32);
    layout.tile_count_offset = layout.group_send_offset + group_send_bytes;
    layout.tile_count_row_elems = AlignUpInt32(rank_count * 2);
    const int32_t tile_count_bytes =
        layout.tile_count_row_elems * kInt32Bytes;
    layout.token_group_count_offset = layout.tile_count_offset + tile_count_bytes;

    const int32_t token_row_bytes =
        AlignUpBytes(tokens_per_tile * kInt32Bytes, 32);
    layout.token_loff_count_offset =
        layout.token_group_count_offset + token_row_bytes;
    layout.token_group_prefix_offset =
        layout.token_loff_count_offset + token_row_bytes;
    layout.token_loff_prefix_offset =
        layout.token_group_prefix_offset + token_row_bytes;
    layout.compact_loff_offset =
        layout.token_loff_prefix_offset + token_row_bytes;
    layout.pack_hist_offset =
        layout.compact_loff_offset +
        AlignUpBytes(layout.route_tile_elems * kInt32Bytes, 32);
    layout.pack_hist_row_elems = AlignUpInt32(rank_count * 2);
    layout.pack_chunk_groups =
        SelectSourceDupPackChunkGroups(rank_count, config.topk);
    const int32_t max_pack_chunk_count =
        (max_group_count + layout.pack_chunk_groups - 1) /
        layout.pack_chunk_groups;
    layout.total_bytes =
        layout.pack_hist_offset +
        max_pack_chunk_count * layout.pack_hist_row_elems * kInt32Bytes;
}

__aicore__ inline void InitSourceDupWorkspaceLayout(
    const SourceDupConfig& config,
    const SourceDupPartition& partition,
    const SourceDupUbLayout& ub_layout,
    SourceDupWorkspaceLayout& layout) {
    layout.tile_count_offset = kMaxPlanningCores * config.rank_count * 2;
    const int32_t max_routes_per_core =
        partition.max_tokens_per_core * config.topk;
    layout.group_stride = (max_routes_per_core / 2) * 3;
    layout.loff_stride = max_routes_per_core;
    const int32_t prefix_elems = kMaxPlanningCores * config.rank_count * 2;
    layout.group_offset = prefix_elems;
    layout.loff_offset =
        layout.group_offset + config.core_num * layout.group_stride;
    const int32_t payload_elems =
        layout.loff_offset + config.core_num * layout.loff_stride;
    layout.prebuilt_group_stride = AlignUpInt32(
        (ub_layout.route_tile_elems / 2) * 3 + config.rank_count * 7);
    layout.prebuilt_loff_stride =
        AlignUpInt32(ub_layout.route_tile_elems + config.rank_count * 7);
    layout.prebuilt_group_offset = AlignUpInt32(payload_elems);
    const int32_t tile_slots =
        config.core_num * partition.max_tiles_per_core;
    layout.prebuilt_loff_offset =
        layout.prebuilt_group_offset +
        tile_slots * layout.prebuilt_group_stride;
}

__aicore__ inline void CleanInvalidDataCache() {
    AscendC::GlobalTensor<uint8_t> global;
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<
        uint8_t,
        AscendC::CacheLine::ENTIRE_DATA_CACHE,
        AscendC::DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

__aicore__ inline void StoreContiguousGmThroughUb(
    AscendC::GlobalTensor<int32_t>& gm,
    AscendC::LocalTensor<int32_t>& ub,
    int32_t gm_offset,
    int32_t elem_count) {
    // Submission only: the caller owns producer and UB-reuse synchronization.
    if (elem_count <= 0) {
        return;
    }

    AscendC::DataCopyExtParams copy_params(
        1, static_cast<uint32_t>(elem_count * sizeof(int32_t)), 0, 0, 0);
    AscendC::DataCopyPad(gm[gm_offset], ub, copy_params);
}

// Partition by 32-byte data blocks so concurrent cores never update different
// parts of the same GM block. Only the final owner may issue a short tail.
__aicore__ inline void FillContiguousGm(
    AscendC::GlobalTensor<int32_t>& gm,
    AscendC::LocalTensor<int32_t>& fill_ub,
    int32_t gm_offset,
    int32_t elem_count,
    int32_t value,
    int32_t core_idx,
    int32_t core_num) {
    (void)fill_ub;
    if (elem_count <= 0) {
        return;
    }

    constexpr int32_t kElemsPerDataBlock = 32 / kInt32Bytes;
    const int32_t block_count =
        (elem_count + kElemsPerDataBlock - 1) / kElemsPerDataBlock;
    const int32_t blocks_per_core = block_count / core_num;
    const int32_t tail_cores = block_count - blocks_per_core * core_num;
    const int32_t block_begin =
        core_idx * blocks_per_core + IntMin(core_idx, tail_cores);
    const int32_t assigned_blocks =
        blocks_per_core + (core_idx < tail_cores ? 1 : 0);
    if (assigned_blocks <= 0) {
        return;
    }
    const int32_t begin = block_begin * kElemsPerDataBlock;
    const int32_t end = IntMin(
        elem_count,
        (block_begin + assigned_blocks) * kElemsPerDataBlock);

    for (int32_t base = begin; base < end; base += kFillTileElems) {
        int32_t len = IntMin(kFillTileElems, end - base);
        AscendC::Duplicate(fill_ub, value, len);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        StoreContiguousGmThroughUb(gm, fill_ub, gm_offset + base, len);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

__aicore__ inline bool ExpertSelectedInUb(
    AscendC::LocalTensor<int32_t>& selected_ub,
    int32_t expert,
    int32_t B) {
    for (int32_t b = 0; b < B; ++b) {
        if (selected_ub.GetValue(b) == expert) {
            return true;
        }
    }
    return false;
}

__aicore__ inline void WaitForSourceDupInbox(
    AscendC::GlobalTensor<int32_t>& local_shmem_gm,
    AscendC::LocalTensor<int32_t>& inbox_ub,
    AscendC::LocalTensor<int32_t>& result_ub,
    int32_t inbox_offset,
    int32_t slot_count) {
    if (slot_count <= 0) {
        return;
    }

    const int32_t inbox_elems = slot_count * kSourceDupInboxSlotElems;
    AscendC::DataCopyExtParams copy_params(
        1,
        static_cast<uint32_t>(inbox_elems * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
    while (true) {
        AscendC::DataCopyPad(
            inbox_ub,
            local_shmem_gm[inbox_offset],
            copy_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::ReduceMin<int32_t>(
            result_ub,
            inbox_ub,
            inbox_ub,
            inbox_elems,
            false);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        if (result_ub.GetValue(0) >= 0) {
            break;
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    }
}

__aicore__ inline void PublishSourceDupPayloadDoneFlags(
    const SourceDupConfig& config,
    __gm__ int32_t* shmem_base,
    AscendC::LocalTensor<int32_t>& signal_ub,
    int32_t payload_done_flag_off) {
    signal_ub.SetValue(0, 1);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    int32_t dest_begin = 0;
    int32_t dest_count = 0;
    GetSourceDupRankPartition(config, dest_begin, dest_count);
    if (dest_count <= 0) {
        return;
    }

    AscendC::DataCopyExtParams flag_store_params(
        1,
        static_cast<uint32_t>(kInt32Bytes),
        0,
        0,
        0);
    for (int32_t local_dest = 0; local_dest < dest_count; ++local_dest) {
        const int32_t dest = dest_begin + local_dest;
        AscendC::GlobalTensor<int32_t> remote_shmem_gm;
        remote_shmem_gm.SetGlobalBuffer(
            static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, dest)));
        AscendC::DataCopyPad(
            remote_shmem_gm[payload_done_flag_off + config.rank],
            signal_ub,
            flag_store_params);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void WaitForSourceDupPayloadDoneFlags(
    const SourceDupConfig& config,
    PlanningResource& resource,
    AscendC::GlobalTensor<int32_t>& local_shmem_gm,
    int32_t payload_done_flag_off) {
    const int32_t ranks_per_core = config.rank_count / config.core_num;
    const int32_t tail_cores =
        config.rank_count - ranks_per_core * config.core_num;
    const int32_t rank_begin =
        config.core_idx * ranks_per_core + IntMin(config.core_idx, tail_cores);
    const int32_t rank_count =
        ranks_per_core + (config.core_idx < tail_cores ? 1 : 0);
    if (rank_count <= 0) {
        return;
    }

    const int32_t flag_bytes = AlignUpBytes(rank_count * kInt32Bytes, 32);
    AscendC::LocalTensor<int32_t> flag_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> result_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(flag_bytes));
    AscendC::DataCopyExtParams flag_load_params(
        1,
        static_cast<uint32_t>(rank_count * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> flag_pad_params{false, 0, 0, 0};
    while (true) {
        AscendC::DataCopyPad(
            flag_ub,
            local_shmem_gm[payload_done_flag_off + rank_begin],
            flag_load_params,
            flag_pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::ReduceMin<int32_t>(
            result_ub,
            flag_ub,
            flag_ub,
            rank_count,
            false);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        if (result_ub.GetValue(0) >= 1) {
            break;
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    }
}

__aicore__ inline void PublishSourceDupCounts(
    const SourceDupConfig& config,
    PlanningResource& resource,
    __gm__ int32_t* shmem_base,
    AscendC::GlobalTensor<int32_t>& dup_workspace_gm,
    int32_t count_inbox_off) {
    SourceDupCountTileLayout layout{};
    InitSourceDupCountTileLayout(config, layout);
    AscendC::LocalTensor<int32_t> core_counts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(layout.core_counts_offset));
    AscendC::LocalTensor<int32_t> count_slots_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(layout.slots_offset));

    int32_t dest_begin = 0;
    int32_t dest_count = 0;
    GetSourceDupRankPartition(config, dest_begin, dest_count);
    if (dest_count <= 0) {
        return;
    }

    LoadSourceDupCoreCountTile(
        config,
        layout,
        dest_begin,
        dest_count,
        core_counts_ub,
        dup_workspace_gm);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    const int32_t slot_elems = dest_count * kSourceDupInboxSlotElems;
    const uint32_t thread_count = static_cast<uint32_t>(
        ((slot_elems + 31) / 32) * 32);
    asc_vf_call<PackSourceDupCountSlotsSimt>(
        dim3(thread_count),
        (__ubuf__ int32_t*)core_counts_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)count_slots_ub.GetPhyAddr(),
        config.core_num,
        dest_count,
        layout.row_elems);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    AscendC::DataCopyExtParams slot_copy_params(
        1,
        static_cast<uint32_t>(kSourceDupInboxSlotElems * kInt32Bytes),
        0,
        0,
        0);
    for (int32_t local_dest = 0; local_dest < dest_count; ++local_dest) {
        const int32_t dest = dest_begin + local_dest;
        AscendC::GlobalTensor<int32_t> remote_shmem_gm;
        remote_shmem_gm.SetGlobalBuffer(
            static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, dest)));
        AscendC::DataCopyPad(
            remote_shmem_gm[
                count_inbox_off +
                config.rank * kSourceDupInboxSlotElems],
            count_slots_ub[local_dest * kSourceDupInboxSlotElems],
            slot_copy_params);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void BuildSourceDupDestinationPrefixes(
    const SourceDupConfig& config,
    PlanningResource& resource,
    AscendC::GlobalTensor<int32_t>& local_shmem_gm,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm,
    AscendC::GlobalTensor<int32_t>& dup_counts_gm,
    int32_t count_inbox_off) {
    const int32_t inbox_elems =
        config.rank_count * kSourceDupInboxSlotElems;
    const int32_t inbox_bytes = inbox_elems * kInt32Bytes;
    AscendC::LocalTensor<int32_t> count_slots_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    // An aligned [G,D,P...] region is reserved after the inbox; its capacity is bound by the host budget and the top
    // assertions.
    AscendC::LocalTensor<int32_t> totals_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(inbox_bytes));
    WaitForSourceDupInbox(
        local_shmem_gm,
        count_slots_ub,
        totals_ub,
        count_inbox_off,
        config.rank_count);

    asc_vf_call<BuildSourceDupDestinationPrefixesSimt>(
        dim3(32),
        (__ubuf__ int32_t*)count_slots_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)totals_ub.GetPhyAddr(),
        config.rank_count);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::DataCopyExtParams prefix_copy_params(
        1,
        static_cast<uint32_t>(inbox_bytes),
        0,
        0,
        0);
    AscendC::DataCopyPad(
        dup_key_workspace_gm[0],
        count_slots_ub,
        prefix_copy_params);
    // Internally the source prefix is still shared; rank outputs carry only G and D, and P must not be written outside
    // its logical bound.
    const int32_t count_size = config.server_dedup ? config.rank_count + 2 : 2;
    AscendC::DataCopyExtParams total_copy_params(
        1,
        static_cast<uint32_t>(count_size * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPad(dup_counts_gm[0], totals_ub, total_copy_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void PublishSourceDupPrefixes(
    const SourceDupConfig& config,
    PlanningResource& resource,
    __gm__ int32_t* shmem_base,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm,
    int32_t prefix_inbox_off) {
    int32_t source_begin = 0;
    int32_t source_count = 0;
    GetSourceDupRankPartition(config, source_begin, source_count);
    if (source_count <= 0) {
        return;
    }

    const int32_t slot_elems = source_count * kSourceDupInboxSlotElems;
    AscendC::LocalTensor<int32_t> prefix_slots_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::DataCopyExtParams load_params(
        1,
        static_cast<uint32_t>(slot_elems * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(
        prefix_slots_ub,
        dup_key_workspace_gm[
            source_begin * kSourceDupInboxSlotElems],
        load_params,
        pad_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

    AscendC::DataCopyExtParams slot_copy_params(
        1,
        static_cast<uint32_t>(kSourceDupInboxSlotElems * kInt32Bytes),
        0,
        0,
        0);
    for (int32_t local_source = 0;
         local_source < source_count;
         ++local_source) {
        const int32_t source = source_begin + local_source;
        AscendC::GlobalTensor<int32_t> remote_shmem_gm;
        remote_shmem_gm.SetGlobalBuffer(
            static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, source)));
        AscendC::DataCopyPad(
            remote_shmem_gm[
                prefix_inbox_off +
                config.rank * kSourceDupInboxSlotElems],
            prefix_slots_ub[local_source * kSourceDupInboxSlotElems],
            slot_copy_params);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void BuildSourceDupCorePrefixes(
    const SourceDupConfig& config,
    PlanningResource& resource,
    AscendC::GlobalTensor<int32_t>& local_shmem_gm,
    AscendC::GlobalTensor<int32_t>& dup_workspace_gm,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm,
    int32_t prefix_inbox_off) {
    SourceDupCountTileLayout layout{};
    InitSourceDupCountTileLayout(config, layout);
    AscendC::LocalTensor<int32_t> core_counts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(layout.core_counts_offset));
    AscendC::LocalTensor<int32_t> prefix_slots_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(layout.slots_offset));
    AscendC::LocalTensor<int32_t> result_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(layout.result_offset));

    int32_t dest_begin = 0;
    int32_t dest_count = 0;
    GetSourceDupRankPartition(config, dest_begin, dest_count);
    if (dest_count <= 0) {
        return;
    }

    WaitForSourceDupInbox(
        local_shmem_gm,
        prefix_slots_ub,
        result_ub,
        prefix_inbox_off + dest_begin * kSourceDupInboxSlotElems,
        dest_count);
    LoadSourceDupCoreCountTile(
        config,
        layout,
        dest_begin,
        dest_count,
        core_counts_ub,
        dup_workspace_gm);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    const int32_t count_elems = dest_count * 2;
    const uint32_t thread_count = static_cast<uint32_t>(
        ((count_elems + 31) / 32) * 32);
    asc_vf_call<BuildSourceDupCorePrefixesSimt>(
        dim3(thread_count),
        (__ubuf__ int32_t*)core_counts_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)prefix_slots_ub.GetPhyAddr(),
        config.core_num,
        dest_count,
        layout.row_elems);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    StoreSourceDupCoreCountTile(
        config,
        layout,
        dest_begin,
        dest_count,
        dup_key_workspace_gm,
        core_counts_ub);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void MaterializeSourceDupOutputs(
    const SourceDupConfig& config,
    const SourceDupUbLayout& ub_layout,
    PlanningResource& resource,
    AscendC::GlobalTensor<int32_t>& local_shmem_gm,
    AscendC::GlobalTensor<int32_t>& dup_counts_gm,
    AscendC::GlobalTensor<int32_t>& dup_groups_gm,
    AscendC::GlobalTensor<int32_t>& dup_loffs_gm,
    int32_t dup_groups_scratch_off,
    int32_t dup_loffs_scratch_off) {
    AscendC::LocalTensor<int32_t> count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> transfer_pingpong_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.count_bytes));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::DataCopyExtParams count_copy_params(
        1,
        static_cast<uint32_t>(2 * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> count_pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(
        count_ub,
        dup_counts_gm[0],
        count_copy_params,
        count_pad_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    const int32_t total_group_elems = count_ub.GetValue(0) * 3;
    const int32_t group_elems_per_core = total_group_elems / config.core_num;
    const int32_t group_tail_cores =
        total_group_elems - group_elems_per_core * config.core_num;
    const int32_t group_begin =
        config.core_idx * group_elems_per_core +
        IntMin(config.core_idx, group_tail_cores);
    const int32_t group_assigned =
        group_elems_per_core + (config.core_idx < group_tail_cores ? 1 : 0);
    const int32_t group_end = group_begin + group_assigned;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    int32_t group_tile_idx = 0;
    for (int32_t base = group_begin;
         base < group_end;
         base += kDupSrcTileElems, ++group_tile_idx) {
        const int32_t len = IntMin(kDupSrcTileElems, group_end - base);
        const int32_t buffer_idx = group_tile_idx & 1;
        const auto event_id = buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<int32_t> transfer_ub =
            transfer_pingpong_ub[buffer_idx * kDupSrcTileElems];
        AscendC::DataCopyExtParams copy_params(
            1,
            static_cast<uint32_t>(len * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopyPad(
            transfer_ub,
            local_shmem_gm[dup_groups_scratch_off + base],
            copy_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::DataCopyPad(dup_groups_gm[base], transfer_ub, copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    const int32_t total_loffs = count_ub.GetValue(1);
    const int32_t loffs_per_core = total_loffs / config.core_num;
    const int32_t loff_tail_cores = total_loffs - loffs_per_core * config.core_num;
    const int32_t loff_begin =
        config.core_idx * loffs_per_core + IntMin(config.core_idx, loff_tail_cores);
    const int32_t loff_assigned =
        loffs_per_core + (config.core_idx < loff_tail_cores ? 1 : 0);
    const int32_t loff_end = loff_begin + loff_assigned;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    int32_t loff_tile_idx = 0;
    for (int32_t base = loff_begin;
         base < loff_end;
         base += kDupSrcTileElems, ++loff_tile_idx) {
        const int32_t len = IntMin(kDupSrcTileElems, loff_end - base);
        const int32_t buffer_idx = loff_tile_idx & 1;
        const auto event_id = buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<int32_t> transfer_ub =
            transfer_pingpong_ub[buffer_idx * kDupSrcTileElems];
        AscendC::DataCopyExtParams copy_params(
            1,
            static_cast<uint32_t>(len * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopyPad(
            transfer_ub,
            local_shmem_gm[dup_loffs_scratch_off + base],
            copy_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::DataCopyPad(dup_loffs_gm[base], transfer_ub, copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void ProcessSourceDupTile(
    const SourceDupConfig& config,
    const SourceDupPartition& partition,
    const SourceDupUbLayout& ub_layout,
    const SourceDupWorkspaceLayout& workspace_layout,
    PlanningResource& resource,
    AscendC::GlobalTensor<int32_t>& dst_gm,
    AscendC::GlobalTensor<int32_t>& dup_workspace_gm,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm,
    int32_t token_base,
    int32_t tile_idx) {
    AscendC::LocalTensor<int32_t> count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> dst_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.count_bytes));
    AscendC::LocalTensor<int32_t> meta_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.meta_offset));
    AscendC::LocalTensor<int32_t> raw_group_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.raw_group_offset));
    AscendC::LocalTensor<int32_t> raw_loff_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.raw_loff_offset));
    AscendC::LocalTensor<int32_t> group_send_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.group_send_offset));
    AscendC::LocalTensor<int32_t> tile_count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.tile_count_offset));
    AscendC::LocalTensor<int32_t> token_group_count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.token_group_count_offset));
    AscendC::LocalTensor<int32_t> token_loff_count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.token_loff_count_offset));
    AscendC::LocalTensor<int32_t> token_group_prefix_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.token_group_prefix_offset));
    AscendC::LocalTensor<int32_t> token_loff_prefix_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.token_loff_prefix_offset));
    AscendC::LocalTensor<int32_t> compact_loff_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.compact_loff_offset));
    AscendC::LocalTensor<int32_t> pack_hist_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.pack_hist_offset));
    const int32_t tile_tokens = IntMin(config.tokens_per_tile, partition.token_end - token_base);
    const int32_t tile_routes = tile_tokens * config.topk;
    AscendC::Duplicate(tile_count_ub, static_cast<int32_t>(0), ub_layout.tile_count_row_elems);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::DataCopyExtParams copy_params(
        1,
        static_cast<uint32_t>(tile_routes * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(
        dst_ub,
        dst_gm[token_base * config.topk],
        copy_params,
        pad_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    const uint32_t simt_thread_count = static_cast<uint32_t>(
        ((tile_tokens + 31) / 32) * 32);
    asc_vf_call<DedupRouteTileSimt>(
        dim3(simt_thread_count),
        (__ubuf__ int32_t*)dst_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)raw_group_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)raw_loff_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)token_group_count_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)token_loff_count_ub.GetPhyAddr(),
        (__ubuf__ int32_t*)tile_count_ub.GetPhyAddr(),
        tile_tokens,
        config.topk,
        config.max_tokens_per_rank,
        config.server_dedup);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    int32_t raw_group_count = 0;
    int32_t raw_loff_count = 0;
    for (int32_t local_token = 0; local_token < tile_tokens; ++local_token) {
        const int32_t token_group_count =
            token_group_count_ub.GetValue(local_token);
        const int32_t token_loff_count =
            token_loff_count_ub.GetValue(local_token);
        token_group_prefix_ub.SetValue(local_token, raw_group_count);
        token_loff_prefix_ub.SetValue(local_token, raw_loff_count);
        raw_group_count += token_group_count;
        raw_loff_count += token_loff_count;
    }

    const bool tile_has_duplicates = raw_group_count > 0;
    if (tile_has_duplicates) {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        asc_vf_call<CompactRouteTileSimt>(
            dim3(simt_thread_count),
            (__ubuf__ int32_t*)raw_group_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)raw_loff_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)token_group_count_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)token_loff_count_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)token_group_prefix_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)token_loff_prefix_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)group_send_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)compact_loff_ub.GetPhyAddr(),
            tile_tokens,
            config.topk);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    if (tile_has_duplicates) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams dst_copy_params(
            1,
            static_cast<uint32_t>(tile_routes * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPad(
            dst_gm[token_base * config.topk],
            dst_ub,
            dst_copy_params);
    }
    int32_t group_tile_prefix = 0;
    int32_t loff_tile_prefix = 0;
    for (int32_t dest = 0; dest < config.rank_count; ++dest) {
        const int32_t meta_offset = dest * 5;
        const int32_t count_offset = dest * 2;
        const int32_t group_count = tile_count_ub.GetValue(count_offset);
        const int32_t loff_count = tile_count_ub.GetValue(count_offset + 1);
        if (group_count > 0) {
            group_tile_prefix = AlignUpInt32(group_tile_prefix);
            loff_tile_prefix = AlignUpInt32(loff_tile_prefix);
        }
        meta_ub.SetValue(meta_offset, group_tile_prefix);
        meta_ub.SetValue(meta_offset + 1, group_tile_prefix);
        meta_ub.SetValue(meta_offset + 3, loff_tile_prefix);
        meta_ub.SetValue(meta_offset + 4, loff_tile_prefix);
        group_tile_prefix += group_count * 3;
        loff_tile_prefix += loff_count;
    }
    if (tile_has_duplicates) {
        const int32_t pack_chunk_count =
            (raw_group_count + ub_layout.pack_chunk_groups - 1) /
            ub_layout.pack_chunk_groups;
        const int32_t pack_hist_elems =
            pack_chunk_count * ub_layout.pack_hist_row_elems;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Duplicate(pack_hist_ub, static_cast<int32_t>(0), pack_hist_elems);
        AscendC::PipeBarrier<PIPE_V>();

        const int32_t pack_work_items =
            pack_chunk_count > config.rank_count
                ? pack_chunk_count
                : config.rank_count;
        const uint32_t pack_thread_count = static_cast<uint32_t>(IntMin(
            kSourceDupSimtMaxThreads, ((pack_work_items + 31) / 32) * 32));
        asc_vf_call<PackSourceDupByDestSimt>(
            dim3(pack_thread_count),
            (__ubuf__ int32_t*)group_send_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)compact_loff_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)meta_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)raw_group_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)raw_loff_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)pack_hist_ub.GetPhyAddr(),
            raw_group_count,
            raw_loff_count,
            config.rank_count,
            ub_layout.pack_chunk_groups,
            ub_layout.pack_hist_row_elems,
            static_cast<int32_t>(pack_thread_count));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    for (int32_t i = 0; i < config.rank_count * 2; ++i) {
        count_ub.SetValue(
            i,
            count_ub.GetValue(i) + tile_count_ub.GetValue(i));
    }
    const int32_t tile_count_gm_offset =
        workspace_layout.tile_count_offset +
        (config.core_idx * partition.max_tiles_per_core + tile_idx) *
            ub_layout.tile_count_row_elems;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyExtParams tile_count_copy_params(
        1,
        static_cast<uint32_t>(config.rank_count * 2 * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPad(
        dup_workspace_gm[tile_count_gm_offset],
        tile_count_ub,
        tile_count_copy_params);

    if (tile_has_duplicates) {
        const int32_t tile_slot =
            config.core_idx * partition.max_tiles_per_core + tile_idx;
        AscendC::DataCopyExtParams group_copy_params(
            1,
            static_cast<uint32_t>(group_tile_prefix * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPad(
            dup_key_workspace_gm[
                workspace_layout.prebuilt_group_offset +
                tile_slot * workspace_layout.prebuilt_group_stride],
            raw_group_ub,
            group_copy_params);
        AscendC::DataCopyExtParams loff_copy_params(
            1,
            static_cast<uint32_t>(loff_tile_prefix * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPad(
            dup_key_workspace_gm[
                workspace_layout.prebuilt_loff_offset +
                tile_slot * workspace_layout.prebuilt_loff_stride],
            raw_loff_ub,
            loff_copy_params);
    }
    // The following tile reuses these UB regions from both V and MTE2. Wait
    // until all count, payload, and dst writebacks have consumed their input.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline void CountAndPrebuildSourceDupTiles(
    const SourceDupConfig& config,
    const SourceDupPartition& partition,
    const SourceDupUbLayout& ub_layout,
    const SourceDupWorkspaceLayout& workspace_layout,
    PlanningResource& resource,
    AscendC::GlobalTensor<int32_t>& dst_gm,
    AscendC::GlobalTensor<int32_t>& dup_workspace_gm,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm) {
    AscendC::LocalTensor<int32_t> count_ub = resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::Duplicate(
        count_ub,
        static_cast<int32_t>(0),
        config.rank_count * 2);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    int32_t tile_idx = 0;
    for (int32_t token_base = partition.token_begin;
         token_base < partition.token_end;
         token_base += config.tokens_per_tile, ++tile_idx) {
        ProcessSourceDupTile(
            config,
            partition,
            ub_layout,
            workspace_layout,
            resource,
            dst_gm,
            dup_workspace_gm,
            dup_key_workspace_gm,
            token_base,
            tile_idx);
    }

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    StoreContiguousGmThroughUb(
        dup_workspace_gm,
        count_ub,
        config.core_idx * config.rank_count * 2,
        config.rank_count * 2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void PublishSourceDupPayloads(
    const SourceDupConfig& config,
    const SourceDupPartition& partition,
    const SourceDupUbLayout& ub_layout,
    const SourceDupWorkspaceLayout& workspace_layout,
    PlanningResource& resource,
    __gm__ int32_t* shmem_base,
    AscendC::GlobalTensor<int32_t>& dup_workspace_gm,
    AscendC::GlobalTensor<int32_t>& dup_key_workspace_gm,
    int32_t payload_done_flag_off,
    int32_t dup_groups_scratch_off,
    int32_t dup_loffs_scratch_off) {
    AscendC::LocalTensor<int32_t> count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> staging_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.count_bytes));
    AscendC::LocalTensor<int32_t> meta_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.meta_offset));
    AscendC::LocalTensor<int32_t> group_send_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.group_send_offset));
    AscendC::LocalTensor<int32_t> tile_count_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(ub_layout.tile_count_offset));

    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::DataCopyExtParams count_copy_params(
        1,
        static_cast<uint32_t>(config.rank_count * 2 * kInt32Bytes),
        0,
        0,
        0);
    AscendC::DataCopyPadExtParams<int32_t> count_pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(
        count_ub,
        dup_key_workspace_gm[config.core_idx * config.rank_count * 2],
        count_copy_params,
        count_pad_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    int32_t tile_idx = 0;
    for (int32_t token_base = partition.token_begin;
         token_base < partition.token_end;
         token_base += config.tokens_per_tile, ++tile_idx) {
        const int32_t tile_count_gm_offset =
            workspace_layout.tile_count_offset +
            (config.core_idx * partition.max_tiles_per_core + tile_idx) *
                ub_layout.tile_count_row_elems;
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::DataCopyPad(
            tile_count_ub,
            dup_workspace_gm[tile_count_gm_offset],
            count_copy_params,
            count_pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        int32_t group_tile_prefix = 0;
        int32_t loff_tile_prefix = 0;
        for (int32_t dest = 0; dest < config.rank_count; ++dest) {
            const int32_t meta_offset = dest * 5;
            const int32_t count_offset = dest * 2;
            const int32_t group_count = tile_count_ub.GetValue(count_offset);
            const int32_t loff_count = tile_count_ub.GetValue(count_offset + 1);
            if (group_count > 0) {
                group_tile_prefix = AlignUpInt32(group_tile_prefix);
                loff_tile_prefix = AlignUpInt32(loff_tile_prefix);
            }
            meta_ub.SetValue(meta_offset, group_tile_prefix);
            meta_ub.SetValue(meta_offset + 1, group_tile_prefix);
            meta_ub.SetValue(meta_offset + 3, loff_tile_prefix);
            meta_ub.SetValue(meta_offset + 4, loff_tile_prefix);
            group_tile_prefix += group_count * 3;
            loff_tile_prefix += loff_count;
        }

        const int32_t tile_slot =
            config.core_idx * partition.max_tiles_per_core + tile_idx;
        if (group_tile_prefix <= 0) {
            continue;
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::DataCopyExtParams group_load_params(
            1,
            static_cast<uint32_t>(group_tile_prefix * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> group_pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(
            group_send_ub,
            dup_key_workspace_gm[
                workspace_layout.prebuilt_group_offset +
                tile_slot * workspace_layout.prebuilt_group_stride],
            group_load_params,
            group_pad_params);
        AscendC::DataCopyExtParams loff_load_params(
            1,
            static_cast<uint32_t>(loff_tile_prefix * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> loff_pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(
            staging_ub,
            dup_key_workspace_gm[
                workspace_layout.prebuilt_loff_offset +
                tile_slot * workspace_layout.prebuilt_loff_stride],
            loff_load_params,
            loff_pad_params);

        for (int32_t dest = 0; dest < config.rank_count; ++dest) {
            const int32_t meta_offset = dest * 5;
            const int32_t count_offset = dest * 2;
            const int32_t group_count = tile_count_ub.GetValue(count_offset);
            if (group_count <= 0) {
                continue;
            }
            const int32_t global_loff_base =
                count_ub.GetValue(count_offset + 1);
            meta_ub.SetValue(meta_offset + 1, global_loff_base);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        const int32_t threads_per_dest = IntMin(
            32,
            config.rank_count < kSourceDupSimtMaxThreads
                ? kSourceDupSimtMaxThreads / config.rank_count
                : 1);
        const int32_t logical_adjust_threads =
            config.rank_count * threads_per_dest;
        const uint32_t adjust_thread_count = static_cast<uint32_t>(IntMin(
            kSourceDupSimtMaxThreads,
            ((logical_adjust_threads + 31) / 32) * 32));
        asc_vf_call<AdjustSourceDupGroupLoffsSimt>(
            dim3(adjust_thread_count),
            (__ubuf__ int32_t*)group_send_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)tile_count_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)meta_ub.GetPhyAddr(),
            config.rank_count,
            threads_per_dest);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

        for (int32_t dest = 0; dest < config.rank_count; ++dest) {
            const int32_t meta_offset = dest * 5;
            const int32_t group_start = meta_ub.GetValue(meta_offset);
            const int32_t count_offset = dest * 2;
            const int32_t group_count = tile_count_ub.GetValue(count_offset);
            if (group_count <= 0) {
                continue;
            }

            const int32_t group_elems = group_count * 3;
            const int32_t global_group_pos =
                count_ub.GetValue(count_offset);
            AscendC::DataCopyExtParams group_store_params(
                1,
                static_cast<uint32_t>(group_elems * kInt32Bytes),
                0,
                0,
                0);
            AscendC::GlobalTensor<int32_t> remote_shmem_gm;
            remote_shmem_gm.SetGlobalBuffer(
                static_cast<__gm__ int32_t*>(
                    aclshmem_ptr(shmem_base, dest)));
            AscendC::DataCopyPad(
                remote_shmem_gm[
                    dup_groups_scratch_off + global_group_pos * 3],
                group_send_ub[group_start],
                group_store_params);
            count_ub.SetValue(
                count_offset,
                global_group_pos + group_count);

            const int32_t loff_start = meta_ub.GetValue(meta_offset + 3);
            const int32_t loff_count = tile_count_ub.GetValue(count_offset + 1);
            const int32_t global_loff_pos =
                count_ub.GetValue(count_offset + 1);
            AscendC::DataCopyExtParams loff_copy_params(
                1,
                static_cast<uint32_t>(loff_count * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPad(
                remote_shmem_gm[dup_loffs_scratch_off + global_loff_pos],
                staging_ub[loff_start],
                loff_copy_params);
            count_ub.SetValue(
                count_offset + 1,
                global_loff_pos + loff_count);
        }
        // The next tile reloads the payload into the same UB regions.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    AscendC::SyncAll<true>();
    PublishSourceDupPayloadDoneFlags(
        config,
        shmem_base,
        count_ub,
        payload_done_flag_off);
}

}  // namespace

__aicore__ __attribute__((noinline)) void RunCombinePlanning(
    GM_ADDR dst, GM_ADDR groups, GM_ADDR dups, GM_ADDR counts,
    ascend_moonep::PlanningTilingData tiling, uint64_t begin, uint64_t baseEnd);

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void planning_kernel(
    GM_ADDR topk_idx,
    GM_ADDR symmetric_buffer,
    GM_ADDR dst,
    GM_ADDR cu_seqlens,
    GM_ADDR experts_to_copy,
    GM_ADDR dup_groups,
    GM_ADDR dup_loffs,
    GM_ADDR dup_counts,
    GM_ADDR tpe_all,
    GM_ADDR tpe_prefix,
    GM_ADDR balance,
    GM_ADDR z_tensor,
    GM_ADDR alloc_tensor,
    GM_ADDR alloc_cumsum,
    GM_ADDR expert_offsets,
    GM_ADDR core_expert_counts,
    GM_ADDR dup_workspace,
    GM_ADDR dup_key_workspace,
    ascend_moonep::PlanningTilingData tiling,
    uint64_t ffts_addr) {
    (void)ffts_addr;
    const uint64_t planning_begin = tiling.combine_mode ? GetSystemCycle() : 0U;
    // Hand-built/synthetic dst uses this SAME entry point, skipping only base routing.
    if (tiling.combine_mode == 2U) {
        RunCombinePlanning(dst, dup_groups, dup_loffs, dup_counts,
                           tiling, planning_begin, planning_begin);
        return;
    }

    const int32_t core_idx = get_block_idx() * get_subblockdim() + get_subblockid();
    const int32_t core_num = get_block_num() * get_subblockdim();

    const int32_t rank = static_cast<int32_t>(tiling.rank);
    const int32_t R = static_cast<int32_t>(tiling.world_size);
    const int32_t S = static_cast<int32_t>(tiling.num_tokens);
    const int32_t K = static_cast<int32_t>(tiling.num_topk);
    const int32_t E = static_cast<int32_t>(tiling.num_experts);
    const int32_t NvS =
        static_cast<int32_t>(tiling.num_max_tokens_per_rank);
    const int32_t B = static_cast<int32_t>(tiling.num_prefetch_slots);
    const int32_t epn = E / R;
    const int32_t N = S * K;
    const int32_t tpe_off =
        static_cast<int32_t>(tiling.symmetric_tpe_offset);
    const int32_t dup_count_inbox_off =
        static_cast<int32_t>(tiling.symmetric_dup_rank_counts_offset);
    const int32_t dup_prefix_inbox_off =
        dup_count_inbox_off + R * kSourceDupInboxSlotElems;
    const int32_t dup_payload_done_flag_off =
        dup_prefix_inbox_off + R * kSourceDupInboxSlotElems;
    const int32_t dup_groups_scratch_off =
        dup_payload_done_flag_off + ((R + 7) / 8) * 8;
    const int32_t dup_loffs_scratch_off =
        dup_groups_scratch_off + ((NvS * 3 + 7) / 8) * 8;
    // The secondary scratch is the last region with an unchanged start; the host sizes it as 8*NvS int32s.
    const int32_t source_dup_tokens_per_tile =
        static_cast<int32_t>(tiling.source_dup_tokens_per_tile);
    const int32_t owner_reduce_tile_elems =
        static_cast<int32_t>(tiling.owner_reduce_tile_elems);
    const int32_t route_core_num = IntMin(core_num, kMaxPlanningCores);
    const int32_t items_per_core = (N + route_core_num - 1) / route_core_num;
    const int32_t begin = core_idx * items_per_core;
    const int32_t end = IntMin(begin + items_per_core, N);
    AscendC::GlobalTensor<int32_t> topk_gm;
    AscendC::GlobalTensor<int32_t> tpe_gm;
    AscendC::GlobalTensor<int32_t> dst_gm;
    AscendC::GlobalTensor<int32_t> cu_gm;
    AscendC::GlobalTensor<int32_t> experts_to_copy_gm;
    AscendC::GlobalTensor<int32_t> dup_groups_gm;
    AscendC::GlobalTensor<int32_t> dup_loffs_gm;
    AscendC::GlobalTensor<int32_t> dup_counts_gm;
    AscendC::GlobalTensor<int32_t> tpe_all_gm;
    AscendC::GlobalTensor<int32_t> tpe_prefix_gm;
    AscendC::GlobalTensor<int32_t> balance_gm;
    AscendC::GlobalTensor<int32_t> z_gm;
    AscendC::GlobalTensor<int32_t> alloc_gm;
    AscendC::GlobalTensor<int32_t> alloc_cumsum_gm;
    AscendC::GlobalTensor<int32_t> expert_offsets_gm;
    AscendC::GlobalTensor<int32_t> core_expert_counts_gm;
    AscendC::GlobalTensor<int32_t> cursor_worker_scratch_gm;
    AscendC::GlobalTensor<int32_t> dup_workspace_gm;
    AscendC::GlobalTensor<int32_t> dup_key_workspace_gm;
    AscendC::GlobalTensor<int32_t> local_shmem_gm;

    topk_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topk_idx));
    dst_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dst));
    cu_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cu_seqlens));
    experts_to_copy_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(experts_to_copy));
    dup_groups_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_groups));
    dup_loffs_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_loffs));
    dup_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_counts));
    tpe_all_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tpe_all));
    tpe_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(tpe_all) + rank * E);
    tpe_prefix_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tpe_prefix));
    balance_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(balance));
    z_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(z_tensor));
    alloc_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(alloc_tensor));
    alloc_cumsum_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(alloc_cumsum));
    expert_offsets_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expert_offsets));
    core_expert_counts_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(core_expert_counts));
    cursor_worker_scratch_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(core_expert_counts) +
        kMaxPlanningCores * E);
    dup_workspace_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_workspace));
    dup_key_workspace_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_key_workspace));

    __gm__ int32_t* shmem_base = reinterpret_cast<__gm__ int32_t*>(symmetric_buffer);
    local_shmem_gm.SetGlobalBuffer(shmem_base);

    using ArchTag = PlanningAscend950;
    Catlass::Arch::Resource<ArchTag> resource;
    AscendC::LocalTensor<int32_t> work_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);

    // The first cross-rank barrier also publishes the mailbox initialization.
    // Count and prefix messages each own one 32-byte slot per peer; payload
    // completion uses one flag per source rank after all local cores finish.
    FillContiguousGm(
        local_shmem_gm,
        work_ub,
        dup_count_inbox_off,
        R * kSourceDupInboxSlotElems * 2,
        -1,
        core_idx,
        core_num);
    FillContiguousGm(
        local_shmem_gm,
        work_ub,
        dup_payload_done_flag_off,
        R,
        0,
        core_idx,
        core_num);

    int32_t tpe_elems_per_core = E / core_num;
    int32_t tpe_tail_cores = E - tpe_elems_per_core * core_num;
    int32_t tpe_begin = core_idx * tpe_elems_per_core + IntMin(core_idx, tpe_tail_cores);
    int32_t tpe_assigned = tpe_elems_per_core + (core_idx < tpe_tail_cores ? 1 : 0);
    int32_t tpe_end = tpe_begin + tpe_assigned;

    // Stage 1: build one expert histogram per route-processing core. The same
    // rows later become exclusive per-core expert prefixes for dst ordinals.
    if (core_idx < route_core_num) {
        const int32_t histogram_bytes = AlignUpBytes(E * kInt32Bytes, 32);
        AscendC::LocalTensor<int32_t> route_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(histogram_bytes));
        AscendC::Duplicate(work_ub, static_cast<int32_t>(0), E);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        for (int32_t base = begin; base < end; base += kRouteHistogramTileElems) {
            const int32_t len =
                IntMin(kRouteHistogramTileElems, end - base);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
            AscendC::DataCopyPad(
                route_tile_ub,
                topk_gm[base],
                copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            const uint32_t thread_count = static_cast<uint32_t>(IntMin(
                ((len + 31) / 32) * 32,
                kSourceDupSimtMaxThreads));
            asc_vf_call<BuildRouteHistogramSimt>(
                dim3(thread_count),
                (__ubuf__ int32_t*)route_tile_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)work_ub.GetPhyAddr(),
                len);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        for (int32_t base = 0; base < E; base += kFillTileElems) {
            const int32_t len = IntMin(kFillTileElems, E - base);
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPad(
                core_expert_counts_gm[core_idx * E + base],
                work_ub[base],
                copy_params);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
    AscendC::SyncAll<true>();

    // Reduce [route_core_num, E] by expert slices. A 64-expert slice is one
    // 256-byte vector width, while multiple cores remove the core-0 bottleneck.
    const int32_t tpe_reduce_core_num = IntMin(
        core_num, (E + kTpeReduceExpertsPerCore - 1) / kTpeReduceExpertsPerCore);

    if (core_idx < tpe_reduce_core_num) {
        constexpr int32_t kTpeReduceTmpOffsetBytes =
            ((kFillTileElems * kInt32Bytes + 31) / 32) * 32;
        AscendC::LocalTensor<int32_t> tpe_reduce_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(kTpeReduceTmpOffsetBytes));
        static_assert(
            kTpeReduceTmpOffsetBytes +
                    kMaxPlanningCores * kTpeReduceRegRowBytes <=
                kPlanningUbSizeBytes);

        // EVENT_ID1 owns tpe_reduce_ub. The next MTE2 load must not overwrite
        // it until the previous Reg VF has finished reading the matrix.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        for (int32_t base = core_idx * kTpeReduceExpertsPerCore;
             base < E;
             base += tpe_reduce_core_num * kTpeReduceExpertsPerCore) {
            const int32_t len = IntMin(kTpeReduceExpertsPerCore, E - base);
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};

            const int32_t aligned_row_bytes =
                ((len * kInt32Bytes + 31) / 32) * 32;
            AscendC::DataCopyExtParams matrix_copy_params(
                static_cast<uint16_t>(route_core_num),
                static_cast<uint32_t>(len * kInt32Bytes),
                static_cast<uint32_t>((E - len) * kInt32Bytes),
                static_cast<uint32_t>(
                    (kTpeReduceRegRowBytes - aligned_row_bytes) / 32),
                0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::DataCopyPad(
                tpe_reduce_ub,
                core_expert_counts_gm[base],
                matrix_copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<ReduceTpeRowsReg>(
                (__ubuf__ int32_t*)work_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)tpe_reduce_ub.GetPhyAddr(),
                static_cast<uint32_t>(route_core_num),
                static_cast<uint32_t>(len),
                static_cast<uint32_t>(kTpeReduceExpertsPerCore));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                local_shmem_gm[tpe_off + base],
                work_ub,
                copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }
    aclshmemx_barrier_all_vec();

    // Stage 2: allgather tpe by contiguous (peer, expert-tile) tasks. This
    // avoids issuing one tiny copy per peer from every core when E is small.
    constexpr int32_t kTpeGatherTileBytes = ((kFillTileElems * kInt32Bytes + 31) / 32) * 32;
    static_assert(kTpeGatherTileBytes * 2 <= kPlanningUbSizeBytes);
    AscendC::LocalTensor<int32_t> tpe_gather_pingpong_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    const int32_t tpe_tiles_per_peer = (E + kFillTileElems - 1) / kFillTileElems;
    const int32_t tpe_gather_task_count = R * tpe_tiles_per_peer;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    int32_t local_task_idx = 0;
    for (int32_t task_idx = core_idx; task_idx < tpe_gather_task_count;
         task_idx += core_num, ++local_task_idx) {
        const int32_t peer = task_idx / tpe_tiles_per_peer;
        const int32_t tile_idx = task_idx - peer * tpe_tiles_per_peer;
        const int32_t base = tile_idx * kFillTileElems;
        const int32_t len = IntMin(kFillTileElems, E - base);
        const int32_t buffer_idx = local_task_idx & 1;
        const auto event_id = buffer_idx == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<int32_t> tpe_gather_ub =
            tpe_gather_pingpong_ub[buffer_idx * kFillTileElems];
        AscendC::GlobalTensor<int32_t> peer_shmem_gm;
        peer_shmem_gm.SetGlobalBuffer(static_cast<__gm__ int32_t*>(aclshmem_ptr(shmem_base, peer)));
        AscendC::DataCopyExtParams copy_params(
            1,
            static_cast<uint32_t>(len * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        AscendC::DataCopyPad(
            tpe_gather_ub,
            peer_shmem_gm[tpe_off + base],
            copy_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);

        AscendC::DataCopyPad(
            tpe_all_gm[peer * E + base],
            tpe_gather_ub,
            copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    // Keep the V-produced fill tile separate from both gather buffers. The
    // gather completion events order MTE3 against MTE2, not against PIPE_V.
    constexpr int32_t kZFillUbOffsetBytes = kTpeGatherTileBytes * 2;
    static_assert(kZFillUbOffsetBytes + kFillTileElems * kInt32Bytes <= kPlanningUbSizeBytes);
    AscendC::LocalTensor<int32_t> z_fill_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(static_cast<uint32_t>(kZFillUbOffsetBytes));
    FillContiguousGm(z_gm, z_fill_ub, 0, R * R, 0, core_idx, core_num);

    AscendC::SyncAll<true>();

    // balance[owner] and the home alloc row are independent per owner rank.
    {
        const int32_t reduce_tile_elems = owner_reduce_tile_elems;
        const int32_t reduce_row_bytes = ((reduce_tile_elems * kInt32Bytes + 31) / 32) * 32;
        const int32_t reduce_row_elems = reduce_row_bytes / kInt32Bytes;
        const int32_t reduce_matrix_bytes = R * reduce_row_bytes;
        // Host selects the tile after reserving the tile total and maximum
        // per-core balance output, keeping all three regions disjoint.
        AscendC::LocalTensor<int32_t> tpe_owner_matrix_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> tile_total_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(reduce_matrix_bytes));
        AscendC::LocalTensor<int32_t> balance_out_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(
                    reduce_matrix_bytes + kOwnerReduceTotalBytes));

        const int32_t owners_per_core = R / core_num;
        const int32_t owner_tail_cores = R - owners_per_core * core_num;
        const int32_t owner_begin =
            core_idx * owners_per_core + IntMin(core_idx, owner_tail_cores);
        const int32_t owner_assigned = owners_per_core + (core_idx < owner_tail_cores ? 1 : 0);
        const int32_t owner_end = owner_begin + owner_assigned;

        // Each core accumulates token counts for its owner_assigned ranks.
        for (int32_t owner = owner_begin; owner < owner_end; ++owner) {
            int32_t total = 0;
            int32_t e_begin = owner * epn;
            int32_t e_end = e_begin + epn;

            for (int32_t base = e_begin; base < e_end; base += reduce_tile_elems) {
                const int32_t len = IntMin(reduce_tile_elems, e_end - base);
                const int32_t aligned_row_bytes =
                    ((len * kInt32Bytes + 31) / 32) * 32;
                AscendC::DataCopyExtParams matrix_copy_params(
                    static_cast<uint16_t>(R),
                    static_cast<uint32_t>(len * kInt32Bytes),
                    static_cast<uint32_t>((E - len) * kInt32Bytes),
                    static_cast<uint32_t>(
                        (reduce_row_bytes - aligned_row_bytes) / 32),
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{
                    false, 0, 0, 0};
                AscendC::DataCopyPad(
                    tpe_owner_matrix_ub,
                    tpe_all_gm[base],
                    matrix_copy_params,
                    pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                asc_vf_call<ReduceTpeOwnerRowsReg>(
                    (__ubuf__ int32_t*)tpe_owner_matrix_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)tile_total_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)tpe_owner_matrix_ub.GetPhyAddr(),
                    static_cast<uint32_t>(R),
                    static_cast<uint32_t>(len),
                    static_cast<uint32_t>(reduce_row_elems));

                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                total += tile_total_ub.GetValue(0);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                StoreContiguousGmThroughUb(
                    alloc_gm,
                    tpe_owner_matrix_ub,
                    owner * E + base,
                    len);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            const int32_t owner_local = owner - owner_begin;
            balance_out_ub.SetValue(owner_local, total - N);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        StoreContiguousGmThroughUb(balance_gm, balance_out_ub, owner_begin, owner_assigned);
    }
    AscendC::SyncAll<true>();

    if (core_idx == 0) {
        // The previous work_ub users have completed and synchronized here, so
        // the balance planner can reuse UB from offset 0 with a semantic name.
        AscendC::LocalTensor<int32_t> balance_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        {
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(R * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
            AscendC::DataCopyPad(balance_ub, balance_gm[0], copy_params, pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        }

        for (int32_t iter = 0; iter < R * R; ++iter) {
            int32_t surplus_rank = 0;
            int32_t deficit_rank = 0;
            int32_t max_balance = balance_ub.GetValue(0);
            int32_t min_balance = balance_ub.GetValue(0);
            for (int32_t r = 1; r < R; ++r) {
                int32_t v = balance_ub.GetValue(r);
                if (v > max_balance) {
                    max_balance = v;
                    surplus_rank = r;
                }
                if (v < min_balance) {
                    min_balance = v;
                    deficit_rank = r;
                }
            }
            if (max_balance <= 0 || min_balance >= 0) {
                break;
            }
            int32_t move_tokens = -min_balance;
            AscendC::WriteGmByPassDCache<int32_t>(
                z_gm.GetPhyAddr(surplus_rank * R + deficit_rank),
                move_tokens);
            balance_ub.SetValue(surplus_rank, max_balance - move_tokens);
            balance_ub.SetValue(deficit_rank, 0);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    AscendC::SyncAll<true>();

    {
        const int32_t s_alloc_row_bytes = ((epn * kInt32Bytes + 31) / 32) * 32;
        const int32_t s_alloc_row_elems = s_alloc_row_bytes / kInt32Bytes;
        const int32_t s_alloc_elems = s_alloc_row_elems * R;
        const int32_t s_alloc_bytes = s_alloc_row_bytes * R;
        AscendC::LocalTensor<int32_t> s_alloc_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> quota_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(static_cast<uint32_t>(s_alloc_bytes));

        for (int32_t owner = core_idx; owner < R; owner += core_num) {
            const int32_t expert_base = owner * epn;
            const int32_t owner_row_offset = owner * s_alloc_row_elems;

            AscendC::Duplicate(s_alloc_ub, static_cast<int32_t>(0), s_alloc_elems);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

            {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams copy_params(
                    1,
                    static_cast<uint32_t>(epn * kInt32Bytes),
                    0,
                    0,
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                AscendC::DataCopyPad(
                    s_alloc_ub[owner_row_offset],
                    alloc_gm[owner * E + expert_base],
                    copy_params,
                    pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            }

            {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams copy_params(
                    1,
                    static_cast<uint32_t>(R * kInt32Bytes),
                    0,
                    0,
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                AscendC::DataCopyPad(quota_ub, z_gm[owner * R], copy_params, pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            }

            // Each owner owns a disjoint expert range, so different cores write
            // disjoint alloc_gm columns.  Inside one owner, follow MoonEP's
            // greedy policy: largest remaining destination quota first, then
            // largest remaining local expert segment.
            for (int32_t iter = 0; iter < R + epn; ++iter) {
                int32_t target_rank = 0;
                int32_t max_quota = quota_ub.GetValue(0);
                for (int32_t r = 1; r < R; ++r) {
                    const int32_t quota = quota_ub.GetValue(r);
                    if (quota > max_quota) {
                        max_quota = quota;
                        target_rank = r;
                    }
                }
                if (max_quota <= 0) {
                    break;
                }

                int32_t selected_local_expert = 0;
                int32_t max_remaining = s_alloc_ub.GetValue(owner_row_offset);
                for (int32_t local_expert = 1; local_expert < epn; ++local_expert) {
                    const int32_t remaining = s_alloc_ub.GetValue(owner_row_offset + local_expert);
                    if (remaining > max_remaining) {
                        max_remaining = remaining;
                        selected_local_expert = local_expert;
                    }
                }
                if (max_remaining <= 0) {
                    break;
                }

                const int32_t take = IntMin(max_remaining, max_quota);
                const int32_t owner_offset = owner_row_offset + selected_local_expert;
                const int32_t target_offset =
                    target_rank * s_alloc_row_elems + selected_local_expert;
                s_alloc_ub.SetValue(
                    target_offset,
                    s_alloc_ub.GetValue(target_offset) + take);
                s_alloc_ub.SetValue(owner_offset, max_remaining - take);
                quota_ub.SetValue(target_rank, max_quota - take);
            }

            // One owner contributes one compact expert segment to every
            // destination row. UB rows are 32-byte aligned; DataCopyPad keeps
            // only the logical bytes when scattering them into compact GM.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams alloc_store_params(
                static_cast<uint16_t>(R),
                static_cast<uint32_t>(epn * kInt32Bytes),
                0,
                static_cast<uint32_t>((E - epn) * kInt32Bytes),
                0);
            AscendC::DataCopyPad(
                alloc_gm[expert_base],
                s_alloc_ub,
                alloc_store_params);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            const uint32_t cumsum_thread_count = static_cast<uint32_t>(
                IntMin(((E + 31) / 32) * 32, kSourceDupSimtMaxThreads));
            asc_vf_call<BuildAllocCumsumSimt>(
                dim3(cumsum_thread_count),
                (__ubuf__ int32_t*)s_alloc_ub.GetPhyAddr(),
                reinterpret_cast<__gm__ int32_t*>(alloc_cumsum),
                R,
                epn,
                s_alloc_row_elems,
                expert_base,
                static_cast<int32_t>(cumsum_thread_count));
        }
    }
    AscendC::SyncAll<true>();
    // The selected experts are moved from their original E slots to the B
    // trailing slots in one in-place E+B count row. The scratch rows hold
    // Top-B reduction candidates, then are reused by prefix/offset building.
    PrefetchUbLayout prefetch_layout;
    InitPrefetchUbLayout(E, B, prefetch_layout);
    AscendC::LocalTensor<int32_t> group_counts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(prefetch_layout.group_counts_offset));
    AscendC::LocalTensor<int32_t> selected_experts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(prefetch_layout.selected_experts_offset));
    AscendC::LocalTensor<int32_t> scratch_counts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(prefetch_layout.scratch_counts_offset));
    AscendC::LocalTensor<int32_t> scratch_experts_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(prefetch_layout.scratch_experts_offset));
    const int32_t prefetch_thread_count = SelectPrefetchSimtThreadCount(E);
    for (int32_t d = core_idx; d < R; d += core_num) {
        AscendC::DataCopyExtParams copy_params(
            1,
            static_cast<uint32_t>(E * kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(
            group_counts_ub,
            alloc_gm[d * E],
            copy_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        const int32_t local_expert_begin = d * epn;
        const int32_t local_expert_end = local_expert_begin + epn;
        asc_vf_call<SelectPrefetchExpertsSimt>(
            dim3(static_cast<uint32_t>(prefetch_thread_count)),
            (__ubuf__ int32_t*)group_counts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)selected_experts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)scratch_counts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)scratch_experts_ub.GetPhyAddr(),
            E,
            B,
            local_expert_begin,
            local_expert_end,
            prefetch_thread_count);
        if ((B * kInt32Bytes) % 32 == 0) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            StoreContiguousGmThroughUb(
                experts_to_copy_gm,
                selected_experts_ub,
                d * B,
                B);
        } else {
            // Packed B-wide rows may share a 32-byte GM block. Concurrent
            // padded MTE3 stores from different cores would then overwrite
            // neighboring rows, so use exact scalar stores for this tail case.
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (int32_t slot = 0; slot < B; ++slot) {
                AscendC::WriteGmByPassDCache<int32_t>(
                    experts_to_copy_gm.GetPhyAddr(d * B + slot),
                    selected_experts_ub.GetValue(slot));
            }
        }

        asc_vf_call<BuildGroupPrefixSimt>(
            dim3(static_cast<uint32_t>(prefetch_thread_count)),
            (__ubuf__ int32_t*)group_counts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)scratch_counts_ub.GetPhyAddr(),
            E + B,
            prefetch_thread_count);
        if (d == rank) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            StoreContiguousGmThroughUb(cu_gm, group_counts_ub, 0, E + B);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }

        asc_vf_call<ConvertGroupPrefixToExpertOffsetsSimt>(
            dim3(static_cast<uint32_t>(prefetch_thread_count)),
            (__ubuf__ int32_t*)group_counts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)selected_experts_ub.GetPhyAddr(),
            (__ubuf__ int32_t*)scratch_counts_ub.GetPhyAddr(),
            E,
            B,
            prefetch_thread_count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        StoreContiguousGmThroughUb(
            expert_offsets_gm,
            group_counts_ub,
            d * E,
            E);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if ((B * kInt32Bytes) % 32 != 0) {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    CleanInvalidDataCache();

    // tpe_prefix[e] = sum_{r < rank} tpe_all[r, e]. Load all contributing
    // rows for one expert tile and keep the cross-rank accumulation in vector
    // registers, so the loop does not materialize every intermediate sum.
    if (rank == 0) {
        FillContiguousGm(
            tpe_prefix_gm,
            work_ub,
            0,
            E,
            0,
            core_idx,
            core_num);
    } else {
        constexpr int32_t kTpePrefixMatrixOffsetBytes = kTpeReduceRegRowBytes;
        const int32_t tpe_prefix_row_bytes = IntMin(
            kTpeReduceRegRowBytes,
            ((kPlanningUbSizeBytes - kTpePrefixMatrixOffsetBytes) / R / 32) * 32);
        const int32_t tpe_prefix_tile_elems = tpe_prefix_row_bytes / kInt32Bytes;
        AscendC::LocalTensor<int32_t> tpe_prefix_matrix_ub = resource.ubBuf.template GetBufferByByte<int32_t>(
            static_cast<uint32_t>(kTpePrefixMatrixOffsetBytes));
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        for (int32_t base = tpe_begin;
             base < tpe_end;
             base += tpe_prefix_tile_elems) {
            const int32_t len =
                IntMin(tpe_prefix_tile_elems, tpe_end - base);
            const int32_t aligned_row_bytes =
                AlignUpBytes(len * kInt32Bytes, 32);
            AscendC::DataCopyExtParams matrix_copy_params(
                static_cast<uint16_t>(rank),
                static_cast<uint32_t>(len * kInt32Bytes),
                static_cast<uint32_t>((E - len) * kInt32Bytes),
                static_cast<uint32_t>(
                    (tpe_prefix_row_bytes - aligned_row_bytes) / 32),
                0);
            AscendC::DataCopyExtParams output_copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::DataCopyPad(
                tpe_prefix_matrix_ub,
                tpe_all_gm[base],
                matrix_copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<ReduceTpeRowsReg>(
                (__ubuf__ int32_t*)work_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)tpe_prefix_matrix_ub.GetPhyAddr(),
                static_cast<uint32_t>(rank),
                static_cast<uint32_t>(len),
                static_cast<uint32_t>(tpe_prefix_row_bytes / kInt32Bytes));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                tpe_prefix_gm[base],
                work_ub,
                output_copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }
    AscendC::SyncAll<true>();

    // Split 32-byte-aligned expert columns across AIV cores. If an expert row
    // itself is not 32-byte aligned, keep one worker so two MTE3 writers cannot
    // touch the same GM cache line at a row boundary.
    constexpr int32_t kExpertsPerDataBlock = 32 / kInt32Bytes;
    const int32_t cursor_expert_block_count =
        (E + kExpertsPerDataBlock - 1) / kExpertsPerDataBlock;
    const int32_t cursor_worker_num = E % kExpertsPerDataBlock == 0
        ? IntMin(route_core_num, cursor_expert_block_count)
        : 1;
    const int32_t cursor_blocks_per_worker =
        cursor_expert_block_count / cursor_worker_num;
    const int32_t cursor_block_tail_workers =
        cursor_expert_block_count -
        cursor_blocks_per_worker * cursor_worker_num;
    const int32_t cursor_block_begin =
        core_idx * cursor_blocks_per_worker +
        IntMin(core_idx, cursor_block_tail_workers);
    const int32_t cursor_block_count =
        cursor_blocks_per_worker +
        (core_idx < cursor_block_tail_workers ? 1 : 0);
    const int32_t cursor_expert_begin = IntMin(
        E,
        cursor_block_begin * kExpertsPerDataBlock);
    const int32_t cursor_expert_end = IntMin(
        E,
        (cursor_block_begin + cursor_block_count) *
            kExpertsPerDataBlock);
    const int32_t cursor_expert_count =
        cursor_expert_end - cursor_expert_begin;

    // Publish one total per worker into a dedicated, 32-byte-strided GM row.
    // This scratch is outside the source-dup workspace and is fully overwritten
    // on every launch before any worker reads it.
    if (core_idx < cursor_worker_num) {
        constexpr int32_t kTotalTileBytes =
            kFillTileElems * kInt32Bytes;
        constexpr int32_t kThreadTotalsOffset = kTotalTileBytes;
        constexpr int32_t kTileTotalOffset =
            kThreadTotalsOffset + kCoreCursorThreadPrefixBytes;
        static_assert(
            kTileTotalOffset + kCoreCursorCarryBytes <=
            kPlanningUbSizeBytes);

        AscendC::LocalTensor<int32_t> tpe_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> thread_totals_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(kThreadTotalsOffset));
        AscendC::LocalTensor<int32_t> tile_total_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(kTileTotalOffset));
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};

        const bool accumulate_total_tiles =
            cursor_expert_count > kFillTileElems;
        int32_t worker_total = 0;
        for (int32_t base = cursor_expert_begin;
             base < cursor_expert_end;
             base += kFillTileElems) {
            const int32_t len =
                IntMin(kFillTileElems, cursor_expert_end - base);
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPad(
                tpe_tile_ub,
                tpe_gm[base],
                copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            const int32_t thread_count = IntMin(
                kCoreCursorSimtMaxThreads,
                ((len + 31) / 32) * 32);
            asc_vf_call<ReduceExpertTileTotalSimt>(
                dim3(static_cast<uint32_t>(thread_count)),
                (__ubuf__ int32_t*)tpe_tile_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)thread_totals_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)tile_total_ub.GetPhyAddr(),
                len,
                thread_count);
            if (accumulate_total_tiles) {
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                worker_total += tile_total_ub.GetValue(0);
            }
        }

        if (accumulate_total_tiles) {
            tile_total_ub.SetValue(0, worker_total);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        }
        AscendC::DataCopyExtParams total_store_params(
            1,
            32,
            0,
            0,
            0);
        AscendC::DataCopyPad(
            cursor_worker_scratch_gm[
                core_idx * kCursorWorkerScratchRowElems],
            tile_total_ub,
            total_store_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
    AscendC::SyncAll<true>();

    // Convert [route_core, expert] counts to expert-major cursors. A worker
    // starts from the sum of all preceding expert partitions and advances its
    // carry across local tiles.
    if (core_idx < cursor_worker_num) {
        const int32_t fixed_ub_bytes =
            kCoreCursorThreadPrefixBytes + kCoreCursorCarryBytes;
        const int32_t max_cursor_row_bytes =
            (((kPlanningUbSizeBytes - fixed_ub_bytes) /
              (route_core_num + 1)) /
             32) *
            32;
        const int32_t cursor_tile_elems = IntMin(
            IntMin(kFillTileElems, cursor_expert_count),
            max_cursor_row_bytes / kInt32Bytes);
        const int32_t cursor_row_bytes =
            AlignUpBytes(cursor_tile_elems * kInt32Bytes, 32);
        const int32_t cursor_matrix_bytes =
            route_core_num * cursor_row_bytes;
        const int32_t tpe_tile_offset = cursor_matrix_bytes;
        const int32_t thread_prefix_offset =
            tpe_tile_offset + cursor_row_bytes;
        const int32_t expert_carry_offset =
            thread_prefix_offset + kCoreCursorThreadPrefixBytes;

        AscendC::LocalTensor<int32_t> cursor_matrix_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> tpe_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(tpe_tile_offset));
        AscendC::LocalTensor<int32_t> thread_prefix_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(thread_prefix_offset));
        AscendC::LocalTensor<int32_t> expert_carry_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(expert_carry_offset));

        AscendC::DataCopyExtParams totals_load_params(
            1,
            static_cast<uint32_t>(
                cursor_worker_num * kCursorWorkerScratchRowElems *
                kInt32Bytes),
            0,
            0,
            0);
        AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(
            cursor_matrix_ub,
            cursor_worker_scratch_gm[0],
            totals_load_params,
            pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        int32_t expert_base = 0;
        for (int32_t worker = 0; worker < core_idx; ++worker) {
            expert_base += cursor_matrix_ub.GetValue(
                worker * kCursorWorkerScratchRowElems);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::Duplicate(
            expert_carry_ub,
            expert_base,
            kCoreCursorCarryBytes / kInt32Bytes);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        for (int32_t base = cursor_expert_begin;
             base < cursor_expert_end;
             base += cursor_tile_elems) {
            const int32_t len =
                IntMin(cursor_tile_elems, cursor_expert_end - base);
            const int32_t active_row_bytes = len * kInt32Bytes;
            const int32_t aligned_active_row_bytes =
                AlignUpBytes(active_row_bytes, 32);
            AscendC::DataCopyExtParams matrix_load_params(
                static_cast<uint16_t>(route_core_num),
                static_cast<uint32_t>(active_row_bytes),
                static_cast<uint32_t>((E - len) * kInt32Bytes),
                static_cast<uint32_t>(
                    (cursor_row_bytes - aligned_active_row_bytes) / 32),
                0);
            AscendC::DataCopyExtParams tpe_load_params(
                1,
                static_cast<uint32_t>(active_row_bytes),
                0,
                0,
                0);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::DataCopyPad(
                cursor_matrix_ub,
                core_expert_counts_gm[base],
                matrix_load_params,
                pad_params);
            AscendC::DataCopyPad(
                tpe_tile_ub,
                tpe_gm[base],
                tpe_load_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            const int32_t thread_count = IntMin(
                kCoreCursorSimtMaxThreads,
                ((len + 31) / 32) * 32);
            asc_vf_call<BuildCoreExpertCursorsSimt>(
                dim3(static_cast<uint32_t>(thread_count)),
                (__ubuf__ int32_t*)cursor_matrix_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)tpe_tile_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)thread_prefix_ub.GetPhyAddr(),
                (__ubuf__ int32_t*)expert_carry_ub.GetPhyAddr(),
                route_core_num,
                len,
                cursor_row_bytes / kInt32Bytes,
                thread_count);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyExtParams matrix_store_params(
                static_cast<uint16_t>(route_core_num),
                static_cast<uint32_t>(active_row_bytes),
                static_cast<uint32_t>(
                    (cursor_row_bytes - aligned_active_row_bytes) / 32),
                static_cast<uint32_t>((E - len) * kInt32Bytes),
                0);
            AscendC::DataCopyPad(
                core_expert_counts_gm[base],
                cursor_matrix_ub,
                matrix_store_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::SyncAll<true>();
    const int32_t expert_row_bytes = AlignUpBytes(E * kInt32Bytes, 32);
    AscendC::LocalTensor<int32_t> expert_ordinal_ub =
        resource.ubBuf.template GetBufferByByte<int32_t>(0);
    if (core_idx < route_core_num) {
        static_assert((kFillTileElems * kInt32Bytes) % 32 == 0);
        constexpr int32_t kOrdinalScratchBytes =
            kFillTileElems * kInt32Bytes;
        AscendC::LocalTensor<int32_t> current_core_count_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(expert_row_bytes));
        AscendC::LocalTensor<int32_t> core_zero_count_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(
                    expert_row_bytes + kOrdinalScratchBytes));
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t base = 0; base < E; base += kFillTileElems) {
            const int32_t len = IntMin(kFillTileElems, E - base);
            AscendC::DataCopyExtParams copy_params(
                1,
                static_cast<uint32_t>(len * kInt32Bytes),
                0,
                0,
                0);
            AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
            AscendC::DataCopyPad(
                expert_ordinal_ub[base],
                tpe_prefix_gm[base],
                copy_params,
                pad_params);
            AscendC::DataCopyPad(
                current_core_count_ub,
                core_expert_counts_gm[core_idx * E + base],
                copy_params,
                pad_params);
            AscendC::DataCopyPad(
                core_zero_count_ub,
                core_expert_counts_gm[base],
                copy_params,
                pad_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::Sub(
                current_core_count_ub,
                current_core_count_ub,
                core_zero_count_ub,
                len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(
                expert_ordinal_ub[base],
                expert_ordinal_ub[base],
                current_core_count_ub,
                len);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        DstUbLayout dst_layout{};
        InitDstUbLayout(R, E, expert_row_bytes, dst_layout);
        AscendC::LocalTensor<int32_t> dst_tile_ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(
                static_cast<uint32_t>(dst_layout.dst_tile_offset));

        if (dst_layout.use_simt) {
            AscendC::LocalTensor<int32_t> alloc_cumsum_lookup_ub =
                resource.ubBuf.template GetBufferByByte<int32_t>(
                    static_cast<uint32_t>(dst_layout.lookup_offset));
            AscendC::LocalTensor<int32_t> expert_offsets_lookup_ub =
                resource.ubBuf.template GetBufferByByte<int32_t>(
                    static_cast<uint32_t>(
                        dst_layout.lookup_offset +
                        dst_layout.lookup_table_bytes));
            {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams copy_params(
                    1,
                    static_cast<uint32_t>(
                        dst_layout.lookup_table_elems * kInt32Bytes),
                    0,
                    0,
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                AscendC::DataCopyPad(alloc_cumsum_lookup_ub, alloc_cumsum_gm[0], copy_params, pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            }
            {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams copy_params(
                    1,
                    static_cast<uint32_t>(
                        dst_layout.lookup_table_elems * kInt32Bytes),
                    0,
                    0,
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                AscendC::DataCopyPad(expert_offsets_lookup_ub, expert_offsets_gm[0], copy_params, pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            }
            {
                AscendC::LocalTensor<int32_t> dst_simt_context_ub =
                    resource.ubBuf.template GetBufferByByte<int32_t>(
                        static_cast<uint32_t>(dst_layout.context_offset));
                AscendC::LocalTensor<int32_t> dst_simt_topk_ub =
                    resource.ubBuf.template GetBufferByByte<int32_t>(
                        static_cast<uint32_t>(dst_layout.simt_topk_offset));
                AscendC::LocalTensor<int32_t> dst_simt_hist_ub =
                    resource.ubBuf.template GetBufferByByte<int32_t>(
                        static_cast<uint32_t>(dst_layout.simt_hist_offset));

                dst_simt_context_ub.SetValue(
                    kDstSimtTopkOffset,
                    dst_layout.simt_topk_offset / kInt32Bytes);
                dst_simt_context_ub.SetValue(
                    kDstSimtHistOffset,
                    dst_layout.simt_hist_offset / kInt32Bytes);
                dst_simt_context_ub.SetValue(
                    kDstSimtOrdinalOffset,
                    0);
                dst_simt_context_ub.SetValue(
                    kDstSimtAllocCumsumOffset,
                    dst_layout.lookup_offset / kInt32Bytes);
                dst_simt_context_ub.SetValue(
                    kDstSimtExpertOffsetsOffset,
                    (dst_layout.lookup_offset +
                     dst_layout.lookup_table_bytes) /
                        kInt32Bytes);
                dst_simt_context_ub.SetValue(
                    kDstSimtDstOffset,
                    dst_layout.dst_tile_offset / kInt32Bytes);
                dst_simt_context_ub.SetValue(
                    kDstSimtChunkSize,
                    dst_layout.simt_chunk_size);
                dst_simt_context_ub.SetValue(
                    kDstSimtHistStride,
                    dst_layout.simt_hist_stride);
                dst_simt_context_ub.SetValue(kDstSimtRankCount, R);
                dst_simt_context_ub.SetValue(kDstSimtExpertCount, E);
                dst_simt_context_ub.SetValue(kDstSimtMaxTokensPerRank, NvS);

                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

                // MTE2 reuses dst_simt_topk_ub on every tile. Keep the next
                // load behind the preceding SIMT reader of that same buffer.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                for (int32_t base = begin; base < end; base += kFillTileElems) {
                    const int32_t len = IntMin(kFillTileElems, end - base);
                    const int32_t chunk_count =
                        (len + dst_layout.simt_chunk_size - 1) /
                        dst_layout.simt_chunk_size;
                    const int32_t hist_elems =
                        chunk_count * dst_layout.simt_hist_stride;

                    for (int32_t clear_base = 0; clear_base < hist_elems;
                         clear_base += kMaxDuplicateInt32Elems) {
                        const int32_t clear_len = IntMin(
                            kMaxDuplicateInt32Elems,
                            hist_elems - clear_base);
                        AscendC::Duplicate(
                            dst_simt_hist_ub[clear_base],
                            static_cast<int32_t>(0),
                            clear_len);
                    }
                    AscendC::PipeBarrier<PIPE_V>();

                    AscendC::DataCopyExtParams copy_params(
                        1,
                        static_cast<uint32_t>(len * kInt32Bytes),
                        0,
                        0,
                        0);
                    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    AscendC::DataCopyPad(
                        dst_simt_topk_ub,
                        topk_gm[base],
                        copy_params,
                        pad_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                    asc_vf_call<BuildDstTileSimt>(
                        dim3(static_cast<uint32_t>(kDstSimtMaxThreads)),
                        (__ubuf__ int32_t*)work_ub.GetPhyAddr(),
                        dst_layout.context_offset / kInt32Bytes,
                        base,
                        len);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

                    const int32_t total_row =
                        (chunk_count - 1) * dst_layout.simt_hist_stride;
                    AscendC::Add(
                        expert_ordinal_ub,
                        expert_ordinal_ub,
                        dst_simt_hist_ub[total_row],
                        E);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                    AscendC::DataCopyPad(
                        dst_gm[base],
                        dst_tile_ub,
                        copy_params);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                }
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            }
        } else {
            // Host validation guarantees that the ordinal row and both route
            // tiles fit when the complete lookup-SIMT layout does not.
            AscendC::LocalTensor<int32_t> dst_gm_simt_topk_ub =
                resource.ubBuf.template GetBufferByByte<int32_t>(
                    static_cast<uint32_t>(
                        dst_layout.gm_simt_topk_offset));

            // alloc_cumsum and expert_offsets are produced by multiple cores.
            // The preceding SyncAll joins those producers, while this fence
            // makes their GM stores visible before SIMT starts direct loads.
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            CleanInvalidDataCache();

            for (int32_t base = begin; base < end;
                 base += kFillTileElems) {
                const int32_t len = IntMin(kFillTileElems, end - base);
                AscendC::DataCopyExtParams copy_params(
                    1,
                    static_cast<uint32_t>(len * kInt32Bytes),
                    0,
                    0,
                    0);
                AscendC::DataCopyPadExtParams<int32_t> pad_params{
                    false, 0, 0, 0};
                AscendC::DataCopyPad(
                    dst_gm_simt_topk_ub,
                    topk_gm[base],
                    copy_params,
                    pad_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                asc_vf_call<BuildDstTileFromGmSimt>(
                    dim3(static_cast<uint32_t>(kDstSimtMaxThreads)),
                    (__ubuf__ int32_t*)dst_gm_simt_topk_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)expert_ordinal_ub.GetPhyAddr(),
                    (__ubuf__ int32_t*)dst_tile_ub.GetPhyAddr(),
                    reinterpret_cast<__gm__ int32_t*>(alloc_cumsum),
                    reinterpret_cast<__gm__ int32_t*>(expert_offsets),
                    len,
                    R,
                    E,
                    NvS);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                AscendC::DataCopyPad(
                    dst_gm[base],
                    dst_tile_ub,
                    copy_params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // Source-dup immediately reads back dst through MTE2. In the aligned
    // partition case each core consumes only the slice it produced, so a DDR
    // visibility barrier is sufficient and avoids an unnecessary grid sync.
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();

    // When tokens divide evenly across cores, dst production and source-dup
    // consumption use the same per-core route interval. Each core can consume
    // its own completed dst slice immediately; uneven partitions retain the
    // grid barrier because their route and token boundaries may differ.
    const bool source_dup_is_core_local = (S % core_num) == 0;
    if (!source_dup_is_core_local) {
        AscendC::SyncAll<true>();
    }
    // Source-side duplicate stages use one shared layout. Cross-core and
    // cross-rank synchronization remains explicit between the phase calls.
    {
        const SourceDupConfig source_dup_config{
            rank,
            R,
            S,
            K,
            NvS,
            core_idx,
            core_num,
            source_dup_tokens_per_tile,
            static_cast<int32_t>(tiling.server_dedup)};
        SourceDupPartition source_dup_partition{};
        SourceDupUbLayout source_dup_ub_layout{};
        SourceDupWorkspaceLayout source_dup_workspace_layout{};
        InitSourceDupPartition(source_dup_config, source_dup_partition);
        InitSourceDupUbLayout(source_dup_config, source_dup_ub_layout);
        InitSourceDupWorkspaceLayout(
            source_dup_config,
            source_dup_partition,
            source_dup_ub_layout,
            source_dup_workspace_layout);

        // 1. The source side scans tokens under the chosen rank/server domain and pre-packs group and secondary
        // addresses by the actual primary card.
        CountAndPrebuildSourceDupTiles(
            source_dup_config,
            source_dup_partition,
            source_dup_ub_layout,
            source_dup_workspace_layout,
            resource,
            dst_gm,
            dup_workspace_gm,
            dup_key_workspace_gm);

        // Per-core counts are local. Finish all rows before reducing and
        // actively publishing one compact count slot to each destination.
        AscendC::SyncAll<true>();
        PublishSourceDupCounts(
            source_dup_config,
            resource,
            shmem_base,
            dup_workspace_gm,
            dup_count_inbox_off);

        // Each destination waits on its local inbox, computes source-rank
        // prefixes, and stages them for a parallel return to the sources.
        // Server mode additionally exports the group starts ordered by src_rank.
        if (core_idx == 0) {
            BuildSourceDupDestinationPrefixes(
                source_dup_config,
                resource,
                local_shmem_gm,
                dup_key_workspace_gm,
                dup_counts_gm,
                dup_count_inbox_off);
        }
        AscendC::SyncAll<true>();
        // 4. Return both prefix kinds to the source side as the receive bases for each source to send groups and
        // secondaries.
        PublishSourceDupPrefixes(
            source_dup_config,
            resource,
            shmem_base,
            dup_key_workspace_gm,
            dup_prefix_inbox_off);

        // Prefix senders read the staging table above. Do not reuse it for the
        // per-core prefix table until every local sender has completed MTE3.
        AscendC::SyncAll<true>();
        BuildSourceDupCorePrefixes(
            source_dup_config,
            resource,
            local_shmem_gm,
            dup_workspace_gm,
            dup_key_workspace_gm,
            dup_prefix_inbox_off);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::SyncAll<true>();

        // 6. Send metadata over non-overlapping intervals and publish this source completion notice only after all
        // local cores finish.
        PublishSourceDupPayloads(
            source_dup_config,
            source_dup_partition,
            source_dup_ub_layout,
            source_dup_workspace_layout,
            resource,
            shmem_base,
            dup_workspace_gm,
            dup_key_workspace_gm,
            dup_payload_done_flag_off,
            dup_groups_scratch_off,
            dup_loffs_scratch_off);

        // 7. After all source metadata arrives, copy into this card output tensor in chunks by the actual G/D.
        WaitForSourceDupPayloadDoneFlags(
            source_dup_config,
            resource,
            local_shmem_gm,
            dup_payload_done_flag_off);
        AscendC::SyncAll<true>();

        MaterializeSourceDupOutputs(
            source_dup_config,
            source_dup_ub_layout,
            resource,
            local_shmem_gm,
            dup_counts_gm,
            dup_groups_gm,
            dup_loffs_gm,
            dup_groups_scratch_off,
            dup_loffs_scratch_off);
    }

    if (tiling.combine_mode == 1U) {
        // Replace the old inter-kernel boundary: finish every base-planner DMA
        // before reusing UB and let every local core observe complete routing.
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::SyncAll<true>();
        RunCombinePlanning(dst, dup_groups, dup_loffs, dup_counts,
                           tiling, planning_begin, GetSystemCycle());
    }
}

extern "C" ASCEND_MOONEP_EXPORT void planning_kernel_do(
    void* stream,
    uint8_t* topk_idx,
    uint8_t* symmetric_buffer,
    uint8_t* dst,
    uint8_t* cu_seqlens,
    uint8_t* experts_to_copy,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* tpe_all,
    uint8_t* tpe_prefix,
    uint8_t* balance,
    uint8_t* z_tensor,
    uint8_t* alloc_tensor,
    uint8_t* alloc_cumsum,
    uint8_t* expert_offsets,
    uint8_t* core_expert_counts,
    uint8_t* dup_workspace,
    uint8_t* dup_key_workspace,
    uint8_t* tiling_host,
    uint64_t ffts_addr) {
    auto* td = reinterpret_cast<ascend_moonep::PlanningTilingData*>(tiling_host);
    planning_kernel<<<td->block_dim, kPlanningUbSizeBytes, stream>>>(
        topk_idx,
        symmetric_buffer,
        dst,
        cu_seqlens,
        experts_to_copy,
        dup_groups,
        dup_loffs,
        dup_counts,
        tpe_all,
        tpe_prefix,
        balance,
        z_tensor,
        alloc_tensor,
        alloc_cumsum,
        expert_offsets,
        core_expert_counts,
        dup_workspace,
        dup_key_workspace,
        *td,
        ffts_addr);
}

namespace CombineExecutionPlanning {

// ---- Begin combine planning SIMT functions ----
// Internal vector functions of planning_kernel, NOT separately launched kernels.
// All SIMT stores target private UB. Packed GM tables are still written by MTE3.
constexpr uint32_t kPlanSimtTokens = 64U;
constexpr uint32_t kPlanSimtColumns = 21U;
constexpr uint32_t kPlanDecodedUb = 16384U; // 64 * 40 words
constexpr uint32_t kPlanPrefixUb = 26624U;  // 21 * 64 words
constexpr uint32_t kPlanTotalsUb = 32000U;  // end[0:21], begin[32:53]
constexpr uint32_t kPlanChunksUb = 32256U; // 21 columns * eight 8-token chunks
constexpr uint32_t kPlanTxUb = 98304U;     // 64 * 16 * 4 words
constexpr uint32_t kPlanContributorUb = 114688U;
constexpr uint32_t kPlanRelayUb = 131072U; // 8 lanes, each 64 * 16 * 2 words
constexpr uint32_t kPlanRelayLaneBytes = 8192U;
static_assert(kPlanRelayUb + 8U * kPlanRelayLaneBytes <= 216U * 1024U);
// Route ping-pong uses [0,16KiB) and [64KiB,80KiB). No writer aliases it.
static_assert(kPlanDecodedUb + 64U * 40U * 4U == kPlanPrefixUb);
static_assert(kPlanPrefixUb + 21U * 64U * 4U == kPlanTotalsUb);
static_assert(kPlanTotalsUb + 64U * 4U <= 65536U);
static_assert(kPlanChunksUb + 21U * 8U * 4U <= 65536U);

// One thread decodes one token. Keep only our active rows and our domain's
// suppressed members; representative order is the original top-k order.
__simt_callee__ inline void DecodeCombineTokensSimt(
    __ubuf__ int32_t* routes, __ubuf__ uint32_t* decoded,
    __ubuf__ uint32_t* prefix, uint32_t tokens, uint32_t k,
    uint32_t n, uint32_t r, uint32_t rank, uint32_t source, uint32_t server) {
    const uint32_t t = threadIdx.x;
    if (t >= kPlanSimtTokens) return;
    for (uint32_t col = 0U; col < kPlanSimtColumns; ++col)
        prefix[col * kPlanSimtTokens + t] = 0U;
    if (t >= tokens) return;
    auto row = decoded + t * 40U;
    uint32_t own = 0U, suppressed = 0U, primaries = 0U, finalBase = 0U;
    uint32_t representative = 0U, representativePeer = 0U, ownRepresentative = 0U;
    uint32_t ownSuppressed = 0U, error = 0U;
    for (uint32_t j = 0U; j < k; ++j) {
        const int32_t encoded = routes[t * k + j];
        const uint32_t bits = static_cast<uint32_t>(encoded);
        const bool forwarded = (bits & 0xc0000000U) == 0x80000000U;
        const uint32_t address = forwarded ? bits & 0x3fffffffU :
            encoded >= 0 ? bits : ~bits;
        const uint32_t peer = address / n;
        const bool primary = encoded >= 0 || forwarded;
        const bool active = primary || (server && peer / 8U == source / 8U);
        error |= uint32_t(peer >= r);
        finalBase += uint32_t(active && peer < rank);
        if (peer == rank && active) {
            if (primary) ownRepresentative = own;
            row[own++] = address - peer * n;
        }
        if (server ? peer / 8U == rank / 8U : peer == rank) {
            if (primary) {
                ++primaries; representative = address; representativePeer = peer;
            } else {
                row[16U + suppressed++] = address;
                ownSuppressed += uint32_t(peer == rank);
            }
        }
    }
    error |= uint32_t(suppressed && primaries != 1U);
    const bool remote = !(server && source / 8U == rank / 8U);
    const bool group = suppressed && representativePeer == rank;
    const uint32_t lane = server ? representativePeer % 8U : 0U;
    row[32] = representative; row[33] = representativePeer;
    row[34] = ownRepresentative; row[35] = own; row[36] = suppressed;
    prefix[t] = own;
    prefix[64U + t] = uint32_t(group);
    prefix[128U + t] = finalBase;
    prefix[192U + t] = own - uint32_t(remote && group);
    if (remote && suppressed) {
        prefix[(4U + lane) * 64U + t] = suppressed + 1U;
        prefix[(12U + lane) * 64U + t] = ownSuppressed + uint32_t(representativePeer == rank);
    }
    prefix[20U * 64U + t] = error;
}

__simt_callee__ inline void EmitCombineTokensSimt(
    __ubuf__ uint32_t* decoded, __ubuf__ uint32_t* prefix,
    __ubuf__ uint32_t* totals, __ubuf__ uint32_t* tx,
    __ubuf__ uint32_t* contributors, __ubuf__ uint32_t* relays,
    uint32_t tokens, uint32_t n, uint32_t rank, uint32_t source, uint32_t server) {
    const uint32_t t = threadIdx.x;
    if (t >= tokens) return;
    auto row = decoded + t * 40U;
    const uint32_t representative = row[32], peer = row[33];
    const uint32_t ownRepresentative = row[34], own = row[35], suppressed = row[36];
    const bool group = suppressed && peer == rank;
    const bool remote = !(server && source / 8U == rank / 8U);
    const uint32_t lane = server ? peer % 8U : 0U;
    const uint32_t txIndex = prefix[t];
    const uint32_t head = prefix[(4U + lane) * 64U + t];
    if (remote && suppressed) {
        uint32_t request = prefix[(12U + lane) * 64U + t] - totals[32U + 12U + lane];
        for (uint32_t member = 0U; member <= suppressed; ++member) {
            const uint32_t address = member ? row[16U + member - 1U] : representative;
            if (peer == rank) {
                const uint32_t pos = (head - totals[32U + 4U + lane] + member) * 4U;
                contributors[pos] = member ? 0U : txIndex + ownRepresentative;
                contributors[pos + 1U] = member ? 0U : suppressed + 1U;
                contributors[pos + 2U] = server ? address : address % n;
                contributors[pos + 3U] = 0U;
            }
            if (address / n == rank) {
                const uint32_t pos = lane * (kPlanRelayLaneBytes / 4U) + request++ * 2U;
                relays[pos] = address % n;
                relays[pos + 1U] = head + member;
            }
        }
    }
    for (uint32_t j = 0U; j < own; ++j) {
        const bool isHead = group && j == ownRepresentative;
        const uint32_t pos = (txIndex - totals[32] + j) * 4U;
        tx[pos] = row[j];
        tx[pos + 1U] = isHead ? prefix[64U + t] : 0xffffffffU;
        tx[pos + 2U] = isHead && remote ? head : 0xffffffffU;
        tx[pos + 3U] = isHead && remote ? suppressed + 1U : 1U;
    }
}
// Three prefix phases: eight independent chunks per column; chunk carries;
// then add each carry. Every phase is separated by a full VF block barrier.
__simt_callee__ inline void CombineChunkPrefixSimt(
    __ubuf__ uint32_t* prefix, __ubuf__ uint32_t* chunks) {
    for (uint32_t job = threadIdx.x; job < 21U * 8U; job += 64U) {
        const uint32_t col = job / 8U, begin = (job % 8U) * 8U;
        uint32_t sum = 0U;
        for (uint32_t t = begin; t < begin + 8U; ++t) {
            const uint32_t count = prefix[col * 64U + t];
            prefix[col * 64U + t] = sum; sum += count;
        }
        chunks[job] = sum;
    }
}
__simt_callee__ inline void CombineChunkCarrySimt(
    __ubuf__ uint32_t* chunks, __ubuf__ uint32_t* totals) {
    const uint32_t col = threadIdx.x;
    if (col >= 21U) return;
    uint32_t sum = totals[col];
    totals[32U + col] = sum;
    for (uint32_t chunk = 0U; chunk < 8U; ++chunk) {
        const uint32_t count = chunks[col * 8U + chunk];
        chunks[col * 8U + chunk] = sum; sum += count;
    }
    totals[col] = sum;
}
__simt_callee__ inline void CombineApplyCarrySimt(
    __ubuf__ uint32_t* prefix, __ubuf__ uint32_t* chunks) {
    const uint32_t t = threadIdx.x;
    for (uint32_t col = 0U; col < 21U; ++col)
        prefix[col * 64U + t] += chunks[col * 8U + t / 8U];
}

__simt_vf__ __launch_bounds__(64) inline void BuildCombineSliceSimt(
    __ubuf__ int32_t* routes, __ubuf__ uint32_t* decoded,
    __ubuf__ uint32_t* prefix, __ubuf__ uint32_t* totals,
    __ubuf__ uint32_t* chunks, __ubuf__ uint32_t* tx,
    __ubuf__ uint32_t* contributors, __ubuf__ uint32_t* relays,
    uint32_t tokens, uint32_t k, uint32_t n, uint32_t r,
    uint32_t rank, uint32_t source, uint32_t server) {
    DecodeCombineTokensSimt(routes, decoded, prefix, tokens, k, n, r, rank, source, server);
    asc_syncthreads();
    CombineChunkPrefixSimt(prefix, chunks);
    asc_syncthreads();
    CombineChunkCarrySimt(chunks, totals);
    asc_syncthreads();
    CombineApplyCarrySimt(prefix, chunks);
    asc_syncthreads();
    // Uniform test AFTER all barriers, including inactive tail threads.
    // Never emit records from invalid routes; host reports the error after VF.
    if (!totals[20U])
        EmitCombineTokensSimt(decoded, prefix, totals, tx, contributors, relays,
                              tokens, n, rank, source, server);
}

__simt_vf__ __launch_bounds__(64) inline void AdjustCombineRowsSimt(
    __ubuf__ uint32_t* rows, uint32_t size, uint32_t kind,
    uint32_t txBase, uint32_t groupBase, uint32_t scratchBase) {
    const uint32_t j = threadIdx.x;
    if (j >= size) return;
    const uint32_t off = j * (kind == 2U ? 2U : 4U);
    if (kind == 0U) {
        if (rows[off + 1U] != 0xffffffffU) rows[off + 1U] += groupBase;
        if (rows[off + 2U] != 0xffffffffU) rows[off + 2U] += scratchBase;
    } else if (kind == 1U) {
        if (rows[off + 1U]) rows[off] += txBase;
    } else rows[off + 1U] += scratchBase;
}

__simt_vf__ __launch_bounds__(32) inline void PrefixCombineTaskStatsSimt(
    __ubuf__ uint32_t* stats, __ubuf__ uint32_t* totals,
    uint32_t tiles, uint32_t width) {
    const uint32_t col = threadIdx.x;
    if (col >= 4U + 2U * width) return;
    const uint32_t lane = col < 4U ? 0U : (col - 4U) % width;
    const uint32_t field = col < 4U ? col : 16U + lane * 4U + uint32_t(col >= 4U + width);
    uint32_t sum = 0U;
    for (uint32_t tile = 0U; tile < tiles; ++tile) {
        auto row = stats + tile * 64U;
        if (col < 2U) row[4U + col] = sum;
        else if (col >= 4U) row[field + 2U] = sum;
        sum += row[field];
    }
    totals[col] = sum;
}

__simt_vf__ __launch_bounds__(256) inline void CountFinalRoutesSimt(
    __ubuf__ int32_t* routes, __ubuf__ uint32_t* counts,
    uint32_t size, uint32_t n, uint32_t rank, uint32_t server) {
    for (uint32_t j = threadIdx.x; j < size; j += 256U) {
        const int32_t v = routes[j];
        const uint32_t bits = static_cast<uint32_t>(v);
        const bool forwarded = (bits & 0xc0000000U) == 0x80000000U;
        const uint32_t raw = forwarded ? bits & 0x3fffffffU : v >= 0 ? bits : ~bits;
        const uint32_t peer = raw / n;
        if (v >= 0 || forwarded || (server && peer / 8U == rank / 8U))
            asc_atomic_add(counts + peer * 8U, 1U);
    }
}

// Each chunk owns two consecutive tokens; its private peer cursors make
// emission deterministic. 32*(R+1) words fit in remaining UB even at R=1024.
constexpr uint32_t kFinalChunksUb = 81920U;
static_assert(kFinalChunksUb + 32U * 1025U * 4U <= 216U * 1024U);
__simt_callee__ inline void CountFinalChunksSimt(
    __ubuf__ int32_t* routes, __ubuf__ uint32_t* hist,
    __ubuf__ uint32_t* packed, uint32_t tokens, uint32_t k,
    uint32_t n, uint32_t r, uint32_t rank, uint32_t server) {
    const uint32_t thread = threadIdx.x;
    for (uint32_t i = thread; i < tokens * 32U; i += 128U) packed[i] = 0U;
    if (thread >= 32U) return;
    auto row = hist + thread * (r + 1U);
    for (uint32_t peer = 0U; peer < r; ++peer) row[peer] = 0U;
    const uint32_t end = thread * 2U + 2U < tokens ? thread * 2U + 2U : tokens;
    for (uint32_t t = thread * 2U; t < end; ++t) for (uint32_t j = 0U; j < k; ++j) {
        const int32_t v = routes[t * k + j];
        const uint32_t bits = static_cast<uint32_t>(v);
        const bool forwarded = (bits & 0xc0000000U) == 0x80000000U;
        const uint32_t raw = forwarded ? bits & 0x3fffffffU : v >= 0 ? bits : ~bits;
        const uint32_t peer = raw / n;
        if (v >= 0 || forwarded || (server && peer / 8U == rank / 8U)) ++row[peer];
    }
}
__simt_callee__ inline void PrefixFinalChunksSimt(
    __ubuf__ uint32_t* hist, __ubuf__ uint32_t* cursors, uint32_t r) {
    for (uint32_t peer = threadIdx.x; peer < r; peer += 128U) {
        uint32_t sum = cursors[peer];
        for (uint32_t chunk = 0U; chunk < 32U; ++chunk) {
            const uint32_t off = chunk * (r + 1U) + peer, count = hist[off];
            hist[off] = sum; sum += count;
        }
        cursors[peer] = sum;
    }
}
__simt_callee__ inline void EmitFinalChunksSimt(
    __ubuf__ int32_t* routes, __ubuf__ uint32_t* hist,
    __ubuf__ uint32_t* packed, uint32_t tokens, uint32_t k,
    uint32_t n, uint32_t r, uint32_t rank, uint32_t server) {
    const uint32_t chunk = threadIdx.x;
    if (chunk >= 32U) return;
    auto row = hist + chunk * (r + 1U);
    const uint32_t end = chunk * 2U + 2U < tokens ? chunk * 2U + 2U : tokens;
    for (uint32_t t = chunk * 2U; t < end; ++t) {
        uint32_t count = 0U;
        for (uint32_t j = 0U; j < k; ++j) {
            const int32_t v = routes[t * k + j];
            const uint32_t bits = static_cast<uint32_t>(v);
            const bool forwarded = (bits & 0xc0000000U) == 0x80000000U;
            const uint32_t raw = forwarded ? bits & 0x3fffffffU : v >= 0 ? bits : ~bits;
            const uint32_t peer = raw / n;
            if (!(v >= 0 || forwarded || (server && peer / 8U == rank / 8U))) continue;
            const uint32_t ordinal = row[peer]++;
            packed[t * 32U + 1U + count++] = peer == rank ? 0x80000000U | (raw % n) : ordinal;
        }
        packed[t * 32U] = count;
    }
}
__simt_vf__ __launch_bounds__(128) inline void BuildFinalRoutesSimt(
    __ubuf__ int32_t* routes, __ubuf__ uint32_t* hist,
    __ubuf__ uint32_t* cursors, __ubuf__ uint32_t* packed,
    uint32_t tokens, uint32_t k, uint32_t n, uint32_t r, uint32_t rank, uint32_t server) {
    CountFinalChunksSimt(routes, hist, packed, tokens, k, n, r, rank, server);
    asc_syncthreads();
    PrefixFinalChunksSimt(hist, cursors, r);
    asc_syncthreads();
    EmitFinalChunksSimt(routes, hist, packed, tokens, k, n, r, rank, server);
}
// ---- End combine planning SIMT functions ----

using namespace ascend_moonep;
constexpr uint32_t kCores = kCombinePutCores, kBatch = kCombinePutBatchRows, kTile = 8192U;

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

__aicore__ inline uint32_t Raw(int32_t v) {
    return ascend_moonep::IsMoonEPServerForwardedRoute(v)
        ? ascend_moonep::DecodeMoonEPServerForwardedRoute(v)
        : v >= 0 ? static_cast<uint32_t>(v)
                 : static_cast<uint32_t>(-static_cast<int64_t>(v) - 1);
}

__aicore__ inline bool Primary(int32_t v) {
    return v >= 0 || ascend_moonep::IsMoonEPServerForwardedRoute(v);
}


struct CombinePlanBuilder {
    __gm__ uint8_t* ws;
    // groups/dups/counts remain launch-ABI inputs for existing callers; the
    // combine metadata builder derives all its tables from published dst.
    const __gm__ int32_t *dst, *groups, *dups, *counts;
    CombinePutLayout p;
    uint32_t n, h, k, r, rank, core;
    bool server;
    __gm__ uint8_t* plan;
    CombineExecutionPlanLayout q;
    uint32_t coreCount = kCores; // Fused planner inherits num_sms (default 32).
    struct TokenRoutes {
        uint32_t raw[16], peer[16], members[16];
        bool primary[16], active[16];
        uint32_t suppressed[16], suppressedCount, ownSuppressed;
    };

    __aicore__ __gm__ int32_t* Meta(uint64_t base, uint64_t index) {
        // Only scalar-written outputs use Meta; packed tables are MTE3-written.
        if (base == p.source_counts)
            return reinterpret_cast<__gm__ int32_t*>(plan + q.source_counts + index * 64U);
        if (base == p.relay_ranges)
            return reinterpret_cast<__gm__ int32_t*>(plan + q.relay_ranges + index * 64U);
        return reinterpret_cast<__gm__ int32_t*>(ws + base + index * 64U);
    }
    __aicore__ uint32_t Width() const { return server ? 8U : 1U; }
    __aicore__ uint32_t Lane(uint32_t peer) const { return server ? peer % 8U : 0U; }
    __aicore__ bool LocalSource(uint32_t source) const {
        return server && source / 8U == rank / 8U;
    }
    __aicore__ bool LocalDomain(uint32_t peer) const {
        return server ? peer / 8U == rank / 8U : peer == rank;
    }
    // Each core owns disjoint source intervals. Adjacent intervals can share a
    // GM cacheline: never scalar-write packed records (DCCI is a full-line RMW).
    // Buffer up to 64 records in private UB, then MTE3 stores only valid bytes.
    // No padding is written to GM, including the final partial batch.
    struct PlanWriter {
        LocalTensor<uint32_t> buf;
        __gm__ uint32_t* out;
        uint32_t words, count, ubOffset, bank;
    };
    // One outstanding MTE3 submission, with two private UB banks per writer.
    // Wait only before the next submission, not immediately after every 64 rows.
    bool writerPending = false;
    __aicore__ void WaitWriter() {
        if (writerPending) {
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            writerPending = false;
        }
    }
    __aicore__ PlanWriter Writer(uint32_t ubOffset, uint64_t table,
                                uint64_t first, uint32_t words) {
        return {Ub<uint32_t>(ubOffset, 64U * words),
                reinterpret_cast<__gm__ uint32_t*>(plan + table) + first * words,
                words, 0U, ubOffset, 0U};
    }
    // During planning relay_hidden has no live payload (Init fences the previous
    // invocation). Use it as source-private staging, then compact only valid rows.
    // Each source owns N TX rows, N contributors and Width()*N relay requests.
    // This removes the count-only route pass, without changing any output order.
    __aicore__ bool UseStagedPlan() const {
        const bool staged = r <= 128U &&
            uint64_t(r) * (32U + 8U * Width()) <= uint64_t(h) * 16U;
        if (staged && uint64_t(r) * n * (32U + 8U * Width()) > p.weights - p.relay_hidden)
            aclshmemi_kernel_abort("combine planning phase pool too small rank=%u\n", rank);
        return staged;
    }
    __aicore__ uint64_t StagedTable(uint32_t kind, uint32_t source,
                                   uint32_t lane = 0U) const {
        const uint64_t segment = uint64_t(r) * n * 16U;
        return kind == 0U ? uint64_t(source) * n * 16U :
               kind == 1U ? segment + uint64_t(source) * n * 16U :
               2U * segment + (uint64_t(lane) * r + source) * n * 8U;
    }
    __aicore__ PlanWriter StagedWriter(uint32_t ubOffset, uint32_t kind,
                                      uint32_t source, uint32_t lane = 0U) {
        const uint32_t words = kind == 2U ? 2U : 4U;
        return {Ub<uint32_t>(ubOffset, 64U * words),
                reinterpret_cast<__gm__ uint32_t*>(ws + p.relay_hidden +
                    StagedTable(kind, source, lane)), words, 0U, ubOffset, 0U};
    }
    __aicore__ void FinishWriter(PlanWriter& w) {
        if (!w.count) return;
        WaitWriter();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        Store(w.out, w.buf, w.count * w.words);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        writerPending = true;
        w.out += w.count * w.words;
        w.count = 0U;
        w.bank ^= 1U;
        w.buf = Ub<uint32_t>(w.ubOffset + w.bank * 65536U, 64U * w.words);
    }
    __aicore__ void Append(PlanWriter& w, uint32_t a, uint32_t b,
                           uint32_t c = 0U, uint32_t d = 0U) {
        const uint32_t off = w.count * w.words;
        w.buf.SetValue(off, a); w.buf.SetValue(off + 1U, b);
        if (w.words == 4U) {
            w.buf.SetValue(off + 2U, c); w.buf.SetValue(off + 3U, d);
        }
        if (++w.count == 64U) FinishWriter(w);
    }
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
    __aicore__ void ClearReadyFlags() {
        // Same 8-byte fields as before, NOT a memset of the padded ready arena.
        // A strided MTE transfer replaces thousands of scalar store+DCCI pairs.
        // UB has one zero-padded 32-byte source block per 8-byte destination.
        auto zero = Ub<int32_t>(0U, 8192U);
        Duplicate(zero, int32_t(0), 8192U);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const uint64_t end = p.magic + uint64_t(kCores) * 512U;
        const uint64_t stride = uint64_t(coreCount) * 512U;
        for (uint64_t off = p.relay_ready + uint64_t(core) * 512U; off < end;) {
            const uint64_t left = (end - 1U - off) / stride + 1U;
            const uint32_t count = left < 1024U ? static_cast<uint32_t>(left) : 1024U;
            GlobalTensor<int32_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ws + off));
            DataCopyExtParams cp{static_cast<uint16_t>(count), 8U, 0U,
                                 static_cast<uint32_t>(stride - 8U), 0U};
            DataCopyPad(gm, zero, cp);
            off += uint64_t(count) * stride;
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        // Ring state is mutable workspace, never part of a cloned plan. Only
        // clear the first uint64 in each independent 512B flag/source slot.
        for (uint64_t off = p.ring_epoch + uint64_t(core) * 512U; off < p.bytes;) {
            const uint64_t left = (p.bytes - 1U - off) / stride + 1U;
            const uint32_t count = left < 1024U ? static_cast<uint32_t>(left) : 1024U;
            GlobalTensor<int32_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ws + off));
            DataCopyExtParams cp{static_cast<uint16_t>(count), 8U, 0U,
                                 static_cast<uint32_t>(stride - 8U), 0U};
            DataCopyPad(gm, zero, cp);
            off += uint64_t(count) * stride;
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
    __aicore__ void Init() {
        aclshmemx_sync_vec_all(); // exclude an earlier plan/payload using this workspace
        // Once per plan, never once per combine. No reverse/group table clears.
        ClearReadyFlags();
        for (uint32_t s = core; s < r; s += coreCount) {
            auto flag = reinterpret_cast<__gm__ uint64_t*>(
                ws + p.source_input_ready + uint64_t(s) * 512U);
            *flag = 0U; Flush(flag);
        }
        if (core == 0U) {
            auto flag = reinterpret_cast<__gm__ uint64_t*>(ws + p.final_gate);
            *flag = 0U; Flush(flag);
        }
    }
    __aicore__ void PublishRoutes() {
        // TileXR-style published base routes. Reuse the retired reverse_send
        // arena, but pack int32 routes contiguously (not 64-byte records).
        auto published = reinterpret_cast<__gm__ int32_t*>(ws + p.reverse_send);
        for (uint32_t first = core * kTile; first < n; first += coreCount * kTile) {
            const uint32_t size = n - first < kTile ? n - first : kTile;
            Copy(published + first, dst + first, size);
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemx_sync_vec_all(); // publish completion and ready initialization on every rank
    }
    __aicore__ LocalTensor<int32_t> LoadRoutes(uint32_t source, uint32_t first,
                                             uint32_t size) {
        auto local = reinterpret_cast<__gm__ int32_t*>(ws + p.reverse_send);
        auto remote = reinterpret_cast<__gm__ int32_t*>(aclshmem_ptr(local, source));
        auto routes = Ub<int32_t>(0U, kTile);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0); // previous scalar route reads retired
        Load(routes, remote + first, size);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        return routes;
    }
    // The retired relay_publish arena is 8*N*64 bytes: it holds R complete
    // int32 route arrays for R<=128. No workspace/plan ABI or allocation change.
    // Larger topologies retain the original second remote read.
    __aicore__ bool CacheRoutesEnabled() const { return r <= 128U; }
    __aicore__ void PrefetchRoutes(uint32_t source, uint32_t first, uint32_t size,
                                   uint32_t bank, bool cached) {
        // Also fences a previous route-cache store before its UB bank is reused.
        WaitWriter();
        const __gm__ int32_t* from;
        if (cached && CacheRoutesEnabled()) {
            from = reinterpret_cast<__gm__ int32_t*>(ws + p.relay_publish) +
                   uint64_t(source) * n;
        } else {
            auto published = reinterpret_cast<__gm__ int32_t*>(ws + p.reverse_send);
            from = reinterpret_cast<__gm__ int32_t*>(aclshmem_ptr(published, source));
        }
        const event_t event = bank == 0U ? EVENT_ID0 : EVENT_ID1;
        SetFlag<HardEvent::S_MTE2>(event);
        WaitFlag<HardEvent::S_MTE2>(event);
        Load(Ub<int32_t>(bank * 65536U, kTile), from + first, size);
        SetFlag<HardEvent::MTE2_S>(event);
    }
    __aicore__ LocalTensor<int32_t> AwaitRoutes(uint32_t bank) {
        WaitFlag<HardEvent::MTE2_S>(bank == 0U ? EVENT_ID0 : EVENT_ID1);
        return Ub<int32_t>(bank * 65536U, kTile);
    }
    __aicore__ void CacheRoutes(uint32_t source, uint32_t first, uint32_t size,
                                LocalTensor<int32_t> routes) {
        if (!CacheRoutesEnabled()) return;
        WaitWriter();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        Store(reinterpret_cast<__gm__ int32_t*>(ws + p.relay_publish) +
              uint64_t(source) * n + first, routes, size);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        writerPending = true;
    }
    __aicore__ void DecodeToken(LocalTensor<int32_t> routes, uint32_t offset,
                               uint32_t source, TokenRoutes& token) {
        uint32_t primaries = 0U, representative = 0U;
        token.suppressedCount = 0U;
        token.ownSuppressed = 0U;
        for (uint32_t j = 0U; j < k; ++j) {
            const int32_t encoded = routes.GetValue(offset + j);
            token.raw[j] = Raw(encoded);
            token.peer[j] = token.raw[j] / n;
            if (token.peer[j] >= r)
                aclshmemi_kernel_abort("combine plan invalid dst source=%u rank=%u\n", source, rank);
            token.primary[j] = Primary(encoded);
            token.active[j] = token.primary[j] ||
                (server && token.peer[j] / 8U == source / 8U);
            token.members[j] = 1U;
            if (LocalDomain(token.peer[j])) {
                if (token.primary[j]) { ++primaries; representative = j; }
                else {
                    token.suppressed[token.suppressedCount++] = j;
                    if (token.peer[j] == rank) ++token.ownSuppressed;
                }
            }
        }
        // Every LocalDomain route is in the same domain. Collect its suppressed
        // members once (in top-k order), instead of rescanning K for every j.
        // Keep the upstream representative, including a primary not at index 0.
        if (token.suppressedCount && primaries != 1U)
            aclshmemi_kernel_abort("combine plan ambiguous primary source=%u rank=%u\n", source, rank);
        if (token.suppressedCount)
            token.members[representative] += token.suppressedCount;
    }
    // Count directly from K encoded routes. Do not materialize TokenRoutes or
    // traverse its arrays again; the full decoding is needed only by the fill.
    __aicore__ void CountToken(LocalTensor<int32_t> routes, uint32_t offset,
        uint32_t source, uint32_t& results, uint32_t& finalBase,
        uint32_t& direct, uint32_t& groupCount, uint32_t* scratch,
        uint32_t* requests) {
        uint32_t primaries = 0U, representative = 0U;
        uint32_t suppressed = 0U, ownSuppressed = 0U;
        const bool local = LocalSource(source);
        for (uint32_t j = 0U; j < k; ++j) {
            const int32_t encoded = routes.GetValue(offset + j);
            const uint32_t peer = Raw(encoded) / n;
            if (peer >= r)
                aclshmemi_kernel_abort("combine plan invalid dst source=%u rank=%u\n", source, rank);
            const bool primary = Primary(encoded);
            const bool active = primary || (server && peer / 8U == source / 8U);
            if (active && peer < rank) ++finalBase;
            if (active && peer == rank) { ++results; ++direct; }
            if (LocalDomain(peer)) {
                if (primary) { ++primaries; representative = peer; }
                else { ++suppressed; ownSuppressed += peer == rank; }
            }
        }
        if (!suppressed) return;
        if (primaries != 1U)
            aclshmemi_kernel_abort("combine plan ambiguous primary source=%u rank=%u\n", source, rank);
        if (representative == rank) {
            ++groupCount;
            if (!local) --direct; // This primary is a reduced, not DIRECT, row.
        }
        if (!local) {
            const uint32_t lane = Lane(representative);
            scratch[lane] += suppressed + 1U;
            requests[lane] += ownSuppressed + uint32_t(representative == rank);
        }
    }
    __aicore__ void CountSources(bool synchronize = true) {
        const uint32_t pageRows = (kTile / k) * k;
        for (uint32_t source = core; source < r; source += coreCount) {
            uint32_t results = 0U, finalBase = 0U, direct = 0U, groupCount = 0U;
            uint32_t scratch[8] = {}, requests[8] = {};
            PrefetchRoutes(source, 0U, n < pageRows ? n : pageRows, 0U, false);
            for (uint32_t first = 0U; first < n; first += pageRows) {
                const uint32_t size = n - first < pageRows ? n - first : pageRows;
                const uint32_t bank = (first / pageRows) & 1U;
                auto routes = AwaitRoutes(bank);
                const uint32_t next = first + size;
                if (next < n)
                    PrefetchRoutes(source, next, n - next < pageRows ? n - next : pageRows,
                                   bank ^ 1U, false);
                CacheRoutes(source, first, size, routes);
                for (uint32_t offset = 0U; offset < size; offset += k) {
                    CountToken(routes, offset, source, results, finalBase,
                               direct, groupCount, scratch, requests);
                }
            }
            WaitWriter();
            auto sc = Meta(p.source_counts, source);
            sc[0] = results; sc[1] = finalBase; sc[2] = 0;
            sc[3] = direct; sc[4] = 0; sc[5] = scratch[Lane(rank)]; sc[6] = 0;
            Flush(sc);
            auto gc = Meta(p.counts_send, source);
            gc[0] = groupCount; gc[1] = 0; Flush(gc);
            for (uint32_t lane = 0U; lane < Width(); ++lane) {
                // Planner-only local stats: {scratchCount, requestCount,
                // scratchPrefix, requestPrefix}; no remote metadata PUT.
                auto stats = Meta(p.relay_ranges_send, uint64_t(lane) * r + source);
                stats[0] = scratch[lane]; stats[1] = requests[lane];
                stats[2] = 0; stats[3] = 0; Flush(stats);
            }
        }
        DataSyncBarrier<MemDsbT::DDR>();
        if (synchronize) aclshmemi_sync_core<true>();
    }
    __aicore__ void BuildSourcePrefixes() {
        for (uint32_t laneIndex = core; laneIndex < Width(); laneIndex += coreCount) {
            uint32_t scratch = 0U, requests = 0U, results = 0U, groups = 0U;
            for (uint32_t source = 0U; source < r; ++source) {
                auto stats = Meta(p.relay_ranges_send, uint64_t(laneIndex) * r + source);
                Flush(stats);
                if (static_cast<uint32_t>(stats[0]) > n ||
                    uint64_t(scratch) + stats[0] > 8ULL * n ||
                    uint64_t(requests) + stats[1] > n)
                    aclshmemi_kernel_abort("combine plan scratch/relay capacity rank=%u source=%u\n", rank, source);
                stats[2] = scratch; stats[3] = requests; Flush(stats);
                auto range = Meta(p.relay_ranges, uint64_t(laneIndex) * r + source);
                range[0] = requests; range[1] = stats[1]; Flush(range);
                if (laneIndex == Lane(rank)) {
                    auto sc = Meta(p.source_counts, source); Flush(sc);
                    if (uint64_t(results) + sc[0] > n)
                        aclshmemi_kernel_abort("combine plan tx capacity rank=%u\n", rank);
                    sc[2] = results; sc[4] = scratch; Flush(sc);
                    results += sc[0];
                    auto gc = Meta(p.counts_send, source); Flush(gc);
                    gc[1] = groups; groups += gc[0]; Flush(gc);
                }
                scratch += stats[0]; requests += stats[1];
            }
        }
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemi_sync_core<true>();
    }
    __aicore__ void WriteMember(uint32_t ingress, uint32_t head, uint32_t txIndex,
        uint32_t count, uint32_t member, uint32_t address, uint32_t* requests,
        PlanWriter& contributors, PlanWriter* relays) {
        if (ingress == rank) {
            Append(contributors, member == 0U ? txIndex : 0U,
                   member == 0U ? count : 0U, server ? address : address % n, 0U);
        }
        if (address / n == rank) {
            const uint32_t lane = Lane(ingress), ordinal = requests[lane]++;
            if (ordinal >= n)
                aclshmemi_kernel_abort("combine plan relay row capacity rank=%u\n", rank);
            Append(relays[lane], address % n, head + member);
        }
    }
    // Fused decode/count/emit for the staged path. Most top-k routes belong to
    // other servers: retain only this rank's active rows and our domain's
    // suppressed contributors, instead of populating and revisiting K-wide
    // raw/peer/active/primary/member arrays for every token.
    __aicore__ void StageToken(LocalTensor<int32_t> routes, uint32_t offset,
        uint32_t source, uint32_t& txIndex, uint32_t& groupIndex,
        uint32_t& finalBase, uint32_t& direct, uint32_t* scratch,
        uint32_t* requests, PlanWriter& txWriter, PlanWriter& contributors,
        PlanWriter* relays) {
        uint32_t ownRows[16], suppressedRows[16];
        uint32_t ownCount = 0U, suppressedCount = 0U, primaries = 0U;
        uint32_t representative = 0U, representativePeer = 0U, ownRepresentative = 0U;
        const bool local = LocalSource(source);
        for (uint32_t j = 0U; j < k; ++j) {
            const int32_t encoded = routes.GetValue(offset + j);
            const uint32_t address = Raw(encoded), peer = address / n;
            if (peer >= r)
                aclshmemi_kernel_abort("combine plan invalid dst source=%u rank=%u\n", source, rank);
            const bool primary = Primary(encoded);
            const bool active = primary || (server && peer / 8U == source / 8U);
            if (active && peer < rank) ++finalBase;
            if (peer == rank && active) {
                if (primary) ownRepresentative = ownCount;
                ownRows[ownCount++] = address % n;
            }
            if (LocalDomain(peer)) {
                if (primary) {
                    ++primaries; representative = address; representativePeer = peer;
                } else {
                    suppressedRows[suppressedCount++] = address;
                }
            }
        }
        direct += ownCount;
        uint32_t group = 0xffffffffU, head = 0xffffffffU, members = 1U;
        if (suppressedCount) {
            if (primaries != 1U)
                aclshmemi_kernel_abort("combine plan ambiguous primary source=%u rank=%u\n", source, rank);
            if (representativePeer == rank) group = groupIndex++;
            if (!local) {
                const uint32_t lane = Lane(representativePeer);
                const uint32_t groupHead = scratch[lane], count = suppressedCount + 1U;
                WriteMember(representativePeer, groupHead, txIndex + ownRepresentative,
                            count, 0U, representative, requests, contributors, relays);
                for (uint32_t member = 0U; member < suppressedCount; ++member)
                    WriteMember(representativePeer, groupHead, txIndex + ownRepresentative,
                                count, member + 1U, suppressedRows[member], requests, contributors, relays);
                scratch[lane] += count;
                if (representativePeer == rank) {
                    head = groupHead; members = count; --direct;
                }
            }
        }
        for (uint32_t j = 0U; j < ownCount; ++j) {
            const bool isHead = group != 0xffffffffU && j == ownRepresentative;
            Append(txWriter, ownRows[j], isHead ? group : 0xffffffffU,
                   isHead ? head : 0xffffffffU, isHead ? members : 1U);
        }
        txIndex += ownCount;
    }
    __aicore__ void BuildCombineMetadata(bool staged = false, bool synchronize = true) {
        const uint32_t pageRows = (kTile / k) * k;
        for (uint32_t source = core; source < r; source += coreCount) {
            auto sc = Meta(p.source_counts, source);
            auto gc = Meta(p.counts_send, source);
            if (!staged) { Flush(sc); Flush(gc); }
            uint32_t txIndex = staged ? 0U : sc[2], groupIndex = staged ? 0U : gc[1];
            uint32_t finalBase = 0U, direct = 0U;
            uint32_t scratch[8] = {}, requests[8] = {};
            // Routes: [0,32KiB), [64,96KiB). Writer banks: [32,38KiB),
            // [96,102KiB). Disjoint banks allow load/store overlap with scalar fill.
            auto txWriter = staged ? StagedWriter(32768U, 0U, source) :
                                     Writer(32768U, q.tx_rows, txIndex, 4U);
            auto contributors = staged ? StagedWriter(33792U, 1U, source) :
                                         Writer(33792U, q.contributor, sc[4], 4U);
            PlanWriter relays[8];
            for (uint32_t lane = 0U; lane < Width(); ++lane) {
                auto stats = Meta(p.relay_ranges_send, uint64_t(lane) * r + source);
                if (!staged) {
                    Flush(stats); scratch[lane] = stats[2]; requests[lane] = stats[3];
                }
                relays[lane] = staged ? StagedWriter(34816U + lane * 512U, 2U, source, lane) :
                    Writer(34816U + lane * 512U, q.relay_send,
                           uint64_t(lane) * n + requests[lane], 2U);
            }
            PrefetchRoutes(source, 0U, n < pageRows ? n : pageRows, 0U, !staged);
            for (uint32_t first = 0U; first < n; first += pageRows) {
                const uint32_t size = n - first < pageRows ? n - first : pageRows;
                const uint32_t bank = (first / pageRows) & 1U;
                auto routes = AwaitRoutes(bank);
                const uint32_t next = first + size;
                if (next < n)
                    PrefetchRoutes(source, next, n - next < pageRows ? n - next : pageRows,
                                   bank ^ 1U, !staged);
                for (uint32_t offset = 0U; offset < size; offset += k) {
                    if (staged) {
                        StageToken(routes, offset, source, txIndex, groupIndex, finalBase, direct,
                                   scratch, requests, txWriter, contributors, relays);
                        continue;
                    }
                    TokenRoutes token;
                    DecodeToken(routes, offset, source, token);
                    for (uint32_t j = 0U; j < k; ++j) {
                        const uint32_t peer = token.peer[j];
                        const bool ownGroup = peer == rank && token.primary[j] && token.members[j] > 1U;
                        const uint32_t group = ownGroup ? groupIndex++ : 0xffffffffU;
                        uint32_t head = 0xffffffffU;
                        if (LocalDomain(peer) && !LocalSource(source) &&
                            token.primary[j] && token.members[j] > 1U) {
                            const uint32_t lane = Lane(peer), count = token.members[j];
                            head = scratch[lane];
                            WriteMember(peer, head, txIndex, count, 0U, token.raw[j], requests,
                                        contributors, relays);
                            for (uint32_t member = 0U; member < token.suppressedCount; ++member)
                                WriteMember(peer, head, txIndex, count, member + 1U,
                                            token.raw[token.suppressed[member]], requests, contributors, relays);
                            scratch[lane] += count;
                        }
                        if (peer == rank && token.active[j]) {
                            Append(txWriter, token.raw[j] % n, group, head,
                                   LocalSource(source) ? 1U : token.members[j]);
                            ++txIndex;
                        }
                    }
                }
            }
            FinishWriter(txWriter);
            FinishWriter(contributors);
            for (uint32_t lane = 0U; lane < Width(); ++lane) FinishWriter(relays[lane]);
            WaitWriter();
            if (staged) {
                sc[0] = txIndex; sc[1] = finalBase; sc[2] = 0;
                sc[3] = direct; sc[4] = 0; sc[5] = scratch[Lane(rank)]; sc[6] = 0;
                Flush(sc);
                gc[0] = groupIndex; gc[1] = 0; Flush(gc);
                for (uint32_t lane = 0U; lane < Width(); ++lane) {
                    auto stats = Meta(p.relay_ranges_send, uint64_t(lane) * r + source);
                    stats[0] = scratch[lane]; stats[1] = requests[lane];
                    stats[2] = 0; stats[3] = 0; Flush(stats);
                }
                continue;
            }
            if (txIndex != uint32_t(sc[2] + sc[0]) || groupIndex != uint32_t(gc[1] + gc[0]))
                aclshmemi_kernel_abort("combine plan count/fill mismatch rank=%u source=%u\n", rank, source);
            for (uint32_t lane = 0U; lane < Width(); ++lane) {
                auto stats = Meta(p.relay_ranges_send, uint64_t(lane) * r + source);
                if (scratch[lane] != uint32_t(stats[2] + stats[0]) ||
                    requests[lane] != uint32_t(stats[3] + stats[1]))
                    aclshmemi_kernel_abort("combine plan relay prefix mismatch rank=%u source=%u\n", rank, source);
            }
        }
        DataSyncBarrier<MemDsbT::DDR>();
        if (synchronize) aclshmemi_sync_core<true>();
    }

    // Split a source into independent 256-token tasks. One source per core
    // otherwise leaves a 2-vs-1 (64P) / 3-vs-2 (128P) tail with 56 cores.
    // Temporary record capacity remains N/source: tile t starts at t*256*K.
    // Prefixes restore exact source/token order; no unordered atomic appends.
    __aicore__ uint32_t SourceTiles() const { return (n / k + 255U) / 256U; }
    __aicore__ bool UseTiledPlan() const {
        return UseStagedPlan() && SourceTiles() > 1U &&
               uint64_t(r) * SourceTiles() * 256U <= uint64_t(n) * 512U;
    }
    __aicore__ __gm__ uint32_t* TileStats(uint32_t task) {
        // relay_publish is otherwise used only by the mutually exclusive
        // fallback route cache. Each task owns isolated cachelines (256B).
        // [0..5]: TX/group/finalBase/direct counts, TX/group prefixes;
        // [16+4*lane..+3]: scratch/request counts and their tile prefixes.
        return reinterpret_cast<__gm__ uint32_t*>(ws + p.relay_publish + uint64_t(task) * 256U);
    }
    __aicore__ void PrefetchTask(uint32_t task, uint32_t bank) {
        const uint32_t source = task / SourceTiles();
        const uint32_t first = (task % SourceTiles()) * 256U * k;
        const uint32_t size = n - first < 256U * k ? n - first : 256U * k;
        PrefetchRoutes(source, first, size, bank, false);
    }
    __aicore__ void StoreSimtRows(uint32_t kind, uint32_t source, uint32_t lane,
        uint32_t first, uint32_t column, uint32_t ubOffset) {
        auto totals = Ub<uint32_t>(kPlanTotalsUb, 64U);
        const uint32_t begin = totals.GetValue(32U + column);
        const uint32_t count = totals.GetValue(column) - begin;
        if (!count) return;
        const uint32_t words = kind == 2U ? 2U : 4U;
        auto out = reinterpret_cast<__gm__ uint32_t*>(ws + p.relay_hidden +
            StagedTable(kind, source, lane)) + uint64_t(first + begin) * words;
        // All output UB ranges are disjoint and VF is already retired. Queue
        // every nonempty table, then one completion covers the entire slice.
        Store(out, Ub<uint32_t>(ubOffset, 64U * 16U * words), count * words);
        writerPending = true;
    }
    __aicore__ void StageTaskSimt(LocalTensor<int32_t> routes, uint32_t source,
                                  uint32_t first, uint32_t size) {
        auto decoded = Ub<uint32_t>(kPlanDecodedUb, 64U * 40U);
        auto prefix = Ub<uint32_t>(kPlanPrefixUb, 21U * 64U);
        auto totals = Ub<uint32_t>(kPlanTotalsUb, 64U);
        for (uint32_t col = 0U; col < kPlanSimtColumns; ++col) totals.SetValue(col, 0U);
        for (uint32_t offset = 0U; offset < size; offset += kPlanSimtTokens * k) {
            const uint32_t left = (size - offset) / k;
            const uint32_t tokens = left < kPlanSimtTokens ? left : kPlanSimtTokens;
            // Finish the preceding slice's byte-exact stores before VF reuses UB.
            WaitWriter();
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            asc_vf_call<BuildCombineSliceSimt>(dim3(64),
                (__ubuf__ int32_t*)routes.GetPhyAddr() + offset,
                (__ubuf__ uint32_t*)decoded.GetPhyAddr(),
                (__ubuf__ uint32_t*)prefix.GetPhyAddr(),
                (__ubuf__ uint32_t*)totals.GetPhyAddr(),
                (__ubuf__ uint32_t*)Ub<uint32_t>(kPlanChunksUb, 21U * 8U).GetPhyAddr(),
                (__ubuf__ uint32_t*)Ub<uint32_t>(kPlanTxUb, 4096U).GetPhyAddr(),
                (__ubuf__ uint32_t*)Ub<uint32_t>(kPlanContributorUb, 4096U).GetPhyAddr(),
                (__ubuf__ uint32_t*)Ub<uint32_t>(kPlanRelayUb, 16384U).GetPhyAddr(),
                tokens, k, n, r, rank, source, static_cast<uint32_t>(server));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if (totals.GetValue(20U))
                aclshmemi_kernel_abort("combine SIMT plan invalid routes rank=%u source=%u\n", rank, source);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            StoreSimtRows(0U, source, 0U, first, 0U, kPlanTxUb);
            StoreSimtRows(1U, source, 0U, first, 4U + Lane(rank), kPlanContributorUb);
            for (uint32_t lane = 0U; lane < Width(); ++lane)
                StoreSimtRows(2U, source, lane, first, 12U + lane,
                              kPlanRelayUb + lane * kPlanRelayLaneBytes);
            if (writerPending) SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
        WaitWriter();
    }
    __aicore__ void BuildTiledMetadata() {
        const uint32_t tiles = SourceTiles(), tasks = r * tiles;
        if (core < tasks) PrefetchTask(core, 0U);
        for (uint32_t task = core; task < tasks; task += coreCount) {
            const uint32_t source = task / tiles, first = (task % tiles) * 256U * k;
            const uint32_t size = n - first < 256U * k ? n - first : 256U * k;
            const uint32_t bank = (task / coreCount) & 1U;
            auto routes = AwaitRoutes(bank);
            if (task + coreCount < tasks) PrefetchTask(task + coreCount, bank ^ 1U);
            StageTaskSimt(routes, source, first, size);
            auto totals = Ub<uint32_t>(kPlanTotalsUb, 64U);
            auto stats = TileStats(task);
            stats[0] = totals.GetValue(0U); stats[1] = totals.GetValue(1U);
            stats[2] = totals.GetValue(2U); stats[3] = totals.GetValue(3U);
            for (uint32_t lane = 0U; lane < Width(); ++lane) {
                stats[16U + lane * 4U] = totals.GetValue(4U + lane);
                stats[17U + lane * 4U] = totals.GetValue(12U + lane);
            }
            Flush(stats); Flush(stats + 16U);
            if (Width() > 4U) Flush(stats + 32U);
        }
        // Caller joins all writers before reading tile statistics. This also
        // permits scan compute time and the load-imbalance wait to be separated.
        DataSyncBarrier<MemDsbT::DDR>();
    }
    __aicore__ bool TryBuildTiledSourceCountsSimt() {
        const uint32_t tiles = SourceTiles();
        if (tiles > 256U) return false; // Bounded UB, scalar fallback for huge S.
        auto rows = Ub<uint32_t>(0U, tiles * 64U);
        auto totals = Ub<uint32_t>(65536U, 32U);
        for (uint32_t source = core; source < r; source += coreCount) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            Load(rows, TileStats(source * tiles), tiles * 64U);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<PrefixCombineTaskStatsSimt>(dim3(32),
                (__ubuf__ uint32_t*)rows.GetPhyAddr(),
                (__ubuf__ uint32_t*)totals.GetPhyAddr(), tiles, Width());
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            Store(TileStats(source * tiles), rows, tiles * 64U);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            auto sc = Meta(p.source_counts, source);
            sc[0] = totals.GetValue(0U); sc[1] = totals.GetValue(2U); sc[2] = 0;
            sc[3] = totals.GetValue(3U); sc[4] = 0;
            sc[5] = totals.GetValue(4U + Lane(rank)); sc[6] = 0; Flush(sc);
            auto gc = Meta(p.counts_send, source);
            gc[0] = totals.GetValue(1U); gc[1] = 0; Flush(gc);
            for (uint32_t lane = 0U; lane < Width(); ++lane) {
                auto stats = Meta(p.relay_ranges_send, uint64_t(lane) * r + source);
                stats[0] = totals.GetValue(4U + lane);
                stats[1] = totals.GetValue(4U + Width() + lane);
                stats[2] = 0; stats[3] = 0; Flush(stats);
            }
        }
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemi_sync_core<true>();
        return true;
    }
    __aicore__ void BuildTiledSourceCounts() {
        if (TryBuildTiledSourceCountsSimt()) return;
        const uint32_t tiles = SourceTiles();
        for (uint32_t source = core; source < r; source += coreCount) {
            uint32_t tx = 0U, groups = 0U, finalBase = 0U, direct = 0U;
            uint32_t scratch[8] = {}, requests[8] = {};
            for (uint32_t tile = 0U; tile < tiles; ++tile) {
                auto stats = TileStats(source * tiles + tile);
                Flush(stats); Flush(stats + 16U);
                if (Width() > 4U) Flush(stats + 32U);
                stats[4] = tx; stats[5] = groups;
                tx += stats[0]; groups += stats[1]; finalBase += stats[2]; direct += stats[3];
                for (uint32_t lane = 0U; lane < Width(); ++lane) {
                    const uint32_t off = 16U + lane * 4U;
                    stats[off + 2U] = scratch[lane]; stats[off + 3U] = requests[lane];
                    scratch[lane] += stats[off]; requests[lane] += stats[off + 1U];
                }
                Flush(stats); Flush(stats + 16U);
                if (Width() > 4U) Flush(stats + 32U);
            }
            auto sc = Meta(p.source_counts, source);
            sc[0] = tx; sc[1] = finalBase; sc[2] = 0;
            sc[3] = direct; sc[4] = 0; sc[5] = scratch[Lane(rank)]; sc[6] = 0;
            Flush(sc);
            auto gc = Meta(p.counts_send, source);
            gc[0] = groups; gc[1] = 0; Flush(gc);
            for (uint32_t lane = 0U; lane < Width(); ++lane) {
                auto stats = Meta(p.relay_ranges_send, uint64_t(lane) * r + source);
                stats[0] = scratch[lane]; stats[1] = requests[lane];
                stats[2] = 0; stats[3] = 0; Flush(stats);
            }
        }
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemi_sync_core<true>();
    }
    __aicore__ void CompactTiledMetadata() {
        const uint32_t tiles = SourceTiles();
        for (uint32_t task = core; task < r * tiles; task += coreCount) {
            const uint32_t source = task / tiles, first = (task % tiles) * 256U * k;
            auto tile = TileStats(task);
            Flush(tile); Flush(tile + 16U);
            if (Width() > 4U) Flush(tile + 32U);
            auto sc = Meta(p.source_counts, source); Flush(sc);
            auto gc = Meta(p.counts_send, source); Flush(gc);
            const uint32_t txBase = sc[2] + tile[4], groupBase = gc[1] + tile[5];
            const uint32_t own = 16U + Lane(rank) * 4U, scratchBase = sc[4] + tile[own + 2U];
            CompactTable(0U, source, 0U, tile[0], q.tx_rows + uint64_t(txBase) * 16U,
                         txBase, groupBase, scratchBase, first);
            CompactTable(1U, source, 0U, tile[own], q.contributor + uint64_t(scratchBase) * 16U,
                         txBase, 0U, 0U, first);
            for (uint32_t lane = 0U; lane < Width(); ++lane) {
                auto stats = Meta(p.relay_ranges_send, uint64_t(lane) * r + source); Flush(stats);
                const uint32_t off = 16U + lane * 4U;
                CompactTable(2U, source, lane, tile[off + 1U],
                    q.relay_send + (uint64_t(lane) * n + stats[3] + tile[off + 3U]) * 8U,
                    0U, 0U, stats[2] + tile[off + 2U], first);
            }
        }
    }
    __aicore__ void CompactTable(uint32_t kind, uint32_t source, uint32_t lane,
        uint32_t count, uint64_t output, uint32_t txBase, uint32_t groupBase,
        uint32_t scratchBase, uint32_t inputFirst = 0U) {
        const uint32_t words = kind == 2U ? 2U : 4U;
        auto input = reinterpret_cast<__gm__ uint32_t*>(ws + p.relay_hidden +
            StagedTable(kind, source, lane)) + uint64_t(inputFirst) * words;
        auto dest = reinterpret_cast<__gm__ uint32_t*>(plan + output);
        // Only valid records are copied. MTE byte stores are required: adjacent
        // source segments may share cachelines and are compacted by different cores.
        for (uint32_t first = 0U; first < count; first += 64U) {
            const uint32_t size = count - first < 64U ? count - first : 64U;
            auto rows = Ub<uint32_t>(((first / 64U) & 1U) * 4096U, 64U * words);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            Load(rows, input + uint64_t(first) * words, size * words);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            asc_vf_call<AdjustCombineRowsSimt>(dim3(64),
                (__ubuf__ uint32_t*)rows.GetPhyAddr(), size, kind, txBase, groupBase, scratchBase);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            WaitWriter();
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            Store(dest + uint64_t(first) * words, rows, size * words);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            writerPending = true;
        }
        WaitWriter();
    }
    __aicore__ void CompactStagedMetadata() {
        for (uint32_t source = core; source < r; source += coreCount) {
            auto sc = Meta(p.source_counts, source); Flush(sc);
            auto gc = Meta(p.counts_send, source); Flush(gc);
            CompactTable(0U, source, 0U, sc[0], q.tx_rows + uint64_t(sc[2]) * 16U,
                         sc[2], gc[1], sc[4]);
            CompactTable(1U, source, 0U, sc[5], q.contributor + uint64_t(sc[4]) * 16U,
                         sc[2], 0U, 0U);
            for (uint32_t lane = 0U; lane < Width(); ++lane) {
                auto stats = Meta(p.relay_ranges_send, uint64_t(lane) * r + source); Flush(stats);
                CompactTable(2U, source, lane, stats[1],
                    q.relay_send + (uint64_t(lane) * n + stats[3]) * 8U,
                    0U, 0U, stats[2]);
            }
        }
    }
    __aicore__ void FinishPlan() {
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemi_sync_core<true>();
        // relay_ranges_send is dead after BuildCombineMetadata. Reuse its exact
        // 8*R*64B capacity as R rows of 512B: 64 partition counts/prefixes,
        // followed by the peer total at word 64. No plan/workspace ABI change.
        // All histogram accesses below use MTE, never scalar GM cacheline RMW:
        // simultaneous column writers share cachelines but write disjoint bytes.
        auto histogram = reinterpret_cast<__gm__ uint32_t*>(ws + p.relay_ranges_send);
        const uint32_t tokenCount = n / k;
        const uint32_t begin = uint64_t(tokenCount) * core / coreCount;
        const uint32_t end = uint64_t(tokenCount) * (core + 1U) / coreCount;
        const uint32_t pageTokens = kTile / k;
        // UB: routes [0,32KiB), padded counters [32,64KiB), cursors
        // [64,68KiB), output [72,80KiB). R<=1024, K<=16.
        auto counts = Ub<uint32_t>(32768U, r * 8U);
        for (uint32_t peer = 0U; peer < r; ++peer) counts.SetValue(peer * 8U, 0U);
        for (uint32_t first = begin; first < end; first += pageTokens) {
            const uint32_t size = end - first < pageTokens ? end - first : pageTokens;
            auto routes = LoadRoutes(rank, first * k, size * k);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            asc_vf_call<CountFinalRoutesSimt>(dim3(256),
                (__ubuf__ int32_t*)routes.GetPhyAddr(),
                (__ubuf__ uint32_t*)counts.GetPhyAddr(), size * k, n, rank,
                static_cast<uint32_t>(server));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
        GlobalTensor<uint32_t> gm;
        gm.SetGlobalBuffer(histogram + core);
        // UB source block stride is in 32B units; GM destination stride in bytes.
        DataCopyExtParams column{static_cast<uint16_t>(r), 4U, 0U, 508U, 0U};
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(gm, counts, column);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemi_sync_core<true>();

        // Parallel prefix by peer. Include empty partitions so tiny token cases
        // and R>64 use exactly the same protocol as the 64P/128P perf cases.
        auto row = Ub<uint32_t>(0U, 128U);
        for (uint32_t peer = core; peer < r; peer += coreCount) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            Load(row, histogram + uint64_t(peer) * 128U, coreCount);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            uint32_t total = 0U;
            for (uint32_t part = 0U; part < coreCount; ++part) {
                const uint32_t count = row.GetValue(part);
                row.SetValue(part, total); total += count;
            }
            for (uint32_t part = coreCount; part < kCores; ++part) row.SetValue(part, 0U);
            row.SetValue(64U, total);
            for (uint32_t i = 65U; i < 128U; ++i) row.SetValue(i, 0U);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            Store(histogram + uint64_t(peer) * 128U, row, 128U);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemi_sync_core<true>();

        // Read this partition's prefix and all peer totals in two strided DMAs.
        auto prefixes = Ub<uint32_t>(0U, r * 8U);
        DataCopyExtParams readColumn{static_cast<uint16_t>(r), 4U, 508U, 0U, 0U};
        DataCopyPadExtParams<uint32_t> pad{false, 0U, 0U, 0U};
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        gm.SetGlobalBuffer(histogram + core);
        DataCopyPad(prefixes, gm, readColumn, pad);
        gm.SetGlobalBuffer(histogram + 64U);
        DataCopyPad(counts, gm, readColumn, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        auto cursors = Ub<uint32_t>(65536U, r);
        uint32_t peerBase = 0U;
        for (uint32_t peer = 0U; peer < r; ++peer) {
            cursors.SetValue(peer, peerBase + prefixes.GetValue(peer * 8U));
            peerBase += counts.GetValue(peer * 8U);
        }

        // Stable final ordinal = all earlier peers + earlier token partitions
        // for this peer + earlier routes in this partition. This is the old
        // final_map result, generated directly without that scalar GM table.
        auto packed = Ub<uint32_t>(73728U, 64U * 32U);
        for (uint32_t first = begin; first < end; first += 64U) {
            const uint32_t size = end - first < 64U ? end - first : 64U;
            auto routes = LoadRoutes(rank, first * k, size * k);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            asc_vf_call<BuildFinalRoutesSimt>(dim3(128),
                (__ubuf__ int32_t*)routes.GetPhyAddr(),
                (__ubuf__ uint32_t*)Ub<uint32_t>(kFinalChunksUb, 32U * (r + 1U)).GetPhyAddr(),
                (__ubuf__ uint32_t*)cursors.GetPhyAddr(),
                (__ubuf__ uint32_t*)packed.GetPhyAddr(), size, k, n, r, rank,
                static_cast<uint32_t>(server));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            Store(reinterpret_cast<__gm__ uint32_t*>(plan + q.final_routes) + uint64_t(first) * 32U,
                  packed, size * 32U);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemi_sync_core<true>();
    }
    // Final metadata is immutable now. Retired workspace contributor records
    // publish {sourcePageSeq,rowOffset} at 64B spacing; each source owns a
    // disjoint scratch interval, so no cacheline RMW can touch another writer.
    __aicore__ void PublishRingScratchMap(uint32_t first, uint32_t count,
                                        uint32_t page) {
        constexpr uint32_t rowsPerBatch = 128U;
        auto values = Ub<uint32_t>(8192U, rowsPerBatch * 8U);
        for (uint32_t begin = 0U; begin < count; begin += rowsPerBatch) {
            const uint32_t size = count - begin < rowsPerBatch ? count - begin : rowsPerBatch;
            for (uint32_t j = 0U; j < size; ++j) {
                // DataCopyPad reads one padded 32B UB block for each 8B record.
                values.SetValue(j * 8U, page);
                values.SetValue(j * 8U + 1U, begin + j);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            GlobalTensor<uint32_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
                ws + p.contributor + uint64_t(first + begin) * 64U));
            DataCopyExtParams cp{static_cast<uint16_t>(size), 8U, 0U, 56U, 0U};
            DataCopyPad(gm, values, cp);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }
    __aicore__ void StoreRingPage(uint32_t index, uint32_t source,
                                 uint32_t first, uint32_t slots, uint32_t groups) {
        auto values = Ub<uint32_t>(4096U, 16U);
        for (uint32_t j = 0U; j < 16U; ++j) values.SetValue(j, 0U);
        values.SetValue(0U, first); values.SetValue(1U, slots);
        values.SetValue(2U, groups); values.SetValue(3U, source);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        Store(reinterpret_cast<__gm__ uint32_t*>(plan + q.ring_pages + uint64_t(index) * 64U),
              values, 16U);
        const uint64_t mirror = p.relay_publish + CombinePutAlign(uint64_t(r) * 64U);
        Store(reinterpret_cast<__gm__ uint32_t*>(ws + mirror + uint64_t(index) * 64U),
              values, 16U);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
    // Count whole-group row-packed pages in parallel before the source prefix.
    // A full non-tail page contains >=64 groups (each group has <=K rows),
    // so the existing ceil(N/64)+R descriptor/ready capacity remains sufficient.
    __aicore__ void CountRingSourcePages() {
        auto rows = Ub<uint32_t>(0U, 64U * 4U);
        for (uint32_t source = core; source < r; source += coreCount) {
            auto sc = Meta(p.source_counts, source); Flush(sc);
            if (sc[0] < 0 || sc[3] < 0 || sc[3] > sc[0] || sc[4] < 0 || sc[5] < 0 ||
                uint64_t(sc[4]) + uint32_t(sc[5]) > 8ULL * n || uint32_t(sc[5]) > n)
                aclshmemi_kernel_abort("combine ring invalid count range rank=%u source=%u\n", rank, source);
            const uint32_t begin = static_cast<uint32_t>(sc[4]);
            const uint32_t count = static_cast<uint32_t>(sc[5]);
            uint32_t cursor = 0U, loadedBegin = 0U, loadedCount = 0U;
            uint32_t pageRows = 0U, pages = 0U, groups = 0U;
            while (cursor < count) {
                if (cursor >= loadedBegin + loadedCount) {
                    loadedBegin = cursor;
                    loadedCount = count - cursor < 64U ? count - cursor : 64U;
                    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    Load(rows, reinterpret_cast<__gm__ uint32_t*>(plan + q.contributor) +
                               uint64_t(begin + cursor) * 4U, loadedCount * 4U);
                    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                }
                const uint32_t members = rows.GetValue((cursor - loadedBegin) * 4U + 1U);
                if (members < 2U || members > k || members > count - cursor)
                    aclshmemi_kernel_abort("combine ring invalid count group rank=%u source=%u\n", rank, source);
                if (pageRows && pageRows + members > p.ring_page_rows) {
                    ++pages; pageRows = 0U;
                }
                pageRows += members; cursor += members; ++groups;
            }
            if (pageRows) ++pages;
            if (groups != uint32_t(sc[0] - sc[3]) || (LocalSource(source) && (groups || count)))
                aclshmemi_kernel_abort("combine ring count groups mismatch rank=%u source=%u\n", rank, source);
            sc[6] = pages; Flush(sc);
        }
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemi_sync_core<true>();
    }
    __aicore__ void BuildRingPagePrefix() {
        if (core == 0U) {
            uint64_t pages = 0U, groups = 0U;
            for (uint32_t source = 0U; source < r; ++source) {
                auto sc = Meta(p.source_counts, source); Flush(sc);
                if (sc[0] < 0 || sc[3] < 0 || sc[3] > sc[0])
                    aclshmemi_kernel_abort("combine ring invalid result counts rank=%u source=%u\n", rank, source);
                const uint32_t heads = static_cast<uint32_t>(sc[0] - sc[3]);
                if (LocalSource(source) && (heads != 0U || sc[5] != 0))
                    aclshmemi_kernel_abort("combine ring local source has scratch rank=%u source=%u\n", rank, source);
                if (sc[6] < 0)
                    aclshmemi_kernel_abort("combine ring invalid page count rank=%u source=%u\n", rank, source);
                const uint32_t count = static_cast<uint32_t>(sc[6]);
                if (pages + count > p.ring_page_capacity || groups + heads > n)
                    aclshmemi_kernel_abort("combine ring page capacity rank=%u source=%u\n", rank, source);
                sc[6] = count; sc[7] = static_cast<uint32_t>(pages);
                for (uint32_t word = 8U; word < 16U; ++word) sc[word] = 0;
                Flush(sc); pages += count; groups += heads;
            }
        }
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemi_sync_core<true>();
    }
    __aicore__ void BuildRingLocalPages() {
        auto rows = Ub<uint32_t>(0U, 64U * 4U);
        for (uint32_t source = core; source < r; source += coreCount) {
            auto sc = Meta(p.source_counts, source); Flush(sc);
            const uint32_t scratchBegin = static_cast<uint32_t>(sc[4]);
            const uint32_t scratchCount = static_cast<uint32_t>(sc[5]);
            const uint32_t expectedGroups = static_cast<uint32_t>(sc[0] - sc[3]);
            const uint32_t pageCount = static_cast<uint32_t>(sc[6]);
            const uint32_t pageBase = static_cast<uint32_t>(sc[7]);
            if (uint64_t(scratchBegin) + scratchCount > 8ULL * n || scratchCount > n)
                aclshmemi_kernel_abort("combine ring invalid scratch range rank=%u source=%u\n", rank, source);
            uint32_t cursor = 0U, groupCount = 0U, page = 0U;
            uint32_t first = 0U, pageGroups = 0U, loadedBegin = 0U, loadedCount = 0U;
            while (cursor < scratchCount) {
                if (cursor >= loadedBegin + loadedCount) {
                    loadedBegin = cursor;
                    loadedCount = scratchCount - cursor < 64U ? scratchCount - cursor : 64U;
                    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    Load(rows, reinterpret_cast<__gm__ uint32_t*>(plan + q.contributor) +
                               uint64_t(scratchBegin + cursor) * 4U, loadedCount * 4U);
                    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                }
                const uint32_t off = (cursor - loadedBegin) * 4U;
                const uint32_t tx = rows.GetValue(off), members = rows.GetValue(off + 1U);
                if (members < 2U || members > k || members > scratchCount - cursor ||
                    tx < static_cast<uint32_t>(sc[2]) || uint64_t(tx) >= uint64_t(sc[2]) + sc[0])
                    aclshmemi_kernel_abort("combine ring invalid contributor group rank=%u source=%u\n", rank, source);
                // Close before the next whole group would overflow the input page.
                if (pageGroups && cursor - first + members > p.ring_page_rows) {
                    if (page >= pageCount)
                        aclshmemi_kernel_abort("combine ring page count overflow rank=%u source=%u\n", rank, source);
                    StoreRingPage(pageBase + page, source, scratchBegin + first,
                                  cursor - first, pageGroups);
                    PublishRingScratchMap(scratchBegin + first, cursor - first, page);
                    ++page; first = cursor; pageGroups = 0U;
                }
                cursor += members; ++groupCount; ++pageGroups;
                if (cursor == scratchCount) {
                    if (page >= pageCount || cursor - first > p.ring_page_rows)
                        aclshmemi_kernel_abort("combine ring oversized page rank=%u source=%u\n", rank, source);
                    StoreRingPage(pageBase + page, source, scratchBegin + first,
                                  cursor - first, pageGroups);
                    PublishRingScratchMap(scratchBegin + first, cursor - first, page);
                    ++page; first = cursor; pageGroups = 0U;
                }
            }
            if (groupCount != expectedGroups || page != pageCount)
                aclshmemi_kernel_abort("combine ring group/page count mismatch rank=%u source=%u\n", rank, source);
            // Mirror the complete cacheline so peer producers can consume the
            // new page count and the old scratch bounds without remote plan access.
            Copy(reinterpret_cast<__gm__ uint32_t*>(ws + p.relay_publish + uint64_t(source) * 64U),
                 reinterpret_cast<__gm__ uint32_t*>(sc), 16U);
        }
    }
    __aicore__ void BuildRingRelayMaps() {
        for (uint32_t task = core; task < Width() * r; task += coreCount) {
            const uint32_t lane = task / r, source = task % r;
            const uint32_t peer = server ? rank / 8U * 8U + lane : rank;
            auto mirror = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(ws + p.relay_publish, peer));
            auto scratch = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(ws + p.contributor, peer));
            auto summary = Ub<uint32_t>(0U, 16U);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            Load(summary, reinterpret_cast<__gm__ uint32_t*>(mirror + uint64_t(source) * 64U), 16U);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            const uint32_t scratchBegin = summary.GetValue(4U), scratchCount = summary.GetValue(5U);
            const uint32_t pageCount = summary.GetValue(6U), pageBase = summary.GetValue(7U);
            if (uint64_t(pageBase) + pageCount > p.ring_page_capacity ||
                uint64_t(scratchBegin) + scratchCount > 8ULL * n)
                aclshmemi_kernel_abort("combine ring invalid peer summary rank=%u peer=%u source=%u\n", rank, peer, source);
            auto range = Meta(p.relay_ranges, uint64_t(lane) * r + source); Flush(range);
            const uint32_t first = static_cast<uint32_t>(range[0]);
            const uint32_t count = static_cast<uint32_t>(range[1]);
            if (uint64_t(first) + count > n)
                aclshmemi_kernel_abort("combine ring invalid relay range rank=%u source=%u\n", rank, source);
            range[2] = pageCount; Flush(range);
            auto requests = Ub<uint32_t>(1024U, 64U * 2U);
            auto output = Ub<uint32_t>(8192U, 64U * 2U);
            for (uint32_t begin = 0U; begin < count; begin += 64U) {
                const uint32_t size = count - begin < 64U ? count - begin : 64U;
                const uint64_t ordinal = uint64_t(lane) * n + first + begin;
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                Load(requests, reinterpret_cast<__gm__ uint32_t*>(plan + q.relay_send) + ordinal * 2U,
                     size * 2U);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                for (uint32_t j = 0U; j < size; ++j) {
                    const uint32_t slot = requests.GetValue(j * 2U + 1U);
                    if (slot < scratchBegin || uint64_t(slot) >= uint64_t(scratchBegin) + scratchCount)
                        aclshmemi_kernel_abort("combine ring relay scratch mismatch rank=%u source=%u\n", rank, source);
                    // A padded 32B UB destination per 8B mapping. All loads
                    // retire before these banks are compacted or reused.
                    Load(Ub<uint32_t>(4096U + j * 32U, 2U),
                         reinterpret_cast<__gm__ uint32_t*>(scratch + uint64_t(slot) * 64U), 2U);
                }
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                for (uint32_t j = 0U; j < size; ++j) {
                    auto mapping = Ub<uint32_t>(4096U + j * 32U, 2U);
                    const uint32_t page = mapping.GetValue(0U), row = mapping.GetValue(1U);
                    if (page >= pageCount || row >= p.ring_page_rows)
                        aclshmemi_kernel_abort("combine ring invalid relay page mapping rank=%u source=%u\n", rank, source);
                    output.SetValue(j * 2U, page); output.SetValue(j * 2U + 1U, row);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                Store(reinterpret_cast<__gm__ uint32_t*>(plan + q.relay_page_map) + ordinal * 2U,
                      output, size * 2U);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }
        }
    }
    __aicore__ void BuildRingPages() {
        // All three old metadata paths join here, after final-route readers
        // retire. No route re-scan or modification of the original record ABI.
        CountRingSourcePages();
        BuildRingPagePrefix();
        BuildRingLocalPages();
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemx_sync_vec_all(); // publish temporary maps on every ingress
        BuildRingRelayMaps();
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemx_sync_vec_all(); // peer reads retire before temporary arena reuse
    }
    __aicore__ void RecordPlanningStages(const uint64_t* elapsed, uint32_t path) {
        // Eight per-core intervals, reduced into reserved plan header bytes
        // 128..415. Plan layout/version and all payload tables are unchanged.
        // Tile/cache scratch is dead after FinishPlan + PublishPlan's join.
        auto detail = reinterpret_cast<__gm__ uint64_t*>(plan + 128U);
        if (uint64_t(n) * 512U < uint64_t(coreCount) * 64U) {
            if (core == 0U) { detail[0] = 0U; Flush(detail); }
            return;
        }
        auto samples = reinterpret_cast<__gm__ uint64_t*>(ws + p.relay_publish);
        for (uint32_t stage = 0U; stage < 8U; ++stage)
            samples[uint64_t(core) * 8U + stage] = elapsed[stage];
        Flush(samples + uint64_t(core) * 8U);
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemi_sync_core<true>();
        if (core == 0U) {
            uint64_t minima[8], sums[8] = {}, maxima[8] = {}, owners[8] = {};
            for (uint32_t stage = 0U; stage < 8U; ++stage) minima[stage] = ~uint64_t(0);
            for (uint32_t c = 0U; c < coreCount; ++c) {
                auto row = samples + uint64_t(c) * 8U;
                Flush(row);
                for (uint32_t stage = 0U; stage < 8U; ++stage) {
                    const uint64_t value = row[stage];
                    sums[stage] += value;
                    if (value < minima[stage]) minima[stage] = value;
                    if (value > maxima[stage]) { maxima[stage] = value; owners[stage] = c; }
                }
            }
            detail[0] = 0x504c414e53544731ULL; // PLANSTG1
            detail[1] = coreCount; detail[2] = 8U; detail[3] = path;
            for (uint32_t stage = 0U; stage < 8U; ++stage) {
                detail[4U + stage * 4U] = minima[stage];
                detail[5U + stage * 4U] = sums[stage];
                detail[6U + stage * 4U] = maxima[stage];
                detail[7U + stage * 4U] = owners[stage];
            }
            for (uint32_t line = 0U; line < 36U; line += 8U) Flush(detail + line);
            DataSyncBarrier<MemDsbT::DDR>();
        }
    }
    __aicore__ void PublishPlan() {
        // All output tables already reside in the final tensor; no full-capacity
        // workspace -> plan snapshot copy. Publish the header only after stores.
        PipeBarrier<PIPE_ALL>();
        aclshmemi_sync_core<true>();
        if (core == 0U) {
            auto header = reinterpret_cast<__gm__ uint64_t*>(plan);
            header[0] = kCombineExecutionPlanVersion;
            header[1] = n; header[2] = h; header[3] = k;
            header[4] = r; header[5] = rank; header[6] = server;
            header[7] = CombineRingLayoutFingerprint(k);
            Flush(header);
        }
        DataSyncBarrier<MemDsbT::DDR>();
        aclshmemx_sync_vec_all();
    }
};
} // namespace CombineExecutionPlanning

__aicore__ __attribute__((noinline)) void RunCombinePlanning(
    GM_ADDR dst, GM_ADDR groups, GM_ADDR dups, GM_ADDR counts,
    ascend_moonep::PlanningTilingData tiling, uint64_t begin, uint64_t baseEnd) {
    using namespace CombineExecutionPlanning;
    ascend_moonep::MoonEPUdmaCombineTilingData t{};
    t.S = tiling.num_tokens; t.K = tiling.num_topk; t.H = tiling.combine_h;
    t.R = tiling.world_size; t.rank = tiling.rank; t.server_dedup = tiling.server_dedup;
    auto workspace = reinterpret_cast<__gm__ uint8_t*>(tiling.combine_workspace);
    auto plan = reinterpret_cast<__gm__ uint8_t*>(tiling.combine_plan);
    if (!t.K || t.K > 16U || !t.R || t.R > 1024U)
        aclshmemi_kernel_abort("planning combine invalid route dimensions\n");
    CombinePlanBuilder x{reinterpret_cast<__gm__ uint8_t*>(workspace),
        reinterpret_cast<const __gm__ int32_t*>(dst),
        reinterpret_cast<const __gm__ int32_t*>(groups),
        reinterpret_cast<const __gm__ int32_t*>(dups),
        reinterpret_cast<const __gm__ int32_t*>(counts),
        ascend_moonep::MakeCombinePutLayout(t.S * t.K, t.H, t.R, t.K),
        t.S * t.K, t.H, t.K, t.R, t.rank, static_cast<uint32_t>(GetBlockIdx()), t.server_dedup != 0U,
        reinterpret_cast<__gm__ uint8_t*>(plan), ascend_moonep::MakeCombineExecutionPlanLayout(t.S * t.K, t.R)};
    x.coreCount = tiling.block_dim;
    uint64_t elapsed[8];
    uint64_t previous = baseEnd;
    x.Init();
    uint64_t now = GetSystemCycle(); elapsed[0] = now - previous; previous = now;
    x.PublishRoutes();
    now = GetSystemCycle(); elapsed[1] = now - previous; previous = now;
    const bool tiled = x.UseTiledPlan(), staged = x.UseStagedPlan();
    if (tiled) x.BuildTiledMetadata();
    else if (staged) x.BuildCombineMetadata(true, false);
    else x.CountSources(false);
    now = GetSystemCycle(); elapsed[2] = now - previous; previous = now;
    DataSyncBarrier<MemDsbT::DDR>();
    aclshmemi_sync_core<true>();
    now = GetSystemCycle(); elapsed[3] = now - previous; previous = now;
    if (tiled) x.BuildTiledSourceCounts();
    x.BuildSourcePrefixes();
    now = GetSystemCycle(); elapsed[4] = now - previous; previous = now;
    if (tiled) x.CompactTiledMetadata();
    else if (staged) x.CompactStagedMetadata();
    else x.BuildCombineMetadata();
    now = GetSystemCycle(); elapsed[5] = now - previous; previous = now;
    x.FinishPlan();
    x.BuildRingPages();
    now = GetSystemCycle(); elapsed[6] = now - previous; previous = now;
    x.PublishPlan();
    now = GetSystemCycle(); elapsed[7] = now - previous;
    x.RecordPlanningStages(elapsed, tiled ? 4U : staged ? 1U : 0U);
    if (GetBlockIdx() == 0U) {
        const uint64_t end = GetSystemCycle();
        // Reserved second header cacheline, disjoint from PublishPlan's header.
        // Ascend950 system counter is 1GHz. No extra kernel/event at the boundary.
        auto timing = reinterpret_cast<__gm__ uint64_t*>(plan) + 8U;
        timing[0] = 0x504c414e54494d45ULL;
        timing[1] = tiling.block_dim;
        timing[2] = baseEnd - begin;
        timing[3] = end - baseEnd;
        timing[4] = end - begin;
        timing[5] = tiling.combine_mode;
        CombineExecutionPlanning::Flush(timing);
        DataSyncBarrier<MemDsbT::DDR>();
    }
}
