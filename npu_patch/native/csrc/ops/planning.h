#pragma once

#include <cstdint>

#include <torch/extension.h>

namespace ascend_moonep::ops {

int64_t PlanningSymmetricWorkspaceBytes(
    int64_t num_experts,
    int64_t num_max_tokens_per_rank,
    int64_t world_size);

int64_t PlanningLocalWorkspaceElements(
    int64_t num_tokens,
    int64_t num_topk,
    int64_t num_experts,
    int64_t world_size,
    int64_t core_num);

torch::Tensor Planning(
    torch::Tensor topk_idx,
    torch::Tensor dst,
    torch::Tensor cu_seqlens,
    torch::Tensor experts_to_copy,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor local_workspace,
    int64_t symmetric_workspace_addr,
    int64_t symmetric_workspace_bytes,
    int64_t num_tokens,
    int64_t num_topk,
    int64_t num_experts,
    int64_t num_max_tokens_per_rank,
    int64_t num_prefetch_slots,
    int64_t core_num,
    int64_t rank,
    int64_t world_size,
    bool use_udma,
    torch::Tensor combine_workspace = {}, int64_t hidden_size = 0);

torch::Tensor AllocateCombinePlan(
    torch::Tensor workspace, torch::Tensor dst, torch::Tensor groups,
    torch::Tensor dups, torch::Tensor counts,
    int64_t S, int64_t H, int64_t K, int64_t R, int64_t rank);

torch::Tensor PlanningPrepareCombinePlan(
    torch::Tensor workspace, torch::Tensor dst, torch::Tensor groups,
    torch::Tensor dups, torch::Tensor counts,
    int64_t S, int64_t H, int64_t K, int64_t R, int64_t rank);

}  // namespace ascend_moonep::ops
