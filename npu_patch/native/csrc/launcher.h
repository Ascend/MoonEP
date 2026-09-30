#pragma once

#include <cstdint>

#include <torch/extension.h>

#include "ascend_moonep/kernels/types.h"

void LaunchDeviceBarrierBench(int32_t iterations, int32_t block_dim);
void SynchronizeCurrentNpuStream();

void LaunchMoonEPCombineKernel(
    torch::Tensor hidden_buf,
    torch::Tensor weight_buf,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_sk_out,
    torch::Tensor route_weights_sk_out,
    const ascend_moonep::MoonEPCombineTilingData& tiling);

void LaunchMoonEPDispatchKernel(
    torch::Tensor hidden_sh,
    torch::Tensor route_weights,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_buf,
    torch::Tensor route_weights_buf,
    const ascend_moonep::MoonEPDispatchTilingData& tiling);

void LaunchMoonEPDispatchUdmaKernel(
    torch::Tensor hidden_sh,
    torch::Tensor route_weights,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_buf,
    torch::Tensor route_weights_buf,
    torch::Tensor sync_buffer,
    const ascend_moonep::MoonEPDispatchTilingData& tiling,
    uint64_t magic);

void LaunchMoonEPDispatchUdmaProfileKernel(
    torch::Tensor hidden_sh,
    torch::Tensor route_weights,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor hidden_buf,
    torch::Tensor route_weights_buf,
    torch::Tensor sync_buffer,
    torch::Tensor profile_cycles,
    const ascend_moonep::MoonEPDispatchTilingData& tiling,
    uint64_t magic);

void LaunchPlanningKernel(
    torch::Tensor topk_idx,
    torch::Tensor dst,
    torch::Tensor cu_seqlens,
    torch::Tensor experts_to_copy,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor tpe_all,
    torch::Tensor tpe_prefix,
    torch::Tensor balance,
    torch::Tensor z_tensor,
    torch::Tensor alloc_tensor,
    torch::Tensor alloc_cumsum,
    torch::Tensor expert_offsets,
    torch::Tensor core_expert_counts,
    torch::Tensor dup_workspace,
    torch::Tensor dup_key_workspace,
    const ascend_moonep::PlanningTilingData& tiling,
    void* symmetric_buffer,
    uint64_t ffts_addr);

void LaunchMoonEPUdmaCombinePlanKernel(
    torch::Tensor hidden, torch::Tensor weights, torch::Tensor dst,
    torch::Tensor groups, torch::Tensor dups, torch::Tensor counts,
    torch::Tensor out, torch::Tensor outw, torch::Tensor workspace,
    torch::Tensor plan,
    const ascend_moonep::MoonEPUdmaCombineTilingData& tiling,
    uint64_t magic);

void LaunchPlanningMetadataOnly(
    torch::Tensor dst, torch::Tensor groups, torch::Tensor dups, torch::Tensor counts,
    const ascend_moonep::PlanningTilingData& tiling);
