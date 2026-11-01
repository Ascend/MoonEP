#pragma once

#include <torch/extension.h>

namespace ascend_moonep::ops {

// Caller-owned-output overload selected by ASCEND_MOONEP_COMBINE_USE_UDMA.
// local_workspace is a Buffer-owned symmetric tensor used as this rank's
// gathered payload staging area, allocated in the same SHMEM heap as inputs.
void MoonEPUdmaCombine(
    torch::Tensor hidden_nvsh,
    c10::optional<torch::Tensor> route_weights_nvs,
    torch::Tensor sync_buffer,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor output_sh,
    c10::optional<torch::Tensor> output_sk,
    int64_t S,
    int64_t H,
    int64_t K,
    int64_t R,
    int64_t rank,
    torch::Tensor local_workspace,
    c10::optional<torch::Tensor> execution_plan = c10::nullopt);

// Legacy rank-level dedup combine. `workspace` is one rank-local byte tensor
// that holds the gathered payload plus its bandwidth-scratch tail; the host op
// splits it, so the Buffer's symmetric PUT workspace is not used.
void MoonEPUdmaCombineLegacy(
    torch::Tensor hidden_nvsh,
    c10::optional<torch::Tensor> route_weights_nvs,
    torch::Tensor sync_buffer,
    torch::Tensor dst,
    torch::Tensor dup_groups,
    torch::Tensor dup_loffs,
    torch::Tensor dup_counts,
    torch::Tensor output_sh,
    c10::optional<torch::Tensor> output_sk,
    int64_t S,
    int64_t H,
    int64_t K,
    int64_t R,
    int64_t rank,
    torch::Tensor workspace);

}  // namespace ascend_moonep::ops
