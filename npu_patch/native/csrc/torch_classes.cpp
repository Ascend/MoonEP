#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <acl/acl.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/script.h>

#include "shmem.h"

#include "csrc/common/check.h"
#include "csrc/launcher.h"
#include "csrc/ops/grad_reduce.h"
#include "csrc/ops/moonep_combine.h"
#include "csrc/ops/moonep_combine_udma.h"
#include "csrc/ops/moonep_dispatch.h"
#include "csrc/ops/planning.h"
#include "csrc/ops/prefetch.h"
#include "csrc/shmem_instance.h"

namespace ascend_moonep::torch_classes {
namespace {

constexpr std::size_t kDeviceLargePageBytes = 2 * 1024 * 1024;

std::size_t CheckedLocalMemSize(int64_t local_mem_size) {
    TORCH_CHECK(
        local_mem_size > 0,
        "local_mem_size must be greater than zero");
    TORCH_CHECK(
        local_mem_size % static_cast<int64_t>(kDeviceLargePageBytes) == 0,
        "local_mem_size must be aligned to DEVICE_LARGE_PAGE_SIZE (",
        kDeviceLargePageBytes,
        " bytes)");
    return static_cast<std::size_t>(local_mem_size);
}

void CheckShmem(const c10::intrusive_ptr<ShmemInstance>& shmem) {
    TORCH_CHECK(shmem != nullptr, "shmem must not be None");
    TORCH_CHECK(shmem->Initialized(), "ShmemInstance is not initialized");
}



}  // namespace

class ShmemWorkspaceOp : public torch::jit::CustomClassHolder {
public:
    explicit ShmemWorkspaceOp(int64_t local_mem_size)
        : shmem_ptr_(nullptr),
          local_mem_size_(CheckedLocalMemSize(local_mem_size)) {
        shmem_ptr_ = aclshmem_malloc(local_mem_size_);
        if (shmem_ptr_ == nullptr) {
            throw std::runtime_error("aclshmem_malloc returned nullptr for op local memory");
        }
        ASCEND_MOONEP_CHECK_ACL(aclrtMemset(shmem_ptr_, local_mem_size_, 0, local_mem_size_));
    }

    ~ShmemWorkspaceOp() override {
        destroy();
    }

    void destroy() {
        if (shmem_ptr_ != nullptr) {
            aclshmem_free(shmem_ptr_);
            shmem_ptr_ = nullptr;
        }
    }

    int64_t local_mem_size() const {
        return static_cast<int64_t>(local_mem_size_);
    }

    void clear_symmetric() {
        ASCEND_MOONEP_CHECK_ACL(aclrtMemset(RawPtr(), local_mem_size_, 0, local_mem_size_));
    }

protected:
    bool HasSymmetricBuffer() const {
        return shmem_ptr_ != nullptr;
    }

    void* SymmetricBuffer() const {
        return RawPtr();
    }

    std::size_t SymmetricBufferBytes() const {
        return local_mem_size_;
    }

private:
    void* RawPtr() const {
        TORCH_CHECK(shmem_ptr_ != nullptr, "op local SHMEM memory has been destroyed");
        return shmem_ptr_;
    }

    void* shmem_ptr_;
    std::size_t local_mem_size_;
};

class PrefetchWeight : public torch::jit::CustomClassHolder {
public:
    explicit PrefetchWeight(int64_t /*local_mem_size*/) {}

    void destroy() {}

    torch::Tensor compute(
        torch::Tensor remote_expert,
        torch::Tensor prefetch_buffers,
        torch::Tensor experts_to_copy,
        int64_t num_experts,
        int64_t num_sms,
        int64_t instance_id,
        bool use_udma) {
        TORCH_CHECK(num_experts == remote_expert.size(0),
                    "num_experts must equal remote_expert.shape[0]");
        return ascend_moonep::ops::LaunchPrefetch(
            remote_expert,
            prefetch_buffers,
            experts_to_copy,
            num_sms,
            instance_id,
            use_udma);
    }
};

class GradReduce : public torch::jit::CustomClassHolder {
public:
    GradReduce() = default;

    void compute(
        torch::Tensor expert_grads,
        torch::Tensor reduce_buffers,
        torch::Tensor experts_to_copy,
        int64_t epn,
        int64_t hidden,
        int64_t projection,
        int64_t ranks,
        int64_t slots,
        int64_t rank,
        int64_t blocks,
        bool use_udma) {
        TORCH_CHECK(rank == aclshmem_my_pe(),
                    "rank does not match the active CANN-SHMEM PE");
        ascend_moonep::ops::GradReduce(
            expert_grads,
            reduce_buffers,
            experts_to_copy,
            epn,
            hidden,
            projection,
            ranks,
            slots,
            rank,
            blocks,
            use_udma);
    }
};

class Planning : public torch::jit::CustomClassHolder {
public:
    Planning() = default;

    torch::Tensor compute_fused(
        torch::Tensor topk_idx, torch::Tensor dst, torch::Tensor cu_seqlens,
        torch::Tensor experts_to_copy, torch::Tensor dup_groups, torch::Tensor dup_loffs,
        torch::Tensor dup_counts, torch::Tensor local_workspace,
        int64_t symmetric_workspace_addr, int64_t symmetric_workspace_bytes,
        int64_t num_tokens, int64_t num_topk, int64_t num_experts,
        int64_t num_max_tokens_per_rank, int64_t num_prefetch_slots,
        int64_t core_num, int64_t rank, int64_t world_size, bool use_udma,
        torch::Tensor combine_workspace, int64_t hidden_size) {
        TORCH_CHECK(rank == aclshmem_my_pe() && world_size == aclshmem_n_pes(),
                    "planning topology must match the active SHMEM runtime");
        return ascend_moonep::ops::Planning(topk_idx, dst, cu_seqlens, experts_to_copy,
            dup_groups, dup_loffs, dup_counts, local_workspace,
            symmetric_workspace_addr, symmetric_workspace_bytes,
            num_tokens, num_topk, num_experts, num_max_tokens_per_rank,
            num_prefetch_slots, core_num, rank, world_size, use_udma,
            combine_workspace, hidden_size);
    }

    torch::Tensor prepare_combine_plan(
        torch::Tensor workspace, torch::Tensor dst, torch::Tensor groups,
        torch::Tensor dups, torch::Tensor counts,
        int64_t S, int64_t H, int64_t K, int64_t R, int64_t rank) {
        TORCH_CHECK(rank == aclshmem_my_pe() && R == aclshmem_n_pes(),
                    "planning topology must match the active SHMEM runtime");
        return ascend_moonep::ops::PlanningPrepareCombinePlan(
            workspace, dst, groups, dups, counts, S, H, K, R, rank);
    }

    int64_t symmetric_workspace_bytes(
        int64_t num_experts,
        int64_t num_max_tokens_per_rank,
        int64_t world_size) const {
        return ascend_moonep::ops::PlanningSymmetricWorkspaceBytes(
            num_experts,
            num_max_tokens_per_rank,
            world_size);
    }

    int64_t local_workspace_elements(
        int64_t num_tokens,
        int64_t num_topk,
        int64_t num_experts,
        int64_t world_size,
        int64_t core_num) const {
        return ascend_moonep::ops::PlanningLocalWorkspaceElements(
            num_tokens,
            num_topk,
            num_experts,
            world_size,
            core_num);
    }

    void compute(
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
        bool use_udma) {
        TORCH_CHECK(
            rank == aclshmem_my_pe(),
            "rank does not match the active CANN-SHMEM PE");
        ascend_moonep::ops::Planning(
            topk_idx,
            dst,
            cu_seqlens,
            experts_to_copy,
            dup_groups,
            dup_loffs,
            dup_counts,
            local_workspace,
            symmetric_workspace_addr,
            symmetric_workspace_bytes,
            num_tokens,
            num_topk,
            num_experts,
            num_max_tokens_per_rank,
            num_prefetch_slots,
            core_num,
            rank,
            world_size,
            use_udma);
    }
};

class Dispatch : public torch::jit::CustomClassHolder {
public:
    Dispatch() = default;

    void compute(
        torch::Tensor hidden_sh,
        c10::optional<torch::Tensor> route_weights,
        torch::Tensor dst,
        torch::Tensor dup_groups,
        torch::Tensor dup_loffs,
        torch::Tensor dup_counts,
        torch::Tensor hidden_buf,
        c10::optional<torch::Tensor> route_weights_buf,
        torch::Tensor sync_buffer,
        int64_t num_tokens,
        int64_t hidden_size,
        int64_t num_topk,
        int64_t world_size,
        int64_t rank,
        bool use_udma) {
        TORCH_CHECK(
            rank == aclshmem_my_pe(),
            "rank does not match the active CANN-SHMEM PE");
        ascend_moonep::ops::MoonEPDispatch(
            hidden_sh,
            route_weights,
            dst,
            dup_groups,
            dup_loffs,
            dup_counts,
            hidden_buf,
            route_weights_buf,
            sync_buffer,
            num_tokens,
            hidden_size,
            num_topk,
            world_size,
            rank,
            use_udma);
    }

    void compute_profile(
        torch::Tensor hidden_sh,
        c10::optional<torch::Tensor> route_weights,
        torch::Tensor dst,
        torch::Tensor dup_groups,
        torch::Tensor dup_loffs,
        torch::Tensor dup_counts,
        torch::Tensor hidden_buf,
        c10::optional<torch::Tensor> route_weights_buf,
        torch::Tensor sync_buffer,
        torch::Tensor profile_cycles,
        int64_t num_tokens,
        int64_t hidden_size,
        int64_t num_topk,
        int64_t world_size,
        int64_t rank,
        bool use_udma) {
        TORCH_CHECK(
            rank == aclshmem_my_pe(),
            "rank does not match the active CANN-SHMEM PE");
        ascend_moonep::ops::MoonEPDispatchProfile(
            hidden_sh,
            route_weights,
            dst,
            dup_groups,
            dup_loffs,
            dup_counts,
            hidden_buf,
            route_weights_buf,
            sync_buffer,
            profile_cycles,
            num_tokens,
            hidden_size,
            num_topk,
            world_size,
            rank,
            use_udma);
    }
};

class Combine : public torch::jit::CustomClassHolder {
public:
    Combine() = default;

    void compute(
        torch::Tensor hidden_buf,
        c10::optional<torch::Tensor> weight_buf,
        torch::Tensor dst,
        torch::Tensor dup_groups,
        torch::Tensor dup_loffs,
        torch::Tensor dup_counts,
        torch::Tensor hidden_sk_out,
        c10::optional<torch::Tensor> route_weights_sk_out,
        int64_t num_tokens,
        int64_t hidden_size,
        int64_t num_topk,
        int64_t world_size,
        int64_t rank,
        bool use_udma) {
        TORCH_CHECK(
            rank == aclshmem_my_pe(),
            "rank does not match the active CANN-SHMEM PE");
        ascend_moonep::ops::MoonEPCombine(
            hidden_buf,
            weight_buf,
            dst,
            dup_groups,
            dup_loffs,
            dup_counts,
            hidden_sk_out,
            route_weights_sk_out,
            num_tokens,
            hidden_size,
            num_topk,
            world_size,
            rank,
            use_udma);
    }
};

// Buffer-facing UDMA combine. Inputs and gather workspace are symmetric tensors
// owned by Buffer; this custom class is intentionally stateless.
class CombineUrma : public torch::jit::CustomClassHolder {
public:
    CombineUrma() = default;
    ~CombineUrma() override = default;

    void destroy() {}

    void compute_prepared(
        torch::Tensor hidden, c10::optional<torch::Tensor> weights,
        torch::Tensor sync, torch::Tensor workspace,
        torch::Tensor dst, torch::Tensor groups, torch::Tensor dups, torch::Tensor counts,
        torch::Tensor out, c10::optional<torch::Tensor> outw, torch::Tensor plan,
        int64_t S, int64_t H, int64_t K, int64_t R, int64_t rank) {
        TORCH_CHECK(rank == aclshmem_my_pe() && R == aclshmem_n_pes(),
                    "plan topology must match the active SHMEM runtime");
        ascend_moonep::ops::MoonEPUdmaCombine(hidden, weights, sync, dst, groups,
            dups, counts, out, outw, S, H, K, R, rank, workspace, plan);
    }

    void compute(
        torch::Tensor hidden_nvsh,
        c10::optional<torch::Tensor> route_weights_nvs,
        torch::Tensor sync_buffer,
        torch::Tensor combine_workspace,
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
        int64_t rank) {
        compute_impl(
            hidden_nvsh, route_weights_nvs, sync_buffer, combine_workspace, dst,
            dup_groups, dup_loffs, dup_counts, output_sh, output_sk,
            S, H, K, R, rank);
    }



private:
    void compute_impl(
        torch::Tensor hidden_nvsh,
        c10::optional<torch::Tensor> route_weights_nvs,
        torch::Tensor sync_buffer,
        torch::Tensor combine_workspace,
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
        int64_t rank) {
        TORCH_CHECK(
            rank == aclshmem_my_pe(),
            "rank does not match the active CANN-SHMEM PE");
        TORCH_CHECK(
            R == aclshmem_n_pes(),
            "world size does not match the active CANN-SHMEM runtime");
        ascend_moonep::ops::MoonEPUdmaCombine(
            hidden_nvsh,
            route_weights_nvs,
            sync_buffer,
            dst,
            dup_groups,
            dup_loffs,
            dup_counts,
            output_sh,
            output_sk,
            S,
            H,
            K,
            R,
            rank,
            combine_workspace);
    }
};

}  // namespace ascend_moonep::torch_classes

static auto shmem_instance_registry =
    torch::jit::class_<ShmemInstance>("AscendMoonEP", "ShmemInstance")
        .def(torch::jit::init<int64_t, int64_t, std::string>())
        .def("init", static_cast<void (ShmemInstance::*)(int64_t)>(&ShmemInstance::Init))
        .def("barrier", &ShmemInstance::Barrier)
        .def("destroy", &ShmemInstance::Destroy)
        .def("is_available", &ShmemInstance::IsAvailable)
        .def("rank", &ShmemInstance::Rank)
        .def("world_size", &ShmemInstance::WorldSize);

static auto prefetch_weight_registry =
    torch::jit::class_<ascend_moonep::torch_classes::PrefetchWeight>("AscendMoonEP", "PrefetchWeight")
        .def(torch::jit::init<int64_t>())
        .def("compute", &ascend_moonep::torch_classes::PrefetchWeight::compute)
        .def("destroy", &ascend_moonep::torch_classes::PrefetchWeight::destroy);

static auto grad_reduce_registry =
    torch::jit::class_<ascend_moonep::torch_classes::GradReduce>("AscendMoonEP", "GradReduce")
        .def(torch::jit::init<>())
        .def("compute", &ascend_moonep::torch_classes::GradReduce::compute);

static auto planning_registry =
    torch::jit::class_<ascend_moonep::torch_classes::Planning>("AscendMoonEP", "Planning")
        .def(torch::jit::init<>())
        .def(
            "symmetric_workspace_bytes",
            &ascend_moonep::torch_classes::Planning::symmetric_workspace_bytes)
        .def(
            "local_workspace_elements",
            &ascend_moonep::torch_classes::Planning::local_workspace_elements)
        .def("prepare_combine_plan", &ascend_moonep::torch_classes::Planning::prepare_combine_plan)
        .def("compute_fused", &ascend_moonep::torch_classes::Planning::compute_fused)
        .def("compute", &ascend_moonep::torch_classes::Planning::compute);

static auto dispatch_registry =
    torch::jit::class_<ascend_moonep::torch_classes::Dispatch>("AscendMoonEP", "Dispatch")
        .def(torch::jit::init<>())
        .def("compute", &ascend_moonep::torch_classes::Dispatch::compute)
        .def("compute_profile", &ascend_moonep::torch_classes::Dispatch::compute_profile);

static auto combine_registry =
    torch::jit::class_<ascend_moonep::torch_classes::Combine>("AscendMoonEP", "Combine")
        .def(torch::jit::init<>())
        .def("compute", &ascend_moonep::torch_classes::Combine::compute);

static auto combine_urma_registry =
    torch::jit::class_<ascend_moonep::torch_classes::CombineUrma>("AscendMoonEP", "CombineUrma")
        .def(torch::jit::init<>())
        .def("compute_prepared", &ascend_moonep::torch_classes::CombineUrma::compute_prepared)

        .def("compute", &ascend_moonep::torch_classes::CombineUrma::compute)
        .def("destroy", &ascend_moonep::torch_classes::CombineUrma::destroy);
