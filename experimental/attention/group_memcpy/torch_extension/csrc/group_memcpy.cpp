/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/extension.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
using std::string;
#include "aclnn_common.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace op_api {
namespace {
bool Supported(at::ScalarType type)
{
    switch (type) {
        case at::kChar:
        case at::kByte:
        case at::kShort:
        case at::kInt:
        case at::kLong:
        case at::kHalf:
        case at::kBFloat16:
        case at::kFloat:
        case at::kBool:
            return true;
        default:
            return false;
    }
}

void CheckMetadata(at::TensorList src, at::TensorList dst)
{
    TORCH_CHECK(!src.empty() && src.size() == dst.size(),
                "group_memcpy: src/dst must have the same nonzero list length");
    TORCH_CHECK(src.size() <= static_cast<size_t>(INT32_MAX), "group_memcpy: too many pairs");
    TORCH_CHECK(src[0].defined(), "group_memcpy: undefined tensor");
    const auto device = src[0].device();
    const auto dtype = src[0].scalar_type();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "group_memcpy: expected NPU tensors");
    TORCH_CHECK(Supported(dtype), "group_memcpy: unsupported dtype");
    for (size_t i = 0; i < src.size(); ++i) {
        for (const auto& tensor : {src[i], dst[i]}) {
            TORCH_CHECK(tensor.defined() && tensor.device() == device,
                        "group_memcpy: all tensors must be on the same device");
            TORCH_CHECK(tensor.layout() == at::kStrided && tensor.is_contiguous(),
                        "group_memcpy: contiguous strided tensors required");
            TORCH_CHECK(!tensor.requires_grad(), "group_memcpy: autograd is unsupported");
            TORCH_CHECK(tensor.scalar_type() == dtype, "group_memcpy: all dtypes must match");
            TORCH_CHECK(tensor.numel() == src[0].numel(), "group_memcpy: all copy byte counts must match");
        }
        TORCH_CHECK(src[i].sizes() == dst[i].sizes(), "group_memcpy: paired shapes must match");
    }
    TORCH_CHECK(
        static_cast<uint64_t>(src[0].numel()) <= (static_cast<uint64_t>(INT64_MAX) - 64) / src[0].element_size(),
        "group_memcpy: copy byte count overflows");
}

void CheckOverlap(at::TensorList src, at::TensorList dst)
{
    struct Range {
        uintptr_t begin;
        uintptr_t end;
        bool output;
    };
    std::vector<Range> ranges;
    ranges.reserve(src.size() * 2);
    auto append = [&](const at::Tensor& tensor, bool output) {
        if (tensor.numel() == 0) {
            return;
        }
        const auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        const auto size = static_cast<uint64_t>(tensor.numel()) * tensor.element_size();
        TORCH_CHECK(begin <= std::numeric_limits<uintptr_t>::max() - size, "group_memcpy: address range overflows");
        ranges.push_back({begin, begin + size, output});
    };
    for (size_t i = 0; i < src.size(); ++i) {
        append(src[i], false);
        append(dst[i], true);
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) { return a.begin < b.begin; });
    uintptr_t srcEnd = 0;
    uintptr_t dstEnd = 0;
    for (const auto& range : ranges) {
        TORCH_CHECK(range.begin >= dstEnd && (!range.output || range.begin >= srcEnd),
                    "group_memcpy: destinations must not overlap any source or another destination");
        if (range.output) {
            dstEnd = std::max(dstEnd, range.end);
        } else {
            srcEnd = std::max(srcEnd, range.end);
        }
    }
}

struct PendingCopy {
    std::vector<at::Tensor> src;
    std::vector<at::Tensor> dst;
    aclTensorList* aclSrc = nullptr;
    aclTensorList* aclDst = nullptr;
    at::Tensor workspace;
    ~PendingCopy()
    {
        Release(aclSrc);
        Release(aclDst);
    }
};

aclTensorList* MakeCopyTensorList(at::TensorList tensors)
{
    static const auto createTensor = GET_OP_API_FUNC(aclCreateTensor);
    static const auto createList = GET_OP_API_FUNC(aclCreateTensorList);
    TORCH_CHECK(createTensor != nullptr && createList != nullptr,
                "group_memcpy: ACL tensor creation interfaces are unavailable");
    using TensorOwner = std::unique_ptr<aclTensor, void (*)(aclTensor*)>;
    std::vector<TensorOwner> owners;
    std::vector<const aclTensor*> descriptors;
    owners.reserve(tensors.size());
    descriptors.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        const int64_t elements = tensor.numel();
        const int64_t stride = 1;
        // The launcher exposes storage shapes to Tiling. Describe only this
        // contiguous slice, rebasing its storage pointer to include the offset.
        // PendingCopy retains the original allocation and records its stream.
        auto* descriptor = createTensor(&elements, 1, ConvertToAclDataType(tensor.scalar_type()), &stride, 0,
                                        ACL_FORMAT_ND, &elements, 1, tensor.data_ptr());
        TORCH_CHECK(descriptor != nullptr, "group_memcpy: failed to create ACL tensor");
        owners.emplace_back(descriptor, [](aclTensor* p) { Release(p); });
        descriptors.push_back(descriptor);
    }
    auto* list = createList(descriptors.data(), descriptors.size());
    TORCH_CHECK(list != nullptr, "group_memcpy: failed to create ACL tensor list");
    for (auto& owner : owners) {
        owner.release(); // aclDestroyTensorList owns its member descriptors.
    }
    return list;
}

void BumpDestinationVersions(at::TensorList dst)
{
    // Custom mutable schemas do not generate an inplace autograd wrapper.
    // Preserve PyTorch's saved-tensor mutation detection for ordinary tensors.
    for (const auto& tensor : dst) {
        if (!tensor.is_inference()) {
            torch::autograd::increment_version(tensor);
        }
    }
}

void GroupMemcpyNpu(at::TensorList src, at::TensorList dst)
{
    CheckMetadata(src, dst);
    const c10::OptionalDeviceGuard guard(src[0].device());
    for (size_t i = 0; i < src.size(); ++i) {
        TORCH_CHECK(IsOpInputBaseFormat(src[i]) && IsOpInputBaseFormat(dst[i]),
                    "group_memcpy: blocked/private NPU formats are unsupported");
    }
    CheckOverlap(src, dst);
    if (src[0].numel() == 0) {
        BumpDestinationVersions(dst);
        return;
    }
    using WorkspaceFn = int (*)(const aclTensorList*, const aclTensorList*, uint64_t*, aclOpExecutor**);
    using LaunchFn = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    static const auto workspaceFn = reinterpret_cast<WorkspaceFn>(GetOpApiFuncAddr("aclnnGroupMemcpyGetWorkspaceSize"));
    static const auto launchFn = reinterpret_cast<LaunchFn>(GetOpApiFuncAddr("aclnnGroupMemcpy"));
    TORCH_CHECK(workspaceFn != nullptr && launchFn != nullptr,
                "group_memcpy: install the GroupMemcpy opapi package and source its environment first");

    const auto stream = c10_npu::getCurrentNPUStream();
    const auto aclStream = stream.stream(false);
    if (c10_npu::check_enqueue_need_use(aclStream)) {
        aclrtUseStreamResInCurrentThread(aclStream);
    }
    auto pending = std::make_shared<PendingCopy>();
    for (size_t i = 0; i < src.size(); ++i) {
        pending->src.push_back(src[i]);
        pending->dst.push_back(dst[i]);
        c10_npu::NPUCachingAllocator::recordStream(src[i].storage().data_ptr(), stream);
        c10_npu::NPUCachingAllocator::recordStream(dst[i].storage().data_ptr(), stream);
    }
    pending->aclSrc = MakeCopyTensorList(at::TensorList(pending->src));
    pending->aclDst = MakeCopyTensorList(at::TensorList(pending->dst));
    TORCH_CHECK(pending->aclSrc != nullptr && pending->aclDst != nullptr,
                "group_memcpy: failed to create ACL tensor lists");
    ApplyDeterministicConfig();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const int status = workspaceFn(pending->aclSrc, pending->aclDst, &workspaceSize, &executor);
    TORCH_CHECK(status == 0, "aclnnGroupMemcpyGetWorkspaceSize failed: ", status, "; ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(INT64_MAX), "group_memcpy: workspace is too large");
    if (workspaceSize != 0) {
        pending->workspace = at::empty({static_cast<int64_t>(workspaceSize)}, src[0].options().dtype(at::kByte));
        c10_npu::NPUCachingAllocator::recordStream(pending->workspace.storage().data_ptr(), stream);
    }
    // Capture Tensor owners AND workspace until the queued handler runs. Recording
    // their stream above also protects allocations until the device work completes.
    auto aclCall = [pending, workspaceSize, executor, aclStream]() -> int {
        if (c10_npu::check_dequeue_need_use(aclStream)) {
            aclrtUseStreamResInCurrentThread(aclStream);
        }
        void* workspace = pending->workspace.defined() ? pending->workspace.data_ptr() : nullptr;
        const int result = launchFn(workspace, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == 0, "aclnnGroupMemcpy failed: ", result, "; ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnGroupMemcpy", aclCall);
    BumpDestinationVersions(dst);
}

} // namespace

void group_memcpy(const std::vector<at::Tensor>& src, const std::vector<at::Tensor>& dst)
{
    GroupMemcpyNpu(at::TensorList(src), at::TensorList(dst));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("group_memcpy", &group_memcpy, "Copy tensor pairs into preallocated destinations", pybind11::arg("src"),
          pybind11::arg("dst"));
}
} // namespace op_api
