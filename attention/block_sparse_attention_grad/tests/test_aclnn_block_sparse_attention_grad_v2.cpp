/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Standalone Ascend 950 API rejection test. Link with the V2 custom op package,
// libascendcl and libnnopbase; select an idle device via ASCEND_RT_VISIBLE_DEVICES.
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "opdev/op_errno.h"
#include "aclnnop/aclnn_block_sparse_attention_grad_v2.h"

namespace {
struct Descriptors {
    std::vector<aclTensor*> tensors;
    aclTensor* Make(aclDataType dtype, const std::vector<int64_t>& shape)
    {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        auto* tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                       shape.data(), shape.size(), nullptr);
        if (tensor == nullptr) {
            throw std::runtime_error("aclCreateTensor failed");
        }
        tensors.push_back(tensor);
        return tensor;
    }
    ~Descriptors()
    {
        for (auto* tensor : tensors) {
            aclDestroyTensor(tensor);
        }
    }
};
} // namespace

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        return 2;
    }
    int passed = 0;
    int result = 0;
    try {
        Descriptors descriptors;
        auto* hif8 = descriptors.Make(ACL_HIFLOAT8, {1, 4, 128, 128});
        auto* fp16 = descriptors.Make(ACL_FLOAT16, {1, 4, 128, 128});
        auto* bf16 = descriptors.Make(ACL_BF16, {1, 4, 128, 128});
        auto* fp32 = descriptors.Make(ACL_FLOAT, {1, 4, 128, 128});
        auto* lse = descriptors.Make(ACL_FLOAT, {1, 4, 128});
        auto* mask = descriptors.Make(ACL_UINT8, {1, 4, 1, 1});
        auto* int8Mask = descriptors.Make(ACL_INT8, {1, 4, 1, 1});
        auto* scale = descriptors.Make(ACL_FLOAT, {1});
        auto* wrongShape = descriptors.Make(ACL_FLOAT, {2});
        auto* wrongType = descriptors.Make(ACL_FLOAT16, {1});
        std::array<const aclTensor*, 6> scales;
        scales.fill(scale);
        std::array<const aclTensor*, 6> noScales{};
        int64_t blockDims[] = {128, 128};
        auto* block = aclCreateIntArray(blockDims, 2);
        auto reject = [&](const char* name, int64_t mode, const aclTensor* q,
                          const std::array<const aclTensor*, 6>& scaleArgs) {
            char layout[] = "BNSD";
            uint64_t workspace = 0;
            aclOpExecutor* executor = nullptr;
            auto* output = q == fp16 ? fp16 : bf16;
            auto status = aclnnBlockSparseAttentionGradV2GetWorkspaceSize(
                q, q, q, q, output, lse, mask, nullptr, block, nullptr, nullptr, scaleArgs[0], scaleArgs[1],
                scaleArgs[2], scaleArgs[3], scaleArgs[4], scaleArgs[5], layout, layout, 4, 0, 0.125, 2147483647,
                2147483647, mode, output, output, output, &workspace, &executor);
            if (status == ACL_SUCCESS) {
                std::fprintf(stderr, "Unexpected success: %s\n", name);
                result = 1;
            } else {
                ++passed;
            }
        };
        for (int64_t mode : {-1, 1, 2, 3, 4, 6}) {
            reject("unsupported mode", mode, hif8, scales);
        }
        reject("HiF8 with mode 0", 0, hif8, scales);
        reject("FP16 with mode 5", 5, fp16, scales);
        reject("BF16 with mode 5", 5, bf16, scales);
        reject("FP32 with mode 0", 0, fp32, noScales);
        reject("null query", 5, nullptr, scales);
        for (size_t i = 0; i < scales.size(); ++i) {
            auto bad = scales;
            bad[i] = nullptr;
            reject("missing HiF8 scale", 5, hif8, bad);
            bad[i] = wrongShape;
            reject("wrong scale shape", 5, hif8, bad);
            bad[i] = wrongType;
            reject("wrong scale dtype", 5, hif8, bad);
            bad = noScales;
            bad[i] = scale;
            reject("scale in FP16 mode", 0, fp16, bad);
        }
        // The original signature must compile without scales and reject HiF8.
        char layout[] = "BNSD";
        uint64_t workspace = 0;
        aclOpExecutor* executor = nullptr;
        auto legacyStatus = aclnnBlockSparseAttentionGradGetWorkspaceSize(
            hif8, hif8, hif8, hif8, bf16, lse, mask, nullptr, block, nullptr, nullptr, layout, layout, 4, 0, 0.125,
            2147483647, 2147483647, bf16, bf16, bf16, &workspace, &executor);
        if (legacyStatus == ACL_SUCCESS) {
            std::fprintf(stderr, "Original API unexpectedly accepted HiF8\n");
            result = 1;
        } else {
            ++passed;
        }
        // Reject INT8 at the public API boundary, before any kernel is queued.
        for (auto* input : {fp16, bf16, hif8}) {
            const bool quantized = input == hif8;
            const auto& args = quantized ? scales : noScales;
            auto* output = quantized ? bf16 : input;
            const auto status = aclnnBlockSparseAttentionGradV2GetWorkspaceSize(
                input, input, input, input, output, lse, int8Mask, nullptr, block, nullptr, nullptr, args[0], args[1],
                args[2], args[3], args[4], args[5], layout, layout, 4, 0, 0.125, 2147483647, 2147483647,
                quantized ? 5 : 0, output, output, output, &workspace, &executor);
            if (status == ACLNN_ERR_PARAM_INVALID) {
                ++passed;
            } else {
                std::fprintf(stderr, "V2 INT8 mask status: %d\n", status);
                result = 1;
            }
        }
        for (auto* input : {fp16, bf16}) {
            const auto status = aclnnBlockSparseAttentionGradGetWorkspaceSize(
                input, input, input, input, input, lse, int8Mask, nullptr, block, nullptr, nullptr, layout, layout, 4,
                0, 0.125, 2147483647, 2147483647, input, input, input, &workspace, &executor);
            if (status == ACLNN_ERR_PARAM_INVALID) {
                ++passed;
            } else {
                std::fprintf(stderr, "Original API INT8 mask status: %d\n", status);
                result = 1;
            }
        }
        aclDestroyIntArray(block);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }
    aclrtResetDevice(0);
    aclFinalize();
    std::printf("API rejection checks passed: %d/41\n", passed);
    return result == 0 && passed == 41 ? 0 : 1;
}
