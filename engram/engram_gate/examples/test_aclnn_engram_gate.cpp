/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_engram_gate.h"

namespace {
void Check(int32_t status)
{
    if (status != 0)
        throw std::runtime_error("ACL/ACLNN failure: " + std::to_string(status));
}
bool CheckRawDot(const std::vector<float>& dot, float expected)
{
    for (auto value : dot) {
        if (!std::isfinite(value) || std::fabs(value - expected) > 1e-5F)
            return false;
    }
    return true;
}
struct Resources {
    aclrtStream stream = nullptr;
    std::vector<void*> buffers;
    std::vector<aclTensor*> tensors;
    ~Resources()
    {
        if (stream)
            aclrtSynchronizeStream(stream);
        for (auto tensor : tensors)
            aclDestroyTensor(tensor);
        for (auto buffer : buffers)
            aclrtFree(buffer);
        if (stream)
            aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }
    void* Allocate(size_t bytes)
    {
        void* ptr = nullptr;
        Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        buffers.push_back(ptr);
        return ptr;
    }
    aclTensor* Tensor(std::vector<int64_t> shape, aclDataType dtype, void* ptr)
    {
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size() - 1; i > 0; --i)
            strides[i - 1] = strides[i] * shape[i];
        auto tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), ptr);
        if (!tensor)
            throw std::runtime_error("aclCreateTensor failed");
        tensors.push_back(tensor);
        return tensor;
    }
};
} // namespace
int main()
{
    try {
        constexpr int64_t t = 2, h = 4, d = 64;
        const float expected = static_cast<float>(d);
        if (!CheckRawDot({expected}, expected) || CheckRawDot({std::numeric_limits<float>::quiet_NaN()}, expected))
            throw std::runtime_error("raw dot check self-test failed");
        Check(aclInit(nullptr));
        Resources r;
        Check(aclrtSetDevice(0));
        Check(aclrtCreateStream(&r.stream));
        // FP16 x=k=v=1; FP32 fused weight=1. H is fixed to 4 and D is aligned to 64.
        std::vector<uint16_t> x(t * h * d, 0x3c00), kv(t * (h + 1) * d, 0x3c00);
        std::vector<float> w(h * d, 1.0F);
        auto xp = r.Allocate(x.size() * sizeof(uint16_t));
        auto kp = r.Allocate(kv.size() * sizeof(uint16_t));
        auto wp = r.Allocate(w.size() * sizeof(float));
        Check(aclrtMemcpy(xp, x.size() * 2, x.data(), x.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
        Check(aclrtMemcpy(kp, kv.size() * 2, kv.data(), kv.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
        Check(aclrtMemcpy(wp, w.size() * 4, w.data(), w.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE));
        auto output = r.Tensor({t, h, d}, ACL_FLOAT16, r.Allocate(t * h * d * 2));
        std::vector<aclTensor*> stats;
        std::vector<void*> statBuffers;
        for (int i = 0; i < 4; ++i) {
            statBuffers.push_back(r.Allocate(t * h * 4));
            stats.push_back(r.Tensor({t, h}, ACL_FLOAT, statBuffers.back()));
        }
        auto xt = r.Tensor({t, h, d}, ACL_FLOAT16, xp);
        auto kt = r.Tensor({t, (h + 1) * d}, ACL_FLOAT16, kp);
        auto wt = r.Tensor({h, d}, ACL_FLOAT, wp);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        Check(aclnnEngramGateGetWorkspaceSize(xt, kt, wt, 1e-6, 1e-6, true, output, stats[0], stats[1], stats[2],
                                              stats[3], &workspaceSize, &executor));
        void* workspace = workspaceSize ? r.Allocate(workspaceSize) : nullptr;
        Check(aclnnEngramGate(workspace, workspaceSize, executor, r.stream));
        Check(aclrtSynchronizeStream(r.stream));
        std::vector<float> dot(t * h);
        Check(aclrtMemcpy(dot.data(), dot.size() * 4, statBuffers[0], dot.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));
        if (!CheckRawDot(dot, expected))
            throw std::runtime_error("raw dot mismatch");
        std::cout << "EngramGate PASS; dot=" << dot[0] << ", workspace=" << workspaceSize << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
