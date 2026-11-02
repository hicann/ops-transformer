/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_chunk_kda_fwd.cpp
 * \brief Minimal BSND example for aclnnChunkKdaFwd.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_kda_fwd.h"

namespace {

constexpr int32_t kDeviceId = 0;
constexpr int64_t kBatch = 1;
constexpr int64_t kSeqLen = 64;
constexpr int64_t kHeadNum = 2;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kChunkSize = 64;

#define CHECK_RET(cond, expr) \
    do { \
        if (!(cond)) { \
            expr; \
        } \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const auto dim : shape) {
        size *= dim;
    }
    return size;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16U);
}

float Bf16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

class AclRuntime {
public:
    ~AclRuntime()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        if (context_ != nullptr) {
            aclrtDestroyContext(context_);
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
        }
        if (initialized_) {
            aclFinalize();
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclInit failed. ERROR: %d\n", ret); return ret);
        initialized_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateContext(&context_, deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);

        ret = aclrtSetCurrentContext(context_);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const
    {
        return stream_;
    }

private:
    int32_t deviceId_ = kDeviceId;
    bool initialized_ = false;
    bool deviceSet_ = false;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
};

class AclTensorHolder {
public:
    AclTensorHolder() = default;
    AclTensorHolder(const AclTensorHolder&) = delete;
    AclTensorHolder& operator=(const AclTensorHolder&) = delete;

    ~AclTensorHolder()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
        if (address_ != nullptr) {
            aclrtFree(address_);
        }
    }

    template <typename T>
    int Create(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
               const char* name)
    {
        if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
            std::printf("%s host data size mismatch\n", name);
            return -1;
        }

        const size_t byteSize = hostData.size() * sizeof(T);
        auto ret = aclrtMalloc(&address_, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("allocate %s failed. ERROR: %d\n", name, ret); return ret);

        ret = aclrtMemcpy(address_, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("copy %s to device failed. ERROR: %d\n", name, ret); return ret);

        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        tensor_ = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), address_);
        CHECK_RET(tensor_ != nullptr, std::printf("create %s tensor failed\n", name); return -1);
        return ACL_SUCCESS;
    }

    aclTensor* Tensor() const
    {
        return tensor_;
    }

    void* Address() const
    {
        return address_;
    }

private:
    void* address_ = nullptr;
    aclTensor* tensor_ = nullptr;
};

class DeviceBuffer {
public:
    ~DeviceBuffer()
    {
        if (address_ != nullptr) {
            aclrtFree(address_);
        }
    }

    int Allocate(uint64_t size)
    {
        if (size == 0) {
            return ACL_SUCCESS;
        }
        auto ret = aclrtMalloc(&address_, size, ACL_MEM_MALLOC_HUGE_FIRST);
        return ret;
    }

    void* Address() const
    {
        return address_;
    }

private:
    void* address_ = nullptr;
};

int PrintFirstValues(const AclTensorHolder& holder, const std::vector<int64_t>& shape, const char* name)
{
    std::vector<uint16_t> hostData(GetShapeSize(shape), 0);
    const size_t byteSize = hostData.size() * sizeof(uint16_t);
    auto ret = aclrtMemcpy(hostData.data(), byteSize, holder.Address(), byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("copy %s to host failed. ERROR: %d\n", name, ret); return ret);

    constexpr int64_t kPrintCount = 5;
    for (int64_t i = 0; i < kPrintCount && i < static_cast<int64_t>(hostData.size()); ++i) {
        const float value = Bf16ToFloat(hostData[i]);
        CHECK_RET(std::isfinite(value), std::printf("%s[%ld] is not finite\n", name, i); return -1);
        std::printf("  %s[%ld] = %f\n", name, i, value);
    }
    return ACL_SUCCESS;
}

} // namespace

int main()
{
    AclRuntime runtime;
    auto ret = runtime.Init();
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const std::vector<int64_t> qkShape = {kBatch, kSeqLen, kHeadNum, kHeadDim};
    const std::vector<int64_t> vShape = {kBatch, kSeqLen, kHeadNum, kHeadDim};
    const std::vector<int64_t> gShape = {kBatch, kSeqLen, kHeadNum, kHeadDim};
    const std::vector<int64_t> betaShape = {kBatch, kSeqLen, kHeadNum};
    const std::vector<int64_t> matrixShape = {kBatch, kHeadNum, kSeqLen, kChunkSize};
    const std::vector<int64_t> aLogShape = {kHeadNum};

    std::vector<uint16_t> qHostData(GetShapeSize(qkShape), FloatToBf16(0.02F));
    std::vector<uint16_t> kHostData(GetShapeSize(qkShape), FloatToBf16(0.03F));
    std::vector<uint16_t> vHostData(GetShapeSize(vShape), FloatToBf16(0.5F));
    std::vector<float> gHostData(GetShapeSize(gShape), -0.5F);
    std::vector<float> betaHostData(GetShapeSize(betaShape), 0.5F);
    std::vector<float> aLogHostData = {0.1F, 0.2F};
    std::vector<uint16_t> attnOutHostData(GetShapeSize(vShape), 0);
    std::vector<uint16_t> aqkHostData(GetShapeSize(matrixShape), 0);
    std::vector<uint16_t> akkHostData(GetShapeSize(matrixShape), 0);

    AclTensorHolder q;
    AclTensorHolder k;
    AclTensorHolder v;
    AclTensorHolder g;
    AclTensorHolder beta;
    AclTensorHolder aLog;
    AclTensorHolder attnOut;
    AclTensorHolder aqk;
    AclTensorHolder akk;

    ret = q.Create(qHostData, qkShape, ACL_BF16, "q");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = k.Create(kHostData, qkShape, ACL_BF16, "k");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = v.Create(vHostData, vShape, ACL_BF16, "v");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = g.Create(gHostData, gShape, ACL_FLOAT, "g");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = beta.Create(betaHostData, betaShape, ACL_FLOAT, "beta");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aLog.Create(aLogHostData, aLogShape, ACL_FLOAT, "aLog");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = attnOut.Create(attnOutHostData, vShape, ACL_BF16, "attnOut");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aqk.Create(aqkHostData, matrixShape, ACL_BF16, "aqkOut");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = akk.Create(akkHostData, matrixShape, ACL_BF16, "akkOut");
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnChunkKdaFwdGetWorkspaceSize(q.Tensor(), k.Tensor(), v.Tensor(), g.Tensor(), beta.Tensor(), aLog.Tensor(),
                                           nullptr, // dtBiasOptional
                                           nullptr, // initialStateOptional
                                           nullptr, // cuSeqlensOptional
                                           nullptr, // chunkIndicesOptional
                                           "BSND", 1.0 / std::sqrt(static_cast<double>(kHeadDim)), kChunkSize,
                                           true,  // safeGate
                                           -5.0,  // lowerBound
                                           true,  // useGateInKernel
                                           false, // stateVFirst
                                           attnOut.Tensor(),
                                           nullptr, // finalStateOut
                                           nullptr, // gkOut
                                           aqk.Tensor(), akk.Tensor(),
                                           nullptr, // wOut
                                           nullptr, // uOut
                                           nullptr, // qgOut
                                           nullptr, // kgOut
                                           nullptr, // vNewOut
                                           nullptr, // hOut
                                           &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclnnChunkKdaFwdGetWorkspaceSize failed. ERROR: %d, message: %s\n", ret,
                                              aclGetRecentErrMsg());
              return ret);

    DeviceBuffer workspace;
    ret = workspace.Allocate(workspaceSize);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("allocate workspace failed. ERROR: %d\n", ret); return ret);

    ret = aclnnChunkKdaFwd(workspace.Address(), workspaceSize, executor, runtime.Stream());
    CHECK_RET(ret == ACL_SUCCESS,
              std::printf("aclnnChunkKdaFwd failed. ERROR: %d, message: %s\n", ret, aclGetRecentErrMsg());
              return ret);

    ret = aclrtSynchronizeStream(runtime.Stream());
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::printf("ChunkKdaFwd execution succeeded.\n");
    ret = PrintFirstValues(attnOut, vShape, "attnOut");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = PrintFirstValues(aqk, matrixShape, "Aqk");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = PrintFirstValues(akk, matrixShape, "Akk");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
