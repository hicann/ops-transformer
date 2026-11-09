/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * TextGatedResidualGrad aclnn example: read binary inputs from a directory and write binary outputs.
 *
 * Usage: test_aclnn_text_gated_residual_grad [data_dir] [deviceId] (defaults: ./data, device 0).
 * Data directory files:
 *   meta.txt: dtype(bf16|fp16) n c h r has_gamma [has_gate] (n=B*S; tensors use B=1, S=n;
 *             has_gate defaults to 1; n=0 empty token input is supported).
 *   grad_mixed.bin [n,h]T  grad_inject.bin [n,c]T  x.bin [n,d]T
 *   w_down.bin [r,d]T      w_up.bin [d,r]T         w_gate.bin [c,d]T
 *   gamma.bin [c*h]fp32 (original weight, used as 1+gamma)  inv_rms.bin [n,c]fp32
 *   inject.bin [n,c]T  down_pre_act.bin [n,r]T (xn@w_down^T, before division by C)  up.bin [n,d]T (upPreAct)
 * Outputs: dx/dw_down/dw_up/dw_gate.bin use T; dgamma.bin uses fp32. Omit dw_gate/dgamma when has_gate/has_gamma is 0.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_text_gated_residual_grad.h"

#define CHECK_RET(cond) ((cond) ? true : (false))
#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

namespace {
constexpr size_t HALF_BYTES = sizeof(uint16_t);
constexpr int64_t METADATA_CHANNEL_LIMIT = 4096; // Tiling additionally checks the actual AIV count.
constexpr int DATA_DIR_ARG = 1;
constexpr int DEVICE_ARG = 2;
enum TensorIndex : size_t {
    GRAD_MIXED,
    GRAD_INJECT,
    X,
    W_DOWN,
    W_UP,
    W_GATE,
    GAMMA,
    INV_RMS,
    INJECT,
    DOWN_PRE,
    UP,
    DX,
    DW_DOWN,
    DW_UP,
    DW_GATE,
    DGAMMA,
    TENSOR_COUNT
};

struct Metadata {
    std::string dtype;
    int64_t n = 0;
    int64_t c = 0;
    int64_t h = 0;
    int64_t r = 0;
    int64_t d = 0;
    int hasGamma = 0;
    int hasGate = 1;
};

struct TensorSpec {
    std::string file;
    std::vector<int64_t> shape;
    aclDataType dtype;
    size_t elemBytes;
    bool enabled = true;
};

struct Resources {
    std::array<aclTensor*, TENSOR_COUNT> tensors{};
    std::array<void*, TENSOR_COUNT> addresses{};
    void* workspace = nullptr;
    aclrtStream stream = nullptr;
    int32_t device = 0;
    bool initialized = false;
    bool deviceSet = false;

    ~Resources()
    {
        for (auto tensor : tensors) {
            if (tensor != nullptr) {
                aclDestroyTensor(tensor);
            }
        }
        for (auto address : addresses) {
            if (address != nullptr) {
                aclrtFree(address);
            }
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(device);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};

bool GetByteSize(const std::vector<int64_t>& shape, size_t elemSize, size_t& bytes)
{
    bytes = elemSize;
    for (auto i : shape) {
        if (i < 0 || (i != 0 && bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(i))) {
            LOG_PRINT("invalid or overflowing tensor shape\n");
            return false;
        }
        bytes *= static_cast<size_t>(i);
    }
    return true;
}

int Init(Resources& resources)
{
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    resources.initialized = true;
    ret = aclrtSetDevice(resources.device);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    resources.deviceSet = true;
    ret = aclrtCreateStream(&resources.stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

bool ReadBin(const std::string& path, std::string& data)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        LOG_PRINT("open %s failed\n", path.c_str());
        return false;
    }
    auto size = ifs.tellg();
    if (size < 0) {
        return false;
    }
    data.resize(size);
    ifs.seekg(0);
    ifs.read(data.data(), size);
    return ifs.good();
}

bool WriteBin(const std::string& path, const std::string& data)
{
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(data.data(), data.size());
    return ofs.good();
}

int CreateTensorFromBin(const std::string& path, const std::vector<int64_t>& shape, aclDataType dtype, size_t elemSize,
                        aclTensor** tensor, void** deviceAddr, std::string& hostBuf)
{
    if (!ReadBin(path, hostBuf)) {
        return -1;
    }
    size_t size = 0;
    if (!GetByteSize(shape, elemSize, size)) {
        return -1;
    }
    if (hostBuf.size() != size) {
        LOG_PRINT("size mismatch %s: file %zu expect %zu\n", path.c_str(), hostBuf.size(), size);
        return -1;
    }
    auto ret = aclrtMalloc(deviceAddr, std::max<size_t>(size, 1), ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return -1;
    }
    if (size != 0) {
        ret = aclrtMemcpy(*deviceAddr, size, hostBuf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            return -1;
        }
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        const size_t last = i - 1;
        strides[last - 1] = shape[last] * strides[last];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return *tensor != nullptr ? 0 : -1;
}

int CreateOutTensor(const std::vector<int64_t>& shape, aclDataType dtype, size_t elemSize, aclTensor** tensor,
                    void** deviceAddr)
{
    size_t size = 0;
    if (!GetByteSize(shape, elemSize, size)) {
        return -1;
    }
    auto ret = aclrtMalloc(deviceAddr, std::max<size_t>(size, 1), ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return -1;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        const size_t last = i - 1;
        strides[last - 1] = shape[last] * strides[last];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return *tensor != nullptr ? 0 : -1;
}

int DumpOut(const std::string& path, void* deviceAddr, const std::vector<int64_t>& shape, size_t elemSize)
{
    size_t size = 0;
    if (!GetByteSize(shape, elemSize, size)) {
        return -1;
    }
    std::string host(size, '\0');
    if (size != 0) {
        auto ret = aclrtMemcpy(host.data(), size, deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            return -1;
        }
    }
    return WriteBin(path, host) ? 0 : -1;
}

bool ValidateMetadata(Metadata& m)
{
    const auto maxDim = std::numeric_limits<uint32_t>::max();
    if ((m.dtype != "bf16" && m.dtype != "fp16") || m.n < 0 || m.n > maxDim || m.c <= 0 ||
        m.c > METADATA_CHANNEL_LIMIT || m.h <= 0 || m.h > maxDim / m.c || m.r <= 0 || m.r > maxDim) {
        LOG_PRINT("invalid dtype or dimensions in meta.txt\n");
        return false;
    }
    if ((m.hasGamma != 0 && m.hasGamma != 1) || (m.hasGate != 0 && m.hasGate != 1)) {
        LOG_PRINT("invalid optional flags in meta.txt\n");
        return false;
    }
    m.d = m.c * m.h;
    const int64_t maxFloatElements = std::numeric_limits<int64_t>::max() / sizeof(float);
    if (m.n > maxFloatElements / m.d || m.n > maxFloatElements / m.r || m.r > maxFloatElements / m.d ||
        m.c > maxFloatElements / m.d) {
        LOG_PRINT("Tensor byte size overflows int64 in meta.txt\n");
        return false;
    }
    return true;
}

bool ReadMetadata(const std::string& dir, Metadata& m)
{
    std::ifstream meta(dir + "/meta.txt");
    if (!(meta >> m.dtype >> m.n >> m.c >> m.h >> m.r >> m.hasGamma)) {
        LOG_PRINT("invalid or missing %s/meta.txt\n", dir.c_str());
        return false;
    }
    meta >> std::ws;
    if (!meta.eof() && !(meta >> m.hasGate)) {
        LOG_PRINT("invalid has_gate in meta.txt\n");
        return false;
    }
    return ValidateMetadata(m);
}

std::array<TensorSpec, TENSOR_COUNT> MakeTensorSpecs(const Metadata& m)
{
    const auto type = m.dtype == "fp16" ? ACL_FLOAT16 : ACL_BF16;
    return {{{"grad_mixed.bin", {1, m.n, m.h}, type, HALF_BYTES},
             {"grad_inject.bin", {1, m.n, m.c}, type, HALF_BYTES, m.hasGate != 0},
             {"x.bin", {1, m.n, m.d}, type, HALF_BYTES},
             {"w_down.bin", {m.r, m.d}, type, HALF_BYTES},
             {"w_up.bin", {m.d, m.r}, type, HALF_BYTES},
             {"w_gate.bin", {m.c, m.d}, type, HALF_BYTES, m.hasGate != 0},
             {"gamma.bin", {m.d}, ACL_FLOAT, sizeof(float), m.hasGamma != 0},
             {"inv_rms.bin", {1, m.n, m.c}, ACL_FLOAT, sizeof(float)},
             {"inject.bin", {1, m.n, m.c}, type, HALF_BYTES, m.hasGate != 0},
             {"down_pre_act.bin", {1, m.n, m.r}, type, HALF_BYTES},
             {"up.bin", {1, m.n, m.d}, type, HALF_BYTES},
             {"dx.bin", {1, m.n, m.d}, type, HALF_BYTES},
             {"dw_down.bin", {m.r, m.d}, type, HALF_BYTES},
             {"dw_up.bin", {m.d, m.r}, type, HALF_BYTES},
             {"dw_gate.bin", {m.c, m.d}, type, HALF_BYTES, m.hasGate != 0},
             {"dgamma.bin", {m.d}, ACL_FLOAT, sizeof(float), m.hasGamma != 0}}};
}

int CreateTensors(const std::string& dir, const std::array<TensorSpec, TENSOR_COUNT>& specs, Resources& resources)
{
    std::string hostBuf;
    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& spec = specs[i];
        if (!spec.enabled) {
            continue;
        }
        auto* tensor = &resources.tensors[i];
        auto* address = &resources.addresses[i];
        const auto status = i < DX ? CreateTensorFromBin(dir + "/" + spec.file, spec.shape, spec.dtype, spec.elemBytes,
                                                         tensor, address, hostBuf) :
                                     CreateOutTensor(spec.shape, spec.dtype, spec.elemBytes, tensor, address);
        if (status != 0) {
            return -1;
        }
    }
    return 0;
}

int RunOperator(Resources& resources, int64_t channels)
{
    auto& t = resources.tensors;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnTextGatedResidualGradGetWorkspaceSize(
        t[GRAD_MIXED], t[X], t[W_DOWN], t[W_UP], t[INV_RMS], t[DOWN_PRE], t[UP], t[GRAD_INJECT], t[W_GATE], t[GAMMA],
        t[INJECT], channels, t[DX], t[DW_DOWN], t[DW_UP], t[DW_GATE], t[DGAMMA], &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnTextGatedResidualGradGetWorkspaceSize failed. ERROR: %d\n", ret);
        return -1;
    }
    if (workspaceSize > 0 &&
        aclrtMalloc(&resources.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return -1;
    }
    ret = aclnnTextGatedResidualGrad(resources.workspace, workspaceSize, executor, resources.stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnTextGatedResidualGrad failed. ERROR: %d\n", ret);
        return -1;
    }
    return aclrtSynchronizeStream(resources.stream) == ACL_SUCCESS ? 0 : -1;
}

int DumpOutputs(const std::string& dir, const std::array<TensorSpec, TENSOR_COUNT>& specs, const Resources& resources)
{
    for (size_t i = DX; i < specs.size(); ++i) {
        const auto& spec = specs[i];
        if (spec.enabled && DumpOut(dir + "/" + spec.file, resources.addresses[i], spec.shape, spec.elemBytes) != 0) {
            return -1;
        }
    }
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    const std::string dir = argc > DATA_DIR_ARG ? argv[DATA_DIR_ARG] : "./data";
    Metadata metadata;
    if (!ReadMetadata(dir, metadata)) {
        return -1;
    }
    LOG_PRINT("meta: dtype=%s n=%ld c=%ld h=%ld r=%ld hasGamma=%d\n", metadata.dtype.c_str(), metadata.n, metadata.c,
              metadata.h, metadata.r, metadata.hasGamma);
    Resources resources;
    resources.device = argc > DEVICE_ARG ? atoi(argv[DEVICE_ARG]) : 0;
    const auto specs = MakeTensorSpecs(metadata);
    if (Init(resources) != 0 || CreateTensors(dir, specs, resources) != 0 || RunOperator(resources, metadata.c) != 0 ||
        DumpOutputs(dir, specs, resources) != 0) {
        return -1;
    }
    LOG_PRINT("run success, outputs dumped to %s\n", dir.c_str());
    return 0;
}
