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
 * \file test_aclnn_quant_lightning_indexer_v2.cpp
 * rief QuantLightningIndexerV2（两级TopK 的第一级 / source 级）aclnn 两段式接口调用示例。
 *        candidate_topk_blocks=2048（开启 source 的唯一有效值）: 除 sparse_indices/sparse_values 外，还输出块级候选索引
 *        candidate_topk_index_out（供 QuantSparseLightningIndexer 消费）。
 *        arch22 (910B/910_93) 用例: INT8 量化, BSND query + PA_BBND 分页 key。
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v2.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v2_metadata.h"
#include "aclnn/opdev/platform.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        (void)printf(message, ##__VA_ARGS__); \
    } while (0)

// 示例规模：BSND 的 query + PA_BBND 的 key（910B/910_93 上 layout_k 仅支持分页布局）
constexpr int64_t B = 2;                       // batch
constexpr int64_t S1 = 4;                      // query 序列长度
constexpr int64_t N1 = 64;                     // query 头数（A2/A3 仅支持 64）
constexpr int64_t N2 = 1;                      // key 头数（仅支持 1）
constexpr int64_t D = 128;                     // head dim
constexpr int64_t PA_BLOCK_SIZE = 16;          // 分页 block 大小（须为 16 的倍数且属于 (0, 1024]）
constexpr int64_t MAX_BLOCK_NUM_PER_BATCH = 1; // block_table 第 1 维
constexpr int64_t S2 = PA_BLOCK_SIZE * MAX_BLOCK_NUM_PER_BATCH; // key 序列长度（由分页布局推导）
constexpr int64_t BLOCK_NUM = B * MAX_BLOCK_NUM_PER_BATCH;      // 分页 block 总数
constexpr int64_t TOPK = 512;                                   // topk 取值 (0, 2048]
constexpr int64_t CANDIDATE_TOPK_BLOCKS = 2048; // 候选块个数(开关): -1=关闭; 2048=开启 source (当前仅支持 -1/2048)
constexpr int64_t CANDIDATE_BLOCK_SIZE = 8; // 候选块大小，当前仅支持 8
constexpr int64_t QUANT_MODE = 2;           // 910B/910_93 仅支持 INT8 量化

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

// 缺省的可选输入按接口约定传 shape (0,) 的 INT32 空 tensor（与 torch 扩展的 get_valid_tensor 一致）:
// metadata 算子的参数校验会解引用这些张量, 传 nullptr 会崩溃
int CreateEmptyOptionalTensor(void** deviceAddr, aclTensor** tensor)
{
    static int64_t emptyShape[1] = {0};
    static int64_t emptyStride[1] = {1};
    auto ret = aclrtMalloc(deviceAddr, static_cast<size_t>(sizeof(int32_t)), ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc for empty optional tensor failed. ERROR: %d\n", ret);
        return ret;
    }
    *tensor = aclCreateTensor(emptyShape, 1, aclDataType::ACL_INT32, emptyStride, 0, aclFormat::ACL_FORMAT_ND,
                              emptyShape, 1, *deviceAddr);
    return static_cast<int>(ACL_SUCCESS);
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    // 空 tensor（如 return_value=0 时的 sparse_values）按 1 个元素申请，避免 0 字节申请失败
    auto allocSize = (size == 0) ? static_cast<size_t>(sizeof(T)) : static_cast<size_t>(size);
    auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }

    if (size > 0) {
        ret =
            aclrtMemcpy(*deviceAddr, allocSize, hostData.data(), static_cast<size_t>(size), ACL_MEMCPY_HOST_TO_DEVICE);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* qScaleDeviceAddr = nullptr;
    void* kScaleDeviceAddr = nullptr;
    void* blockTableDeviceAddr = nullptr;
    void* sequsedKDeviceAddr = nullptr;
    void* metadataDeviceAddr = nullptr;
    void* candidateTopkIndexOutDeviceAddr = nullptr;
    void* candidateBlockLengthOutDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* qScaleTensor = nullptr;
    aclTensor* kScaleTensor = nullptr;
    aclTensor* blockTableTensor = nullptr;
    aclTensor* sequsedKTensor = nullptr;
    aclTensor* metadataTensor = nullptr;
    aclTensor* candidateTopkIndexOutTensor = nullptr;
    aclTensor* candidateBlockLengthOutTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources)
{
    std::vector<int64_t> queryShape = {B, S1, N1, D};
    std::vector<int64_t> keyShape = {BLOCK_NUM, PA_BLOCK_SIZE, N2, D};
    std::vector<int64_t> weightsShape = {B, S1, N1};
    std::vector<int64_t> qScaleShape = {B, S1, N1};
    std::vector<int64_t> kScaleShape = {BLOCK_NUM, PA_BLOCK_SIZE, N2};
    std::vector<int64_t> blockTableShape = {B, MAX_BLOCK_NUM_PER_BATCH};
    std::vector<int64_t> sequsedKShape = {B};
    std::vector<int64_t> metadataShape = {1024};
    std::vector<int64_t> candidateOutShape = {B, S1, N2, CANDIDATE_TOPK_BLOCKS};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, TOPK};
    std::vector<int64_t> sparseValuesShape = {0}; // return_value=0 时 sparse_values 为空 tensor

    std::vector<int8_t> queryHostData(GetShapeSize(queryShape), 1);
    std::vector<int8_t> keyHostData(GetShapeSize(keyShape), 1);
    std::vector<uint16_t> weightsHostData(GetShapeSize(weightsShape), 0x3C00); // fp16 1.0
    std::vector<uint16_t> qScaleHostData(GetShapeSize(qScaleShape), 0x3C00);
    std::vector<uint16_t> kScaleHostData(GetShapeSize(kScaleShape), 0x3C00);
    std::vector<int32_t> blockTableHostData(GetShapeSize(blockTableShape), 0);
    std::vector<int32_t> sequsedKHostData(GetShapeSize(sequsedKShape), static_cast<int32_t>(S2));
    std::vector<int32_t> metadataHostData(GetShapeSize(metadataShape), 0);
    std::vector<int32_t> candidateOutHostData(GetShapeSize(candidateOutShape), 0);
    std::vector<int32_t> sparseIndicesHostData(GetShapeSize(sparseIndicesShape), 0);
    std::vector<uint16_t> sparseValuesHostData(0, 0);

    for (int64_t b = 0; b < B; b++) { // block_table 第 b 行指向第 b 个分页 block
        blockTableHostData[static_cast<size_t>(b)] = static_cast<int32_t>(b);
    }

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, aclDataType::ACL_INT8,
                              &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, aclDataType::ACL_INT8, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr, aclDataType::ACL_FLOAT16,
                          &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(qScaleHostData, qScaleShape, &resources.qScaleDeviceAddr, aclDataType::ACL_FLOAT16,
                          &resources.qScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(kScaleHostData, kScaleShape, &resources.kScaleDeviceAddr, aclDataType::ACL_FLOAT16,
                          &resources.kScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(blockTableHostData, blockTableShape, &resources.blockTableDeviceAddr, aclDataType::ACL_INT32,
                          &resources.blockTableTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(sequsedKHostData, sequsedKShape, &resources.sequsedKDeviceAddr, aclDataType::ACL_INT32,
                          &resources.sequsedKTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(metadataHostData, metadataShape, &resources.metadataDeviceAddr, aclDataType::ACL_INT32,
                          &resources.metadataTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(candidateOutHostData, candidateOutShape, &resources.candidateTopkIndexOutDeviceAddr,
                          aclDataType::ACL_INT32, &resources.candidateTopkIndexOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // candidate_block_length: 预留接口输出, 恒为空 tensor
    std::vector<int64_t> candidateBlockLengthOutShape = {0};
    std::vector<int32_t> candidateBlockLengthOutHostData(0, 0);
    ret = CreateAclTensor(candidateBlockLengthOutHostData, candidateBlockLengthOutShape,
                          &resources.candidateBlockLengthOutDeviceAddr, aclDataType::ACL_INT32,
                          &resources.candidateBlockLengthOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                          aclDataType::ACL_BF16, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    return ACL_SUCCESS;
}

// metadata 需与本次调用的 query/key 形状匹配：layout_k 为 PA_BBND 时 seqused_k 必传
int GenerateMetadata(TensorResources& resources, aclrtStream stream)
{
    constexpr const char layoutQ[] = "BSND";
    constexpr const char layoutK[] = "PA_BBND";
    constexpr size_t layoutQLen = sizeof(layoutQ);
    constexpr size_t layoutKLen = sizeof(layoutK);
    char layoutQCopy[layoutQLen];
    char layoutKCopy[layoutKLen];
    errno_t memcpyRet = memcpy_s(layoutQCopy, sizeof(layoutQCopy), layoutQ, layoutQLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("metadata memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKCopy, sizeof(layoutKCopy), layoutK, layoutKLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("metadata memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
        return -1;
    }

    // 缺省的可选输入（cu_seqlens_q/k、seqused_q、cmp_residual_k）传 shape (0,) 的空 tensor,
    // 不能传 nullptr
    aclTensor* emptyOptTensors[4] = {nullptr, nullptr, nullptr, nullptr};
    void* emptyOptDevAddr[4] = {nullptr, nullptr, nullptr, nullptr};
    int ret = ACL_SUCCESS;
    for (int i = 0; i < 4; ++i) {
        ret = CreateEmptyOptionalTensor(&emptyOptDevAddr[i], &emptyOptTensors[i]);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("create empty optional tensor failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    ret = aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize(
        emptyOptTensors[0], emptyOptTensors[1], emptyOptTensors[2], resources.sequsedKTensor, emptyOptTensors[3], N1,
        N2, D, TOPK, QUANT_MODE, B, S1, S2, layoutQCopy, layoutKCopy, 0, 1, resources.metadataTensor, &workspaceSize,
        &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    void* metadataWsAddr = nullptr;
    if (workspaceSize > 0ULL) {
        ret = aclrtMalloc(&metadataWsAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("metadata allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnQuantLightningIndexerV2Metadata(metadataWsAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2Metadata failed. ERROR: %d\n", ret);
        if (metadataWsAddr) {
            (void)aclrtFree(metadataWsAddr);
        }
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("metadata synchronize stream failed. ERROR: %d\n", ret);
        if (metadataWsAddr) {
            (void)aclrtFree(metadataWsAddr);
        }
        return ret;
    }

    if (metadataWsAddr) {
        (void)aclrtFree(metadataWsAddr);
    }
    for (int i = 0; i < 4; ++i) {
        if (emptyOptTensors[i] != nullptr) {
            (void)aclDestroyTensor(emptyOptTensors[i]);
        }
        if (emptyOptDevAddr[i] != nullptr) {
            (void)aclrtFree(emptyOptDevAddr[i]);
        }
    }
    return ACL_SUCCESS;
}

int ExecuteQuantLightningIndexerV2(TensorResources& resources, aclrtStream stream, void** workspaceAddr,
                                   uint64_t* workspaceSize)
{
    constexpr const char layoutQStr[] = "BSND";
    constexpr const char layoutKStr[] = "PA_BBND";
    constexpr size_t layoutQLen = sizeof(layoutQStr);
    constexpr size_t layoutKLen = sizeof(layoutKStr);
    char layoutQ[layoutQLen];
    char layoutK[layoutKLen];
    errno_t memcpyRet = memcpy_s(layoutQ, sizeof(layoutQ), layoutQStr, layoutQLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutK, sizeof(layoutK), layoutKStr, layoutKLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor = nullptr;

    int ret = aclnnQuantLightningIndexerV2GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor, resources.qScaleTensor,
        resources.kScaleTensor, nullptr /* cuSeqLensQ: BSND 不传 */, nullptr /* cuSeqLensK: PA_BBND 不传 */,
        nullptr /* sequsedQ */, resources.sequsedKTensor, nullptr /* cmpResidualK */, resources.blockTableTensor,
        nullptr /* outputIdxOffset */, resources.metadataTensor, TOPK, QUANT_MODE, -1 /* maxSeqlenQ */, layoutQ,
        layoutK, 0 /* maskMode */, 1 /* cmpRatio */, 0 /* returnValue */, CANDIDATE_TOPK_BLOCKS, CANDIDATE_BLOCK_SIZE,
        resources.sparseIndicesTensor, resources.sparseValuesTensor, resources.candidateTopkIndexOutTensor,
        resources.candidateBlockLengthOutTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnQuantLightningIndexerV2(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(const std::vector<int64_t>& shape, void* deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    LOG_PRINT("sparse_indices result (first 10 elements):\n");
    for (int64_t i = 0; i < size && i < 10; i++) {
        LOG_PRINT("  [%ld] = %d\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

int PrintCandidateOutResult(const std::vector<int64_t>& shape, void* deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy candidate_topk_index_out from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    LOG_PRINT("candidate_topk_index_out (first 8 elements, -1 = 无效槽):\n");
    for (int64_t i = 0; i < size && i < 8; i++) {
        LOG_PRINT("  [%ld] = %d\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr, aclrtStream stream, int32_t deviceId)
{
    aclTensor* tensors[] = {
        resources.queryTensor,         resources.keyTensor,         resources.weightsTensor,
        resources.qScaleTensor,        resources.kScaleTensor,      resources.blockTableTensor,
        resources.sequsedKTensor,      resources.metadataTensor,    resources.candidateTopkIndexOutTensor,
        resources.sparseIndicesTensor, resources.sparseValuesTensor};
    for (auto* tensor : tensors) {
        if (tensor) {
            aclDestroyTensor(tensor);
        }
    }

    void* deviceAddrs[] = {resources.queryDeviceAddr,
                           resources.keyDeviceAddr,
                           resources.weightsDeviceAddr,
                           resources.qScaleDeviceAddr,
                           resources.kScaleDeviceAddr,
                           resources.blockTableDeviceAddr,
                           resources.sequsedKDeviceAddr,
                           resources.metadataDeviceAddr,
                           resources.candidateTopkIndexOutDeviceAddr,
                           resources.candidateBlockLengthOutDeviceAddr,
                           resources.sparseValuesDeviceAddr};
    for (auto* addr : deviceAddrs) {
        if (addr) {
            aclrtFree(addr);
        }
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    if (stream) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, TOPK};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 本算子仅支持 arch22 (910B/910_93); Ascend 950 的 arch35 实现已移除, def 不再注册 ascend950
    // 注: 架构查询需在 Init (aclInit + aclrtSetDevice) 之后, 否则 GetCurNpuArch 返回未初始化值
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201) {
        LOG_PRINT("QuantLightningIndexerV2 only supports ascend910b/ascend910_93 (arch22), skip this sample.\n");
        return 0;
    }

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = GenerateMetadata(resources, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("GenerateMetadata failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteQuantLightningIndexerV2(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteQuantLightningIndexerV2 failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintOutResult(sparseIndicesShape, resources.sparseIndicesDeviceAddr);

    std::vector<int64_t> candidateOutShapeMain = {B, S1, N2, CANDIDATE_TOPK_BLOCKS};
    (void)PrintCandidateOutResult(candidateOutShapeMain, resources.candidateTopkIndexOutDeviceAddr);

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
