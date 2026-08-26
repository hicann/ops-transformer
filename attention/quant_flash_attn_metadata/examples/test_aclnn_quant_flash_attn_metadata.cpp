/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_flash_attn_metadata.h"
#include "../op_kernel_aicpu/quant_flash_attn_metadata.h"
#include "securec.h"

using namespace std;
using namespace optiling;

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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

static void DumpMeta(void *data)
{
    int32_t sectionNum = static_cast<int32_t>(((int32_t *)data)[0]);
    optiling::detail::FaMetaData faMetadata(data, sectionNum);
    printf("sectionNum:%d\n", faMetadata.GetHeadMedata(optiling::HEAD_SECTION_NUM_INDEX));
    printf("isFd:%d\n", faMetadata.GetHeadMedata(optiling::HEAD_IS_FD_INDEX));
    printf("mBaseSize:%d\n", faMetadata.GetHeadMedata(optiling::HEAD_M_BASE_SIZE_INDEX));
    printf("s2BaseSize:%d\n", faMetadata.GetHeadMedata(optiling::HEAD_S2_BASE_SIZE_INDEX));
    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        printf("sectionIdx:%d\n", sectionId);
        for (size_t i = 0; i < AIC_CORE_NUM; ++i) {
            printf("bn2 start: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_BN_START_INDEX));
            printf("m start: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_M_START_INDEX));
            printf("s2 start: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX));
            printf("bn2 end: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_BN_END_INDEX));
            printf("m end: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_M_END_INDEX));
            printf("s2 end: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX));
            printf("first fd data ws idx: %d\n",
                   faMetadata.GetFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX));
        }

        for (size_t i = 0; i < AIV_CORE_NUM; ++i) {
            printf("bn2 idx: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_BN_IDX_INDEX));
            printf("m idx: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_M_IDX_INDEX));
            printf("fd workspace idx: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_IDX_INDEX));
            printf("fd workspace num: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_NUM_INDEX));
            printf("m start: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_M_START_INDEX));
            printf("m num: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_M_NUM_INDEX));
        }
    }
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    int32_t batchSize = 4;
    int32_t numHeads = 32;
    int32_t numKeyValueHeads = numHeads;
    int32_t qS = 1;
    int32_t kvS = 8192;
    int32_t headDim = 128;
    int32_t quantMode = 1;

    int32_t preTokens = -1;
    int32_t nextTokens = -1;
    string layout_query = "BSND";
    char layoutQuery[layout_query.length() + 1];
    strcpy(layoutQuery, layout_query.c_str());
    string layout_kv = "BSND";
    char layoutKv[layout_kv.length() + 1];
    strcpy(layoutKv, layout_kv.c_str());
    int32_t sparseMode = 3;

    int64_t metadataSize = ((36 + 72) * batchSize * numKeyValueHeads + 1) * 16;
    int64_t alignedSize = ((metadataSize + 4095) / 4096) * 4096;

    std::vector<int64_t> actualSeqLengthsQueryShape = {batchSize};
    std::vector<int64_t> actualSeqLengthsKvShape = {batchSize};
    std::vector<int64_t> metadataShape = {alignedSize};
    void *actualSeqLengthsQueryDeviceAddr = nullptr;
    void *actualSeqLengthsKvDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    aclTensor *actualSeqLengthsQueryTensor = nullptr;
    aclTensor *actualSeqLengthsKvTensor = nullptr;
    aclTensor *metadataTensor = nullptr;
    int64_t actualSeqLengthsQueryShapeSize = GetShapeSize(actualSeqLengthsQueryShape);
    int64_t actualSeqLengthsKvShapeSize = GetShapeSize(actualSeqLengthsKvShape);
    int64_t metadataShapeSize = GetShapeSize(metadataShape);
    std::vector<int32_t> actualSeqLengthsQueryHostData(actualSeqLengthsQueryShapeSize, qS);
    std::vector<int32_t> actualSeqLengthsKvHostData(actualSeqLengthsKvShapeSize, kvS);
    std::vector<int32_t> metadataHostData(metadataShapeSize, 0);

    ret = CreateAclTensor(actualSeqLengthsQueryHostData, actualSeqLengthsQueryShape, &actualSeqLengthsQueryDeviceAddr,
                          aclDataType::ACL_INT32, &actualSeqLengthsQueryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(actualSeqLengthsKvHostData, actualSeqLengthsKvShape, &actualSeqLengthsKvDeviceAddr,
                          aclDataType::ACL_INT32, &actualSeqLengthsKvTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret =
        CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadataTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    printf("start aclnnQuantFlashAttnMetadata\n");
    ret = aclnnQuantFlashAttnMetadataGetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, batchSize, qS, kvS, numHeads, numKeyValueHeads, headDim, quantMode,
        sparseMode, preTokens, nextTokens, "BSND", "BSND", "BSND", "BSND", metadataTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("aclnnQuantFlashAttnMetadataGetWorkspaceSize %d\n", ret);
        return -1;
    }

    ret = aclnnQuantFlashAttnMetadata(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("aclnnQuantFlashAttnMetadata %d\n", ret);
        return -1;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSynchronizeStream %d\n", ret);
        return -1;
    }

    ret = aclrtMemcpy(metadataHostData.data(), metadataHostData.size() * sizeof(metadataHostData[0]),
                      metadataDeviceAddr, metadataShapeSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        printf("aclrtMemcpy %d\n", ret);
        return -1;
    }

    DumpMeta(&metadataHostData[0]);

    aclDestroyTensor(metadataTensor);
    aclrtFree(metadataDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
