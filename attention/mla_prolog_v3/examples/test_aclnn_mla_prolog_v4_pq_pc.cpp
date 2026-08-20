/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_mla_prolog_v4_pq_pc.cpp
 * \brief Example: 部分量化 kvCache per-channel, PA_BLK_BSND, BS合轴, queryNormFlag=true
 */

#include <iostream>
#include <cstring>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <random>
#include "acl/acl.h"
#include "aclnn/opdev/bfloat16.h"
#include "aclnnop/aclnn_mla_prolog_v4_weight_nz.h"
#include <unistd.h>

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensorND(const std::vector<T> &shape, void **deviceAddr, void **hostAddr, aclDataType dataType,
                      aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMallocHost申请host侧内存
    ret = aclrtMallocHost(hostAddr, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
    memset(*hostAddr, 0, size);
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensorNZ(const std::vector<T> &shape, void **deviceAddr, void **hostAddr, aclDataType dataType,
                      aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMallocHost申请host侧内存
    ret = aclrtMallocHost(hostAddr, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
    memset(*hostAddr, 0, size);
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              shape.data(), shape.size(), *deviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int TransToNZShape(std::vector<int64_t> &shapeND, size_t typeSize)
{
    if (typeSize == static_cast<size_t>(0)) {
        return 0;
    }
    int64_t h = shapeND[0];
    int64_t w = shapeND[1];
    int64_t h0 = static_cast<int64_t>(16);
    int64_t w0 = static_cast<int64_t>(32) / static_cast<int64_t>(typeSize);
    int64_t h1 = h / h0;
    int64_t w1 = w / w0;
    shapeND[0] = w1;
    shapeND[1] = h1;
    shapeND.emplace_back(h0);
    shapeND.emplace_back(w0);
    return 0;
}

template <typename T>
int FillHostDataAndCopy(void *deviceAddr, void *hostAddr, int64_t num, T value)
{
    if (hostAddr == nullptr || deviceAddr == nullptr) {
        LOG_PRINT("FillHostDataAndCopy: null address.\n");
        return -1;
    }
    auto *hostData = static_cast<T *>(hostAddr);
    for (int64_t i = 0; i < num; ++i) {
        hostData[i] = value;
    }
    auto ret = aclrtMemcpy(deviceAddr, num * sizeof(T), hostAddr, num * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy host data to device failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int main()
{
    // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口定义构造
    // 部分量化 kvCache per-channel场景, PA_BLK_BSND, BS合轴
    // B=4, seqlens=[2,3,1,2], T=8, BlockSize=16
    std::vector<int64_t> tokenXShape = {8, 7168};             // T,He (BS合轴)
    std::vector<int64_t> weightDqShape = {7168, 1536};        // He,Hcq
    std::vector<int64_t> weightUqQrShape = {1536, 24576};     // Hcq,N*(D+Dr)
    std::vector<int64_t> weightUkShape = {128, 128, 512};     // N,D,Hckv
    std::vector<int64_t> weightDkvKrShape = {7168, 576};      // He,Hckv+Dr
    std::vector<int64_t> rmsnormGammaCqShape = {1536};        // Hcq
    std::vector<int64_t> rmsnormGammaCkvShape = {512};        // Hckv
    std::vector<int64_t> ropeSinShape = {8, 64};              // T,Dr (BS合轴)
    std::vector<int64_t> ropeCosShape = {8, 64};              // T,Dr (BS合轴)
    std::vector<int64_t> kvCacheShape = {4, 16, 1, 512};      // BlockNum,BlockSize,Nkv,Hckv
    std::vector<int64_t> krCacheShape = {4, 16, 1, 64};       // BlockNum,BlockSize,Nkv,Dr
    std::vector<int64_t> cacheIndexShape = {4};               // Sum(Ceil(S_i/BlockSize))
    std::vector<int64_t> dequantScaleWUqQrShape = {1, 24576}; // 1,N*(D+Dr)
    std::vector<int64_t> quantScaleCkvShape = {1, 512};       // 1,Hckv (per-channel)
    std::vector<int64_t> quantScaleCkrShape = {1, 64};        // 1,Dr (per-channel)
    std::vector<int64_t> smoothScalesCqShape = {1, 1536};     // 1,Hcq
    std::vector<int64_t> actualSeqLenShape = {4};             // B (prefix sum: [2,5,6,8])
    std::vector<int64_t> queryShape = {8, 128, 512};          // T,N,Hckv (BS合轴)
    std::vector<int64_t> queryRopeShape = {8, 128, 64};       // T,N,Dr (BS合轴)
    std::vector<int64_t> queryNormOutShape = {8, 1536};       // T,Hcq (BS合轴)
    std::vector<int64_t> dequantScaleQNormOutShape = {8, 1};  // T,1 (BS合轴)

    double rmsnormEpsilonCq = 1e-5;
    double rmsnormEpsilonCkv = 1e-5;
    char cacheMode[] = "PA_BLK_BSND";

    void *tokenXDeviceAddr = nullptr;
    void *weightDqDeviceAddr = nullptr;
    void *weightUqQrDeviceAddr = nullptr;
    void *weightUkDeviceAddr = nullptr;
    void *weightDkvKrDeviceAddr = nullptr;
    void *rmsnormGammaCqDeviceAddr = nullptr;
    void *rmsnormGammaCkvDeviceAddr = nullptr;
    void *ropeSinDeviceAddr = nullptr;
    void *ropeCosDeviceAddr = nullptr;
    void *cacheIndexDeviceAddr = nullptr;
    void *kvCacheDeviceAddr = nullptr;
    void *krCacheDeviceAddr = nullptr;
    void *dequantScaleWUqQrDeviceAddr = nullptr;
    void *quantScaleCkvDeviceAddr = nullptr;
    void *quantScaleCkrDeviceAddr = nullptr;
    void *smoothScalesCqDeviceAddr = nullptr;
    void *actualSeqLenDeviceAddr = nullptr;
    void *queryDeviceAddr = nullptr;
    void *queryRopeDeviceAddr = nullptr;
    void *queryNormOutDeviceAddr = nullptr;
    void *dequantScaleQNormOutDeviceAddr = nullptr;

    void *tokenXHostAddr = nullptr;
    void *weightDqHostAddr = nullptr;
    void *weightUqQrHostAddr = nullptr;
    void *weightUkHostAddr = nullptr;
    void *weightDkvKrHostAddr = nullptr;
    void *rmsnormGammaCqHostAddr = nullptr;
    void *rmsnormGammaCkvHostAddr = nullptr;
    void *ropeSinHostAddr = nullptr;
    void *ropeCosHostAddr = nullptr;
    void *cacheIndexHostAddr = nullptr;
    void *kvCacheHostAddr = nullptr;
    void *krCacheHostAddr = nullptr;
    void *dequantScaleWUqQrHostAddr = nullptr;
    void *quantScaleCkvHostAddr = nullptr;
    void *quantScaleCkrHostAddr = nullptr;
    void *smoothScalesCqHostAddr = nullptr;
    void *actualSeqLenHostAddr = nullptr;
    void *queryHostAddr = nullptr;
    void *queryRopeHostAddr = nullptr;
    void *queryNormOutHostAddr = nullptr;
    void *dequantScaleQNormOutHostAddr = nullptr;

    aclTensor *tokenX = nullptr;
    aclTensor *weightDq = nullptr;
    aclTensor *weightUqQr = nullptr;
    aclTensor *weightUk = nullptr;
    aclTensor *weightDkvKr = nullptr;
    aclTensor *rmsnormGammaCq = nullptr;
    aclTensor *rmsnormGammaCkv = nullptr;
    aclTensor *ropeSin = nullptr;
    aclTensor *ropeCos = nullptr;
    aclTensor *kvCache = nullptr;
    aclTensor *krCache = nullptr;
    aclTensor *cacheIndex = nullptr;
    aclTensor *dequantScaleWUqQr = nullptr;
    aclTensor *quantScaleCkv = nullptr;
    aclTensor *quantScaleCkr = nullptr;
    aclTensor *smoothScalesCq = nullptr;
    aclTensor *actualSeqLen = nullptr;
    int64_t weightQuantMode = 1;
    int64_t kvQuantMode = 2;
    int64_t queryQuantMode = 0;
    int64_t ckvkrRepoMode = 0;
    int64_t quantScaleRepoMode = 0;
    int64_t tileSize = 128;
    double qcQrScale = 1.0f;
    double kcScale = 1.0f;
    bool doRope = true;
    aclTensor *query = nullptr;
    aclTensor *queryRope = nullptr;
    aclTensor *queryNormOut = nullptr;
    aclTensor *dequantScaleQNormOut = nullptr;

    // 转换NZ格式变量的shape
    constexpr size_t EXAMPLE_INT8_SIZE = sizeof(int8_t);
    constexpr size_t EXAMPLE_BFLOAT16_SIZE = sizeof(int16_t);
    // weightDq和weightDkvKr为BF16, weightUqQr为INT8
    ret = TransToNZShape(weightDqShape, EXAMPLE_BFLOAT16_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
    ret = TransToNZShape(weightUqQrShape, EXAMPLE_INT8_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
    ret = TransToNZShape(weightDkvKrShape, EXAMPLE_BFLOAT16_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);

    // 创建tokenX aclTensor
    ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_BF16, &tokenX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置tokenX数据为1
    ret = FillHostDataAndCopy(tokenXDeviceAddr, tokenXHostAddr, GetShapeSize(tokenXShape), op::bfloat16(1.0f));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightDq aclTensor
    ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_BF16, &weightDq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置weightDq数据为1
    ret = FillHostDataAndCopy(weightDqDeviceAddr, weightDqHostAddr, GetShapeSize(weightDqShape), op::bfloat16(1.0f));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightUqQr aclTensor (部分量化场景下为INT8)
    ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_INT8,
                            &weightUqQr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置weightUqQr数据为1
    ret = FillHostDataAndCopy(weightUqQrDeviceAddr, weightUqQrHostAddr, GetShapeSize(weightUqQrShape), int8_t(1));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightUk aclTensor
    ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置weightUk数据为1
    ret = FillHostDataAndCopy(weightUkDeviceAddr, weightUkHostAddr, GetShapeSize(weightUkShape), op::bfloat16(1.0f));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightDkvKr aclTensor
    ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_BF16,
                            &weightDkvKr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置weightDkvKr数据为1
    ret = FillHostDataAndCopy(weightDkvKrDeviceAddr, weightDkvKrHostAddr, GetShapeSize(weightDkvKrShape),
                              op::bfloat16(1.0f));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rmsnormGammaCq aclTensor
    ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr,
                            aclDataType::ACL_BF16, &rmsnormGammaCq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置rmsnormGammaCq数据为1
    ret = FillHostDataAndCopy(rmsnormGammaCqDeviceAddr, rmsnormGammaCqHostAddr, GetShapeSize(rmsnormGammaCqShape),
                              op::bfloat16(1.0f));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rmsnormGammaCkv aclTensor
    ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr,
                            aclDataType::ACL_BF16, &rmsnormGammaCkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置rmsnormGammaCkv数据为1
    ret = FillHostDataAndCopy(rmsnormGammaCkvDeviceAddr, rmsnormGammaCkvHostAddr, GetShapeSize(rmsnormGammaCkvShape),
                              op::bfloat16(1.0f));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建ropeSin aclTensor
    ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置ropeSin数据为1
    ret = FillHostDataAndCopy(ropeSinDeviceAddr, ropeSinHostAddr, GetShapeSize(ropeSinShape), op::bfloat16(0.5f));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建ropeCos aclTensor
    ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置ropeCos数据为1
    ret = FillHostDataAndCopy(ropeCosDeviceAddr, ropeCosHostAddr, GetShapeSize(ropeCosShape), op::bfloat16(0.5f));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建cacheIndex aclTensor
    ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64,
                            &cacheIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置cacheIndex，为每个token分配不同的cache slot，避免所有token写入同一slot
    CHECK_RET(cacheIndexHostAddr != nullptr, LOG_PRINT("cacheIndexHostAddr is null.\n"); return 0);
    int64_t *cacheIndexData = static_cast<int64_t *>(cacheIndexHostAddr);
    for (int64_t i = 0; i < cacheIndexShape[0]; ++i) {
        cacheIndexData[i] = i;
    }
    std::shuffle(cacheIndexData, cacheIndexData + cacheIndexShape[0], std::mt19937(0));
    ret = aclrtMemcpy(cacheIndexDeviceAddr, sizeof(int64_t) * cacheIndexShape[0], cacheIndexHostAddr,
                      sizeof(int64_t) * cacheIndexShape[0], ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy cacheIndex to device failed. ERROR: %d\n", ret); return ret);
    // 创建kvCache aclTensor (per-channel量化为INT8)
    ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_INT8, &kvCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置kvCache数据为1
    ret = FillHostDataAndCopy(kvCacheDeviceAddr, kvCacheHostAddr, GetShapeSize(kvCacheShape), int8_t(1));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建krCache aclTensor (per-channel量化为INT8)
    ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_INT8, &krCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置krCache数据为1
    ret = FillHostDataAndCopy(krCacheDeviceAddr, krCacheHostAddr, GetShapeSize(krCacheShape), int8_t(1));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dequantScaleWUqQr aclTensor
    ret = CreateAclTensorND(dequantScaleWUqQrShape, &dequantScaleWUqQrDeviceAddr, &dequantScaleWUqQrHostAddr,
                            aclDataType::ACL_FLOAT, &dequantScaleWUqQr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置dequantScaleWUqQr数据为1
    ret = FillHostDataAndCopy(dequantScaleWUqQrDeviceAddr, dequantScaleWUqQrHostAddr,
                              GetShapeSize(dequantScaleWUqQrShape), 1.0f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建quantScaleCkv aclTensor (per-channel)
    ret = CreateAclTensorND(quantScaleCkvShape, &quantScaleCkvDeviceAddr, &quantScaleCkvHostAddr,
                            aclDataType::ACL_FLOAT, &quantScaleCkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置quantScaleCkv数据为1
    ret = FillHostDataAndCopy(quantScaleCkvDeviceAddr, quantScaleCkvHostAddr, GetShapeSize(quantScaleCkvShape), 1.0f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建quantScaleCkr aclTensor (per-channel)
    ret = CreateAclTensorND(quantScaleCkrShape, &quantScaleCkrDeviceAddr, &quantScaleCkrHostAddr,
                            aclDataType::ACL_FLOAT, &quantScaleCkr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置quantScaleCkr数据为1
    ret = FillHostDataAndCopy(quantScaleCkrDeviceAddr, quantScaleCkrHostAddr, GetShapeSize(quantScaleCkrShape), 1.0f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建smoothScalesCq aclTensor
    ret = CreateAclTensorND(smoothScalesCqShape, &smoothScalesCqDeviceAddr, &smoothScalesCqHostAddr,
                            aclDataType::ACL_FLOAT, &smoothScalesCq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置smoothScalesCq数据为1
    ret =
        FillHostDataAndCopy(smoothScalesCqDeviceAddr, smoothScalesCqHostAddr, GetShapeSize(smoothScalesCqShape), 1.0f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建actualSeqLen aclTensor
    ret = CreateAclTensorND(actualSeqLenShape, &actualSeqLenDeviceAddr, &actualSeqLenHostAddr, aclDataType::ACL_INT32,
                            &actualSeqLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 设置actualSeqLen数据为prefix sum [2,5,6,8]
    int32_t *actualSeqLenData = static_cast<int32_t *>(actualSeqLenHostAddr);
    int32_t prefixSum[4] = {2, 5, 6, 8};
    for (int64_t i = 0; i < actualSeqLenShape[0]; ++i) {
        actualSeqLenData[i] = prefixSum[i];
    }
    ret = aclrtMemcpy(actualSeqLenDeviceAddr, sizeof(int32_t) * actualSeqLenShape[0], actualSeqLenHostAddr,
                      sizeof(int32_t) * actualSeqLenShape[0], ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy actualSeqLen to device failed. ERROR: %d\n", ret); return ret);
    // 创建query aclTensor
    ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_BF16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建queryRope aclTensor
    ret =
        CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建queryNormOut aclTensor (部分量化场景下为INT8)
    ret = CreateAclTensorND(queryNormOutShape, &queryNormOutDeviceAddr, &queryNormOutHostAddr, aclDataType::ACL_INT8,
                            &queryNormOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dequantScaleQNormOut aclTensor
    ret = CreateAclTensorND(dequantScaleQNormOutShape, &dequantScaleQNormOutDeviceAddr, &dequantScaleQNormOutHostAddr,
                            aclDataType::ACL_FLOAT, &dequantScaleQNormOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    // 调用aclnnMlaPrologV4WeightNz第一段接口
    ret = aclnnMlaPrologV4WeightNzGetWorkspaceSize(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache,
        krCache, cacheIndex, nullptr, nullptr, dequantScaleWUqQr, nullptr, quantScaleCkv, quantScaleCkr, smoothScalesCq,
        actualSeqLen, nullptr, rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode, weightQuantMode, kvQuantMode,
        queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize, qcQrScale, kcScale, doRope, query, queryRope,
        nullptr, queryNormOut, dequantScaleQNormOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV4WeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMlaPrologV4WeightNz第二段接口
    ret = aclnnMlaPrologV4WeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV4WeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(queryShape);
    auto copySize = size * aclDataTypeSize(aclDataType::ACL_BF16);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), copySize, queryDeviceAddr, copySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    uint64_t printBufferSize = 32;
    for (int64_t i = 0; i < printBufferSize; i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
    }

    // 6. 释放aclTensor
    aclDestroyTensor(tokenX);
    aclDestroyTensor(weightDq);
    aclDestroyTensor(weightUqQr);
    aclDestroyTensor(weightUk);
    aclDestroyTensor(weightDkvKr);
    aclDestroyTensor(rmsnormGammaCq);
    aclDestroyTensor(rmsnormGammaCkv);
    aclDestroyTensor(ropeSin);
    aclDestroyTensor(ropeCos);
    aclDestroyTensor(cacheIndex);
    aclDestroyTensor(kvCache);
    aclDestroyTensor(krCache);
    aclDestroyTensor(dequantScaleWUqQr);
    aclDestroyTensor(quantScaleCkv);
    aclDestroyTensor(quantScaleCkr);
    aclDestroyTensor(smoothScalesCq);
    aclDestroyTensor(actualSeqLen);
    aclDestroyTensor(query);
    aclDestroyTensor(queryRope);
    aclDestroyTensor(queryNormOut);
    aclDestroyTensor(dequantScaleQNormOut);

    // 7. 释放device 资源
    aclrtFree(tokenXDeviceAddr);
    aclrtFree(weightDqDeviceAddr);
    aclrtFree(weightUqQrDeviceAddr);
    aclrtFree(weightUkDeviceAddr);
    aclrtFree(weightDkvKrDeviceAddr);
    aclrtFree(rmsnormGammaCqDeviceAddr);
    aclrtFree(rmsnormGammaCkvDeviceAddr);
    aclrtFree(ropeSinDeviceAddr);
    aclrtFree(ropeCosDeviceAddr);
    aclrtFree(cacheIndexDeviceAddr);
    aclrtFree(kvCacheDeviceAddr);
    aclrtFree(krCacheDeviceAddr);
    aclrtFree(dequantScaleWUqQrDeviceAddr);
    aclrtFree(quantScaleCkvDeviceAddr);
    aclrtFree(quantScaleCkrDeviceAddr);
    aclrtFree(smoothScalesCqDeviceAddr);
    aclrtFree(actualSeqLenDeviceAddr);
    aclrtFree(queryDeviceAddr);
    aclrtFree(queryRopeDeviceAddr);
    aclrtFree(queryNormOutDeviceAddr);
    aclrtFree(dequantScaleQNormOutDeviceAddr);

    // 8. 释放host 资源
    aclrtFreeHost(tokenXHostAddr);
    aclrtFreeHost(weightDqHostAddr);
    aclrtFreeHost(weightUqQrHostAddr);
    aclrtFreeHost(weightUkHostAddr);
    aclrtFreeHost(weightDkvKrHostAddr);
    aclrtFreeHost(rmsnormGammaCqHostAddr);
    aclrtFreeHost(rmsnormGammaCkvHostAddr);
    aclrtFreeHost(ropeSinHostAddr);
    aclrtFreeHost(ropeCosHostAddr);
    aclrtFreeHost(cacheIndexHostAddr);
    aclrtFreeHost(kvCacheHostAddr);
    aclrtFreeHost(krCacheHostAddr);
    aclrtFreeHost(dequantScaleWUqQrHostAddr);
    aclrtFreeHost(quantScaleCkvHostAddr);
    aclrtFreeHost(quantScaleCkrHostAddr);
    aclrtFreeHost(smoothScalesCqHostAddr);
    aclrtFreeHost(actualSeqLenHostAddr);
    aclrtFreeHost(queryHostAddr);
    aclrtFreeHost(queryRopeHostAddr);
    aclrtFreeHost(queryNormOutHostAddr);
    aclrtFreeHost(dequantScaleQNormOutHostAddr);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
