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
 * \file test_all_gather_add_common.h
 * \brief AllGatherAdd 测试共享基础设施：宏、公共结构体、与 shape 无关的 helper。
 *        由 test_aclnn_all_gather_add.cpp（入口）和各用例 header 共同 include。
 */

#ifndef TEST_ALL_GATHER_ADD_COMMON_H
#define TEST_ALL_GATHER_ADD_COMMON_H

#include <thread>
#include <future>
#include <atomic>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "../op_host/op_api/aclnn_all_gather_add.h"
#include "acl/acl.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            LOG_PRINT("Example failed.\n"); \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int RANK_DIM = 2;

typedef struct {
    std::vector<op::fp16_t> rank0_a;
    std::vector<op::fp16_t> rank0_b;
    std::vector<op::fp16_t> rank1_a;
    std::vector<op::fp16_t> rank1_b;

    std::vector<op::fp16_t> gatherOut;
    std::vector<op::fp16_t> rank0_c;
    std::vector<op::fp16_t> rank1_c;
} TestData;

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
};

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// 一个用例的 tensor + device 内存句柄（a/b/out/gatherOut）
struct TensorSet {
    void* aDeviceAddr = nullptr;
    void* bDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* gatherOutDeviceAddr = nullptr;
    aclTensor* a = nullptr;
    aclTensor* b = nullptr;
    aclTensor* out = nullptr;
    aclTensor* gatherOut = nullptr;
};

// 创建一组 a/b/out/gatherOut tensor（host 数据用 0 初始化，真实输入由 RunOneRound 填充）
static int CreateTensorSet(TensorSet& ts, const std::vector<int64_t>& aShape, const std::vector<int64_t>& bShape)
{
    int64_t aSize = GetShapeSize(aShape);
    int64_t bSize = GetShapeSize(bShape);
    std::vector<op::fp16_t> aHost(aSize, 0);
    std::vector<op::fp16_t> bHost(bSize, 0);
    std::vector<op::fp16_t> outHost(bSize, 0);
    std::vector<op::fp16_t> gatherOutHost(bSize, 0);

    int ret = CreateAclTensor(aHost, aShape, &ts.aDeviceAddr, aclDataType::ACL_FLOAT16, &ts.a);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(bHost, bShape, &ts.bDeviceAddr, aclDataType::ACL_FLOAT16, &ts.b);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHost, bShape, &ts.outDeviceAddr, aclDataType::ACL_FLOAT16, &ts.out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gatherOutHost, bShape, &ts.gatherOutDeviceAddr, aclDataType::ACL_FLOAT16, &ts.gatherOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (ts.aDeviceAddr == nullptr || ts.bDeviceAddr == nullptr || ts.outDeviceAddr == nullptr ||
        ts.gatherOutDeviceAddr == nullptr) {
        return -1;
    }
    return 0;
}

// 释放一组 tensor + device 内存
static void DestroyTensorSet(TensorSet& ts)
{
    if (ts.a != nullptr) {
        aclDestroyTensor(ts.a);
    }
    if (ts.b != nullptr) {
        aclDestroyTensor(ts.b);
    }
    if (ts.out != nullptr) {
        aclDestroyTensor(ts.out);
    }
    if (ts.gatherOut != nullptr) {
        aclDestroyTensor(ts.gatherOut);
    }
    if (ts.aDeviceAddr != nullptr) {
        aclrtFree(ts.aDeviceAddr);
    }
    if (ts.bDeviceAddr != nullptr) {
        aclrtFree(ts.bDeviceAddr);
    }
    if (ts.outDeviceAddr != nullptr) {
        aclrtFree(ts.outDeviceAddr);
    }
    if (ts.gatherOutDeviceAddr != nullptr) {
        aclrtFree(ts.gatherOutDeviceAddr);
    }
}

static int CompareVector(std::vector<op::fp16_t>& vec1, const std::vector<op::fp16_t>& vec2)
{
    const float tolerance = 0.001f; // 千分之一
    for (size_t i = 0; i < vec1.size(); ++i) {
        float a = static_cast<float>(vec1[i]);
        float b = static_cast<float>(vec2[i]);

        float diff = std::fabs(a - b);
        float maxAbs = std::max(std::fabs(a), std::fabs(b));
        bool mismatch = false;
        if (maxAbs > 1e-6f) {
            mismatch = (diff / maxAbs) > tolerance;
        } else {
            mismatch = diff > tolerance;
        }
        if (mismatch) {
            return 1;
        }
    }
    return 0;
}

static int RandomVectorGenerator(std::vector<op::fp16_t>& vec)
{
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);
    for (auto& elem : vec) {
        elem = static_cast<op::fp16_t>(distribution(generator));
    }
    return 0;
}

// 拼接两个大小相同的 vector
static int GatherVectors(std::vector<op::fp16_t>& vec1, std::vector<op::fp16_t>& vec2, std::vector<op::fp16_t>& vec3)
{
    vec3.clear();
    vec3.reserve(vec1.size() + vec2.size());
    vec3.insert(vec3.end(), vec1.begin(), vec1.end());
    vec3.insert(vec3.end(), vec2.begin(), vec2.end());
    return 0;
}

// 对两个大小相同的 vector 执行加法
static int AddVectors(std::vector<op::fp16_t>& vec1, std::vector<op::fp16_t>& vec2, std::vector<op::fp16_t>& vec3)
{
    vec3.clear();
    vec3.resize(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        vec3[i] = vec1[i] + vec2[i];
    }
    return 0;
}

static int GenerateTestData(TestData& testData)
{
    // 随机生成输入
    int ret = RandomVectorGenerator(testData.rank0_a);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank0_a failed. ret = %d \n", ret);
              return ret);
    ret = RandomVectorGenerator(testData.rank0_b);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank0_b failed. ret = %d \n", ret);
              return ret);
    ret = RandomVectorGenerator(testData.rank1_a);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank1_a failed. ret = %d \n", ret);
              return ret);
    ret = RandomVectorGenerator(testData.rank1_b);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank1_b failed. ret = %d \n", ret);
              return ret);

    // 计算 golden 数据
    ret = GatherVectors(testData.rank0_a, testData.rank1_a, testData.gatherOut);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate gatherOut failed. ret = %d \n", ret); return ret);
    ret = AddVectors(testData.gatherOut, testData.rank0_b, testData.rank0_c);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate rank0 output failed. ret = %d \n", ret); return ret);
    ret = AddVectors(testData.gatherOut, testData.rank1_b, testData.rank1_c);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate rank1 output failed. ret = %d \n", ret); return ret);

    return 0;
}

// 生成确定性测试数据：a 全为 va0/va1，b 全为 vb0/vb1，并计算 golden。
static int GenerateConstTestData(TestData& testData, float va0, float va1, float vb0, float vb1)
{
    std::fill(testData.rank0_a.begin(), testData.rank0_a.end(), op::fp16_t(va0));
    std::fill(testData.rank1_a.begin(), testData.rank1_a.end(), op::fp16_t(va1));
    std::fill(testData.rank0_b.begin(), testData.rank0_b.end(), op::fp16_t(vb0));
    std::fill(testData.rank1_b.begin(), testData.rank1_b.end(), op::fp16_t(vb1));

    int ret = GatherVectors(testData.rank0_a, testData.rank1_a, testData.gatherOut);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate gatherOut failed. ret = %d \n", ret); return ret);
    ret = AddVectors(testData.gatherOut, testData.rank0_b, testData.rank0_c);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate rank0 output failed. ret = %d \n", ret); return ret);
    ret = AddVectors(testData.gatherOut, testData.rank1_b, testData.rank1_c);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate rank1 output failed. ret = %d \n", ret); return ret);

    return 0;
}

#endif // TEST_ALL_GATHER_ADD_COMMON_H
