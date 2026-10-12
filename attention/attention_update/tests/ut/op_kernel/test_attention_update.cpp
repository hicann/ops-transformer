/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "attention_update_tiling.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"

namespace {
template <typename T1, typename T2>
inline T1 CeilA2B(T1 a, T2 b)
{
    return (a + b - 1) / b;
}

template <typename T>
uint8_t* CreateTensorList(const std::vector<std::vector<uint64_t>>& shapeInfos)
{
    uint64_t tensorListDescCount = 1 + shapeInfos.size() * 2;
    for (const auto& shape : shapeInfos) {
        tensorListDescCount += shape.size();
    }
    std::vector<uint64_t> shapeSizeList;
    auto* tensorListDesc = reinterpret_cast<uint64_t*>(AscendC::GmAlloc(tensorListDescCount * sizeof(uint64_t)));
    *tensorListDesc = (tensorListDescCount - shapeInfos.size()) * sizeof(uint64_t);
    uint64_t addrIndex = 0;
    for (size_t i = 0; i < shapeInfos.size(); ++i) {
        ++addrIndex;
        uint16_t dimCount = shapeInfos[i].size();
        *(tensorListDesc + addrIndex) = (static_cast<uint64_t>(i) << 32) + dimCount;
        uint64_t shapeSize = 1;
        for (size_t j = 0; j < dimCount; ++j) {
            ++addrIndex;
            *(tensorListDesc + addrIndex) = shapeInfos[i][j];
            shapeSize *= shapeInfos[i][j];
        }
        shapeSizeList.push_back(shapeSize);
    }
    for (size_t i = 0; i < shapeInfos.size(); ++i) {
        ++addrIndex;
        uint64_t dataSize = shapeSizeList[i] * sizeof(T);
        auto* dataPtr = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilA2B(dataSize, 32) * 32));
        *(tensorListDesc + addrIndex) = reinterpret_cast<uint64_t>(dataPtr);
    }
    return reinterpret_cast<uint8_t*>(tensorListDesc);
}

void FreeTensorList(uint8_t* addr, const std::vector<std::vector<uint64_t>>& shapeInfos)
{
    uint64_t dataPtrOffset = *reinterpret_cast<uint64_t*>(addr);
    uint8_t* dataAddr = addr + dataPtrOffset;
    for (size_t i = 0; i < shapeInfos.size(); ++i) {
        auto* tensorAddr = reinterpret_cast<uint8_t*>(*(reinterpret_cast<uint64_t*>(dataAddr) + i));
        AscendC::GmFree(tensorAddr);
    }
    AscendC::GmFree(addr);
}
} // namespace

extern "C" __global__ __aicore__ void attention_update(GM_ADDR les, GM_ADDR in, GM_ADDR out, GM_ADDR lseout,
                                                       GM_ADDR workspace, GM_ADDR tiling);

struct AttentionUpdateTestParam {
    std::string caseName;
    size_t dataTypeSize;
    uint32_t tilingKey;
    DecodeUpdateTilingData tiling;
};

class AttentionUpdateTest : public testing::TestWithParam<AttentionUpdateTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AttentionUpdateTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AttentionUpdateTest TearDown" << std::endl;
    }
};

TEST_P(AttentionUpdateTest, KernelSmoke)
{
    AttentionUpdateTestParam param = GetParam();
    const DecodeUpdateTilingData& tilingParam = param.tiling;

    int64_t outputShapeSize = tilingParam.totalLength * tilingParam.hDim;
    int64_t lseoutShapeSize = tilingParam.totalLength;
    int64_t outputByteSize = outputShapeSize * param.dataTypeSize;
    int64_t lseoutByteSize = lseoutShapeSize * sizeof(float);

    std::vector<std::vector<uint64_t>> lseShapeInfos(tilingParam.sp, {tilingParam.totalLength});
    std::vector<std::vector<uint64_t>> goShapeInfos(tilingParam.sp, {tilingParam.totalLength, tilingParam.hDim});

    constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024;
    int64_t tilingSize = sizeof(DecodeUpdateTilingData);

    uint8_t* lse = CreateTensorList<float>(lseShapeInfos);
    uint8_t* go = CreateTensorList<float>(goShapeInfos);
    auto* output = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputByteSize));
    auto* lseout = tilingParam.updateType == 0 ? nullptr : reinterpret_cast<uint8_t*>(AscendC::GmAlloc(lseoutByteSize));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(WORKSPACE_SIZE));
    auto* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingSize));

    *reinterpret_cast<DecodeUpdateTilingData*>(tiling) = tilingParam;

    uint32_t blockDim = tilingParam.formerNum + tilingParam.tailNum;
    ICPU_SET_TILING_KEY(param.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(attention_update, blockDim, lse, go, output, lseout, workspace, tiling);

    FreeTensorList(lse, lseShapeInfos);
    FreeTensorList(go, goShapeInfos);
    AscendC::GmFree(output);
    if (lseout != nullptr) {
        AscendC::GmFree(lseout);
    }
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

static AttentionUpdateTestParam cases[] = {
    {
        "test_case_float32",
        sizeof(float),
        0,
        {4, 3, 8, 32, 512, 0, 2, 128},
    },
    {
        "test_case_float32_updateType1_case0",
        sizeof(float),
        0,
        {4, 3, 8, 32, 512, 1, 2, 128},
    },
};

INSTANTIATE_TEST_CASE_P(AttentionUpdate, AttentionUpdateTest, testing::ValuesIn(cases));
