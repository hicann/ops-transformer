/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <numeric>
#include "cpu_kernel_utils.h"
#define private public
#include "../../../op_kernel_aicpu/quant_flash_attn_metadata_aicpu.h"
#undef private
#define QFA_WORKLOAD_INLINE inline
#include "../../../../quant_flash_attn/op_kernel/arch22/quant_flash_attn_workload_int8.h"
#undef QFA_WORKLOAD_INLINE

namespace aicpu {
using namespace BaseApi::Int8GqaTnd;
namespace {
std::shared_ptr<Tensor> IntTensor(std::vector<int32_t>& values)
{
    auto tensor = CpuKernelUtils::CreateTensor();
    auto shape = CpuKernelUtils::CreateTensorShape();
    shape->SetDimSizes({static_cast<int64_t>(values.size())});
    tensor->SetTensorShape(shape.get());
    tensor->SetDataType(DT_INT32);
    tensor->SetData(values.data());
    tensor->SetDataSize(values.size() * sizeof(int32_t));
    return tensor;
}
class QuantFlashAttnInt8MetadataCpu : public testing::Test {
protected:
    QuantFlashAttnMetadataCpuKernel kernel;
    std::vector<int32_t> cq{0, 32, 20480}, ck{0, 32, 20480}, data;
    std::shared_ptr<Tensor> q, k, output;
    void SetUp() override
    {
        kernel.batchSize_ = 2;
        kernel.numHeadsQ_ = 24;
        kernel.numHeadsKv_ = 4;
        kernel.headDim_ = kernel.headDimV_ = 128;
        kernel.layoutQ_ = kernel.layoutKv_ = kernel.layoutOut_ = "TND";
        kernel.maskMode_ = 3;
        kernel.quantMode_ = 12;
        kernel.aicCoreNum_ = 20;
        data.resize(QfaMetadataSize(2, 128), -1);
        ResetTensors();
    }
    void ResetTensors()
    {
        q = IntTensor(cq);
        k = IntTensor(ck);
        output = IntTensor(data);
        kernel.cuSeqlensQ_ = q.get();
        kernel.cuSeqlensKv_ = k.get();
        kernel.metaData_ = output.get();
        kernel.metadataRowSize_ = data.size();
    }
    void CheckCoverage()
    {
        const uint32_t cores = data[QFA_METADATA_USED_CORES], m = data[QFA_METADATA_BLOCK_M];
        uint32_t tasks = 0;
        for (uint32_t b = 0; b < kernel.batchSize_; ++b)
            tasks += ((cq[b + 1] - cq[b]) * kernel.numHeadsQ_ / kernel.numHeadsKv_ + m - 1) / m * kernel.numHeadsKv_;
        const uint32_t ranges = QfaMetadataRangeOffset(kernel.batchSize_);
        EXPECT_EQ(data[ranges], 0);
        EXPECT_EQ(data[ranges + 2 * (cores - 1) + 1], tasks);
        for (uint32_t core = 0; core < cores; ++core) {
            EXPECT_LE(data[ranges + 2 * core], data[ranges + 2 * core + 1]);
            if (core)
                EXPECT_EQ(data[ranges + 2 * core], data[ranges + 2 * core - 1]);
        }
        if (data[QFA_METADATA_TASK_MODE] == QFA_METADATA_MAPPED) {
            const uint32_t offset = QfaMetadataMapOffset(kernel.batchSize_, 128);
            std::vector<int32_t> ids(data.begin() + offset, data.begin() + offset + tasks);
            std::sort(ids.begin(), ids.end());
            for (uint32_t i = 0; i < tasks; ++i)
                ASSERT_EQ(ids[i], i);
        }
    }
};

TEST_F(QuantFlashAttnInt8MetadataCpu, MixedLengthsAndEmptyMembers)
{
    for (int mask : {0, 3}) {
        kernel.maskMode_ = mask;
        for (int first : {0, 32, 10240}) {
            cq[1] = ck[1] = first;
            ASSERT_TRUE(kernel.GenInt8MetaData());
            EXPECT_EQ(data[0], 0x51464133);
            EXPECT_EQ(data[1], 3);
            EXPECT_EQ(data[QFA_METADATA_HEADER + 2], first);
            CheckCoverage();
        }
    }
}

TEST_F(QuantFlashAttnInt8MetadataCpu, SlidingWindowsKeepLengthValidation)
{
    kernel.winLeft_ = 127;
    kernel.winRight_ = 0;
    ASSERT_TRUE(kernel.GenInt8MetaData());
    EXPECT_EQ(data[QFA_METADATA_CONFIG] & 256, 256);
    CheckCoverage();
    cq[1] = -1;
    EXPECT_FALSE(kernel.GenInt8MetaData());
}

TEST_F(QuantFlashAttnInt8MetadataCpu, RejectsBadCumulativeLengthsAndBounds)
{
    for (const auto& values : {std::vector<int32_t>{1, 32, 20480}, {0, -1, 20480}, {0, 20481, 20480}, {0, 0, 0}}) {
        cq = values;
        ResetTensors();
        EXPECT_FALSE(kernel.GenInt8MetaData());
    }
    cq = {0, 32, 20480};
    ck = {0, 0, 20480};
    ResetTensors();
    EXPECT_FALSE(kernel.GenInt8MetaData());
    ck = cq;
    ResetTensors();
    kernel.maxSeqlenQ_ = 100;
    EXPECT_FALSE(kernel.GenInt8MetaData());
}

TEST_F(QuantFlashAttnInt8MetadataCpu, RejectsInvalidDescriptorsAndOutputCapacity)
{
    q->SetDataType(DT_INT64);
    EXPECT_FALSE(kernel.GenInt8MetaData());
    q->SetDataType(DT_INT32);
    q->SetDataSize(sizeof(int32_t));
    EXPECT_FALSE(kernel.GenInt8MetaData());
    ResetTensors();
    output->SetDataSize(sizeof(int32_t));
    EXPECT_FALSE(kernel.GenInt8MetaData());
    ResetTensors();
    --kernel.metadataRowSize_;
    EXPECT_FALSE(kernel.GenInt8MetaData());
}

TEST_F(QuantFlashAttnInt8MetadataCpu, TaskMapCapacityAndRangeFallback)
{
    kernel.numHeadsQ_ = 64;
    kernel.headDim_ = kernel.headDimV_ = 64;
    for (int tail : {65504, 65536}) {
        cq = ck = {0, 32, 32 + tail};
        ResetTensors();
        ASSERT_TRUE(kernel.GenInt8MetaData());
        EXPECT_EQ(data[QFA_METADATA_TASK_MODE], tail == 65504 ? QFA_METADATA_MAPPED : QFA_METADATA_NORMAL);
        CheckCoverage();
    }
}

TEST_F(QuantFlashAttnInt8MetadataCpu, RejectsUnsupportedAttributes)
{
    kernel.headDim_ = 257;
    EXPECT_FALSE(kernel.GenInt8MetaData());
    kernel.headDim_ = 128;
    kernel.maskMode_ = 1;
    EXPECT_FALSE(kernel.GenInt8MetaData());
    kernel.maskMode_ = 3;
    kernel.winLeft_ = -2;
    EXPECT_FALSE(kernel.GenInt8MetaData());
    kernel.winLeft_ = -1;
    kernel.layoutKv_ = "BNSD";
    EXPECT_FALSE(kernel.GenInt8MetaData());
}
} // namespace
} // namespace aicpu
