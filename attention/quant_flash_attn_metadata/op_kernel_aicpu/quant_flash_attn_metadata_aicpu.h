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
 * \file quant_flash_attn_metadata_aicpu.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_METADATA_AICPU_H
#define QUANT_FLASH_ATTN_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>
#include <limits>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "quant_flash_attn_metadata.h"
#include "../../common/op_kernel/load_balance/section_stream_k/section_stream_k.h"
#include "../../common/op_kernel/aicpu_common.h"

using namespace optiling;
using namespace std;
using namespace load_balance;

namespace aicpu {

static const int64_t NUM_8192 = 8192L;
static const int64_t NUM_4096 = 4096L;
static const int64_t NUM_2048 = 2048L;
static const int64_t NUM_1024 = 1024L;
static const int64_t NUM_512 = 512L;
static const int64_t NUM_256 = 256L;
static const int64_t NUM_128 = 128L;
static const int64_t NUM_64 = 64L;
static const int64_t NUM_32 = 32L;

class QuantFlashAttnMetadataCpuKernel : public CpuKernel {
public:
    QuantFlashAttnMetadataCpuKernel() = default;
    ~QuantFlashAttnMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool BalanceSchedule(SectionStreamKResult &splitRes);
    bool GenMetaData(SectionStreamKResult &splitRes);
    bool ParamsInit();
    bool CheckNeedInitOutput();
    std::vector<int64_t> GetTensorDataAsInt64(Tensor *tensor, size_t size);
    uint32_t GetS1SeqSize(uint32_t bIdx);
    uint32_t GetS2SeqSize(uint32_t bIdx);
    int64_t CalDeterMaxRound();

private:
    CpuKernelContext *context_ = nullptr;
    Tensor *cuSeqlensQ_ = nullptr;
    Tensor *cuSeqlensKv_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedKv_ = nullptr;
    Tensor *metaData_ = nullptr;

    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = -1;
    int32_t maxSeqlenKv_ = -1;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsKv_ = 0;
    int32_t headDim_ = 0;
    int32_t headDimV_ = 0;
    int32_t quantMode_ = 1;
    int32_t maskMode_ = 1;
    int32_t winLeft_ = -1;
    int32_t winRight_ = -1;
    std::string layoutQ_ = "BSND";
    std::string layoutQDescale_ = "BSND";
    std::string layoutKv_ = "BSND";
    std::string layoutOut_ = "BSND";
    std::string socVersion_ = "";
    int32_t aicCoreNum_ = 36U;
    int32_t aivCoreNum_ = 72U;
    uint32_t s1Size_ = 0;
    uint32_t s2Size_ = 0;
    bool isGradEnabled_ = false;

    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = NUM_64;
    uint32_t s2BaseSize_ = NUM_128;
    bool needInitOutput_ = false;
    load_balance::DeviceInfo deviceInfo;
    load_balance::BaseInfo baseInfo;
    load_balance::SectionStreamKParam param;

private:
    enum class ParamId : uint32_t {
        cuSeqlensQ = 0,
        cuSeqlensKv = 1,
        sequsedQ = 2,
        sequsedKv = 3,
        metaData = 0,
    };
};
} // namespace aicpu

#endif
