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
 * \file mixed_quant_flash_attn_metadata_aicpu.h
 * \brief
 */

#ifndef MIXED_QUANT_FLASH_ATTN_METADATA_AICPU_H
#define MIXED_QUANT_FLASH_ATTN_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>
#include <limits>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "mixed_quant_flash_attn_metadata.h"
#include "../../common/op_kernel/load_balance/section_stream_k/section_stream_k.h"
#include "../../common/op_kernel/aicpu_common.h"
#include "../../mixed_quant_flash_attn/op_host/mqfa_fa_adjust_sinner_souter.h"

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

class MixedQuantFlashAttnMetadataCpuKernel : public CpuKernel {
public:
    MixedQuantFlashAttnMetadataCpuKernel() = default;
    ~MixedQuantFlashAttnMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    bool Prepare(CpuKernelContext& ctx);
    bool BalanceSchedule(SectionStreamKResult& splitRes);
    bool GenMetaData(SectionStreamKResult& splitRes);
    bool ParamsInit();
    inline static int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
    std::vector<int64_t> GetTensorDataAsInt64(Tensor* tensor, size_t size);

private:
    CpuKernelContext* context_ = nullptr;
    // input tensor
    Tensor* cuSeqlensQ_ = nullptr;
    Tensor* sequsedQ_ = nullptr;
    Tensor* sequsedKv_ = nullptr;
    // output tensor
    Tensor* metaData_ = nullptr;

    // input attr
    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = -1;
    int32_t maxSeqlenKv_ = -1;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsKv_ = 0;
    int32_t headDim_ = 0;
    int32_t quantMode_ = 1;
    int32_t maskMode_ = 1;
    int32_t winLeft_ = -1;
    int32_t winRight_ = -1;
    std::string layoutQ_ = "BSND";
    std::string layoutKv_ = "PA_BBND";
    std::string layoutAttnOut_ = "BSND";
    std::string socVersion_ = "";
    int32_t aicCoreNum_ = 36U;
    int32_t aivCoreNum_ = 72U;

    // SplitParams
    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = NUM_64;
    uint32_t s2BaseSize_ = NUM_128;
    load_balance::DeviceInfo deviceInfo;
    load_balance::BaseInfo baseInfo;
    load_balance::SectionStreamKParam param;

private:
    enum class ParamId : uint32_t {
        // input
        cuSeqlensQ = 0,
        sequsedQ = 1,
        sequsedKv = 2,
        // output
        metaData = 0,
    };
};
} // namespace aicpu

#endif
