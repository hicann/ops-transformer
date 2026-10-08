/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_with_kvcache_metadata_aicpu.h
 * \brief QuantFlashMlaWithKvcacheMetadata AICPU算子: 负责负载均衡分核, 生成metadata
 */

#ifndef QUANT_FLASH_MLA_WITH_KVCACHE_METADATA_AICPU_H
#define QUANT_FLASH_MLA_WITH_KVCACHE_METADATA_AICPU_H

#include <string>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "quant_flash_mla_with_kvcache_metadata.h"
#include "../../common/op_kernel/load_balance/section_stream_k/section_stream_k.h"
#include "../../common/op_kernel/aicpu_common.h"

using namespace optiling;
using namespace std;
using namespace load_balance;

namespace aicpu {

// MLA固定切分: SOuter=32, SInner=128 (与FIA MLA tiling保持一致)
static const int64_t NUM_128 = 128L;
static const int64_t NUM_32 = 32L;

class QuantFlashMlaWithKvcacheMetadataCpuKernel : public CpuKernel {
public:
    QuantFlashMlaWithKvcacheMetadataCpuKernel() = default;
    ~QuantFlashMlaWithKvcacheMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    bool Prepare(CpuKernelContext& ctx);
    bool BalanceSchedule(SectionStreamKResult& splitRes);
    bool GenMetaData(SectionStreamKResult& splitRes);
    bool ParamsInit();
    bool CheckNeedInitOutput();
    std::vector<int64_t> GetTensorDataAsInt64(Tensor* tensor, size_t size);

private:
    Tensor* cacheSeqlens_ = nullptr;
    Tensor* cuSeqlensQ_ = nullptr;
    Tensor* sequsedQ_ = nullptr;
    Tensor* metaData_ = nullptr;

    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = -1;
    int32_t maxSeqlenKv_ = -1;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsKv_ = 1;
    int32_t headDimQk_ = 576;
    int32_t headDimV_ = 512;
    int32_t quantMode_ = 1;
    int32_t maskMode_ = 0;
    std::string layoutQ_ = "BSND";
    std::string socVersion_ = "";
    int32_t aicCoreNum_ = 36;
    int32_t aivCoreNum_ = 72;

    uint32_t mBaseSize_ = 0;
    uint32_t s2BaseSize_ = 0;
    bool needInitOutput_ = false;
    load_balance::DeviceInfo deviceInfo;
    load_balance::BaseInfo baseInfo;
    load_balance::SectionStreamKParam param;

private:
    enum class ParamId : uint32_t {
        cacheSeqlens = 0,
        cuSeqlensQ = 1,
        sequsedQ = 2,
        metaData = 0,
    };
};
} // namespace aicpu

#endif
