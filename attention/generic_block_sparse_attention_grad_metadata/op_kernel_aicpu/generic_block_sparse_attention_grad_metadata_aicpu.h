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
 * \file generic_block_sparse_attention_grad_metadata_aicpu.h
 * \brief
 */

#ifndef GENERIC_SPARSE_ATTENTION_GRAD_METADATA_AICPU_H
#define GENERIC_SPARSE_ATTENTION_GRAD_METADATA_AICPU_H

#include <cstdint>
#include <string>
#include <vector>
#include "log.h"
#include "status.h"
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "../op_kernel/generic_block_sparse_attention_grad_metadata.h"
#include "../../common/op_kernel/aicpu_common.h"

namespace aicpu {

struct GsagTask {
    uint32_t b{0U};
    uint32_t n2{0U};
    uint32_t j{0U};
    uint32_t g{0U};
    int32_t count{0};
};

// One KV block (b, n2, j): G tasks share the same K/V tile.
struct GsagKvBlockGroup {
    uint32_t b{0U};
    uint32_t n2{0U};
    uint32_t j{0U};
    int32_t count{0};
    uint32_t kvBlockLen{0U};
    uint64_t blockCost{0U};
};

template <typename T>
inline T CeilDiv(T num, T rnd)
{
    return ((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd));
}

class GenericBlockSparseAttentionGradMetadataCpuKernel : public CpuKernel {
public:
    GenericBlockSparseAttentionGradMetadataCpuKernel() = default;
    ~GenericBlockSparseAttentionGradMetadataCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool ParamsCheck();
    bool ParamsInit();
    uint32_t GetKvSeqLen(uint32_t bIdx) const;
    uint32_t GetKvBlockLen(uint32_t bIdx, uint32_t jIdx) const;
    uint64_t CalcGroupBlockCost(int32_t count, uint32_t kvBlockLen) const;
    bool BuildKvBlockGroups();
    bool BalanceKvBlockGroups();
    bool ExpandTaskList();
    bool GenMetadata();

private:
    Tensor *rsvdBlockIdx_ = nullptr;
    Tensor *rsvdBlockCount_ = nullptr;
    Tensor *cuSeqLengthsQ_ = nullptr;
    Tensor *cuSeqLengthsKv_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedKv_ = nullptr;
    Tensor *metadata_ = nullptr;

    int32_t maxQSeqlen_ = 0;
    int32_t maxKvSeqlen_ = 0;
    int32_t numQHeads_ = 0;
    int32_t numKvHeads_ = 0;
    int32_t headDim_ = 0;
    int32_t blockShapeX_ = 1;
    int32_t blockShapeY_ = 128;
    int32_t isPackedGQA_ = 1;
    int32_t maskType_ = 0;
    int32_t softmaxPrecision_ = 0;
    int64_t winLeft_ = -1;
    int64_t winRight_ = -1;
    std::string layoutQ_ = "TND";
    std::string layoutKv_ = "TND";
    uint32_t aicCoreNum_ = optiling::AIC_CORE_MAX_NUM;
    uint32_t aivCoreNum_ = optiling::AIV_CORE_MAX_NUM;
    std::string socVersion_ = "";

    uint32_t batchSize_ = 0U;
    uint32_t n2Size_ = 0U;
    uint32_t jSize_ = 0U;
    uint32_t maxS1_ = 0U;
    uint32_t groupSize_ = 0U;
    uint32_t baseM_ = optiling::GSAG_DEFAULT_BASE_M;
    uint32_t baseN_ = optiling::GSAG_DEFAULT_BASE_N;
    uint32_t totalNum_ = 0U;
    uint32_t usedCoreNum_ = 0U;
    uint32_t metadataCapacity_ = 0U; // actual metadata tensor length (int64 elems)
    uint64_t totalBlockCost_ = 0U;
    uint64_t maxCoreBlockCost_ = 0U;
    int32_t maxTaskCount_ = 0;
    std::vector<GsagKvBlockGroup> kvBlockGroups_;
    std::vector<uint32_t> coreGroupStart_;
    std::vector<uint32_t> coreGroupEnd_;
    std::vector<GsagTask> taskList_;

private:
    enum class ParamId : uint32_t {
        rsvdBlockIdx = 0,
        rsvdBlockCount = 1,
        cuSeqLengthsQ = 2,
        cuSeqLengthsKv = 3,
        sequsedQ = 4,
        sequsedKv = 5,
        metadata = 0,
    };
};

} // namespace aicpu

#endif
