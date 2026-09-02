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
 * \file sparse_lightning_indexer_grad_kl_loss_tiling_regbase.h
 * \brief
 */
#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_REGBASE_TILING_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_REGBASE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
namespace optiling {

// ------------------公共定义--------------------------
const uint32_t SLI_MAX_AIC_CORE_NUM_REGBASE = 36;
constexpr uint32_t MAX_CORE_NUM_REGBASE = 36; // 目前使用AIC核的最大值为36

// -----------算子TilingData定义---------------
class SLIGradKLLossBaseParamsRegbase {
public:
    // 基础输入参数
    int32_t bSize;
    int32_t n2Size;
    int32_t gSizeQuery;
    int32_t gSizeQueryIndex;
    int32_t s1Size;
    int32_t s2Size;
    int32_t dSizeQuery;
    int32_t dSizeQueryIndex;
    int32_t kSize;
    int32_t sparseMode;
    int32_t rsvd;
    float scaleValue;

    int64_t t1Size;
    int64_t t2Size;
    int32_t totalAivNum; // 实际使用的AIV核数（非平台总核数）

    uint8_t layoutType; // 0: BSND, 1: TND

    int32_t get_bSize() const
    {
        return bSize;
    }
    void set_bSize(int32_t bSizeParam)
    {
        this->bSize = bSizeParam;
    }

    int32_t get_n2Size() const
    {
        return n2Size;
    }
    void set_n2Size(int32_t n2SizeParam)
    {
        this->n2Size = n2SizeParam;
    }

    int32_t get_gSizeQuery() const
    {
        return gSizeQuery;
    }
    void set_gSizeQuery(int32_t gSizeQueryParam)
    {
        this->gSizeQuery = gSizeQueryParam;
    }

    int32_t get_gSizeQueryIndex() const
    {
        return gSizeQueryIndex;
    }
    void set_gSizeQueryIndex(int32_t gSizeQueryIndexParam)
    {
        this->gSizeQueryIndex = gSizeQueryIndexParam;
    }

    int32_t get_s1Size() const
    {
        return s1Size;
    }
    void set_s1Size(int32_t s1SizeParam)
    {
        this->s1Size = s1SizeParam;
    }

    int32_t get_s2Size() const
    {
        return s2Size;
    }
    void set_s2Size(int32_t s2SizeParam)
    {
        this->s2Size = s2SizeParam;
    }

    int32_t get_dSizeQuery() const
    {
        return dSizeQuery;
    }
    void set_dSizeQuery(int32_t dSizeQueryParam)
    {
        this->dSizeQuery = dSizeQueryParam;
    }

    int32_t get_dSizeQueryIndex() const
    {
        return dSizeQueryIndex;
    }
    void set_dSizeQueryIndex(int32_t dSizeQueryIndexParam)
    {
        this->dSizeQueryIndex = dSizeQueryIndexParam;
    }

    int32_t get_kSize() const
    {
        return kSize;
    }
    void set_kSize(int32_t kSizeParam)
    {
        this->kSize = kSizeParam;
    }

    int32_t get_sparseMode() const
    {
        return sparseMode;
    }
    void set_sparseMode(int32_t sparseModeParam)
    {
        this->sparseMode = sparseModeParam;
    }

    int32_t get_rsvd() const
    {
        return rsvd;
    }
    void set_rsvd(int32_t rsvdParam)
    {
        this->rsvd = rsvdParam;
    }

    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float scaleValueParam)
    {
        this->scaleValue = scaleValueParam;
    }

    int64_t get_t1Size() const
    {
        return t1Size;
    }
    void set_t1Size(int64_t t1SizeParam)
    {
        this->t1Size = t1SizeParam;
    }

    int64_t get_t2Size() const
    {
        return t2Size;
    }
    void set_t2Size(int64_t t2SizeParam)
    {
        this->t2Size = t2SizeParam;
    }

    int32_t get_totalAivNum() const
    {
        return totalAivNum;
    }
    void set_totalAivNum(int32_t totalAivNumParam)
    {
        this->totalAivNum = totalAivNumParam;
    }

    uint8_t get_layoutType() const
    {
        return layoutType;
    }
    void set_layoutType(uint8_t layoutTypeParam)
    {
        this->layoutType = layoutTypeParam;
    }
};

class SLIGradKLLossMultiCoreParamsRegbase {
public:
    uint32_t coreNum;
    uint32_t rsvd;
    int64_t splitFactorSize;
    int64_t totalSize;                      // 表明有多少个S1
    int64_t bS1Index[MAX_CORE_NUM_REGBASE]; // 每个核B,S1合轴之后的起始和结束位置
    uint32_t syKTotalSize;
    uint32_t pKTotalSize;

    int32_t get_coreNum() const
    {
        return coreNum;
    }
    void set_coreNum(uint32_t coreNumParam)
    {
        this->coreNum = coreNumParam;
    }

    int64_t get_splitFactorSize() const
    {
        return splitFactorSize;
    }
    void set_splitFactorSize(int64_t splitFactorSizeParam)
    {
        this->splitFactorSize = splitFactorSizeParam;
    }

    int64_t get_totalSize() const
    {
        return totalSize;
    }
    void set_totalSize(int64_t totalSizeParam)
    {
        this->totalSize = totalSizeParam;
    }

    int64_t *get_bS1Ptr()
    {
        return bS1Index;
    }

    void set_syKTotalSize(uint32_t syKTotalSizeParam)
    {
        this->syKTotalSize = syKTotalSizeParam;
    }
    void set_pKTotalSize(uint32_t pKTotalSizeParam)
    {
        this->pKTotalSize = pKTotalSizeParam;
    }
};

class SLIGradKLLossInitOutputParamsRegbase {
public:
    uint32_t singleCoreSize; // 单核需要初始化的元素个数
    uint32_t rsvd;
    int64_t totalOutputSize; // 总的需要初始化的元素个数，等于bSize * s2Size * D 或者T2 * D

    uint32_t get_singleCoreSize() const
    {
        return singleCoreSize;
    }
    void set_singleCoreSize(uint32_t singleCoreSizeParam)
    {
        this->singleCoreSize = singleCoreSizeParam;
    }

    int64_t get_totalOutputSize() const
    {
        return totalOutputSize;
    }
    void set_totalOutputSize(int64_t totalOutputSizeParam)
    {
        this->totalOutputSize = totalOutputSizeParam;
    }
};

class SLIGradKLLossVecApiParamsRegbase {
public:
    SoftMaxTiling softmaxYTilingData;
    SoftMaxTiling simpleSoftmaxPTilingData;
};

class SparseLightningIndexerGradKLLossRegBaseTilingData {
public:
    SLIGradKLLossBaseParamsRegbase baseParams;
    SLIGradKLLossMultiCoreParamsRegbase multiCoreParams;
    SLIGradKLLossInitOutputParamsRegbase initOutputParams;
    SLIGradKLLossVecApiParamsRegbase vectorParams;
};

} // namespace optiling
#endif // SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_H
