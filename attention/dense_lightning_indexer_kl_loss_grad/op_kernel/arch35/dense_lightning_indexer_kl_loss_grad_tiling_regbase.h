/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file dense_lightning_indexer_kl_loss_grad_tiling_regbase.h
 * \brief
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_REGBASE_TILING_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_REGBASE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
namespace optiling {
// ------------------公共定义--------------------------
const uint32_t DLIKG_MAX_AIC_CORE_NUM_REGBASE = 36;
constexpr uint32_t MAX_CORE_NUM_REGBASE = 36; // 目前使用AIC核的最大值为36

// -----------算子TilingData定义---------------
class DLIKGradBaseParamsRegbase {
public:
    // 基础输入参数
    int32_t bSize;
    int32_t t1Size;
    int32_t t2Size;
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
    uint32_t cmpRatio;
    uint8_t layoutType; // 0: BSND, 1: TND
    uint32_t maxSeqlenK;
    int64_t mm3ResWorkSpaceOffset;

    int32_t get_bSize() const
    {
        return bSize;
    }
    void set_bSize(int32_t bSizeParam)
    {
        this->bSize = bSizeParam;
    }

    int32_t get_t1Size() const
    {
        return t1Size;
    }
    void set_t1Size(int32_t t1SizeParam)
    {
        this->t1Size = t1SizeParam;
    }

    int32_t get_t2Size() const
    {
        return t2Size;
    }
    void set_t2Size(int32_t t2SizeParam)
    {
        this->t2Size = t2SizeParam;
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

    uint8_t get_layoutType() const
    {
        return layoutType;
    }
    void set_layoutType(uint8_t layoutTypeParam)
    {
        this->layoutType = layoutTypeParam;
    }

    uint32_t get_cmpRatio() const
    {
        return cmpRatio;
    }
    void set_cmpRatio(uint32_t cmpRatioParam)
    {
        this->cmpRatio = cmpRatioParam;
    }

    uint32_t get_maxSeqlenK() const
    {
        return maxSeqlenK;
    }
    void set_maxSeqlenK(uint32_t maxSeqlenKParam)
    {
        this->maxSeqlenK = maxSeqlenKParam;
    }

    int64_t get_mm3ResWorkSpaceOffset() const
    {
        return mm3ResWorkSpaceOffset;
    }
    void set_mm3ResWorkSpaceOffset(uint32_t mm3ResWorkSpaceOffset)
    {
        this->mm3ResWorkSpaceOffset = mm3ResWorkSpaceOffset;
    }
};

class DLIKGradMultiCoreParamsRegbase {
public:
    uint32_t coreNum;
    uint32_t rsvd;
    int64_t splitFactorSize;
    int64_t totalSize;                      // 表明有多少个S1
    int64_t bS1Index[MAX_CORE_NUM_REGBASE]; // 每个核B,S1合轴之后的起始和结束位置

    int64_t postUbBaseSize;
    int64_t qPreBlockFactor;
    int64_t qPreBlockTotal;
    int64_t qPreBlockTail;
    int64_t kPreBlockFactor;
    int64_t kPreBlockTotal;
    int64_t kPreBlockTail;
    int64_t weightPreBlockFactor;
    int64_t weightPreBlockTotal;
    int64_t weightPreBlockTail;
    int64_t softmaxOutPreBlockFactor;
    int64_t softmaxOutPreBlockTotal;
    int64_t softmaxOutPreBlockTail;
    int64_t kPostBlockFactor;
    int64_t kPostBlockTotal;
    int64_t kPostBlockTail;

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

    int64_t get_postUbBaseSize() const
    {
        return postUbBaseSize;
    }
    int64_t get_qPreBlockFactor() const
    {
        return qPreBlockFactor;
    }
    int64_t get_qPreBlockTotal() const
    {
        return qPreBlockTotal;
    }
    int64_t get_qPreBlockTail() const
    {
        return qPreBlockTail;
    }
    int64_t get_kPreBlockFactor() const
    {
        return kPreBlockFactor;
    }
    int64_t get_kPreBlockTotal() const
    {
        return kPreBlockTotal;
    }
    int64_t get_kPreBlockTail() const
    {
        return kPreBlockTail;
    }
    int64_t get_weightPreBlockFactor() const
    {
        return weightPreBlockFactor;
    }
    int64_t get_weightPreBlockTotal() const
    {
        return weightPreBlockTotal;
    }
    int64_t get_weightPreBlockTail() const
    {
        return weightPreBlockTail;
    }
    int64_t get_softmaxOutPreBlockFactor() const
    {
        return softmaxOutPreBlockFactor;
    }
    int64_t get_softmaxOutPreBlockTotal() const
    {
        return softmaxOutPreBlockTotal;
    }
    int64_t get_softmaxOutPreBlockTail() const
    {
        return softmaxOutPreBlockTail;
    }
    int64_t get_kPostBlockFactor() const
    {
        return kPostBlockFactor;
    }
    int64_t get_kPostBlockTotal() const
    {
        return kPostBlockTotal;
    }
    int64_t get_kPostBlockTail() const
    {
        return kPostBlockTail;
    }

    void set_postUbBaseSize(int64_t value)
    {
        this->postUbBaseSize = value;
    }
    void set_qPreBlockFactor(int64_t value)
    {
        this->qPreBlockFactor = value;
    }
    void set_qPreBlockTotal(int64_t value)
    {
        this->qPreBlockTotal = value;
    }
    void set_qPreBlockTail(int64_t value)
    {
        this->qPreBlockTail = value;
    }
    void set_kPreBlockFactor(int64_t value)
    {
        this->kPreBlockFactor = value;
    }
    void set_kPreBlockTotal(int64_t value)
    {
        this->kPreBlockTotal = value;
    }
    void set_kPreBlockTail(int64_t value)
    {
        this->kPreBlockTail = value;
    }
    void set_weightPreBlockFactor(int64_t value)
    {
        this->weightPreBlockFactor = value;
    }
    void set_weightPreBlockTotal(int64_t value)
    {
        this->weightPreBlockTotal = value;
    }
    void set_weightPreBlockTail(int64_t value)
    {
        this->weightPreBlockTail = value;
    }
    void set_softmaxOutPreBlockFactor(int64_t value)
    {
        this->softmaxOutPreBlockFactor = value;
    }
    void set_softmaxOutPreBlockTotal(int64_t value)
    {
        this->softmaxOutPreBlockTotal = value;
    }
    void set_softmaxOutPreBlockTail(int64_t value)
    {
        this->softmaxOutPreBlockTail = value;
    }
    void set_kPostBlockFactor(int64_t value)
    {
        this->kPostBlockFactor = value;
    }
    void set_kPostBlockTotal(int64_t value)
    {
        this->kPostBlockTotal = value;
    }
    void set_kPostBlockTail(int64_t value)
    {
        this->kPostBlockTail = value;
    }
};

class DLIKGradInitOutputParamsRegbase {
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

class DLIKGradVecApiParamsRegbase {
public:
    SoftMaxTiling softmaxYTilingData;
    SoftMaxTiling simpleSoftmaxPTilingData;
};

class DenseLightningIndexerGradRegBaseTilingData {
public:
    DLIKGradBaseParamsRegbase baseParams;
    DLIKGradMultiCoreParamsRegbase multiCoreParams;
    DLIKGradInitOutputParamsRegbase initOutputParams;
    DLIKGradVecApiParamsRegbase vectorParams;
};
} // namespace optiling
#endif // DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_H
