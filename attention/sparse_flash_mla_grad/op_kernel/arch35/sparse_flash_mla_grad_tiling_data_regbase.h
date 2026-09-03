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
 * \file sparse_flash_mla_grad_tiling_data_regbase.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_GRAD_TILING_DATA_REGBASE_H_
#define SPARSE_FLASH_MLA_GRAD_TILING_DATA_REGBASE_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
namespace smlag {
constexpr int64_t MAX_CORE_NUM = 36;

class SparseFlashMlaGradBaseParamsRegbase {
public:
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t s3;
    int64_t d;
    int64_t d1;
    int64_t oriselectedBlockCount;
    int64_t cmpselectedBlockCount;
    int64_t usedCoreNum;
    int64_t formerCoreNum;
    int64_t formerCoreProcessNNum;
    int64_t remainCoreProcessNNum;
    int64_t layout;
    int32_t cmpSparseMode;
    int32_t oriSparseMode;
    int64_t oriWinLeft;
    int64_t oriWinRight;
    float scaleValue;
    int64_t totalSize;
    int64_t selectedKWorkSpaceOffset;
    int64_t mm4ResWorkSpaceOffset;
    int64_t mm5ResWorkSpaceOffset;
    int64_t dSinkWorkSpaceOffset;
    int32_t cmpRatio;
    bool hasUsedSeqQ;
    bool hasUsedSeqOriKV;
    bool hasUsedSeqCmpKV;
    bool hasOriTopK;
    bool hasCmpTopK;
    bool isSink;
    bool tmp1;
    bool tmp2;

    int64_t get_b() const
    {
        return b;
    }
    int64_t get_n2() const
    {
        return n2;
    }
    int64_t get_g() const
    {
        return g;
    }
    int64_t get_s1() const
    {
        return s1;
    }
    int64_t get_s2() const
    {
        return s2;
    }
    int64_t get_s3() const
    {
        return s3;
    }
    int64_t get_d() const
    {
        return d;
    }
    int64_t get_d1() const
    {
        return d1;
    }
    int64_t get_oriselectedBlockCount() const
    {
        return oriselectedBlockCount;
    }
    int64_t get_cmpselectedBlockCount() const
    {
        return cmpselectedBlockCount;
    }
    int64_t get_usedCoreNum() const
    {
        return usedCoreNum;
    }
    int64_t get_formerCoreNum() const
    {
        return formerCoreNum;
    }
    int64_t get_formerCoreProcessNNum() const
    {
        return formerCoreProcessNNum;
    }
    int64_t get_remainCoreProcessNNum() const
    {
        return remainCoreProcessNNum;
    }
    float get_scaleValue() const
    {
        return scaleValue;
    }
    int64_t get_totalSize() const
    {
        return totalSize;
    }
    int64_t get_layout() const
    {
        return layout;
    }
    int32_t get_cmpSparseMode() const
    {
        return cmpSparseMode;
    }
    int32_t get_oriSparseMode() const
    {
        return oriSparseMode;
    }
    int32_t get_oriWinLeft() const
    {
        return oriWinLeft;
    }
    int32_t get_oriWinRight() const
    {
        return oriWinRight;
    }
    int64_t get_selectedKWorkSpaceOffset() const
    {
        return selectedKWorkSpaceOffset;
    }
    int64_t get_mm4ResWorkSpaceOffset() const
    {
        return mm4ResWorkSpaceOffset;
    }
    int64_t get_mm5ResWorkSpaceOffset() const
    {
        return mm5ResWorkSpaceOffset;
    }
    int64_t get_dSinkWorkSpaceOffset() const
    {
        return dSinkWorkSpaceOffset;
    }
    int32_t get_cmpRatio() const
    {
        return cmpRatio;
    }
    bool get_hasUsedSeqQ() const
    {
        return hasUsedSeqQ;
    }
    bool get_hasUsedSeqOriKV() const
    {
        return hasUsedSeqOriKV;
    }
    bool get_hasUsedSeqCmpKV() const
    {
        return hasUsedSeqCmpKV;
    }
    bool get_hasOriTopK() const
    {
        return hasOriTopK;
    }
    bool get_hasCmpTopK() const
    {
        return hasCmpTopK;
    }
    bool get_isSink() const
    {
        return isSink;
    }

    void set_b(int64_t bParam)
    {
        this->b = bParam;
    }
    void set_n2(int64_t n2Param)
    {
        this->n2 = n2Param;
    }
    void set_g(int64_t gParam)
    {
        this->g = gParam;
    }
    void set_s1(int64_t s1Param)
    {
        this->s1 = s1Param;
    }
    void set_s2(int64_t s2Param)
    {
        this->s2 = s2Param;
    }
    void set_s3(int64_t s3Param)
    {
        this->s3 = s3Param;
    }
    void set_d(int64_t dParam)
    {
        this->d = dParam;
    }
    void set_d1(int64_t d1Param)
    {
        this->d1 = d1Param;
    }
    void set_oriselectedBlockCount(int64_t oriselectedBlockCountParam)
    {
        this->oriselectedBlockCount = oriselectedBlockCountParam;
    }
    void set_cmpselectedBlockCount(int64_t cmpselectedBlockCountParam)
    {
        this->cmpselectedBlockCount = cmpselectedBlockCountParam;
    }
    void set_usedCoreNum(int64_t usedCoreNumParam)
    {
        this->usedCoreNum = usedCoreNumParam;
    }
    void set_formerCoreNum(int64_t formerCoreNumParam)
    {
        this->formerCoreNum = formerCoreNumParam;
    }
    void set_formerCoreProcessNNum(int64_t formerCoreProcessNNumParam)
    {
        this->formerCoreProcessNNum = formerCoreProcessNNumParam;
    }
    void set_remainCoreProcessNNum(int64_t remainCoreProcessNNumParam)
    {
        this->remainCoreProcessNNum = remainCoreProcessNNumParam;
    }
    void set_scaleValue(float scaleValueParam)
    {
        this->scaleValue = scaleValueParam;
    }
    void set_totalSize(int64_t totalSizeParam)
    {
        this->totalSize = totalSizeParam;
    }
    void set_layout(int64_t layoutParam)
    {
        this->layout = layoutParam;
    }
    void set_cmpSparseMode(int32_t cmpSparseModeParam)
    {
        this->cmpSparseMode = cmpSparseModeParam;
    }
    void set_oriSparseMode(int32_t oriSparseModeParam)
    {
        this->oriSparseMode = oriSparseModeParam;
    }
    void set_oriWinLeft(int32_t oriWinLeftParam)
    {
        this->oriWinLeft = oriWinLeftParam;
    }
    void set_oriWinRight(int32_t oriWinRightParam)
    {
        this->oriWinRight = oriWinRightParam;
    }
    void set_selectedKWorkSpaceOffset(int64_t selectedKWorkSpaceOffsetParam)
    {
        this->selectedKWorkSpaceOffset = selectedKWorkSpaceOffsetParam;
    }
    void set_mm4ResWorkSpaceOffset(int64_t mm4ResWorkSpaceOffsetParam)
    {
        this->mm4ResWorkSpaceOffset = mm4ResWorkSpaceOffsetParam;
    }
    void set_mm5ResWorkSpaceOffset(int64_t mm5ResWorkSpaceOffsetParam)
    {
        this->mm5ResWorkSpaceOffset = mm5ResWorkSpaceOffsetParam;
    }
    void set_dSinkWorkSpaceOffset(int64_t dSinkWorkSpaceOffset)
    {
        this->dSinkWorkSpaceOffset = dSinkWorkSpaceOffset;
    }
    void set_cmpRatio(int32_t cmpRatio)
    {
        this->cmpRatio = cmpRatio;
    }
    void set_hasUsedSeqQ(bool hasUsedSeqQ)
    {
        this->hasUsedSeqQ = hasUsedSeqQ;
    }
    void set_hasUsedSeqOriKV(bool hasUsedSeqOriKV)
    {
        this->hasUsedSeqOriKV = hasUsedSeqOriKV;
    }
    void set_hasUsedSeqCmpKV(bool hasUsedSeqCmpKV)
    {
        this->hasUsedSeqCmpKV = hasUsedSeqCmpKV;
    }
    void set_hasOriTopK(bool hasOriTopK)
    {
        this->hasOriTopK = hasOriTopK;
    }
    void set_hasCmpTopK(bool hasCmpTopK)
    {
        this->hasCmpTopK = hasCmpTopK;
    }
    void set_isSink(bool isSink)
    {
        this->isSink = isSink;
    }
};

class PreParamsRegbase {
public:
    int64_t qPreBlockFactor;
    int64_t qPreBlockTotal;
    int64_t qPreBlockTail;
    int64_t oriKVPreBlockFactor;
    int64_t oriKVPreBlockTotal;
    int64_t oriKVPreBlockTail;
    int64_t cmpKVPreBlockFactor;
    int64_t cmpKVPreBlockTotal;
    int64_t cmpKVPreBlockTail;
    int64_t dSinksPreBlockFactor;
    int64_t dSinksPreBlockTotal;
    int64_t dSinksPreBlockTail;
    int64_t oriSoftmaxL1PreBlockFactor;
    int64_t oriSoftmaxL1PreBlockTotal;
    int64_t oriSoftmaxL1PreBlockTail;
    int64_t cmpSoftmaxL1PreBlockFactor;
    int64_t cmpSoftmaxL1PreBlockTotal;
    int64_t cmpSoftmaxL1PreBlockTail;

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
    int64_t get_oriKVPreBlockFactor() const
    {
        return oriKVPreBlockFactor;
    }
    int64_t get_oriKVPreBlockTotal() const
    {
        return oriKVPreBlockTotal;
    }
    int64_t get_oriKVPreBlockTail() const
    {
        return oriKVPreBlockTail;
    }
    int64_t get_cmpKVPreBlockFactor() const
    {
        return cmpKVPreBlockFactor;
    }
    int64_t get_cmpKVPreBlockTotal() const
    {
        return cmpKVPreBlockTotal;
    }
    int64_t get_cmpKVPreBlockTail() const
    {
        return cmpKVPreBlockTail;
    }
    int64_t get_dSinksPreBlockFactor() const
    {
        return dSinksPreBlockFactor;
    }
    int64_t get_dSinksPreBlockTotal() const
    {
        return dSinksPreBlockTotal;
    }
    int64_t get_dSinksPreBlockTail() const
    {
        return dSinksPreBlockTail;
    }
    int64_t get_oriSoftmaxL1PreBlockFactor() const
    {
        return oriSoftmaxL1PreBlockFactor;
    }
    int64_t get_oriSoftmaxL1PreBlockTotal() const
    {
        return oriSoftmaxL1PreBlockTotal;
    }
    int64_t get_oriSoftmaxL1PreBlockTail() const
    {
        return oriSoftmaxL1PreBlockTail;
    }
    int64_t get_cmpSoftmaxL1PreBlockFactor() const
    {
        return cmpSoftmaxL1PreBlockFactor;
    }
    int64_t get_cmpSoftmaxL1PreBlockTotal() const
    {
        return cmpSoftmaxL1PreBlockTotal;
    }
    int64_t get_cmpSoftmaxL1PreBlockTail() const
    {
        return cmpSoftmaxL1PreBlockTail;
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
    void set_oriKVPreBlockFactor(int64_t value)
    {
        this->oriKVPreBlockFactor = value;
    }
    void set_oriKVPreBlockTotal(int64_t value)
    {
        this->oriKVPreBlockTotal = value;
    }
    void set_oriKVPreBlockTail(int64_t value)
    {
        this->oriKVPreBlockTail = value;
    }
    void set_cmpKVPreBlockFactor(int64_t value)
    {
        this->cmpKVPreBlockFactor = value;
    }
    void set_cmpKVPreBlockTotal(int64_t value)
    {
        this->cmpKVPreBlockTotal = value;
    }
    void set_cmpKVPreBlockTail(int64_t value)
    {
        this->cmpKVPreBlockTail = value;
    }
    void set_dSinksPreBlockFactor(int64_t value)
    {
        this->dSinksPreBlockFactor = value;
    }
    void set_dSinksPreBlockTotal(int64_t value)
    {
        this->dSinksPreBlockTotal = value;
    }
    void set_dSinksPreBlockTail(int64_t value)
    {
        this->dSinksPreBlockTail = value;
    }
    void set_oriSoftmaxL1PreBlockFactor(int64_t value)
    {
        this->oriSoftmaxL1PreBlockFactor = value;
    }
    void set_oriSoftmaxL1PreBlockTotal(int64_t value)
    {
        this->oriSoftmaxL1PreBlockTotal = value;
    }
    void set_oriSoftmaxL1PreBlockTail(int64_t value)
    {
        this->oriSoftmaxL1PreBlockTail = value;
    }
    void set_cmpSoftmaxL1PreBlockFactor(int64_t value)
    {
        this->cmpSoftmaxL1PreBlockFactor = value;
    }
    void set_cmpSoftmaxL1PreBlockTotal(int64_t value)
    {
        this->cmpSoftmaxL1PreBlockTotal = value;
    }
    void set_cmpSoftmaxL1PreBlockTail(int64_t value)
    {
        this->cmpSoftmaxL1PreBlockTail = value;
    }
};

class PostParamsRegbase {
public:
    int64_t postUbBaseSize;
    int64_t qPostBlockFactor;
    int64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    int64_t oriKVPostBlockFactor;
    int64_t oriKVPostBlockTotal;
    int64_t oriKVPostBaseNum;
    int64_t oriKVPostTailNum;
    int64_t cmpKVPostBlockFactor;
    int64_t cmpKVPostBlockTotal;
    int64_t cmpKVPostBaseNum;
    int64_t cmpKVPostTailNum;
    int64_t dqWorkSpaceOffset;
    int64_t dOriKVWorkSpaceOffset;
    int64_t dCmpKVWorkSpaceOffset;
    int64_t dOriVWorkSpaceOffset;
    int64_t dCmpVWorkSpaceOffset;
    int64_t sinkReduceAxis;
    int64_t sinkPostBlockTotal;
    int64_t sinkPostBlockFactor;
    int64_t sinkPostTailNum;

    int64_t get_postUbBaseSize() const
    {
        return postUbBaseSize;
    }
    int64_t get_qPostBlockFactor() const
    {
        return qPostBlockFactor;
    }
    int64_t get_qPostBlockTotal() const
    {
        return qPostBlockTotal;
    }
    int64_t get_qPostBaseNum() const
    {
        return qPostBaseNum;
    }
    int64_t get_qPostTailNum() const
    {
        return qPostTailNum;
    }
    int64_t get_oriKVPostBlockFactor() const
    {
        return oriKVPostBlockFactor;
    }
    int64_t get_oriKVPostBlockTotal() const
    {
        return oriKVPostBlockTotal;
    }
    int64_t get_oriKVPostBaseNum() const
    {
        return oriKVPostBaseNum;
    }
    int64_t get_oriKVPostTailNum() const
    {
        return oriKVPostTailNum;
    }
    int64_t get_cmpKVPostBlockFactor() const
    {
        return cmpKVPostBlockFactor;
    }
    int64_t get_cmpKVPostBlockTotal() const
    {
        return cmpKVPostBlockTotal;
    }
    int64_t get_cmpKVPostBaseNum() const
    {
        return cmpKVPostBaseNum;
    }
    int64_t get_cmpKVPostTailNum() const
    {
        return cmpKVPostTailNum;
    }
    int64_t get_dqWorkSpaceOffset() const
    {
        return dqWorkSpaceOffset;
    }
    int64_t get_dOriKVWorkSpaceOffset() const
    {
        return dOriKVWorkSpaceOffset;
    }
    int64_t get_dCmpKVWorkSpaceOffset() const
    {
        return dCmpKVWorkSpaceOffset;
    }
    int64_t get_dOriVWorkSpaceOffset() const
    {
        return dOriVWorkSpaceOffset;
    }
    int64_t get_dCmpVWorkSpaceOffset() const
    {
        return dCmpVWorkSpaceOffset;
    }
    int64_t get_sinkReduceAxis() const
    {
        return sinkReduceAxis;
    }
    int64_t get_sinkPostBlockTotal() const
    {
        return sinkPostBlockTotal;
    }
    int64_t get_sinkPostBlockFactor() const
    {
        return sinkPostBlockFactor;
    }
    int64_t get_sinkPostTailNum() const
    {
        return sinkPostTailNum;
    }

    void set_postUbBaseSize(int64_t value)
    {
        this->postUbBaseSize = value;
    }
    void set_qPostBlockFactor(int64_t value)
    {
        this->qPostBlockFactor = value;
    }
    void set_qPostBlockTotal(int64_t value)
    {
        this->qPostBlockTotal = value;
    }
    void set_qPostBaseNum(int64_t value)
    {
        this->qPostBaseNum = value;
    }
    void set_qPostTailNum(int64_t value)
    {
        this->qPostTailNum = value;
    }
    void set_oriKVPostBlockFactor(int64_t value)
    {
        this->oriKVPostBlockFactor = value;
    }
    void set_oriKVPostBlockTotal(int64_t value)
    {
        this->oriKVPostBlockTotal = value;
    }
    void set_oriKVPostBaseNum(int64_t value)
    {
        this->oriKVPostBaseNum = value;
    }
    void set_oriKVPostTailNum(int64_t value)
    {
        this->oriKVPostTailNum = value;
    }
    void set_cmpKVPostBlockFactor(int64_t value)
    {
        this->cmpKVPostBlockFactor = value;
    }
    void set_cmpKVPostBlockTotal(int64_t value)
    {
        this->cmpKVPostBlockTotal = value;
    }
    void set_cmpKVPostBaseNum(int64_t value)
    {
        this->cmpKVPostBaseNum = value;
    }
    void set_cmpKVPostTailNum(int64_t value)
    {
        this->cmpKVPostTailNum = value;
    }
    void set_dqWorkSpaceOffset(int64_t value)
    {
        this->dqWorkSpaceOffset = value;
    }
    void set_dOriKVWorkSpaceOffset(int64_t value)
    {
        this->dOriKVWorkSpaceOffset = value;
    }
    void set_dCmpKVWorkSpaceOffset(int64_t value)
    {
        this->dCmpKVWorkSpaceOffset = value;
    }
    void set_dOriVWorkSpaceOffset(int64_t value)
    {
        this->dOriVWorkSpaceOffset = value;
    }
    void set_dCmpVWorkSpaceOffset(int64_t value)
    {
        this->dCmpVWorkSpaceOffset = value;
    }
    void set_sinkReduceAxis(int64_t value)
    {
        this->sinkReduceAxis = value;
    }
    void set_sinkPostBlockTotal(int64_t value)
    {
        this->sinkPostBlockTotal = value;
    }
    void set_sinkPostBlockFactor(int64_t value)
    {
        this->sinkPostBlockFactor = value;
    }
    void set_sinkPostTailNum(int64_t value)
    {
        this->sinkPostTailNum = value;
    }
};

class SparseFlashMlaGradTilingDataRegbase {
public:
    SparseFlashMlaGradBaseParamsRegbase baseParams;
    PreParamsRegbase preTilingData;
    PostParamsRegbase postTilingData;
};

} // namespace smlag
} // namespace optiling
#endif
