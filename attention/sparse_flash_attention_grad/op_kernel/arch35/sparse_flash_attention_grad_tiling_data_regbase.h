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
 * \file sparse_flash_attention_grad_tiling_data_regbase.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_GRAD_TILING_DATA_REGBASE_H_
#define SPARSE_FLASH_ATTENTION_GRAD_TILING_DATA_REGBASE_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
namespace sfag {
constexpr int64_t MAX_CORE_NUM = 36;

constexpr int64_t MAX_N1 = 128;

class SparseFlashAttentionGradBaseParamsRegbase {
public:
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t d1;
    int64_t totalSize;
    int64_t selectedBlockCount;
    int64_t usedCoreNum;
    int64_t formerCoreNum;
    int64_t formerCoreProcessNNum;
    int64_t remainCoreProcessNNum;
    int64_t layout;
    int32_t sparseMode;
    float scaleValue;
    int64_t selectedKWorkSpaceOffset;
    int64_t mm4ResWorkSpaceOffset;
    int64_t mm5ResWorkSpaceOffset;
    int64_t isHeadNLe64;
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
    int64_t get_d() const
    {
        return d;
    }
    int64_t get_d1() const
    {
        return d1;
    }
    int64_t get_totalSize() const
    {
        return totalSize;
    }
    int64_t get_selectedBlockCount() const
    {
        return selectedBlockCount;
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
    int64_t get_layout() const
    {
        return layout;
    }
    int32_t get_sparseMode() const
    {
        return sparseMode;
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
    int64_t get_isHeadNLe64() const
    {
        return isHeadNLe64;
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
    void set_d(int64_t dParam)
    {
        this->d = dParam;
    }
    void set_d1(int64_t d1Param)
    {
        this->d1 = d1Param;
    }
    void set_totalSize(int64_t totalSizeParam)
    {
        this->totalSize = totalSizeParam;
    }
    void set_selectedBlockCount(int64_t selectedBlockCountParam)
    {
        this->selectedBlockCount = selectedBlockCountParam;
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
    void set_layout(int64_t layoutParam)
    {
        this->layout = layoutParam;
    }
    void set_sparseMode(int32_t sparseModeParam)
    {
        this->sparseMode = sparseModeParam;
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
    void set_isHeadNLe64(int64_t isHeadNLe64Param)
    {
        this->isHeadNLe64 = isHeadNLe64Param;
    }
};

class PreParamsRegbase {
public:
    int64_t qPreBlockFactor;
    int64_t qPreBlockTotal;
    int64_t qPreBlockTail;
    int64_t kPreBlockFactor;
    int64_t kPreBlockTotal;
    int64_t kPreBlockTail;
    int64_t vPreBlockFactor;
    int64_t vPreBlockTotal;
    int64_t vPreBlockTail;

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
    int64_t get_vPreBlockFactor() const
    {
        return vPreBlockFactor;
    }
    int64_t get_vPreBlockTotal() const
    {
        return vPreBlockTotal;
    }
    int64_t get_vPreBlockTail() const
    {
        return vPreBlockTail;
    }

    void set_qPreBlockFactor(int64_t val)
    {
        this->qPreBlockFactor = val;
    }
    void set_qPreBlockTotal(int64_t val)
    {
        this->qPreBlockTotal = val;
    }
    void set_qPreBlockTail(int64_t val)
    {
        this->qPreBlockTail = val;
    }
    void set_kPreBlockFactor(int64_t val)
    {
        this->kPreBlockFactor = val;
    }
    void set_kPreBlockTotal(int64_t val)
    {
        this->kPreBlockTotal = val;
    }
    void set_kPreBlockTail(int64_t val)
    {
        this->kPreBlockTail = val;
    }
    void set_vPreBlockFactor(int64_t val)
    {
        this->vPreBlockFactor = val;
    }
    void set_vPreBlockTotal(int64_t val)
    {
        this->vPreBlockTotal = val;
    }
    void set_vPreBlockTail(int64_t val)
    {
        this->vPreBlockTail = val;
    }
};

class PostParamsRegbase {
public:
    int64_t postUbBaseSize;
    int64_t qPostBlockFactor;
    int64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    int64_t kPostBlockFactor;
    int64_t kPostBlockTotal;
    int64_t kPostBaseNum;
    int64_t kPostTailNum;
    int64_t vPostBlockFactor;
    int64_t vPostBlockTotal;
    int64_t vPostBaseNum;
    int64_t vPostTailNum;
    int64_t dqWorkSpaceOffset;
    int64_t dkWorkSpaceOffset;
    int64_t dvWorkSpaceOffset;
    int64_t dSinkWorkSpaceOffset;

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
    int64_t get_kPostBlockFactor() const
    {
        return kPostBlockFactor;
    }
    int64_t get_kPostBlockTotal() const
    {
        return kPostBlockTotal;
    }
    int64_t get_kPostBaseNum() const
    {
        return kPostBaseNum;
    }
    int64_t get_kPostTailNum() const
    {
        return kPostTailNum;
    }
    int64_t get_vPostBlockFactor() const
    {
        return vPostBlockFactor;
    }
    int64_t get_vPostBlockTotal() const
    {
        return vPostBlockTotal;
    }
    int64_t get_vPostBaseNum() const
    {
        return vPostBaseNum;
    }
    int64_t get_vPostTailNum() const
    {
        return vPostTailNum;
    }
    int64_t get_dqWorkSpaceOffset() const
    {
        return dqWorkSpaceOffset;
    }
    int64_t get_dkWorkSpaceOffset() const
    {
        return dkWorkSpaceOffset;
    }
    int64_t get_dvWorkSpaceOffset() const
    {
        return dvWorkSpaceOffset;
    }
    int64_t get_dSinkWorkSpaceOffset() const
    {
        return dSinkWorkSpaceOffset;
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
    void set_kPostBlockFactor(int64_t value)
    {
        this->kPostBlockFactor = value;
    }
    void set_kPostBlockTotal(int64_t value)
    {
        this->kPostBlockTotal = value;
    }
    void set_kPostBaseNum(int64_t value)
    {
        this->kPostBaseNum = value;
    }
    void set_kPostTailNum(int64_t value)
    {
        this->kPostTailNum = value;
    }
    void set_vPostBlockFactor(int64_t value)
    {
        this->vPostBlockFactor = value;
    }
    void set_vPostBlockTotal(int64_t value)
    {
        this->vPostBlockTotal = value;
    }
    void set_vPostBaseNum(int64_t value)
    {
        this->vPostBaseNum = value;
    }
    void set_vPostTailNum(int64_t value)
    {
        this->vPostTailNum = value;
    }
    void set_dqWorkSpaceOffset(int64_t value)
    {
        this->dqWorkSpaceOffset = value;
    }
    void set_dkWorkSpaceOffset(int64_t value)
    {
        this->dkWorkSpaceOffset = value;
    }
    void set_dvWorkSpaceOffset(int64_t value)
    {
        this->dvWorkSpaceOffset = value;
    }
    void set_dSinkWorkSpaceOffset(int64_t value)
    {
        this->dSinkWorkSpaceOffset = value;
    }
};

class SparseFlashAttentionGradTilingDataRegbase {
public:
    SparseFlashAttentionGradBaseParamsRegbase baseParams;
    PreParamsRegbase preTilingData;
    PostParamsRegbase postTilingData;
};

} // namespace sfag
} // namespace optiling
#endif
