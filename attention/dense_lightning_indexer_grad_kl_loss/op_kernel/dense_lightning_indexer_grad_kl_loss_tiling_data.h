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
 * \file dense_lightning_indexer_grad_kl_loss_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILLING_DATA_H__
#define __DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILLING_DATA_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t QUERY_INDEX_INPUT_INDEX = 2;
constexpr uint32_t KEY_INDEX_INPUT_INDEX = 3;
constexpr uint32_t WEIGHT_INPUT_INDEX = 4;
constexpr uint32_t SOFTMAX_MAX_INPUT_INDEX = 5;
constexpr uint32_t SOFTMAX_SUM_INPUT_INDEX = 6;
constexpr uint32_t SOFTMAX_MAX_INDEX_INPUT_INDEX = 7;
constexpr uint32_t SOFTMAX_SUM_INDEX_INPUT_INDEX = 8;
constexpr uint32_t QUERY_ROPE_INPUT_INDEX = 9;
constexpr uint32_t KEY_ROPE_INPUT_INDEX = 10;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_QUERY_INPUT_INDEX = 11;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_KEY_INPUT_INDEX = 12;

// Outputs Index
constexpr uint32_t D_QUERY_INDEX_OUTPUT_INDEX = 0;
constexpr uint32_t D_KEY_INDEX_OUTPUT_INDEX = 1;
constexpr uint32_t D_WEIGHTS_OUTPUT_INDEX = 2;
constexpr uint32_t LOSS_OUTPUT_INDEX = 3;

// Attributes Index
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 0;
constexpr uint32_t LAYOUT_ATTR_INDEX = 1;
constexpr uint32_t SPARSE_MODE_ATTR_INDEX = 2;

// Dim Num
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t DIM_NUM_FOUR = 4;
// 常量
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;
const uint32_t SLI_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐
constexpr uint32_t S1_BASE_STEP = 128;
constexpr uint32_t S2_BASE_STEP = 1024;
constexpr uint32_t S1_VEC_SIZE_8 = 8;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t AIC_AIV_RATIO = 2;
constexpr int64_t C0_SIZE = 16;
constexpr int64_t DETER_CORE_INFO_TMP_GM_NUM = 16;
constexpr int64_t DETER_LOSS_TMP_GM_NUM = 8;

// ------------------公共定义--------------------------
constexpr uint32_t MAX_CORE_NUM = 36;

struct InnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

enum class SparseMode : uint32_t {
    RIGHT_DOWN_CAUSAL = 3 // 右下角点划分的下三角部分
};

// -----------算子TilingData定义---------------
class DLIGradKLLossBaseParams {
public:
    // 基础输入参数
    int32_t bSize;
    int32_t n2Size;
    int32_t n2IndexSize;
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
    int64_t maxLoopSize;

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

    int32_t get_n2IndexSize() const
    {
        return n2IndexSize;
    }
    void set_n2IndexSize(int32_t n2IndexSizeParam)
    {
        this->n2IndexSize = n2IndexSizeParam;
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

    uint8_t get_layoutType() const
    {
        return layoutType;
    }
    void set_layoutType(uint8_t layoutTypeParam)
    {
        this->layoutType = layoutTypeParam;
    }

    int64_t get_maxLoopSize() const
    {
        return maxLoopSize;
    }
    void set_maxLoopSize(int32_t maxLoopSizeParam)
    {
        this->maxLoopSize = maxLoopSizeParam;
    }
};

class DLIGradKLLossMultiCoreParams {
public:
    uint32_t coreNum;
    uint32_t rsvd;
    int64_t splitFactorSize;
    int64_t totalSize;              // 表明有多少个S1
    int64_t padTotalSize;           // TND场景下的T有可能大于实际长度
    int64_t bS1Index[MAX_CORE_NUM]; // 每个核B,S1合轴之后的起始和结束位置

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

    int64_t get_padTotalSize() const
    {
        return padTotalSize;
    }
    void set_padTotalSize(int64_t padTotalSizeParam)
    {
        this->padTotalSize = padTotalSizeParam;
    }

    int64_t *get_bS1Ptr()
    {
        return bS1Index;
    }
};

class DLIGradKLLossInitOutputParams {
public:
    uint32_t singleCoreSize; // 单核需要初始化的元素个数
    uint32_t rsvd;
    int64_t totalOutputSize;     // 总的需要初始化的元素个数，等于bSize * s2Size * D 或者T2 * D
    int64_t t1PadSingleCoreSize; // q的t维度pad部份进行清零，每个核的任务数
    int64_t t1PadRemainderSize;  // q的t维度pad部份进行清零，每个核处理的余数
    int64_t t2PadSingleCoreSize; // k的t维度pad部份进行清零，每个核的任务数
    int64_t t2PadRemainderSize;  // k的t维度pad部份进行清零，每个核的任务数

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

    int64_t get_t1PadSingleCoreSize() const
    {
        return t1PadSingleCoreSize;
    }
    void set_t1PadSingleCoreSize(int64_t t1PadSingleCoreSizeParam)
    {
        this->t1PadSingleCoreSize = t1PadSingleCoreSizeParam;
    }

    int64_t get_t1PadRemainderSize() const
    {
        return t1PadRemainderSize;
    }
    void set_t1PadRemainderSize(int64_t t1PadRemainderSizeParam)
    {
        this->t1PadRemainderSize = t1PadRemainderSizeParam;
    }

    int64_t get_t2PadSingleCoreSize() const
    {
        return t2PadSingleCoreSize;
    }
    void set_t2PadSingleCoreSize(int64_t t2PadSingleCoreSizeParam)
    {
        this->t2PadSingleCoreSize = t2PadSingleCoreSizeParam;
    }

    int64_t get_t2PadRemainderSize() const
    {
        return t2PadRemainderSize;
    }
    void set_t2PadRemainderSize(int64_t t2PadRemainderSizeParam)
    {
        this->t2PadRemainderSize = t2PadRemainderSizeParam;
    }
};

class DLIGradKLLossVecApiParams {
public:
    SoftMaxTiling softmaxYTilingData;
    SoftMaxTiling simpleSoftmaxPTilingData;
};

class DenseLightningIndexerGradKLLossTilingData {
public:
    DLIGradKLLossBaseParams baseParams;
    DLIGradKLLossMultiCoreParams multiCoreParams;
    DLIGradKLLossInitOutputParams initOutputParams;
    DLIGradKLLossVecApiParams vectorParams;
};

template <typename T>
inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

} // namespace optiling
#endif
