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
 * \file fia_tiling_data_public.h
 * \brief
 */

// TODO，文件头，宏定义整体调整与刷新

// TODO，变量命名优化

#ifndef FIA_TILING_DATA_PUBLIC_H_
#define FIA_TILING_DATA_PUBLIC_H_

namespace optiling {

// 数组长度
// TODO，host和device宏定义不一样，如何通过编译宏隔离？
// #if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
constexpr uint32_t NPU_AIC_CORE_NUM = 64;
constexpr uint32_t NPU_AIV_CORE_NUM = 64;
// #else
// constexpr uint32_t NPU_AIC_CORE_NUM = 24;
// constexpr uint32_t NPU_AIV_CORE_NUM = 48;
// #endif

constexpr uint32_t FA_METADATA_SIZE = 8;
constexpr uint32_t FD_METADATA_SIZE = 8;

// 索引数组含义
//  FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;

// FD Metadata Index Definitions
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_S2_SPLIT_NUM_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

struct FiaBaseParams {
    uint32_t bSize;
    uint32_t t1Size;
    uint32_t t2Size;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dSizeRope;
    uint32_t actualSeqLengthsQSize;
    uint32_t actualSeqLengthsKVSize;
    float scaleValue;
    uint8_t isActualSeqLengthsNull;   // TODO，未使用，是否要删除？
    uint8_t isActualSeqLengthsKVNull; // TODO，未使用，是否要删除？
    uint8_t isKvContinuous;
    uint8_t isSoftMaxLseEnable;
    uint32_t coreNum;
    uint32_t outputLayout;
};

struct FiaAttenMaskParams {
    uint8_t sparseMode;
    int32_t preTokens;
    int32_t nextTokens;
    uint32_t attenMaskBatch;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    uint8_t isRowInvalidOpen;
    uint8_t isExistRowInvalid;
};

struct FiaPseParams {
    uint8_t pseShiftByBatch;
    uint32_t pseS1Size;
    uint32_t pseS2Size;
    uint32_t pseStride;
    uint32_t qStartIdx;
    uint32_t kvStartIdx;
};

struct FiaSystemPrefixParams {
    uint8_t isActualSharedPrefixLenNull;
    uint32_t prefixSeqInnerSize;
};

struct FiaPageAttentionParams {
    uint8_t paLayoutType = 0;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
};

struct FiaLeftPaddingParams {
    uint8_t isQHasLeftPadding;
    uint8_t isKVHasLeftPadding;
};

struct FiaPostQuantParams {
    uint8_t isPostQuantPerChnl;
    uint8_t isPostQuantBF16;
};

struct FiaWorkspaceParams {
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};

struct FiaS1OuterSplitCoreParams {
    bool enableS1OutSplit;
    uint64_t totalSize;
};

struct FiaEmptyTensorParams {
    uint32_t singleCoreSize;
    uint8_t needInit;
    uint64_t totalOutputSize;
    uint64_t totalSoftMaxLseOutputSize;
};

} // namespace optiling
#endif
