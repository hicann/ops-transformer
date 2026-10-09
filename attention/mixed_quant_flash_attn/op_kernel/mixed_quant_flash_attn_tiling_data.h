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
 * \file mixed_quant_flash_attn_tiling_data.h
 * \brief MixedQuantFlashAttn TilingData定义（伪量化，Q为BF16/FP16，K/V为mxFP4）
 */

#ifndef MIXED_QUANT_FLASH_ATTN_TILING_DATA_H_
#define MIXED_QUANT_FLASH_ATTN_TILING_DATA_H_

namespace optiling {
constexpr uint32_t MIXED_QUANT_FLASH_ATTN_METADATA_SIZE = 16;
constexpr uint32_t MQFA_FD_METADATA_SIZE = 16;
constexpr uint32_t MIXED_QUANT_FLASH_ATTN_METADATA_HEAD_SIZE = 16;
constexpr uint32_t MIXED_QUANT_FLASH_ATTN_IS_FD_INDEX = 1;

constexpr uint32_t MIXED_QUANT_FLASH_ATTN_BN2_START_INDEX = 0;
constexpr uint32_t MIXED_QUANT_FLASH_ATTN_M_START_INDEX = 1;
constexpr uint32_t MIXED_QUANT_FLASH_ATTN_S2_START_INDEX = 2;
constexpr uint32_t MIXED_QUANT_FLASH_ATTN_BN2_END_INDEX = 3;
constexpr uint32_t MIXED_QUANT_FLASH_ATTN_M_END_INDEX = 4;
constexpr uint32_t MIXED_QUANT_FLASH_ATTN_S2_END_INDEX = 5;
constexpr uint32_t MIXED_QUANT_FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6;

constexpr uint32_t MQFA_FD_BN2_IDX_INDEX = 0;
constexpr uint32_t MQFA_FD_M_IDX_INDEX = 1;
constexpr uint32_t MQFA_FD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t MQFA_FD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t MQFA_FD_M_START_INDEX = 4;
constexpr uint32_t MQFA_FD_M_NUM_INDEX = 5;

struct MixedQuantFlashAttnBaseParams {
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
    uint32_t cuSeqLensQSize;
    uint32_t cuSeqLensKVSize;
    uint32_t seqUsedQSize;
    uint32_t seqUsedKvSize;
    float scaleValue;
    uint8_t iscuSeqLengthsNull;
    uint8_t iscuSeqLengthsKVNull;
    uint8_t isKvContinuous;
    uint8_t isSoftMaxLseEnable;
    uint32_t coreNum;
    uint32_t outputLayout;
    bool needInitOutput;
};

struct MixedQuantFlashAttnAttenMaskParams {
    uint8_t sparseMode;
    int32_t winLefts;
    int32_t winRights;
    uint32_t attenMaskBatch = 0;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    uint8_t isExistRowInvalid = 0;
};

struct MixedQuantFlashAttnPageAttentionParams {
    uint8_t paLayoutType;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
};

struct MixedQuantFlashAttnWorkspaceParams {
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};

struct MixedQuantFlashAttnS1OuterSplitCoreParams {
    bool enableS1OutSplit = 0;
    uint64_t totalSize;
};

struct MixedQuantFlashAttnQuantParams {
    uint32_t quantComputeMode;
};

class MixedQuantFlashAttnTiling {
public:
    MixedQuantFlashAttnBaseParams mixedQuantFlashAttnBaseParams;
    MixedQuantFlashAttnAttenMaskParams mixedQuantFlashAttnAttenMaskParams;
    MixedQuantFlashAttnPageAttentionParams mixedQuantFlashAttnPageAttentionParams;
    MixedQuantFlashAttnWorkspaceParams mixedQuantFlashAttnWorkspaceParams;
    MixedQuantFlashAttnS1OuterSplitCoreParams mixedQuantFlashAttnS1OuterSplitCoreParams;
    MixedQuantFlashAttnQuantParams mixedQuantFlashAttnQuantParams;
};

class MixedQuantFlashAttnTilingData {
public:
    MixedQuantFlashAttnTiling baseTiling;
};

} // namespace optiling
#endif // MIXED_QUANT_FLASH_ATTN_TILING_DATA_H_
