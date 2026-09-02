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
 * \file quant_flash_attn_tiling_data.h
 * \brief QuantFlashAttn tiling data structures
 */

#ifndef QUANT_FLASH_ATTN_TILING_DATA_H_
#define QUANT_FLASH_ATTN_TILING_DATA_H_

namespace optiling {
constexpr uint32_t QFA_AIC_CORE_NUM = 36;
constexpr uint32_t QFA_AIV_CORE_NUM = 72;

constexpr uint32_t QFA_METADATA_SIZE = 16;
constexpr uint32_t QFA_FD_METADATA_SIZE = 16;

constexpr uint32_t QFA_BN2_START_INDEX = 0;
constexpr uint32_t QFA_M_START_INDEX = 1;
constexpr uint32_t QFA_S2_START_INDEX = 2;
constexpr uint32_t QFA_BN2_END_INDEX = 3;
constexpr uint32_t QFA_M_END_INDEX = 4;
constexpr uint32_t QFA_S2_END_INDEX = 5;
constexpr uint32_t QFA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6;

constexpr uint32_t QFA_FD_BN2_IDX_INDEX = 0;
constexpr uint32_t QFA_FD_M_IDX_INDEX = 1;
constexpr uint32_t QFA_FD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t QFA_FD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t QFA_FD_M_START_INDEX = 4;
constexpr uint32_t QFA_FD_M_NUM_INDEX = 5;

struct StridesParams {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;

    void set_bnStride(uint64_t bnStride)
    {
        this->bnStride = bnStride;
    }
    uint64_t get_bnStride() const
    {
        return bnStride;
    }
    void set_n2Stride(uint64_t n2Stride)
    {
        this->n2Stride = n2Stride;
    }
    uint64_t get_n2Stride() const
    {
        return n2Stride;
    }
};

struct QuantFlashAttnBaseParams {
    uint32_t bSize;
    uint32_t t1Size;
    uint32_t t2Size;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t dSize;
    uint32_t dSizeV;
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
    // strides参数
    StridesParams keyStrides;
    StridesParams valueStrides;
    StridesParams kDescaleStrides;
    StridesParams vDescaleStrides;
};

struct QuantFlashAttnAttenMaskParams {
    uint8_t sparseMode;
    int32_t winLefts;
    int32_t winRights;
    uint32_t attenMaskBatch = 0;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
};

struct QuantFlashAttnPageAttentionParams {
    uint8_t paLayoutType;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
};

struct QuantFlashAttnWorkspaceParams {
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};

class QuantFlashAttnQuantTilingArch35 {
public:
    QuantFlashAttnBaseParams quantFlashAttnBaseParams;
    QuantFlashAttnAttenMaskParams quantFlashAttnAttenMaskParams;
    QuantFlashAttnPageAttentionParams quantFlashAttnPageAttentionParams;
    QuantFlashAttnWorkspaceParams quantFlashAttnWorkspaceParams;
};

class QuantFlashAttnTilingData {
public:
    QuantFlashAttnQuantTilingArch35 baseTiling;
};

} // namespace optiling
#endif
