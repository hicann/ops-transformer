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
 * \file quant_flash_mla_with_kvcache_tiling_data.h
 * \brief QuantFlashMlaWithKvcache tiling data 定义（arch35）
 */

#ifndef QUANT_FLASH_MLA_WITH_KVCACHE_TILING_DATA_H_
#define QUANT_FLASH_MLA_WITH_KVCACHE_TILING_DATA_H_

namespace optiling {

// 数组长度
constexpr uint32_t QMLA_AIC_CORE_NUM = 36;
constexpr uint32_t QMLA_AIV_CORE_NUM = 72;

// metadata内存布局: header(16个uint32) + FA区(sectionNum*36*16) + FD区(sectionNum*72*16)
// 与 quant_flash_mla_with_kvcache_metadata 算子写入布局一致
constexpr uint32_t QMLA_FA_METADATA_SIZE = 16;
constexpr uint32_t QMLA_FD_METADATA_SIZE = 16;
constexpr uint32_t QMLA_METADATA_HEADER_SIZE = 16; // uint32个数

// Head Metadata Index Definitions
constexpr uint32_t QMLA_HEAD_SECTION_NUM_INDEX = 0;
constexpr uint32_t QMLA_HEAD_IS_FD_INDEX = 1;
constexpr uint32_t QMLA_HEAD_M_BASE_SIZE_INDEX = 2;
constexpr uint32_t QMLA_HEAD_S2_BASE_SIZE_INDEX = 3;
constexpr uint32_t QMLA_HEAD_NEED_INIT_OUTPUT_INDEX = 15;

// FA Metadata Index Definitions（多section: sectionIdx*36*16 + aicIdx*16 + idx）
constexpr uint32_t QMLA_FA_BN_START_INDEX = 0;
constexpr uint32_t QMLA_FA_M_START_INDEX = 1;
constexpr uint32_t QMLA_FA_S2_START_INDEX = 2;
constexpr uint32_t QMLA_FA_BN_END_INDEX = 3;
constexpr uint32_t QMLA_FA_M_END_INDEX = 4;
constexpr uint32_t QMLA_FA_S2_END_INDEX = 5;
constexpr uint32_t QMLA_FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6;

// FD Metadata Index Definitions（多section: sectionIdx*72*16 + aivIdx*16 + idx）
constexpr uint32_t QMLA_FD_BN_IDX_INDEX = 0;
constexpr uint32_t QMLA_FD_M_IDX_INDEX = 1;
constexpr uint32_t QMLA_FD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t QMLA_FD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t QMLA_FD_M_START_INDEX = 4;
constexpr uint32_t QMLA_FD_M_NUM_INDEX = 5;

// MLA 输出layout
enum QmlaOutLayout : uint32_t {
    QMLA_LAYOUT_BSND = 0,
    QMLA_LAYOUT_BNSD = 1,
    QMLA_LAYOUT_TND = 2,
    QMLA_LAYOUT_NTD = 3,
};

// KV cache 非连续stride: 0表示未提供, kernel侧按shape推连续stride
struct QuantFlashMlaWithKvcacheStridesParams {
    uint64_t bnStride = 0; // PA块间(batch维)步长
    uint64_t n2Stride = 0; // head间步长
};

struct QuantFlashMlaWithKvcacheBaseParams {
    uint32_t bSize = 0;
    uint32_t t1Size = 0; // Q_T, layout_q为TND时有效
    uint32_t t2Size = 0; // KV total
    uint32_t n1Size = 0; // Q_N
    uint32_t n2Size = 0; // KV_N, MLA固定为1
    uint32_t gSize = 0;
    uint32_t s1Size = 0;         // max Q seq len
    uint32_t s2Size = 0;         // max KV seq len
    uint32_t dSize = 0;          // head_dim_qk = 576
    uint32_t dSizeV = 0;         // head_dim_v = 512
    uint32_t cuSeqLensQSize = 0; // cu_seqlens_q 元素个数, 0表示未传入
    uint32_t seqUsedQSize = 0;   // seqused_q 元素个数, 0表示未传入
    float scaleValue = 1.0f;
    uint8_t isSoftMaxLseEnable = 0;
    uint8_t isMetadataEnable = 0;
    uint8_t l2CacheOffFlag = 0;
    uint32_t coreNum = 0;
    uint32_t outputLayout = QMLA_LAYOUT_BSND;
    QuantFlashMlaWithKvcacheStridesParams keyStrides; // k_cache非连续stride, V=K_nope复用
};

struct QuantFlashMlaWithKvcacheAttenMaskParams {
    uint8_t maskMode = 0; // 0: NO_MASK 3: CAUSAL
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
};

struct QuantFlashMlaWithKvcachePageAttentionParams {
    uint8_t paLayoutType = 0; // 0: PA_BBND 1: PA_BNBD 2: PA_NZ
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
};

struct QuantFlashMlaWithKvcacheWorkspaceParams {
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

struct QuantFlashMlaWithKvcacheEmptyTensorParams {
    uint32_t singleCoreSize = 0;
    uint8_t needInit = 0;
    uint64_t totalOutputSize = 0;
    uint64_t totalSoftMaxLseOutputSize = 0;
};

class QuantFlashMlaWithKvcacheTilingData {
public:
    QuantFlashMlaWithKvcacheBaseParams baseParams;
    QuantFlashMlaWithKvcacheAttenMaskParams attenMaskParams;
    QuantFlashMlaWithKvcachePageAttentionParams pageAttentionParams;
    QuantFlashMlaWithKvcacheWorkspaceParams workspaceParams;
    QuantFlashMlaWithKvcacheEmptyTensorParams emptyTensorParams;
};

} // namespace optiling
#endif // QUANT_FLASH_MLA_WITH_KVCACHE_TILING_DATA_H_
