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
 * \file quant_block_sparse_attn_metadata.h
 * \brief metadata layout helpers for quant block sparse attention arch35 kernel.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_METADATA_ARCH35_H_
#define QUANT_BLOCK_SPARSE_ATTN_METADATA_ARCH35_H_

#include <cstdint>
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

// metadata获取: head[8] + qbsa[sectionNum][AIC_CORE_NUM][8] + fd[AIV_CORE_NUM][8]
constexpr uint32_t QBSA_FA_HEAD_METADATA_SIZE = 8U;
constexpr uint32_t QBSA_FA_HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t QBSA_FA_AIC_CORE_NUM = 36U;
constexpr uint32_t QBSA_FA_AIV_CORE_NUM = 72U;
constexpr uint32_t QBSA_FA_METADATA_SIZE = 8U;
constexpr uint32_t QBSA_FA_FD_METADATA_SIZE = 8U;

constexpr uint32_t QBSA_FA_CORE_ENABLE_INDEX = 0U;
constexpr uint32_t QBSA_FA_BN1_START_INDEX = 1U;
constexpr uint32_t QBSA_FA_S1_START_INDEX = 2U;
constexpr uint32_t QBSA_FA_S2_START_INDEX = 3U;
constexpr uint32_t QBSA_FA_BN1_END_INDEX = 4U;
constexpr uint32_t QBSA_FA_S1_END_INDEX = 5U;
constexpr uint32_t QBSA_FA_S2_END_INDEX = 6U;
constexpr uint32_t QBSA_FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7U;

struct BsaFaCoreMetadata {
    uint32_t coreEnable{0U};
    uint32_t bn1StartIdx{0U};
    uint32_t s1StartIdx{0U};
    uint32_t s2StartIdx{0U};
    uint32_t bn1EndIdx{0U};
    uint32_t s1EndIdx{0U};
    uint32_t s2EndIdx{0U};
    uint32_t firstFdDataWorkspaceIdx{0U};
};

__aicore__ inline uint32_t GetBsaSectionNum(AscendC::GlobalTensor<int32_t> &metadataGm)
{
    return metadataGm.GetValue(QBSA_FA_HEAD_SECTION_NUM_INDEX);
}

__aicore__ inline uint64_t GetBsaCoreMetadataOffset(uint32_t sectionIdx, uint32_t coreIdx)
{
    return QBSA_FA_HEAD_METADATA_SIZE +
           static_cast<uint64_t>(sectionIdx) * QBSA_FA_AIC_CORE_NUM * QBSA_FA_METADATA_SIZE +
           static_cast<uint64_t>(coreIdx) * QBSA_FA_METADATA_SIZE;
}

__aicore__ inline uint32_t GetBsaAttrMetadata(AscendC::GlobalTensor<int32_t> &metadataGm, uint32_t sectionIdx,
                                              uint32_t coreIdx, uint32_t metaIdx)
{
    return metadataGm.GetValue(GetBsaCoreMetadataOffset(sectionIdx, coreIdx) + metaIdx);
}

__aicore__ inline bool IsBsaCoreEnabled(AscendC::GlobalTensor<int32_t> &metadataGm, uint32_t sectionIdx,
                                        uint32_t coreIdx)
{
    return GetBsaAttrMetadata(metadataGm, sectionIdx, coreIdx, QBSA_FA_CORE_ENABLE_INDEX) != 0U;
}

__aicore__ inline uint32_t GetBsaLastValidSectionIdx(AscendC::GlobalTensor<int32_t> &metadataGm, uint32_t sectionNum,
                                                     uint32_t coreIdx)
{
    uint32_t lastValidSectionIdx = sectionNum;
    for (uint32_t sectionIdx = 0U; sectionIdx < sectionNum; ++sectionIdx) {
        if (IsBsaCoreEnabled(metadataGm, sectionIdx, coreIdx)) {
            lastValidSectionIdx = sectionIdx;
        }
    }
    return lastValidSectionIdx;
}

__aicore__ inline BsaFaCoreMetadata GetBsaCoreMetadata(AscendC::GlobalTensor<int32_t> &metadataGm, uint32_t sectionIdx,
                                                       uint32_t coreIdx)
{
    uint64_t baseOffset = GetBsaCoreMetadataOffset(sectionIdx, coreIdx);
    BsaFaCoreMetadata coreMetadata;
    coreMetadata.coreEnable = metadataGm.GetValue(baseOffset + QBSA_FA_CORE_ENABLE_INDEX);
    coreMetadata.bn1StartIdx = metadataGm.GetValue(baseOffset + QBSA_FA_BN1_START_INDEX);
    coreMetadata.s1StartIdx = metadataGm.GetValue(baseOffset + QBSA_FA_S1_START_INDEX);
    coreMetadata.s2StartIdx = metadataGm.GetValue(baseOffset + QBSA_FA_S2_START_INDEX);
    coreMetadata.bn1EndIdx = metadataGm.GetValue(baseOffset + QBSA_FA_BN1_END_INDEX);
    coreMetadata.s1EndIdx = metadataGm.GetValue(baseOffset + QBSA_FA_S1_END_INDEX);
    coreMetadata.s2EndIdx = metadataGm.GetValue(baseOffset + QBSA_FA_S2_END_INDEX);
    coreMetadata.firstFdDataWorkspaceIdx = metadataGm.GetValue(baseOffset + QBSA_FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX);
    return coreMetadata;
}

#endif // QUANT_BLOCK_SPARSE_ATTN_METADATA_ARCH35_H_
