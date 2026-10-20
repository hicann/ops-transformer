/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_METADATA_9102_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_METADATA_9102_H_

#include "../common/quant_block_sparse_attn_metadata.h"

namespace regbasemx {
// [STC060_METADATA_LAYOUT] Keep these ABI constants in sync with the AICPU
// producer and CPU generator. No tiling-data or public operator ABI changes.
constexpr uint32_t MX_METADATA_AIC_CORE_NUM_INDEX = 1U;
constexpr uint32_t MX_METADATA_AIV_CORE_NUM_INDEX = 2U;
constexpr uint32_t MX_METADATA_LAYOUT_TAG_INDEX = 3U;
constexpr uint32_t MX_METADATA_LAYOUT_V1 = 0x51425301U;
constexpr uint32_t MX_METADATA_LEGACY_AIC_CORE_NUM = 64U;

__aicore__ inline uint32_t GetMxMetadataAicCoreNum(AscendC::GlobalTensor<int32_t>& metadataGm)
{
    const uint32_t tag = static_cast<uint32_t>(metadataGm.GetValue(MX_METADATA_LAYOUT_TAG_INDEX));
    if (tag == 0U) {
        // Only the historical 64-slot CPU ABI is supported without a tag.
        // Old untagged AICPU output using 54 slots MUST be regenerated.
        return MX_METADATA_LEGACY_AIC_CORE_NUM;
    }
    if (tag != MX_METADATA_LAYOUT_V1) {
        return 0U; // Unsupported header: do not address records with a guessed stride.
    }
    const int32_t layoutAicCoreNum = metadataGm.GetValue(MX_METADATA_AIC_CORE_NUM_INDEX);
    // Valid producers and the Python loader require a positive layout count.
    return layoutAicCoreNum > 0 ? static_cast<uint32_t>(layoutAicCoreNum) : 0U;
}

__aicore__ inline uint64_t GetMxCoreMetadataOffset(uint32_t sectionIdx, uint32_t slotIdx, uint32_t layoutAicCoreNum)
{
    return QBSA_FA_HEAD_METADATA_SIZE +
           (static_cast<uint64_t>(sectionIdx) * layoutAicCoreNum + slotIdx) * QBSA_FA_METADATA_SIZE;
}

__aicore__ inline BsaFaCoreMetadata GetMxCoreMetadata(AscendC::GlobalTensor<int32_t>& metadataGm, uint32_t sectionIdx,
                                                      uint32_t slotIdx, uint32_t layoutAicCoreNum)
{
    const uint64_t baseOffset = GetMxCoreMetadataOffset(sectionIdx, slotIdx, layoutAicCoreNum);
    BsaFaCoreMetadata result;
    result.coreEnable = metadataGm.GetValue(baseOffset + QBSA_FA_CORE_ENABLE_INDEX);
    result.bn1StartIdx = metadataGm.GetValue(baseOffset + QBSA_FA_BN1_START_INDEX);
    result.s1StartIdx = metadataGm.GetValue(baseOffset + QBSA_FA_S1_START_INDEX);
    result.s2StartIdx = metadataGm.GetValue(baseOffset + QBSA_FA_S2_START_INDEX);
    result.bn1EndIdx = metadataGm.GetValue(baseOffset + QBSA_FA_BN1_END_INDEX);
    result.s1EndIdx = metadataGm.GetValue(baseOffset + QBSA_FA_S1_END_INDEX);
    result.s2EndIdx = metadataGm.GetValue(baseOffset + QBSA_FA_S2_END_INDEX);
    result.firstFdDataWorkspaceIdx = metadataGm.GetValue(baseOffset + QBSA_FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX);
    return result;
}
} // namespace regbasemx

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_METADATA_9102_H_
