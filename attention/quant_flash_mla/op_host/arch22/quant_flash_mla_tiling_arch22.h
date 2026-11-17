/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_MLA_TILING_IMPL_INT8_ARCH22_H_
#define QUANT_FLASH_MLA_TILING_IMPL_INT8_ARCH22_H_

#include <cstdint>
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/tiling_api.h"
#include "../qfmla_tiling_info.h"
#include "../quant_flash_mla_tiling_common.h"
#include "../../../incre_flash_attention/op_host/incre_flash_attention_tiling_mla.h"

namespace optiling {
namespace quant_flash_mla {

constexpr uint64_t QFMLA_ARCH22_KEY_QUANT_MODE_SHIFT = 0;
constexpr uint64_t QFMLA_ARCH22_KEY_IN_OUT_LAYOUT_SHIFT = 4;
constexpr uint64_t QFMLA_ARCH22_KEY_KV_LAYOUT_SHIFT = 8;
constexpr uint64_t QFMLA_ARCH22_KEY_HAS_ATTEN_MASK_SHIFT = 12;
constexpr uint64_t QFMLA_ARCH22_KEY_FIELD_MASK = 0xFULL;

constexpr uint64_t QFMLA_ARCH22_QUANT_MODE_INT8 = 2;
constexpr uint64_t QFMLA_ARCH22_INOUT_LAYOUT_BSH = 0;
constexpr uint64_t QFMLA_ARCH22_INOUT_LAYOUT_TND = 1;
constexpr uint64_t QFMLA_ARCH22_KV_LAYOUT_NZ = 1;

constexpr uint32_t IFA_LAYOUT_BSH_BSND = 0;
constexpr uint32_t IFA_LAYOUT_TND = 3;

class QuantFlashMlaTilingInt8Impl : public FiaTilingBase {
public:
    explicit QuantFlashMlaTilingInt8Impl(gert::TilingContext* context)
        : FiaTilingBase(context)
    {}
    ~QuantFlashMlaTilingInt8Impl() override = default;

    void InitTilingInfo(TilingInfo* tilingInfo) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus SetPlatMemoryInfo();
    void CalcNumBlocks(uint32_t aicNum);
    void FillTilingData();
    void FillBaseParams();
    void FillSplitKVParams();
    void FillSingleCoreParams();
    void CalcScheduleMode();
    void CalcWorkspaceSize();
    void GenTilingKey();
    uint64_t GetMaxActualseqUpperBound() const;
    void PrintAllTilingData();
    ge::graphStatus SetTilingData(IncreFlashAttentionTilingDataMla& tilingData);

    IncreFlashAttentionTilingDataMla tilingData_;
    QfmlaPlatFormInfo platformInfo_;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    uint32_t numBlocks_ = 0;
    uint32_t qSeqSizeUpper_ = 0;
    uint64_t fdAccumOutSize_ = 0;
    uint64_t fdLogSumExpSize_ = 0;

    QfmlaTilingInfo* qfmlaInfo_ = nullptr;
};

} // namespace quant_flash_mla
} // namespace optiling
#endif
