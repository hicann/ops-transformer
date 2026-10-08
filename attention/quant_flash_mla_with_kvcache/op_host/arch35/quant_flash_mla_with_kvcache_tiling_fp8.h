/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_with_kvcache_tiling_fp8.h
 * \brief QuantFlashMlaWithKvcache arch35 tiling implementation (MLA_FP8_E4M3_FULLQUANT)
 */
#ifndef QUANT_FLASH_MLA_WITH_KVCACHE_TILING_IMPL_FP8_H_
#define QUANT_FLASH_MLA_WITH_KVCACHE_TILING_IMPL_FP8_H_

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../qmla_tiling_info.h"
#include "../quant_flash_mla_with_kvcache_tiling_common.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/quant_flash_mla_with_kvcache_tiling_data.h"
#include "../../op_kernel/arch35/quant_flash_mla_with_kvcache_template_tiling_key.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

class QuantFlashMlaWithKvcacheTilingFp8Impl : public FiaTilingBase {
public:
    explicit QuantFlashMlaWithKvcacheTilingFp8Impl(gert::TilingContext* context)
        : FiaTilingBase(context)
    {}
    ~QuantFlashMlaWithKvcacheTilingFp8Impl() override = default;

    void InitTilingInfo(TilingInfo* tilingInfo) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus SetPlatMemoryInfo();
    void ComputeTilingData();
    void GenTilingKey();
    void CalcWorkspaceSize();
    void UpdateTilingKeyLayout();
    void UpdateTilingKeyConfig();
    void UpdateTilingKeyKvLayout();
    void UpdateTilingKeyQuantMode();
    void UpdateTilingKeyInfo();
    void SetQmlaTilingData();
    void InitImplParam();
    void PrintAllTilingData();
    void CalcScheduleMode();
    void CalcNumBlocks(uint32_t aicNum);
    void FillTiling();
    ge::graphStatus SetTilingData(QuantFlashMlaWithKvcacheTilingData& tilingData);
    bool CheckNeedInitOutput() const;

    QuantFlashMlaWithKvcacheTilingData tilingData_;
    QmlaTilingKeyInfo tilingKeyInfo_;
    QmlaPlatFormInfo platformInfo_;
    uint32_t sOuterFactor_ = 0;
    uint32_t sInnerFactor_ = 0;
    bool cuSeqLenQFlag_ = false;
    bool seqUsedQFlag_ = false;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    uint32_t numBlocks_ = 0;

    QmlaTilingInfo* qmlaInfo_ = nullptr;
};

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling
#endif // QUANT_FLASH_MLA_WITH_KVCACHE_TILING_IMPL_FP8_H_
