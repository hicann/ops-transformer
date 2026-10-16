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
 * \file flash_mla_tiling_arch35.h
 * \brief FlashMla arch35 tiling implementation
 */
#ifndef FLASH_MLA_TILING_ARCH35_H_
#define FLASH_MLA_TILING_ARCH35_H_

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../flash_mla_tiling_info.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/flash_mla_tiling_data.h"
#include "../../op_kernel/arch35/flash_mla_template_tiling_key.h"

namespace optiling {
namespace flash_mla {

// 4 字段 tiling key —— 与 flash_mla_template_tiling_key.h 的 ASCENDC_TPL_ARGS_DECL(FlashMla,
// ...) 声明序一致（InOutLayoutType 8bw / KvLayoutType 8bw / HasAttenMask 1bw / Config 3bw）。
// 键内含语义按限定路由裁剪：InOutLayoutType 恒 BSND/BNSD/TND、Config 恒 MLA(config 0)、KvLayoutType 恒 PA 三类。
// 无 isFd——FD 能力恒实例化，运行时由 metadata mLen>0 决定。
struct FlashMlaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t kvLayoutType = 0;
    bool hasAttenMask = false;
    uint64_t config = 0;
};

struct FlashMlaPlatFormInfo {
    uint32_t coreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t cvRatio = 0;
    uint64_t ubSize = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0aSize = 0;
    uint64_t defaultSysWorkspaceSize = 0;
};

class FlashMlaTilingImpl : public FiaTilingBase {
public:
    explicit FlashMlaTilingImpl(gert::TilingContext* context)
        : FiaTilingBase(context)
    {}
    ~FlashMlaTilingImpl() override = default;

protected:
    void InitTilingInfo(TilingInfo* tilingInfo) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus SetPlatMemoryInfo();
    void InitImplParam();
    void SplitPolicy();
    void ComputeTilingData();
    void GenTilingKey();
    void CalcNumBlocks(uint32_t aicNum);
    void CalcWorkspaceSize();
    void UpdateTilingKeyConfig();
    void UpdateTilingKeyLayout();
    void UpdateTilingKeyKvLayout();
    void UpdateTilingKeyInfo();
    void SetFATilingData();
    void FillTiling();
    void PrintAllTilingData();
    void CalcScheduleMode();
    bool CheckNeedInitOutput() const;
    ge::graphStatus SetTilingData(FlashMlaTilingData& tilingData);

    FlashMlaTilingData tilingData_;
    FlashMlaTilingKeyInfo tilingKeyInfo_;
    FlashMlaPlatFormInfo platformInfo_;
    uint32_t sOuterFactor_ = 0;
    uint32_t sInnerFactor_ = 0;
    bool flashDecodeFlag_ = false;
    bool cuSeqLenQFlag_ = false;
    bool seqUsedQFlag_ = false;
    bool cacheSeqlensFlag_ = false;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    uint32_t numBlocks_ = 0;

    FlashMlaTilingInfo* faInfo_ = nullptr;
};

} // namespace flash_mla
} // namespace optiling
#endif // FLASH_MLA_TILING_ARCH35_H_
