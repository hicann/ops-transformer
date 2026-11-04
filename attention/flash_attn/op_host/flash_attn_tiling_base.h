/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLASH_ATTN_TILING_BASE_H_
#define FLASH_ATTN_TILING_BASE_H_

#include <cstdint>
#include <vector>
#include "fa_tiling_info.h"
#include "../op_kernel/flash_attn_tiling_data.h"

namespace optiling {
namespace flash_attn {

struct FaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t kvLayoutType = 0;
    bool hasAttenMask = false;
    uint64_t templateId = 0;
    uint64_t config = 0;
};

struct FaPlatFormInfo {
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

// arch35/arch92 共用 tiling 基类：公共流程在此实现，架构差异通过虚函数覆写
class FlashAttnTilingBase : public FiaTilingBase {
public:
    explicit FlashAttnTilingBase(gert::TilingContext* context)
        : FiaTilingBase(context)
    {}
    ~FlashAttnTilingBase() override;

    void InitTilingInfo(TilingInfo* tilingInfo) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

protected:
    // 架构差异：由 arch35 / arch92 子类实现
    virtual void SplitPolicy() = 0;
    virtual void UpdateTilingKeyConfig() = 0;
    virtual void UpdateTilingKeyInfo() = 0;
    virtual void UpdateTilingKeyTemplateId() {}

    // 公共流程
    ge::graphStatus SetPlatMemoryInfo();
    void ComputeTilingData();
    void GenTilingKey();
    void CalcWorkspaceSize();
    void UpdateTilingKeyLayout();
    void UpdateTilingKeyKvLayout();
    void SetFATilingData();
    void InitImplParam();
    void PrintAllTilingData();
    void CalcScheduleMode();
    void CalcNumBlocks(uint32_t aicNum);
    void FillTiling();
    ge::graphStatus SetTilingData(FlashAttnTilingData& tilingData);
    bool CheckNeedInitOutput() const;

    FlashAttnTilingData tilingData_;
    FaTilingKeyInfo tilingKeyInfo_;
    FaPlatFormInfo platformInfo_;
    uint32_t sOuterFactor_;
    uint32_t sInnerFactor_;
    bool flashDecodeFlag_ = false;
    bool dnFlag_ = false;
    bool cuSeqLenQFlag_ = false;
    bool cuSeqLenKVFlag_ = false;
    bool seqUsedQFlag_ = false;
    bool seqUsedKvFlag_ = false;
    std::vector<int64_t> cuSeqLengthsQ_ = {};
    std::vector<int64_t> cuSeqLengthsKV_ = {};
    bool needInit_ = false;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    uint32_t numBlocks_ = 0;

    FaTilingInfo* faInfo_ = nullptr;
};

} // namespace flash_attn
} // namespace optiling

#endif // FLASH_ATTN_TILING_BASE_H_
