/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attention_score_tiling_regbase_host.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace FA {
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
class FlashAttentionScoreTilingDropMaskRegbase : public FlashAttentionScoreTilingRegbase {
public:
    explicit FlashAttentionScoreTilingDropMaskRegbase(gert::TilingContext *context) :
        FlashAttentionScoreTilingRegbase(context)
    {
        // manually initialize tiling data in hostapi scenario
        OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                            context_->GetRawTilingData()->GetCapacity()) != EOK,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "Fail to memset tiling data"), return;);
    }
    ~FlashAttentionScoreTilingDropMaskRegbase() override = default;

protected:
    ge::graphStatus DoOpTiling() override
    {
        if (!inputParamsRegbase_->get_needDropMaskOp()) {
            return ge::GRAPH_PARAM_INVALID;
        }

        int64_t shapeTotalSize = inputParamsRegbase_->get_bSize() * inputParamsRegbase_->get_n2Size() *
                                 inputParamsRegbase_->get_gSize() * inputParamsRegbase_->get_s1Size() *
                                 inputParamsRegbase_->get_s2Size();
        auto layoutType = inputParamsRegbase_->get_layoutType();
        if (layoutType == static_cast<uint8_t>(LayoutType::LAYOUT_TND)) {
            for (auto i = 0; i < bSize; i++) {
                dropTotalSize += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeTotalSize = inputParamsRegbase_->get_n2Size() * inputParamsRegbase_->get_gSize() * dropTotalSize;
            OP_LOGD(context_, "shapeTotalSize %ld dropTotalSize %ld.", shapeTotalSize, dropTotalSize);
        }
        // 保证每核计算数据量256倍数，2048表示bit位，256 * 8
        const int64_t ubCalFactor = 2048;
        int64_t shapeSplitCoreSize = CeilDivision(shapeTotalSize, ubCalFactor);
        int64_t shapeSingleCoreSize = CeilDivision(shapeSplitCoreSize, static_cast<int64_t>(aivNum));

        // ub能计算的最大元素数, 向下对齐
        // 单次ub计算量为x个元素,空间占用：x/8 * 1 [1个 uint8]+ 2x * 2[2个fp16,select的src和res] + x * 1 [1个uint8]共6份
        int64_t baseUbCalSize = AlignDown(CeilDivision(static_cast<int64_t>(aicoreParams_.ubSize),
                                          DROP_MASK_CAL_SIZE), ubCalFactor);
        baseUbCalSize = std::min(baseUbCalSize, shapeSingleCoreSize * ubCalFactor);
        // ub 的外层循环次数
        int64_t multiCoreFactorSize = CeilDivision(shapeSingleCoreSize * ubCalFactor, baseUbCalSize);
        shapeTotalSize = AlignUp(shapeTotalSize, GM_ALIGN);

        dropmaskParamsRegbase_->set_shapeTotalSize(shapeTotalSize);
        dropmaskParamsRegbase_->set_multiCoreFactorSize(static_cast<int32_t>(multiCoreFactorSize));
        dropmaskParamsRegbase_->set_multiCoreTotalSize(CeilDivision(shapeSplitCoreSize * ubCalFactor, baseUbCalSize));
        dropmaskParamsRegbase_->set_baseUbCalSize(static_cast<int32_t>(baseUbCalSize));
        context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
        return ge::GRAPH_PARAM_INVALID;
    }

    uint64_t GetTilingKey() const override
    {
        return 0UL;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    void CalcS1S2BasicBlock() override {}

    void CalcDBasicBlock() override {}
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingDropMaskRegbase, static_cast<int32_t>(NpuArch::DAV_3510), 81);
} // namespace FA
} // namespace optiling
