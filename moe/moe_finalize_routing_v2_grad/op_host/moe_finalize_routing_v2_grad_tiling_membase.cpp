/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_finalize_routing_v2_grad_tiling_membase.cpp
 * \brief
 */
#include "moe_finalize_routing_v2_grad_tiling.h"

namespace optiling {
constexpr int64_t TILING_KEY_WITHOUT_SCALE_NOT_CUT_H = 10001;
constexpr int64_t TILING_KEY_WITHOUT_SCALE_CUT_H = 10002;
constexpr int64_t TILING_KEY_WITH_SCALE_NOT_CUT_H = 20001;
constexpr int64_t TILING_KEY_WITH_SCALE_CUT_H = 20002;
constexpr int64_t BYTE_BLOCK = 32;
constexpr int64_t FP32_SIZE = 4;

class MoeFinalizeRoutingV2GradMembase : public MoeFinalizeRoutingV2GradTiling {
public:
    explicit MoeFinalizeRoutingV2GradMembase(gert::TilingContext* context)
        : MoeFinalizeRoutingV2GradTiling(context)
    {}
    ~MoeFinalizeRoutingV2GradMembase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        MoeFinalizeRoutingV2GradTiling::Reset(context);
    }

protected:
    ge::graphStatus CalcTilingKey() override;

    bool IsCapable() override
    {
        if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
            return false;
        }
        return true;
    }

private:
    void CalcTilingKeyWithScales();
};

ge::graphStatus MoeFinalizeRoutingV2GradMembase::CalcTilingKey()
{
    if (isScalesExist_) {
        CalcTilingKeyWithScales();
    } else {
        int64_t expandedRowIdxUbSize = BYTE_BLOCK;
        hiddenPrePart_ = (aicoreParams_.ubSize - expandedRowIdxUbSize) / BYTE_BLOCK * BYTE_BLOCK / gradYTypeByteSize_;
        if (hidden_ <= hiddenPrePart_) {
            tilingKey_ = TILING_KEY_WITHOUT_SCALE_NOT_CUT_H;
        } else {
            tilingKey_ = TILING_KEY_WITHOUT_SCALE_CUT_H;
        }
    }

    OP_CHECK_IF(
        (hiddenPrePart_ <= 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName_, "hiddenPrePart", std::to_string(hiddenPrePart_).c_str(),
                                              "hiddenPrePart must be greater than 0"),
        return ge::GRAPH_FAILED);

    if (hidden_ <= hiddenPrePart_) {
        hiddenInnerLoops_ = 0;
        hiddenLastPart_ = 0;
    } else {
        hiddenInnerLoops_ = hidden_ / hiddenPrePart_;
        hiddenLastPart_ = hidden_ % hiddenPrePart_;
    }

    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingV2GradMembase::CalcTilingKeyWithScales()
{
    // gradY和expandedX计算时需要升精度到FP32
    int64_t gradYElementSize = FP32_SIZE;
    int64_t expandedXElementSize = FP32_SIZE;
    // gradExpandedX计算时无需升精度
    int64_t gradExpandedXElementSize = gradYTypeByteSize_;
    int64_t elementTotalSize = gradYElementSize + expandedXElementSize + gradExpandedXElementSize;
    if (isBiasExist_) {
        uint64_t biasTypePart = gradYElementSize;
        elementTotalSize = elementTotalSize + biasTypePart;
    }

    int64_t expandedRowIdxUbSize = BYTE_BLOCK;
    int64_t scaleUbSize = BYTE_BLOCK;
    int64_t expertIdxUbSize = BYTE_BLOCK;
    int64_t gradScalesUbSize = BYTE_BLOCK;
    hiddenPrePart_ = (aicoreParams_.ubSize - expandedRowIdxUbSize - scaleUbSize - expertIdxUbSize - gradScalesUbSize) /
                     elementTotalSize * gradYTypeByteSize_ / BYTE_BLOCK * BYTE_BLOCK / gradYTypeByteSize_;
    if (hidden_ <= hiddenPrePart_) {
        tilingKey_ = TILING_KEY_WITH_SCALE_NOT_CUT_H;
    } else {
        tilingKey_ = TILING_KEY_WITH_SCALE_CUT_H;
    }
}

REGISTER_OPS_TILING_TEMPLATE(MoeFinalizeRoutingV2Grad, MoeFinalizeRoutingV2GradMembase, 10000);

} // namespace optiling
