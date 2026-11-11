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
 * \file mqfa_common_checker.h
 * \brief Common checker for layout, shape, dtype, and scalar attr parameters
 */

#ifndef MQFA_COMMON_CHECKER_H
#define MQFA_COMMON_CHECKER_H

#include <map>
#include <memory>
#include <numeric>
#include "tiling/tiling_api.h"
#include "mqfa_base_checker.h"
#include "../fa_tiling_shape.h"

namespace optiling {
namespace mixed_quant_flash_attn {

class CommonChecker : public FABaseChecker {
public:
    CommonChecker() = default;
    ~CommonChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo& faInfo) override;

    ge::graphStatus CheckParaExistence(const FaTilingInfo& faInfo) override;

    ge::graphStatus CheckMultiPara(const FaTilingInfo& faInfo) override;

    ge::graphStatus CheckEmptyTensorPara(const FaTilingInfo& faInfo);

    void SetFaShapeCompare(const FaTilingInfo& faInfo);
    ge::graphStatus CheckQueryShape(const FaTilingInfo& faInfo) const;
    ge::graphStatus CheckKVShape(const FaTilingInfo& faInfo) const;
    ge::graphStatus CheckAttnOutShape(const FaTilingInfo& faInfo) const;

private:
    // --- Layout checks ---
    ge::graphStatus CheckSingleParaLayout(const FaTilingInfo& faInfo);
    ge::graphStatus CheckMultiParaLayout(const FaTilingInfo& faInfo);

    // --- Stride checks ---
    ge::graphStatus CheckStride(const FaTilingInfo& faInfo) const;
    ge::graphStatus CheckParamStridesPositive(const FaTilingInfo& faInfo, const char* paraName,
                                              const gert::Stride* strides) const;
    ge::graphStatus CheckParamStridedViewOverlap(const FaTilingInfo& faInfo, const char* paraName,
                                                 const gert::Shape& shape, const gert::Stride* strides) const;
    // kv cache / kv descale的0/1轴非连续支持范围校验
    ge::graphStatus CheckKVContiguous(const FaTilingInfo& faInfo) const;
    ge::graphStatus CheckDescaleContiguous(const FaTilingInfo& faInfo) const;
    // 非连续轴支持范围校验的Reason构造
    std::string BuildNonContiguousReason(const FaTilingInfo& faInfo, const char* paraName, int32_t maxAllowedAxis,
                                         int32_t dimIndex) const;

    // --- Shape/dt checks ---
    ge::graphStatus CheckMixedQuantDataType(const FaTilingInfo& faInfo);
    ge::graphStatus CheckHeadNum(const FaTilingInfo& faInfo);
    ge::graphStatus CheckDtypeConsistency(const FaTilingInfo& faInfo);
    ge::graphStatus CheckAxis(const FaTilingInfo& faInfo);
    ge::graphStatus CheckShapeConsistency(const FaTilingInfo& faInfo);

    ge::graphStatus CheckKVShapeForContinuous(const FaTilingInfo& faInfo) const;
    ge::graphStatus CheckKVShapeForPageAttention(const FaTilingInfo& faInfo) const;

    // --- Attr checks ---

    std::shared_ptr<FaTilingShapeCompare> queryShapeCmp_;
    std::shared_ptr<FaTilingShapeCompare> keyShapeCmp_;
    std::shared_ptr<FaTilingShapeCompare> valueShapeCmp_;
    std::shared_ptr<FaTilingShapeCompare> attnOutShapeCmp_;
};

} // namespace mixed_quant_flash_attn
} // namespace optiling
#endif // MQFA_COMMON_CHECKER_H
