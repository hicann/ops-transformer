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
 * \file common_checker.h
 * \brief Common checker for layout, shape, dtype, and scalar attr parameters
 */

#ifndef FLASH_MLA_COMMON_CHECKER_H
#define FLASH_MLA_COMMON_CHECKER_H

#include <map>
#include <memory>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace flash_mla {

class CommonChecker : public FlashMlaBaseChecker {
public:
    CommonChecker() = default;
    ~CommonChecker() override = default;

    ge::graphStatus CheckSinglePara(const FlashMlaTilingInfo& faInfo) override;

    ge::graphStatus CheckParaExistence(const FlashMlaTilingInfo& faInfo) override;

    ge::graphStatus CheckMultiPara(const FlashMlaTilingInfo& faInfo) override;

    void SetFaShapeCompare(const FlashMlaTilingInfo& faInfo);
    ge::graphStatus CheckQueryShape(const FlashMlaTilingInfo& faInfo) const;
    ge::graphStatus CheckKVShape(const FlashMlaTilingInfo& faInfo) const;
    ge::graphStatus CheckAttnOutShape(const FlashMlaTilingInfo& faInfo) const;

private:
    // --- Layout checks ---
    ge::graphStatus CheckSingleParaLayout(const FlashMlaTilingInfo& faInfo);
    ge::graphStatus CheckMultiParaLayout(const FlashMlaTilingInfo& faInfo);

    // --- Shape/dt checks ---
    ge::graphStatus CheckNonQuantDataType(const FlashMlaTilingInfo& faInfo);
    ge::graphStatus CheckNonQuantHeadNum(const FlashMlaTilingInfo& faInfo);
    ge::graphStatus CheckDtypeConsistency(const FlashMlaTilingInfo& faInfo);
    ge::graphStatus CheckAxis(const FlashMlaTilingInfo& faInfo);
    ge::graphStatus CheckMlaGeometry(const FlashMlaTilingInfo& faInfo);
    ge::graphStatus CheckShapeConsistency(const FlashMlaTilingInfo& faInfo);

    ge::graphStatus CheckKVShapeForPageAttention(const FlashMlaTilingInfo& faInfo) const;

    // --- Attr checks ---

    std::shared_ptr<FlashMlaTilingShapeCompare> queryShapeCmp_;
    std::shared_ptr<FlashMlaTilingShapeCompare> keyShapeCmp_;
    std::shared_ptr<FlashMlaTilingShapeCompare> attnOutShapeCmp_;
};

} // namespace flash_mla
} // namespace optiling
#endif // FLASH_MLA_COMMON_CHECKER_H
