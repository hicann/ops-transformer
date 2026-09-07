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
 * \file base_checker_lightning_indexer_v2.h
 * \brief Declares shared validation helpers for Lightning Indexer V2 checkers.
 */

#ifndef BASE_CHECKER_LIGHTNING_INDEXER_V2_H
#define BASE_CHECKER_LIGHTNING_INDEXER_V2_H

#include <string>
#include <vector>
#include "tiling/tiling_api.h"
#include "checker_context_lightning_indexer_v2.h"

namespace optiling {
namespace lightning_indexer_v2_checker {

class LightningIndexerV2BaseChecker {
public:
    virtual ~LightningIndexerV2BaseChecker() = default;
    virtual ge::graphStatus CheckParaExistence(const LightningIndexerV2CheckerInfo &info) const;
    virtual ge::graphStatus CheckSinglePara(const LightningIndexerV2CheckerInfo &info) const;
    virtual ge::graphStatus CheckFeature(const LightningIndexerV2CheckerInfo &info) const;
    virtual ge::graphStatus CheckMultiPara(const LightningIndexerV2CheckerInfo &info) const;

protected:
    ge::graphStatus CheckDtype(const LightningIndexerV2CheckerInfo &info, const CheckerTensor &tensor, const char *name,
                               const std::vector<ge::DataType> &expected) const;
    ge::graphStatus CheckRank(const LightningIndexerV2CheckerInfo &info, const CheckerTensor &tensor, const char *name,
                              uint32_t expected, const std::string &condition = "") const;
    ge::graphStatus CheckElementCount(const LightningIndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                      const char *name, int64_t expected) const;
    static std::string DtypesToString(const std::vector<ge::DataType> &dtypes);
    static std::string ShapeToString(const CheckerTensor &tensor);
};

} // namespace lightning_indexer_v2_checker
} // namespace optiling
#endif // BASE_CHECKER_LIGHTNING_INDEXER_V2_H
