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
 * \file paged_attention_checker_lightning_indexer_v2.h
 * \brief Declares Paged Attention parameter checks for Lightning Indexer V2 operators.
 */

#ifndef PAGED_ATTENTION_CHECKER_LIGHTNING_INDEXER_V2_H
#define PAGED_ATTENTION_CHECKER_LIGHTNING_INDEXER_V2_H

#include "base_checker_lightning_indexer_v2.h"

namespace optiling {
namespace lightning_indexer_v2_checker {

class PagedAttentionChecker : public LightningIndexerV2BaseChecker {
public:
    ge::graphStatus CheckParaExistence(const LightningIndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckSinglePara(const LightningIndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckMultiPara(const LightningIndexerV2CheckerInfo &info) const override;

private:
    ge::graphStatus CheckContiguous(const LightningIndexerV2CheckerInfo &info, const CheckerTensor &tensor,
                                    const char *name, uint32_t firstCheckedAxis) const;
};

} // namespace lightning_indexer_v2_checker
} // namespace optiling
#endif // PAGED_ATTENTION_CHECKER_LIGHTNING_INDEXER_V2_H
