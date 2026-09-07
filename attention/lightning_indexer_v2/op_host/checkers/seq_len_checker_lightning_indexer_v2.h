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
 * \file seq_len_checker_lightning_indexer_v2.h
 * \brief Declares sequence-length parameter checks for Lightning Indexer V2 operators.
 */

#ifndef SEQ_LEN_CHECKER_LIGHTNING_INDEXER_V2_H
#define SEQ_LEN_CHECKER_LIGHTNING_INDEXER_V2_H

#include "base_checker_lightning_indexer_v2.h"

namespace optiling {
namespace lightning_indexer_v2_checker {

class SeqLenChecker : public LightningIndexerV2BaseChecker {
public:
    ge::graphStatus CheckSinglePara(const LightningIndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckParaExistence(const LightningIndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckMultiPara(const LightningIndexerV2CheckerInfo &info) const override;
};

} // namespace lightning_indexer_v2_checker
} // namespace optiling
#endif // SEQ_LEN_CHECKER_LIGHTNING_INDEXER_V2_H
