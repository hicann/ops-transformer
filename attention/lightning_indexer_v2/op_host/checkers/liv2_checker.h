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
 * \file liv2_checker.h
 * \brief Declares the Lightning Indexer V2 checker entry point.
 */

#ifndef LIGHTNING_INDEXER_V2_CHECKER_H
#define LIGHTNING_INDEXER_V2_CHECKER_H

#include "checker_context_lightning_indexer_v2.h"

namespace optiling {
namespace lightning_indexer_v2_checker {

class LIV2Checker {
public:
    explicit LIV2Checker(const LightningIndexerV2CheckerInfo &info)
        : info_(info)
    {}
    ge::graphStatus Process() const;

private:
    LightningIndexerV2CheckerInfo info_;
};

} // namespace lightning_indexer_v2_checker
} // namespace optiling
#endif
