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
 * \file qliv2_variant_checker.h
 * \brief Declares variant-specific checks for Quant Lightning Indexer V2.
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_VARIANT_CHECKER_H
#define QUANT_LIGHTNING_INDEXER_V2_VARIANT_CHECKER_H

#include "../../../lightning_indexer_v2/op_host/checkers/base_checker_lightning_indexer_v2.h"

namespace optiling {
namespace lightning_indexer_v2_checker {

class QLIV2VariantChecker : public LightningIndexerV2BaseChecker {
public:
    ge::graphStatus CheckParaExistence(const LightningIndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckSinglePara(const LightningIndexerV2CheckerInfo &info) const override;
    ge::graphStatus CheckMultiPara(const LightningIndexerV2CheckerInfo &info) const override;

private:
    ge::graphStatus CheckScaleShape(const LightningIndexerV2CheckerInfo &info, const CheckerTensor &input,
                                    const CheckerTensor &scale, const char *inputName, const char *scaleName) const;
};

} // namespace lightning_indexer_v2_checker
} // namespace optiling
#endif // QUANT_LIGHTNING_INDEXER_V2_VARIANT_CHECKER_H
