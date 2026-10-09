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
 * \file mqfa_seq_len_checker.h
 * \brief
 */

#ifndef MQFA_SEQ_LEN_CHECKER_H
#define MQFA_SEQ_LEN_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "mqfa_base_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {

class ActualSeqLenChecker : public FABaseChecker {
public:
    ActualSeqLenChecker() = default;
    ~ActualSeqLenChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo& faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo& faInfo) override;
    ge::graphStatus CheckFeature(const FaTilingInfo& faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo& faInfo) override;

private:
    ge::graphStatus CheckSingleParaSequsedQ(const FaTilingInfo& faInfo);
    ge::graphStatus CheckSingleParaSequsedKv(const FaTilingInfo& faInfo);
    ge::graphStatus CheckSingleParaCuSeqlensQ(const FaTilingInfo& faInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenQ(const FaTilingInfo& faInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenKv(const FaTilingInfo& faInfo);
};

} // namespace mixed_quant_flash_attn
} // namespace optiling
#endif // MQFA_SEQ_LEN_CHECKER_H
