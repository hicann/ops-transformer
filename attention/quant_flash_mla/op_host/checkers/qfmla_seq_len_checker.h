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
 * \file qfmla_seq_len_checker.h
 * \brief Checker for cu_seqlens_q, seqused_q, max_seqlen_q/kv ( SeqLengths参数组)
 */

#ifndef QFMLA_SEQ_LEN_CHECKER_H
#define QFMLA_SEQ_LEN_CHECKER_H

#include "qfmla_base_checker.h"

namespace optiling {
namespace quant_flash_mla {

class SeqLenChecker : public QfmlaBaseChecker {
public:
    SeqLenChecker() = default;
    ~SeqLenChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo) override;
    ge::graphStatus CheckFeature(const QfmlaTilingInfo& qfmlaInfo) override;

private:
    ge::graphStatus CheckSingleParaSequsedQ(const QfmlaTilingInfo& qfmlaInfo);
    ge::graphStatus CheckSingleParaCuSeqlensQ(const QfmlaTilingInfo& qfmlaInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenQ(const QfmlaTilingInfo& qfmlaInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenKv(const QfmlaTilingInfo& qfmlaInfo);

    // --- Feature: cu_seqlens_q 与 layout 关系 ---
    ge::graphStatus CheckCuSeqlensLayoutConsistency(const QfmlaTilingInfo& qfmlaInfo);
    // --- Feature: TND 时 seqused_q 与 max_seqlen_q 至少传1个 ---
    ge::graphStatus CheckSequsedMaxSeqlenAtLeastOne(const QfmlaTilingInfo& qfmlaInfo);
};

} // namespace quant_flash_mla
} // namespace optiling
#endif // QFMLA_SEQ_LEN_CHECKER_H
