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
 * \file seq_len_checker.h
 * \brief Checker for cu_seqlens_q, seqused_q, max_seqlen_q/kv ( SeqLengths参数组)
 */

#ifndef QMLA_SEQ_LEN_CHECKER_H
#define QMLA_SEQ_LEN_CHECKER_H

#include "base_checker.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

class SeqLenChecker : public QmlaBaseChecker {
public:
    SeqLenChecker() = default;
    ~SeqLenChecker() override = default;

    ge::graphStatus CheckSinglePara(const QmlaTilingInfo& qmlaInfo) override;
    ge::graphStatus CheckFeature(const QmlaTilingInfo& qmlaInfo) override;

private:
    ge::graphStatus CheckSingleParaSequsedQ(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckSingleParaCuSeqlensQ(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenQ(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenKv(const QmlaTilingInfo& qmlaInfo);

    // --- Feature: cu_seqlens_q 与 layout 关系 ---
    ge::graphStatus CheckCuSeqlensLayoutConsistency(const QmlaTilingInfo& qmlaInfo);
    // --- Feature: TND 时 seqused_q 与 max_seqlen_q 至少传1个 ---
    ge::graphStatus CheckSequsedMaxSeqlenAtLeastOne(const QmlaTilingInfo& qmlaInfo);
};

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling
#endif // QMLA_SEQ_LEN_CHECKER_H
