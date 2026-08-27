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
 * \file seq_len_checker_quant_flash_attn.h
 * \brief Checker for cu_seqlens_q/kv, seqused_q/kv, max_seqlen_q/kv ( SeqLens参数组)
 */

#ifndef SEQ_LEN_CHECKER_QUANT_FLASH_ATTN_H
#define SEQ_LEN_CHECKER_QUANT_FLASH_ATTN_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {

class SeqLenChecker : public QfaBaseChecker {
public:
    SeqLenChecker() = default;
    ~SeqLenChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckParaExistence(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckFeature(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckMultiPara(const QfaTilingInfo &qfaInfo) override;

private:
    ge::graphStatus CheckSingleParaSequsedQ(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaSequsedKv(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaCuSeqlensQ(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaCuSeqlensKv(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenQ(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenKv(const QfaTilingInfo &qfaInfo);

    // --- Feature: cu_seqlens_q/kv 与 layout 的关系约束 ---
    ge::graphStatus CheckCuSeqlensLayoutConsistency(const QfaTilingInfo &qfaInfo);

    // --- Feature: 非TND时 seqused 与 max_seqlen 至少传1组 ---
    //  layout_q/kv 不为TND时, seqused_q/kv 与 max_seqlen_q/kv 至少传入其中一个;
    //          PA场景下 seqused_kv 必传(由 paged_attention_checker 负责, 此处不重复)
    ge::graphStatus CheckSequsedMaxSeqlenAtLeastOne(const QfaTilingInfo &qfaInfo);
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // SEQ_LEN_CHECKER_QUANT_FLASH_ATTN_H
