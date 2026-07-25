/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file seq_len_checker.h
 * \brief Checker for cu_seqlens_q/kv, seqused_q/kv, max_seqlen_q/kv (文档约束: SeqLengths参数组)
 */

#ifndef QUANT_FLASH_ATTN_SEQ_LEN_CHECKER_H
#define QUANT_FLASH_ATTN_SEQ_LEN_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace quant_flash_attn {

class SeqLenChecker : public QfaBaseChecker {
public:
    SeqLenChecker() = default;
    ~SeqLenChecker() override = default;

    ge::graphStatus CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo) override;

private:
    // --- SinglePara ---
    ge::graphStatus CheckSingleParaSequsedQ(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaSequsedKv(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaCuSeqlensQ(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaCuSeqlensKv(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenQ(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaMaxSeqlenKv(const QuantFlashAttnTilingInfo &qfaInfo);

    // --- Feature: 非TND时 seqused 与 max_seqlen 至少传1组 ---
    // 文档约束:
    //   layout_q 不为 TND 时, seqused_q 与 max_seqlen_q 至少传入其中一个
    //   layout_kv 不为 TND 且不为 PA 场景时, seqused_kv 与 max_seqlen_kv 至少传入其中一个
    ge::graphStatus CheckSequsedMaxSeqlenAtLeastOne(const QuantFlashAttnTilingInfo &qfaInfo);
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_SEQ_LEN_CHECKER_H
