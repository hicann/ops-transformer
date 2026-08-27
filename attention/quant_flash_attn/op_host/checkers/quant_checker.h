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
 * \file quant_checker.h
 * \brief Checker for quant_mode, q_descale, k_descale, v_descale, p_scale ( 全量化参数组)
 */

#ifndef QUANT_FLASH_ATTN_QUANT_CHECKER_H
#define QUANT_FLASH_ATTN_QUANT_CHECKER_H

#include <map>
#include <memory>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker_quant_flash_attn.h"
#include "../qfa_tiling_shape.h"

namespace optiling {
namespace quant_flash_attn {

class QuantChecker : public QfaBaseChecker {
public:
    QuantChecker() = default;
    ~QuantChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckParaExistence(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckFeature(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckMultiPara(const QfaTilingInfo &qfaInfo) override;

private:
    ge::graphStatus CheckSingleParaQuantMode(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaQDescale(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaKDescale(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaVDescale(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaPScale(const QfaTilingInfo &qfaInfo);

    // --- SinglePara: descale shape dim 校验 (按量化场景分发) ---
    ge::graphStatus CheckQDescaleDimMxFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckQDescaleDimGqaFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckQDescaleDimHif8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckKDescaleDimMxFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckKDescaleDimGqaFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckKDescaleDimHif8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckVDescaleDimMxFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckVDescaleDimGqaFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckVDescaleDimHif8(const QfaTilingInfo &qfaInfo) const;

    // --- ParaExistence: 场景化必选参数 ---
    ge::graphStatus CheckParaExistenceGqaFp8(const QfaTilingInfo &qfaInfo) const;

    // --- MultiPara: descale shape consistency (descale_shape匹配关系表) ---
    ge::graphStatus CheckDescaleShape(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckQDescaleShape(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckQDescaleShapeMxFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckQDescaleShapeGqaFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckKDescaleShape(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckKDescaleShapeMxFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckKDescaleShapeGqaFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckVDescaleShape(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckVDescaleShapeMxFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckVDescaleShapeGqaFp8(const QfaTilingInfo &qfaInfo) const;

    // --- MultiPara: descale dtype 校验 ---
    // MxFP8 场景下, q/k/v descale 的 tensor_type 仅支持 FLOAT8_E8M0
    ge::graphStatus CheckDescaleDtype(const QfaTilingInfo &qfaInfo) const;

    // TND(非PA)场景下 v_descale 第一维 KV_T/64 的实际计算:
    // 各 batch 实际 KV 序列长度按 64 向上取整后累加，即 Σ ceil(cu_seqlens_kv[b+1]-cu_seqlens_kv[b], 64)
    int64_t CalcVDescaleTndDim0(const QfaTilingInfo &qfaInfo) const;

    // --- Feature: q/k/v dtype 与 quant_mode 精确匹配校验 ---
    ge::graphStatus CheckQkvDtype(const QfaTilingInfo &qfaInfo) const;

    // --- Feature: q/out ShapeDim 与 quant_mode 精确匹配校验 ---
    ge::graphStatus CheckQkvShapeDim(const QfaTilingInfo &qfaInfo) const;

    // --- Feature: layout 匹配关系校验 (文档: layout匹配关系表) ---
    // MxFP8: layout_q=TND, layout_kv∈{TND,PA_BBND,PA_BNBD,PA_NZ}, layout_out=TND, layout_q_descale∈{TND,N2TGD}
    ge::graphStatus CheckLayoutConstraint(const QfaTilingInfo &qfaInfo) const;

    // --- Feature: q/k/v/attn_out shape 校验 (文档: q/k/v/attn_out shape匹配关系表) ---
    ge::graphStatus CheckShapeMatch(const QfaTilingInfo &qfaInfo);
    void SetQfaShapeCompare(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckQueryShape(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckKVShape(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckAttnOutShape(const QfaTilingInfo &qfaInfo) const;

    // 校验实际 shape 与期望 shape 的每个维度是否相等
    ge::graphStatus CheckShapeEqual(const gert::StorageShape &actual, const std::vector<int64_t> &expected,
                                    const std::string &paraName, const char *opName) const;

    // --- Feature: N1/N2/G 上限校验 (全量化场景) ---
    ge::graphStatus CheckN1SizeFullquant(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckN2SizeFullquant(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckGSizeFullquant(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckInputAxisFullquant(const QfaTilingInfo &qfaInfo) const;

private:
    std::shared_ptr<QfaTilingShapeCompare> queryShapeCmp_ = nullptr;
    std::shared_ptr<QfaTilingShapeCompare> keyShapeCmp_ = nullptr;
    std::shared_ptr<QfaTilingShapeCompare> valueShapeCmp_ = nullptr;
    std::shared_ptr<QfaTilingShapeCompare> attnOutShapeCmp_ = nullptr;
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_QUANT_CHECKER_H
