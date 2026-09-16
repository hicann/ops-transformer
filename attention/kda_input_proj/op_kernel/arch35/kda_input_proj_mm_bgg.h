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
 * \file kda_input_proj_mm_bgg.h
 * \brief Stage1 AIC: Matmul(beta / gate / g)
 */

#ifndef KDA_INPUT_PROJ_MM_BGG_H
#define KDA_INPUT_PROJ_MM_BGG_H

#include "kernel_operator.h"
#include "kda_input_proj_common.h"
#include "../kda_input_proj_tiling_data.h"

namespace KdaInputProj {

template <typename TypePack>
class KdaInputProjMmBgg {
public:
    __aicore__ inline KdaInputProjMmBgg() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *weightBeta, __gm__ uint8_t *weightGate,
                                __gm__ uint8_t *weightG, __gm__ uint8_t *beta, __gm__ uint8_t *gate, __gm__ uint8_t *g,
                                __gm__ uint8_t *workspace, const optiling::KdaInputProjTilingData *__restrict tiling);
    __aicore__ inline void Process();

protected:
    using DtypeX = typename TypePack::DtypeX;
    using DtypeBeta = typename TypePack::DtypeBeta;
    using DtypeGate = typename TypePack::DtypeGate;
    using DtypeG = typename TypePack::DtypeG;

    const optiling::KdaInputProjTilingData *tiling_{nullptr};
    __gm__ uint8_t *x_{nullptr};
    __gm__ uint8_t *weightBeta_{nullptr};
    __gm__ uint8_t *weightGate_{nullptr};
    __gm__ uint8_t *weightG_{nullptr};
    __gm__ uint8_t *beta_{nullptr};
    __gm__ uint8_t *gate_{nullptr};
    __gm__ uint8_t *g_{nullptr};
    __gm__ uint8_t *workspace_{nullptr};
};

template <typename TypePack>
__aicore__ inline void KdaInputProjMmBgg<TypePack>::Init(__gm__ uint8_t *x, __gm__ uint8_t *weightBeta,
                                                         __gm__ uint8_t *weightGate, __gm__ uint8_t *weightG,
                                                         __gm__ uint8_t *beta, __gm__ uint8_t *gate, __gm__ uint8_t *g,
                                                         __gm__ uint8_t *workspace,
                                                         const optiling::KdaInputProjTilingData *__restrict tiling)
{
    x_ = x;
    weightBeta_ = weightBeta;
    weightGate_ = weightGate;
    weightG_ = weightG;
    beta_ = beta;
    gate_ = gate;
    g_ = g;
    workspace_ = workspace;
    tiling_ = tiling;
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMmBgg<TypePack>::Process()
{}
} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_MM_BGG_H
