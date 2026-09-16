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
 * \file kda_input_proj_mx_quant.h
 * \brief Stage1 AIV: DynamicMxQuant on x
 */

#ifndef KDA_INPUT_PROJ_MX_QUANT_H
#define KDA_INPUT_PROJ_MX_QUANT_H

#include "kernel_operator.h"
#include "kda_input_proj_common.h"
#include "../kda_input_proj_tiling_data.h"

namespace KdaInputProj {

template <typename TypePack>
class KdaInputProjMxQuant {
public:
    __aicore__ inline KdaInputProjMxQuant() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *quantX, __gm__ uint8_t *xScale,
                                const optiling::KdaInputProjTilingData *__restrict tiling);
    __aicore__ inline void Process();

protected:
    using DtypeX = typename TypePack::DtypeX;

    const optiling::KdaInputProjTilingData *tiling_{nullptr};
    __gm__ uint8_t *x_{nullptr};
    __gm__ uint8_t *quantX_{nullptr};
    __gm__ uint8_t *xScale_{nullptr};
};

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::Init(__gm__ uint8_t *x, __gm__ uint8_t *quantX,
                                                           __gm__ uint8_t *xScale,
                                                           const optiling::KdaInputProjTilingData *__restrict tiling)
{
    x_ = x;
    quantX_ = quantX;
    xScale_ = xScale;
    tiling_ = tiling;
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::Process()
{}
} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_MX_QUANT_H
