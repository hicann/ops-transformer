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
 * \file rotary_position_embedding_grad_apt.cpp
 * \brief
 */

#include "atvoss/reduce/reduce_sch.h"
#include "arch35/rotary_position_embedding_grad_dag.h"
#include "arch35/rotary_position_embedding_grad_tiling_key.h"
#include "arch35/rotary_position_embedding_grad_tiling_data.h"
#include "arch35/rotary_position_embedding_grad_bab.h"
#include "arch35/rotary_position_embedding_grad_ab.h"
#include "arch35/rotary_position_embedding_grad_aba_and_ba.h"
#include "arch35/rotary_position_embedding_grad_a_and_b.h"
#include "arch35/rotary_position_embedding_grad_rotary_x.h"
#include "arch35/rotary_position_embedding_grad_a_dcos_dsin.h"
#include "arch35/rotary_position_embedding_grad_zero_fill.h"

enum class ropeGradTilingKey : uint32_t {
    TILING_KEY_ABA = 201,
    TILING_KEY_BA = 202,
    TILING_KEY_BAB = 203,
    TILING_KEY_AB = 204,
    TILING_KEY_A = 205,
    TILING_KEY_B = 206
};

using namespace Ops::Base::ReduceOpTmpl;
using namespace AscendC;

template <REDUCE_TPL_PARAM, uint32_t DxTilingKey, uint32_t DcosFlag>
__global__ __aicore__ void rotary_position_embedding_grad(
    GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR x, GM_ADDR xGrad, GM_ADDR cosGrad, GM_ADDR sinGrad,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(RopeGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(RopeGradTilingData, tilingData, tiling);
    TPipe pipe;
    if (tilingData.isEmptyDy == 1) {
        // dy/x 为空 tensor：空进空出，不做任何有效计算；dcos/dsin 为空集求和，输出全 0
        if constexpr (DcosFlag) {
            RotaryPositionEmbeddingGrad::RopeGradZeroFill<DTYPE_DY> zeroFill(&pipe);
            zeroFill.Init(cosGrad, sinGrad, tilingData.cosShapeSize);
            zeroFill.Process();
        }
        return;
    }
    if constexpr (DxTilingKey == static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_ABA)) {
        RotaryPositionEmbeddingGrad::RotaryPositionEmbeddingGradABAAndBA<DTYPE_DY, false> op(
            &pipe, &tilingData.ropeGradParams);
        op.Init(grad, cos, sin, xGrad);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_BA)) {
        RotaryPositionEmbeddingGrad::RotaryPositionEmbeddingGradABAAndBA<DTYPE_DY, true> op(
            &pipe, &tilingData.ropeGradParams);
        op.Init(grad, cos, sin, xGrad);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_BAB)) {
        RotaryPositionEmbeddingGrad::RotaryPositionEmbeddingGradBAB<DTYPE_DY> op(&pipe, &tilingData.ropeGradParams);
        op.Init(grad, cos, sin, xGrad);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_AB)) {
        RotaryPositionEmbeddingGrad::RotaryPositionEmbeddingGradAB<DTYPE_DY> op(&pipe, &tilingData.ropeGradABParams);
        op.Init(grad, cos, sin, xGrad);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_A)) {
        RotaryPositionEmbeddingGrad::RotaryPositionEmbeddingGradAAndB<DTYPE_DY, false> op(
            &pipe, &tilingData.ropeGradParams);
        op.Init(grad, cos, sin, xGrad);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_B)) {
        RotaryPositionEmbeddingGrad::RotaryPositionEmbeddingGradAAndB<DTYPE_DY, true> op(
            &pipe, &tilingData.ropeGradParams);
        op.Init(grad, cos, sin, xGrad);
        op.Process();
        pipe.Reset();
    }

    if constexpr (DcosFlag) {
        PipeBarrier<PIPE_ALL>();
        SetSysWorkspace(workspace);
        GM_ADDR usrWorkSpace = AscendC::GetUserWorkspace(workspace);
        RotaryPositionEmbeddingGrad::RopeGradRotaryX<DTYPE_DY> op(&pipe, &tilingData.rotaryXParams);
        op.Init(x, usrWorkSpace);
        op.Process();
        pipe.Reset();
        SyncAll();

        int64_t workSpaceSize = tilingData.rotaryXParams.b * tilingData.rotaryXParams.s * tilingData.rotaryXParams.n *
                                tilingData.rotaryXParams.d;
        // A 模版不需要做reduce
        if constexpr (DxTilingKey == static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_A)) {
            RotaryPositionEmbeddingGrad::ComputeADcosDsin<DTYPE_DY> opGrad(&pipe, &tilingData.rotaryXParams);
            if (tilingData.rotaryXParams.rotaryMode ==
                static_cast<int64_t>(RotaryPositionEmbeddingGrad::RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
                GM_ADDR dSinWorkSpace = usrWorkSpace + workSpaceSize * sizeof(DTYPE_DY);
                opGrad.Init(grad, usrWorkSpace, dSinWorkSpace, cosGrad, sinGrad);
            } else {
                opGrad.Init(grad, x, usrWorkSpace, cosGrad, sinGrad);
            }
            opGrad.Process();
            pipe.Reset();
        } else {
            // 调用reduce模版，做reduce操作
            using ReduceOp = ReduceSch<REDUCE_TPL_VALUE,
                                       RotaryPositionEmbeddingGrad::RotaryPositionEmbeddingGradDag<DTYPE_DY, float>::OpDag>;
            ReduceOp reduceOp0(&tilingData.reduceTiling);
            ReduceOp reduceOp1(&tilingData.reduceTiling);
            if (tilingData.rotaryXParams.rotaryMode ==
                static_cast<int64_t>(RotaryPositionEmbeddingGrad::RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
                GM_ADDR reduceWorkSpace = usrWorkSpace + workSpaceSize * 2 * sizeof(DTYPE_DY);
                GM_ADDR dSinWorkSpace = usrWorkSpace + workSpaceSize * sizeof(DTYPE_DY);
                reduceOp0.Init(&pipe, grad, usrWorkSpace, cosGrad, reduceWorkSpace);
                reduceOp0.Process();
                pipe.Reset();
                PipeBarrier<PIPE_ALL>();
                reduceOp1.Init(&pipe, grad, dSinWorkSpace, sinGrad, reduceWorkSpace);
                reduceOp1.Process();
                pipe.Reset();
            } else {
                GM_ADDR reduceWorkSpace = usrWorkSpace + workSpaceSize * sizeof(DTYPE_DY);
                reduceOp0.Init(&pipe, grad, x, cosGrad, reduceWorkSpace);
                reduceOp0.Process();
                pipe.Reset();
                PipeBarrier<PIPE_ALL>();
                reduceOp1.Init(&pipe, grad, usrWorkSpace, sinGrad, reduceWorkSpace);
                reduceOp1.Process();
                pipe.Reset();
            }
        }
    }
}
