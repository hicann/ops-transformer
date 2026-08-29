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
 * \file apply_rotary_pos_emb_grad_apt.cpp
 * \brief
 */

#include "atvoss/reduce/reduce_sch.h"
#include "arch35/apply_rotary_pos_emb_grad_tiling_key.h"
#include "arch35/apply_rotary_pos_emb_grad_tiling_data.h"
#include "arch35/apply_rotary_pos_emb_grad_bab.h"
#include "arch35/apply_rotary_pos_emb_grad_ab.h"
#include "arch35/apply_rotary_pos_emb_grad_a.h"
#include "arch35/apply_rotary_pos_emb_grad_dag.h"
#include "arch35/apply_rotary_pos_emb_grad_dcos_dsin.h"

enum class ApplyRopeGradDxTilingKey : uint32_t {
    TILING_KEY_BAB = 203,
    TILING_KEY_AB = 204,
    TILING_KEY_A = 205
};

using namespace ApplyRotaryPosEmbGrad;
using namespace Ops::Base::ReduceOpTmpl;
using namespace AscendC;

constexpr int64_t INPUT_OUTPUT_NUM = 2; // grad_cos_partial + grad_sin_partial
constexpr int64_t PARTIAL_TYPE_SIZE = sizeof(float);

template <REDUCE_TPL_PARAM, uint32_t DxTilingKey, uint32_t DcosFlag>
__global__ __aicore__ void apply_rotary_pos_emb_grad(GM_ADDR gradQueryEmbed, GM_ADDR gradKeyEmbed, GM_ADDR cos,
                                                     GM_ADDR sin, GM_ADDR query, GM_ADDR key, GM_ADDR gradQueryOut,
                                                     GM_ADDR gradKeyOut, GM_ADDR gradCosOut, GM_ADDR gradSinOut,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    if constexpr (DxTilingKey != static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_A)) {
        if (workspace == nullptr) {
            return;
        }
    }

    REGISTER_TILING_DEFAULT(ApplyRopeGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyRopeGradTilingData, tilingData, tiling);
    TPipe pipe;

    // 部分积 workspace 基址（DcosFlag=1 且非 A 模板时，BAB/AB 需在 Phase 1 写 grad_cos/grad_sin 部分积到 workspace）
    GM_ADDR usrWorkSpace = nullptr;
    int64_t partSize = 0;
    if constexpr (DcosFlag && DxTilingKey != static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_A)) {
        SetSysWorkspace(workspace);
        usrWorkSpace = AscendC::GetUserWorkspace(workspace);
        int64_t maxN = max(tilingData.ropeGradParams.nQ, tilingData.ropeGradParams.nK);
        partSize = tilingData.ropeGradParams.b * tilingData.ropeGradParams.s * maxN * tilingData.ropeGradParams.d;
    }

    // ================================================================
    // Phase 1: 计算 grad_query / grad_key
    //          当 DcosFlag=1 时，同步在 kernel 内累加 grad_cos / grad_sin 部分积
    // ================================================================
    if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_BAB)) {
        ApplyRotaryPosEmbGradBAB<DTYPE_GRAD_QUERY_EMBED> op(&pipe, &tilingData.ropeGradParams);
        if constexpr (DcosFlag) {
            GM_ADDR dSinWorkSpace = usrWorkSpace + partSize * PARTIAL_TYPE_SIZE;
            op.Init(gradQueryEmbed, gradKeyEmbed, cos, sin, query, key, gradQueryOut, gradKeyOut, usrWorkSpace,
                    dSinWorkSpace);
        } else {
            op.Init(gradQueryEmbed, gradKeyEmbed, cos, sin, query, key, gradQueryOut, gradKeyOut, nullptr, nullptr);
        }
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_AB)) {
        ApplyRotaryPosEmbGradAB<DTYPE_GRAD_QUERY_EMBED, (DcosFlag != 0)> op(&pipe, &tilingData.ropeGradABParams);
        if constexpr (DcosFlag) {
            GM_ADDR dSinWorkSpace = usrWorkSpace + partSize * PARTIAL_TYPE_SIZE;
            op.Init(gradQueryEmbed, gradKeyEmbed, cos, sin, query, key, gradQueryOut, gradKeyOut, usrWorkSpace,
                    dSinWorkSpace);
        } else {
            op.Init(gradQueryEmbed, gradKeyEmbed, cos, sin, query, key, gradQueryOut, gradKeyOut, nullptr, nullptr);
        }
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_A)) {
        if constexpr (DcosFlag) {
            // dcos/dsin 走 workspace 三阶段路径，需要 workspace 非空
            if (workspace == nullptr) {
                return;
            }
        }
        // 阶段 1: 计算 dx (grad_query / grad_key)
        ApplyRotaryPosEmbGradA<DTYPE_GRAD_QUERY_EMBED> op(&pipe, &tilingData.ropeGradParams);
        op.Init(gradQueryEmbed, gradKeyEmbed, cos, sin, gradQueryOut, gradKeyOut);
        op.Process();
        pipe.Reset();

        if constexpr (DcosFlag) {
            PipeBarrier<PIPE_ALL>();
            SetSysWorkspace(workspace);
            GM_ADDR usrWorkSpace = AscendC::GetUserWorkspace(workspace);
            // 阶段 2: 预计算 rotate(query)/rotate(key) -> workspace（ws_q / ws_k 各 b_*d 元素）
            ApplyRotaryXDual<DTYPE_GRAD_QUERY_EMBED> rotOp(&pipe, &tilingData.ropeGradParams);
            rotOp.Init(query, key, usrWorkSpace);
            rotOp.Process();
            pipe.Reset();
            SyncAll(); // 确保 rotate 的 MTE3 写 GM 完成，阶段 3 才能读

            // 阶段 3: dcos/dsin 高层 Mul + Q/K 累加
            int64_t bTotal = tilingData.ropeGradParams.b;
            int64_t dVal = tilingData.ropeGradParams.d;
            GM_ADDR wsQ = usrWorkSpace;
            GM_ADDR wsK = usrWorkSpace + bTotal * dVal * sizeof(DTYPE_GRAD_QUERY_EMBED);
            ApplyDcosDsin<DTYPE_GRAD_QUERY_EMBED> dcosOp(&pipe, &tilingData.ropeGradParams);
            dcosOp.Init(gradQueryEmbed, gradKeyEmbed, query, key, wsQ, wsK, gradCosOut, gradSinOut);
            dcosOp.Process();
            pipe.Reset();
        }
        // A 模板无需 Reduce：grad_cos/grad_sin 已在阶段 3 直接写回 GM
        return;
    }

    // ================================================================
    // Phase 2: DcosFlag=1 → Reduce 跨广播轴求和（仅广播模板 BAB/AB 需要）
    // A 模板 (205): grad_cos/grad_sin 已在 Phase 1 融合计算直接写回 GM，无 reduce
    // ================================================================
    if constexpr (DcosFlag) {
        constexpr bool isBroadcastPartialFloatTemplate =
            DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_BAB) ||
            DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_AB);
        SyncAll();

        // 第一份 workspace 存 dcos_partial，第二份存 dsin_partial
        // gradCosOut = reduce(dcos_partial), gradSinOut = reduce(dsin_partial)
        if constexpr (isBroadcastPartialFloatTemplate) {
            using ReduceOp =
                ReduceSch<REDUCE_TPL_VALUE,
                          ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradDag<float, DTYPE_GRAD_QUERY_EMBED, float>::OpDag>;
            ReduceOp reduceOp0(&tilingData.reduceTiling);
            ReduceOp reduceOp1(&tilingData.reduceTiling);
            GM_ADDR reduceWorkSpace = usrWorkSpace + partSize * INPUT_OUTPUT_NUM * PARTIAL_TYPE_SIZE;

            reduceOp0.Init(&pipe, usrWorkSpace, gradCosOut, reduceWorkSpace);
            reduceOp0.Process();
            pipe.Reset();
            PipeBarrier<PIPE_ALL>();

            GM_ADDR dSinWorkSpace = usrWorkSpace + partSize * PARTIAL_TYPE_SIZE;
            reduceOp1.Init(&pipe, dSinWorkSpace, gradSinOut, reduceWorkSpace);
            reduceOp1.Process();
            pipe.Reset();
        }
    }
}
