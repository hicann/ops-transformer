/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_workspace.h"

#include <algorithm>

namespace optiling {
namespace {
constexpr size_t FP32_SIZE = 4;
constexpr int64_t FAG_WS_ALIGN_ROWS = 128;
} // namespace

// 分区必须与 kernel 的 make_ptr 切分完全一致：
//   [0]        libapi 预留（系统占用）
//   [+libapi]  dq 累加器  B*S1*N1*D  fp32（N1 = N2*G）
//   [+...]     dk 累加器  B*S2*N2*D  fp32
//   [+...]     dv 累加器  B*S2*N2*Dv fp32
// dq 跟随 q 的 head 数（N1），dk/dv 跟随 k/v 的（N2）—— GQA 下同一个 KV head 的
// G 个 Q head 会 atomicAdd 到同一块 dk/dv。kernel 用 fp32 atomicAdd 累加，之后
// 由 post 阶段乘 scale 并 cast 回 fp16/bf16 输出；dv 不乘 scale。
// dq/dk 宽 D，dv 宽 Dv（两者可以不等，例如 192 与 128）。
FagWorkspacePlan BuildWorkspacePlan(const FagParsedInfo &info)
{
    FagWorkspacePlan plan;
    plan.libapiSize = info.libapiWorkspaceSize;

    const size_t dqElems = static_cast<size_t>(info.b) * info.s1 * info.n1 * info.d;
    const size_t dkElems = static_cast<size_t>(info.b) * info.s2 * info.n2 * info.d;
    const size_t dvElems = static_cast<size_t>(info.b) * info.s2 * info.n2 * info.dv;

    size_t cursor = plan.libapiSize;
    plan.dq.offset = cursor;
    plan.dq.size = dqElems * FP32_SIZE;
    cursor += plan.dq.size;

    plan.dk.offset = cursor;
    plan.dk.size = dkElems * FP32_SIZE;
    cursor += plan.dk.size;

    plan.dv.offset = cursor;
    plan.dv.size = dvElems * FP32_SIZE;
    cursor += plan.dv.size;

    // 一块 128 行对齐垫在 dv 后面；dq/dk/dv 基址仍是紧凑 B*S*N*D。
    const size_t alignRowElems =
        static_cast<size_t>(FAG_WS_ALIGN_ROWS) * static_cast<size_t>(std::max(info.d, info.dv));
    cursor += alignRowElems * FP32_SIZE;

    plan.totalSize = cursor;
    return plan;
}

} // namespace optiling
