/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_TILING_H_
#define FLASH_ATTN_GRAD_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

// 这份声明只用于 REGISTER_TILING_DATA_CLASS，让框架知道 tiling data 该分配多大。
// host 实际写入的是 codegen 出的 FlashAttnGradTilingData（由 op_kernel/flash_attn_grad.py
// 的 dataclass 生成），两者没有任何编译期关联。字段增删必须两边同步 ——
// flash_attn_grad_tiling.cpp 里有一组 static_assert 钉住了 codegen 侧的布局，
// 只改 py 侧会在那里断编译；只改这里则由同文件的 nullptr 检查在运行时拦住。
BEGIN_TILING_DATA_DEF(FlashAttnGradTilingDataTmp)
TILING_DATA_FIELD_DEF(int64_t, b)
TILING_DATA_FIELD_DEF(int64_t, s1)
TILING_DATA_FIELD_DEF(int64_t, s2)
TILING_DATA_FIELD_DEF(int64_t, n1)
TILING_DATA_FIELD_DEF(int64_t, n2)
TILING_DATA_FIELD_DEF(int64_t, d)
TILING_DATA_FIELD_DEF(int64_t, dv)
TILING_DATA_FIELD_DEF(float, scaleValue)
// GQA group size: N1 = N2 * gSize. gSize == 1 is plain MHA.
TILING_DATA_FIELD_DEF(int64_t, gSize)
// Layout view parameters. Both BSND and BNSD are expressed as ONE 4D compact
// view [viewD0, S, viewD2, D] so the kernel needs a single pl.load form with a
// fixed order= -- the layout difference reduces to these numbers, evaluated at
// runtime with no branch and no extra tilingkey bit.
//   BSND [B,S,N,D] -> view [B,   S, N, D], index [b,     s, n, 0]
//   BNSD [B,N,S,D] -> view [B*N, S, 1, D], index [b*N+n, s, 0, 0]
// index0 = b * coefB0 + head * coefN0 ; index2 = head * coefN2
//   BSND: coefB0=1, coefN0=0, coefN2=1
//   BNSD: coefB0=N, coefN0=1, coefN2=0
// Under GQA the Q side (q/dout/attn_out/dq) has N1 heads and is indexed by
// head = n2*G+g, while the KV side (k/v/dk/dv) has N2 heads and is indexed by
// head = n2. So viewD0/viewD2/coefB0 come in two sets; coefN0/coefN2 depend
// only on the layout (not on the head count) and are therefore shared.
// Note: a plain stride remap does NOT work -- make_tensor silently ignores
// non-compact strides and reads with the compact ones instead.
TILING_DATA_FIELD_DEF(int64_t, viewD0Q)
TILING_DATA_FIELD_DEF(int64_t, viewD2Q)
TILING_DATA_FIELD_DEF(int64_t, coefB0Q)
TILING_DATA_FIELD_DEF(int64_t, viewD0KV)
TILING_DATA_FIELD_DEF(int64_t, viewD2KV)
TILING_DATA_FIELD_DEF(int64_t, coefB0KV)
TILING_DATA_FIELD_DEF(int64_t, coefN0)
TILING_DATA_FIELD_DEF(int64_t, coefN2)
// mask_mode 3=causal / 4=band. sparseType is the host remap
// (0=DENSE, 1=CASUAL, 2=BAND). s1Token/s2Token are ProcessTokensInfo
// corrected windows. totalPerBatchNum is valid 128x128 tiles per head.
TILING_DATA_FIELD_DEF(int64_t, maskMode)
TILING_DATA_FIELD_DEF(int64_t, winLeft)
TILING_DATA_FIELD_DEF(int64_t, winRight)
TILING_DATA_FIELD_DEF(int64_t, sparseType)
TILING_DATA_FIELD_DEF(int64_t, s1Token)
TILING_DATA_FIELD_DEF(int64_t, s2Token)
TILING_DATA_FIELD_DEF(int64_t, totalPerBatchNum)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FlashAttnGrad, FlashAttnGradTilingDataTmp)

// 编译期缓存的平台信息。图模式下 Tiling 阶段可能拿不到 PlatformInfo，那时只能
// 从这里回退取值，所以 ParsePlatform 是双路的（见 info/flash_attn_grad_tiling_info_parser.cpp）。
// 只放当前真正用到的字段：核数与 L2。L2 是 swizzle 判据的分母，各 ascend950 变体
// 从 16MB 到 128MB 不等，不能硬编码。
struct FlashAttnGradCompileInfo {
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t l2CacheSize = 0;
    uint64_t libapiWorkspaceSize = 0;

    static ge::graphStatus ParamCheck(gert::TilingContext *context)
    {
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_H_
