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
 * \file common_add_rms_norm_tiling.h
 * \brief
 */
#ifndef COMMON_ADD_RMS_NORM_H_
#define COMMON_ADD_RMS_NORM_H_

#include "add_rms_norm/op_host/add_rms_norm_tiling.h"
#include "context_transfer.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "mc2_log.h"
#include "../../../op_kernel/matmul_all_reduce_add_rms_norm_tiling_data.h"

namespace optiling {
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024; // 16M
struct AddRmsNormTilingInputFromMM {
    uint32_t m;
    uint32_t n;
    ge::DataType x1Dtype;
};
struct AddRMSNormTilingDepend {
    AddRMSNormTilingDepend(const char *name, fe::PlatFormInfos &infos, ARNCtxInfo info, AddRmsNormTilingInputFromMM mm,
                           bool b, bool half)
        : nodeName(name),
          platFormInfos(infos),
          arnCtxInfo(info),
          addRmsNormTilingInputFromMm(mm),
          useMmOutputAsX1Input(b),
          useHalfNumBlocks(half) {};
    const char *nodeName;
    fe::PlatFormInfos &platFormInfos;
    ARNCtxInfo arnCtxInfo;
    AddRmsNormTilingInputFromMM addRmsNormTilingInputFromMm{};
    bool useMmOutputAsX1Input{false};
    // 全量化场景下，因为mm的核函数认为aic和aiv配比是1：1,
    // 所以使用到vector的addrms的tiling也需要做一下处理，感知到这个配比
    bool useHalfNumBlocks{false};
};
struct TilingOut {
    uint32_t tilingKey;
    uint32_t workSpaceSize;
    uint32_t numBlocks;
};
struct AddRMSNormTilingOutput {
    Mc2Tiling::AddRMSNormTilingData &addRmsNormTilingData;
    TilingOut &tilingOut;
};

namespace CommonAddResNormTiling {
enum ModeKey : uint32_t {
    K_NORMAL,
    K_SPLIT_D,
    K_MERGE_N,
    K_SINGLE_N,
    K_MULTI_N,
};
ge::graphStatus Tiling4AddRmsNorm(const AddRMSNormTilingDepend &addRmsNormTilingDepend,
                                  AddRMSNormTilingOutput &addRmsNormTilingOutput);
ge::graphStatus CheckAddRmsNormInput(const gert::TilingContext *context, const ARNCtxInfo &arnCtxInfo);
ge::graphStatus SetAddRmsNormTilingData(const AddRMSNormTilingDepend &addRmsNormTilingDepend, const uint32_t numRow,
                                        const int64_t numCol, const uint32_t blockFactor,
                                        AddRMSNormTilingOutput &addRmsNormTilingOutput);
} // namespace CommonAddResNormTiling
} // namespace optiling
#endif // COMMON_ADD_RMS_NORM_H_
