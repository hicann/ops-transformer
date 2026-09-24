/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_MATMUL_V2_OP_DEF_COMMON_H
#define MC2_MATMUL_V2_OP_DEF_COMMON_H

#include <vector>

#include "register/op_def_registry.h"

namespace ops {
namespace Mc2OpDef {

inline OpAICoreConfig &ConfigureMatmulV2DynamicAicore(OpAICoreConfig &config)
{
    return config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false") // 动态shape,复用二进制,后续图支持后修改
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
}

inline const std::vector<ge::DataType> &GetMatmulV2ScaleDataTypes()
{
    static const std::vector<ge::DataType> dataTypes = {
        ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
        ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
        ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
        ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
        ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
        ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
        ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};
    return dataTypes;
}

} // namespace Mc2OpDef
} // namespace ops

#endif // MC2_MATMUL_V2_OP_DEF_COMMON_H
