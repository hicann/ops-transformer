/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "nlohmann/json.hpp"

using namespace ge;
using json = nlohmann::json;

namespace domi {
static Status ParseParamsMoeFinalizeRouting(const ge::Operator& op_src, ge::Operator& op_dest)
{
    return SUCCESS;
}

// register npu_moe_finalize_routing op info to GE
REGISTER_CUSTOM_OP("MoeFinalizeRouting")
    .FrameworkType(ONNX)
    .OriginOpType({
        ge::AscendString("npu::1::NPUMoeFinalizeRouting"),
        ge::AscendString("ai.onnx::11::NPUMoeFinalizeRouting"),
        ge::AscendString("ai.onnx::12::NPUMoeFinalizeRouting"),
        ge::AscendString("ai.onnx::13::NPUMoeFinalizeRouting"),
        ge::AscendString("ai.onnx::14::NPUMoeFinalizeRouting"),
        ge::AscendString("ai.onnx::15::NPUMoeFinalizeRouting"),
        ge::AscendString("ai.onnx::16::NPUMoeFinalizeRouting"),
        ge::AscendString("ai.onnx::17::NPUMoeFinalizeRouting"),
        ge::AscendString("ai.onnx::18::NPUMoeFinalizeRouting"),
    })
    .ParseParamsByOperatorFn(ParseParamsMoeFinalizeRouting)
    .ImplyType(ImplyType::TVM);
} // namespace domi
