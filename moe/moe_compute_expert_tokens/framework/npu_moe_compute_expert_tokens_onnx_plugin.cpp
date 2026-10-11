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
static const int REQ_ATTR_NNUM = 1;
static Status ParseParamsMoeComputeExpertTokens(const ge::Operator& op_src, ge::Operator& op_dest)
{
    AscendString attrs_string;
    int attrCount = 0;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        json attrs = json::parse(attrs_string.GetString());
        for (json& attr : attrs["attribute"]) {
            if (attr["name"] == "num_experts") {
                int num_experts = attr["i"];
                op_dest.SetAttr("num_experts", num_experts);
                attrCount++;
            }
        }
    }

    if (attrCount != REQ_ATTR_NNUM) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr num_experts.");
        return FAILED;
    }

    return SUCCESS;
}

// register npu_moe_finalize_routing op info to GE
REGISTER_CUSTOM_OP("MoeComputeExpertTokens")
    .FrameworkType(ONNX)
    .OriginOpType({
        ge::AscendString("npu::1::NPUMoeComputeExpertTokens"),
        ge::AscendString("ai.onnx::11::NPUMoeComputeExpertTokens"),
        ge::AscendString("ai.onnx::12::NPUMoeComputeExpertTokens"),
        ge::AscendString("ai.onnx::13::NPUMoeComputeExpertTokens"),
        ge::AscendString("ai.onnx::14::NPUMoeComputeExpertTokens"),
        ge::AscendString("ai.onnx::15::NPUMoeComputeExpertTokens"),
        ge::AscendString("ai.onnx::16::NPUMoeComputeExpertTokens"),
        ge::AscendString("ai.onnx::17::NPUMoeComputeExpertTokens"),
        ge::AscendString("ai.onnx::18::NPUMoeComputeExpertTokens"),
    })
    .ParseParamsByOperatorFn(ParseParamsMoeComputeExpertTokens)
    .ImplyType(ImplyType::TVM);
} // namespace domi
