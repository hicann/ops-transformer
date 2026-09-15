/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>

#include "nlohmann/json.hpp"
#include "graph/operator.h"
#include "register/register.h"

namespace domi {
static Status ParseParamsNpuMoeInitRouting(const ge::Operator &op_src, ge::Operator &op_dest)
{
    int64_t parsed_value = 0;
    bool found = false;
    ge::AscendString attributes;
    // ONNX repeated AttributeProto messages are exposed as JSON in "attribute".
    if (op_src.GetAttr("attribute", attributes) == ge::GRAPH_SUCCESS) {
        if (attributes.GetString() == nullptr) {
            return FAILED;
        }
        try {
            const auto root = nlohmann::json::parse(attributes.GetString());
            if (!root.is_object()) {
                return FAILED;
            }
            if (root.contains("attribute")) {
                const auto &attrs = root.at("attribute");
                if (!attrs.is_array()) {
                    return FAILED;
                }
                for (const auto &attr : attrs) {
                    if (!attr.is_object()) {
                        return FAILED;
                    }
                    // ONNX AttributeProto::INT is 2. Other types were ignored by the old parser.
                    if (attr.value("name", "") != "active_num" || attr.value("type", 0) != 2) {
                        continue;
                    }
                    if (found) {
                        return FAILED;
                    }
                    found = true;
                    // An omitted protobuf integer field has the default value zero.
                    if (!attr.contains("i")) {
                        parsed_value = 0;
                        continue;
                    }
                    const auto &value = attr.at("i");
                    if (!value.is_number_integer() ||
                        (value.is_number_unsigned() &&
                         value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
                        return FAILED;
                    }
                    parsed_value = value.get<int64_t>();
                }
            }
        } catch (const nlohmann::json::exception &) {
            return FAILED;
        }
    }
    if (!found) {
        return FAILED;
    }
    op_dest.SetAttr("active_num", parsed_value);
    return SUCCESS;
}

// register npu_moe_init_routing op info to GE
REGISTER_CUSTOM_OP("MoeInitRouting")
    .FrameworkType(ONNX)
    .OriginOpType({
        ge::AscendString("npu::1::NPUMoeInitRouting"),
        ge::AscendString("ai.onnx::11::NPUMoeInitRouting"),
        ge::AscendString("ai.onnx::12::NPUMoeInitRouting"),
        ge::AscendString("ai.onnx::13::NPUMoeInitRouting"),
        ge::AscendString("ai.onnx::14::NPUMoeInitRouting"),
        ge::AscendString("ai.onnx::15::NPUMoeInitRouting"),
        ge::AscendString("ai.onnx::16::NPUMoeInitRouting"),
        ge::AscendString("ai.onnx::17::NPUMoeInitRouting"),
        ge::AscendString("ai.onnx::18::NPUMoeInitRouting"),
    })
    .ParseParamsByOperatorFn(ParseParamsNpuMoeInitRouting)
    .ImplyType(ImplyType::TVM);
} // namespace domi
