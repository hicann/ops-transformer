/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
#include "ops_proto_legacy.h"
#include "ge_api.h"
#include "graph/error_codes.h"
#include "graph.h"
#include "tensor.h"
#include "../op_graph/engram_gate_proto.h"

namespace {
constexpr int64_t TOKENS = 2;
constexpr float EPS = 1e-6F;

void Check(ge::Status status, const char* operation)
{
    if (status != ge::SUCCESS) {
        std::cerr << ge::GEGetErrorMsgV2().GetString() << '\n';
        throw std::runtime_error(operation);
    }
}

// Data inputs are host tensors. GE copies them to the execution device.
ge::Tensor Ones(const ge::TensorDesc& desc, size_t count)
{
    ge::Tensor tensor(desc);
    if (desc.GetDataType() == ge::DT_FLOAT) {
        std::vector<float> values(count, 1.0F);
        Check(tensor.SetData(reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float)),
              "SetData FP32");
    } else {
        // Exact IEEE encodings of 1.0; do not reuse BF16 bits for FP16.
        const uint16_t one = desc.GetDataType() == ge::DT_BF16 ? 0x3f80 : 0x3c00;
        std::vector<uint16_t> values(count, one);
        Check(tensor.SetData(reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(uint16_t)),
              "SetData FP16/BF16");
    }
    return tensor;
}

float ReadValue(const ge::Tensor& tensor, size_t index)
{
    float value;
    if (tensor.GetTensorDesc().GetDataType() == ge::DT_FLOAT) {
        std::memcpy(&value, tensor.GetData() + index * sizeof(float), sizeof(float));
        return value;
    }
    uint16_t bits;
    std::memcpy(&bits, tensor.GetData() + index * sizeof(uint16_t), sizeof(uint16_t));
    if (tensor.GetTensorDesc().GetDataType() == ge::DT_BF16) {
        const uint32_t wide = static_cast<uint32_t>(bits) << 16;
        std::memcpy(&value, &wide, sizeof(float));
        return value;
    }
    const int exponent = (bits >> 10) & 0x1f;
    const float magnitude = exponent == 0 ?
                                std::ldexp(static_cast<float>(bits & 0x3ff), -24) :
                                std::ldexp(1.0F + static_cast<float>(bits & 0x3ff) / 1024.0F, exponent - 15);
    if (exponent == 0x1f) {
        throw std::runtime_error("Nonfinite FP16 output");
    }
    return (bits & 0x8000) ? -magnitude : magnitude;
}

void CheckOutput(const ge::Tensor& tensor, const std::vector<int64_t>& shape, ge::DataType dtype, float expected,
                 float tolerance)
{
    if (tensor.GetTensorDesc().GetShape().GetDims() != shape || tensor.GetTensorDesc().GetDataType() != dtype) {
        throw std::runtime_error("Output shape/dtype mismatch");
    }
    size_t count = 1;
    for (auto dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    const size_t bytes = count * (dtype == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t));
    if (tensor.GetSize() != bytes) {
        throw std::runtime_error("Output byte size mismatch");
    }
    for (size_t i = 0; i < count; ++i) {
        const float value = ReadValue(tensor, i);
        if (!std::isfinite(value) || std::fabs(value - expected) > tolerance) {
            throw std::runtime_error("Output value mismatch at element " + std::to_string(i));
        }
    }
}

std::vector<int64_t> SetOutputDescs(ge::op::EngramGate& gate, const std::vector<int64_t>& outputShape,
                                    ge::DataType dtype, int64_t heads, bool save)
{
    const std::vector<int64_t> statsShape = save ? std::vector<int64_t>{TOKENS, heads} : std::vector<int64_t>{0};
    Check(gate.update_output_desc_output(ge::TensorDesc(ge::Shape(outputShape), ge::FORMAT_ND, dtype)), "output desc");
    const ge::TensorDesc statsDesc(ge::Shape(statsShape), ge::FORMAT_ND, ge::DT_FLOAT);
    Check(gate.update_output_desc_dot(statsDesc), "dot desc");
    Check(gate.update_output_desc_gate_score(statsDesc), "gate desc");
    Check(gate.update_output_desc_rstd_x(statsDesc), "rstd_x desc");
    Check(gate.update_output_desc_rstd_k(statsDesc), "rstd_k desc");
    return statsShape;
}

void RunCase(ge::Session& session, uint32_t id, int64_t heads, int64_t dim, ge::DataType dtype, bool save)
{
    ge::Graph graph("engram_gate_example");
    const std::vector<std::vector<int64_t>> shapes = {{TOKENS, heads, dim}, {TOKENS, (heads + 1) * dim}, {heads, dim}};
    std::vector<ge::Operator> inputs;
    std::vector<ge::Tensor> tensors;
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto name = "input_" + std::to_string(i);
        auto data = ge::op::Data(name.c_str()).set_attr_index(static_cast<int64_t>(i));
        ge::TensorDesc desc(ge::Shape(shapes[i]), ge::FORMAT_ND, i == 2 ? ge::DT_FLOAT : dtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetRealDimCnt(shapes[i].size());
        Check(data.update_input_desc_x(desc), "Data input desc");
        Check(data.update_output_desc_y(desc), "Data output desc");
        size_t count = 1;
        for (auto size : shapes[i]) {
            count *= static_cast<size_t>(size);
        }
        tensors.push_back(Ones(desc, count));
        inputs.push_back(data);
    }
    // Exactly three inputs: image_mask is not part of this operator.
    auto gate = ge::op::EngramGate("engram_gate");
    gate.set_input_hidden_states(inputs[0]).set_input_kv(inputs[1]).set_input_weight_fused(inputs[2]);
    gate.set_attr_eps(EPS).set_attr_clamp_value(EPS).set_attr_save_for_backward(save);
    const auto statsShape = SetOutputDescs(gate, shapes[0], dtype, heads, save);
    graph.SetInputs(inputs).SetOutputs(
        std::vector<std::pair<ge::Operator, std::vector<size_t>>>{{gate, {0, 1, 2, 3, 4}}});
    Check(session.AddGraph(id, graph, std::map<ge::AscendString, ge::AscendString>{}), "AddGraph");
    std::vector<ge::Tensor> outputs;
    Check(session.RunGraph(id, tensors, outputs), "RunGraph");
    if (outputs.size() != 5) {
        throw std::runtime_error("Expected five graph outputs");
    }
    // x=k=v=w=1: saved raw_dot=D, normalized_dot=sqrt(D)/(1+eps).
    const float rstd = 1.0F / std::sqrt(1.0F + EPS);
    const float normalized = static_cast<float>(dim) * rstd * rstd / std::sqrt(static_cast<float>(dim));
    const float expectedGate = 1.0F / (1.0F + std::exp(-std::sqrt(normalized)));
    CheckOutput(outputs[0], shapes[0], dtype, 1.0F + expectedGate, dtype == ge::DT_BF16 ? 8e-3F : 1e-3F);
    const float expected[] = {static_cast<float>(dim), expectedGate, rstd, rstd};
    for (size_t i = 0; i < 4; ++i) {
        CheckOutput(outputs[i + 1], statsShape, ge::DT_FLOAT, expected[i], 2e-5F);
    }
    Check(session.RemoveGraph(id), "RemoveGraph");
    std::cout << "PASS dtype=" << (dtype == ge::DT_BF16 ? "BF16" : "FP16") << " T=" << TOKENS << " H=" << heads
              << " D=" << dim << " save=" << save << '\n';
}
} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> options = {
        {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}, {"ge.jit_compile", "0"}};
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed: " << ge::GEGetErrorMsgV2().GetString() << '\n';
        return 1;
    }
    int result = 0;
    try {
        // Destroy Session before GEFinalize, including on failure.
        ge::Session session(options);
        uint32_t id = 0;
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
            for (bool save : {true, false}) {
                RunCase(session, id++, 4, 64, dtype, save);    // Minimum aligned D.
                RunCase(session, id++, 4, 5120, dtype, save);  // Network shape.
                RunCase(session, id++, 4, 32768, dtype, save); // Chunked long-D.
            }
        }
        std::cout << "EngramGate GE IR: 12/12 PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    if (ge::GEFinalize() != ge::SUCCESS) {
        std::cerr << "GEFinalize failed\n";
        result = 1;
    }
    return result;
}
