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
 * \file test_geir_sparse_flash_attention.cpp
 * \brief GE graph construction sample for SparseFlashAttention (MLA-absorb mode, TND layout, fp16).
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/sparse_flash_attention_proto.h"

#define FAILED (-1)
#define SUCCESS 0

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int64_t kQTokens = 1;                 // T1
constexpr int64_t kQHeads = 16;                 // N1
constexpr int64_t kKvTokens = 2048;             // T2
constexpr int64_t kKvHeads = 1;                 // N2
constexpr int64_t kHeadDim = 512;               // D (MLA latent)
constexpr int64_t kRopeDim = 64;                // Dr
constexpr double kScale = 0.041666666666666664; // 1.0 / sqrt(512 + 64)
constexpr float kRtol = 3e-2f;
constexpr float kAtol = 3e-2f;

std::string GetGeError()
{
    const AscendString errorMessage = GEGetErrorMsgV2();
    return errorMessage.GetString() == nullptr ? "" : errorMessage.GetString();
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

// Deterministic value in [-0.45, 0.45]; the magnitude stays in the normal fp16 range.
float GenValue(uint64_t seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    return ((static_cast<float>(seed & 0xFFU) / 255.0f) - 0.5f) * 0.9f;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | (mantissa >> 13));
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    const uint32_t exponent = (value & 0x7C00U) >> 10;
    const uint32_t mantissa = value & 0x03FFU;
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -24) * ((sign != 0) ? -1.0f : 1.0f);
    }
    const uint32_t bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T>
int32_t AddInput(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, const string& name,
                 int32_t index, const vector<int64_t>& shape, DataType dtype, const vector<T>& hostData)
{
    auto dataOp = op::Data(name).set_attr_index(index - 1);
    TensorDesc desc(Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    const int64_t elementCount = GetShapeSize(shape);
    CHECK_RET(static_cast<int64_t>(hostData.size()) == elementCount,
              LOG_PRINT("[ERROR] %s data size mismatch\n", name.c_str());
              return FAILED);
    Tensor tensor;
    auto ret = tensor.SetTensorDesc(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s SetTensorDesc failed\n", name.c_str()); return FAILED);
    ret = tensor.SetData(reinterpret_cast<const uint8_t*>(hostData.data()), hostData.size() * sizeof(T));
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s SetData failed\n", name.c_str()); return FAILED);
    ret = dataOp.update_input_desc_x(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s update_input_desc_x failed\n", name.c_str()); return FAILED);
    ret = dataOp.update_output_desc_y(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s update_output_desc_y failed\n", name.c_str()); return FAILED);
    ret = graph.AddOp(dataOp);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] Graph::AddOp failed for %s\n", name.c_str()); return FAILED);
    inputTensors.push_back(tensor);
    inputOps.push_back(dataOp);
    return SUCCESS;
}

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps,
                   vector<float>& qData, vector<float>& kData, vector<float>& vData, vector<float>& qRopeData,
                   vector<float>& kRopeData)
{
    // TND layout.
    const vector<int64_t> qShape = {kQTokens, kQHeads, kHeadDim};
    const vector<int64_t> kvShape = {kKvTokens, kKvHeads, kHeadDim};
    const vector<int64_t> indicesShape = {kQTokens, kKvHeads, kKvTokens};
    const vector<int64_t> qRopeShape = {kQTokens, kQHeads, kRopeDim};
    const vector<int64_t> kRopeShape = {kKvTokens, kKvHeads, kRopeDim};

    qData.resize(GetShapeSize(qShape));
    kData.resize(GetShapeSize(kvShape));
    vData.resize(GetShapeSize(kvShape));
    qRopeData.resize(GetShapeSize(qRopeShape));
    kRopeData.resize(GetShapeSize(kRopeShape));
    for (int64_t i = 0; i < GetShapeSize(qShape); ++i) {
        qData[i] = GenValue(static_cast<uint64_t>(i) * 7 + 1);
    }
    for (int64_t i = 0; i < GetShapeSize(qRopeShape); ++i) {
        qRopeData[i] = GenValue(static_cast<uint64_t>(i) * 23 + 3);
    }
    for (int64_t i = 0; i < GetShapeSize(kvShape); ++i) {
        kData[i] = GenValue(static_cast<uint64_t>(i) * 13 + 5);
        vData[i] = GenValue(static_cast<uint64_t>(i) * 17 + 9);
    }
    for (int64_t i = 0; i < GetShapeSize(kRopeShape); ++i) {
        kRopeData[i] = GenValue(static_cast<uint64_t>(i) * 29 + 7);
    }
    // Identity selection: every kv token is picked in order.
    vector<int32_t> indicesData(kKvTokens);
    std::iota(indicesData.begin(), indicesData.end(), 0);
    // TND sequence lengths are prefix sums.
    const vector<int32_t> actSeqQData = {static_cast<int32_t>(kQTokens)};
    const vector<int32_t> actSeqKvData = {static_cast<int32_t>(kKvTokens)};

    vector<uint16_t> qHalf(qData.size());
    vector<uint16_t> kHalf(kData.size());
    vector<uint16_t> vHalf(vData.size());
    vector<uint16_t> qRopeHalf(qRopeData.size());
    vector<uint16_t> kRopeHalf(kRopeData.size());
    for (size_t i = 0; i < qData.size(); ++i) {
        qHalf[i] = FloatToHalf(qData[i]);
    }
    for (size_t i = 0; i < kData.size(); ++i) {
        kHalf[i] = FloatToHalf(kData[i]);
        vHalf[i] = FloatToHalf(vData[i]);
    }
    for (size_t i = 0; i < qRopeData.size(); ++i) {
        qRopeHalf[i] = FloatToHalf(qRopeData[i]);
    }
    for (size_t i = 0; i < kRopeData.size(); ++i) {
        kRopeHalf[i] = FloatToHalf(kRopeData[i]);
    }

    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT16, qHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT16, kHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "value", 3, kvShape, DT_FLOAT16, vHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "sparse_indices", 4, indicesShape, DT_INT32, indicesData) == SUCCESS,
        return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "actual_seq_lengths_query", 5, {1}, DT_INT32, actSeqQData) == SUCCESS,
        return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "actual_seq_lengths_kv", 6, {1}, DT_INT32, actSeqKvData) == SUCCESS,
        return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query_rope", 7, qRopeShape, DT_FLOAT16, qRopeHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key_rope", 8, kRopeShape, DT_FLOAT16, kRopeHalf) == SUCCESS,
              return FAILED);

    auto node = op::SparseFlashAttention("sparse_flash_attention");
    node.set_input_query(inputOps[0]);
    node.set_input_key(inputOps[1]);
    node.set_input_value(inputOps[2]);
    node.set_input_sparse_indices(inputOps[3]);
    node.set_input_actual_seq_lengths_query(inputOps[4]);
    node.set_input_actual_seq_lengths_kv(inputOps[5]);
    node.set_input_query_rope(inputOps[6]);
    node.set_input_key_rope(inputOps[7]);
    node.set_attr_scale_value(kScale);
    node.set_attr_sparse_block_size(1);
    node.set_attr_layout_query("TND");
    node.set_attr_layout_kv("TND");
    node.set_attr_sparse_mode(0);
    node.set_attr_pre_tokens(9223372036854775807);
    node.set_attr_next_tokens(9223372036854775807);
    node.set_attr_attention_mode(2);
    node.set_attr_return_softmax_lse(false);

    TensorDesc attentionOutDesc(Shape(qShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc emptyFp32Desc(Shape({0}), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_attention_out(attentionOutDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_max(emptyFp32Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_sum(emptyFp32Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

// CPU reference (MLA-absorb): score = scale * (q @ k^T + qRope @ kRope^T), out = softmax(score) @ v.
void ComputeReference(const vector<float>& qData, const vector<float>& kData, const vector<float>& vData,
                      const vector<float>& qRopeData, const vector<float>& kRopeData, vector<float>& attentionOut)
{
    const int64_t t1 = kQTokens;
    const int64_t n1 = kQHeads;
    const int64_t t2 = kKvTokens;
    const int64_t n2 = kKvHeads;
    const int64_t d = kHeadDim;
    const int64_t dr = kRopeDim;
    const int64_t group = n1 / n2;
    attentionOut.assign(t1 * n1 * d, 0.0f);
    for (int64_t i = 0; i < t1; ++i) {
        for (int64_t n = 0; n < n1; ++n) {
            const int64_t kvN = n / group;
            vector<float> score(t2);
            for (int64_t j = 0; j < t2; ++j) {
                float dot = 0.0f;
                for (int64_t t = 0; t < d; ++t) {
                    dot += qData[(i * n1 + n) * d + t] * kData[(j * n2 + kvN) * d + t];
                }
                for (int64_t t = 0; t < dr; ++t) {
                    dot += qRopeData[(i * n1 + n) * dr + t] * kRopeData[(j * n2 + kvN) * dr + t];
                }
                score[j] = static_cast<float>(kScale) * dot;
            }
            float maxScore = score[0];
            for (int64_t j = 1; j < t2; ++j) {
                maxScore = std::fmax(maxScore, score[j]);
            }
            float sum = 0.0f;
            for (int64_t j = 0; j < t2; ++j) {
                score[j] = std::exp(score[j] - maxScore);
                sum += score[j];
            }
            for (int64_t t = 0; t < d; ++t) {
                float y = 0.0f;
                for (int64_t j = 0; j < t2; ++j) {
                    y += (score[j] / sum) * vData[(j * n2 + kvN) * d + t];
                }
                attentionOut[(i * n1 + n) * d + t] = y;
            }
        }
    }
}

bool CheckOutput(const Tensor& tensor, const string& name, const vector<int64_t>& shape, const vector<float>& expected)
{
    const auto desc = tensor.GetTensorDesc();
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    CHECK_RET(desc.GetShape().GetDims() == shape && desc.GetDataType() == DT_FLOAT16 &&
                  tensor.GetSize() == count * sizeof(uint16_t),
              LOG_PRINT("[CHECK] %s FAIL: unexpected shape, dtype, or data size\n", name.c_str());
              return false);
    vector<uint16_t> halfValues(count);
    std::memcpy(halfValues.data(), tensor.GetData(), count * sizeof(uint16_t));
    size_t mismatches = 0;
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float value = HalfToFloat(halfValues[i]);
        const float tolerance = kAtol + kRtol * std::fabs(expected[i]);
        const float err = std::fabs(value - expected[i]);
        maxAbsErr = std::fmax(maxAbsErr, err);
        if (err > tolerance) {
            if (mismatches < 4) {
                LOG_PRINT("[CHECK] %s[%zu]=%.7f expected=%.7f tolerance=%.7f\n", name.c_str(), i, value, expected[i],
                          tolerance);
            }
            ++mismatches;
        }
    }
    LOG_PRINT("[CHECK] %s: count=%zu, maxAbsErr=%.7g, mismatches=%zu: %s\n", name.c_str(), count, maxAbsErr, mismatches,
              mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

int32_t ValidateOutputs(const vector<Tensor>& outputs, const vector<float>& qData, const vector<float>& kData,
                        const vector<float>& vData, const vector<float>& qRopeData, const vector<float>& kRopeData)
{
    CHECK_RET(outputs.size() == 3, LOG_PRINT("[CHECK] FAIL: expected 3 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> attentionOut;
    ComputeReference(qData, kData, vData, qRopeData, kRopeData, attentionOut);
    const vector<int64_t> qShape = {kQTokens, kQHeads, kHeadDim};
    const bool ok = CheckOutput(outputs[0], "attention_out", qShape, attentionOut);
    CHECK_RET(outputs[1].GetSize() == 0, LOG_PRINT("[CHECK] softmax_max expected empty\n"); return FAILED);
    CHECK_RET(outputs[2].GetSize() == 0, LOG_PRINT("[CHECK] softmax_sum expected empty\n"); return FAILED);
    LOG_PRINT("[CHECK] total: %s\n", ok ? "PASS" : "FAIL");
    return ok ? SUCCESS : FAILED;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    auto status = GEInitialize(globalOptions);
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEInitialize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);

    Graph graph("sparse_flash_attention_graph");
    vector<Tensor> inputTensors;
    vector<Operator> inputOps;
    vector<Operator> outputOps;
    vector<float> qData;
    vector<float> kData;
    vector<float> vData;
    vector<float> qRopeData;
    vector<float> kRopeData;
    int32_t ret = BuildGraph(graph, inputTensors, inputOps, outputOps, qData, kData, vData, qRopeData, kRopeData);
    if (ret == SUCCESS) {
        graph.SetInputs(inputOps).SetOutputs(outputOps);
        std::map<AscendString, AscendString> buildOptions;
        Session* session = new (std::nothrow) Session(buildOptions);
        CHECK_RET(session != nullptr, LOG_PRINT("[ERROR] Session allocation failed.\n"); return FAILED);
        constexpr uint32_t graphId = 0;
        std::map<AscendString, AscendString> graphOptions;
        auto addRet = session->AddGraph(graphId, graph, graphOptions);
        CHECK_RET(addRet == GRAPH_SUCCESS,
                  LOG_PRINT("[ERROR] Session::AddGraph failed, status=%u, error=%s\n", addRet, GetGeError().c_str());
                  delete session; return FAILED);
        vector<Tensor> outputTensors;
        auto runRet = session->RunGraph(graphId, inputTensors, outputTensors);
        CHECK_RET(runRet == GRAPH_SUCCESS,
                  LOG_PRINT("[ERROR] Session::RunGraph failed, status=%u, error=%s\n", runRet, GetGeError().c_str());
                  delete session; return FAILED);
        LOG_PRINT("SparseFlashAttention graph run success, output count: %zu\n", outputTensors.size());
        ret = ValidateOutputs(outputTensors, qData, kData, vData, qRopeData, kRopeData);
        delete session;
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
