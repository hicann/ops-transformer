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
 * \file test_geir_fused_infer_attention_score.cpp
 * \brief GE graph construction sample for FusedInferAttentionScore (BNSD layout, fp16, no-quant path).
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/fused_infer_attention_score_proto.h"

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
constexpr int64_t kBatch = 1;
constexpr int64_t kQSeqLen = 2;
constexpr int64_t kKvSeqLen = 2;
constexpr int64_t kNumHeads = 2;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kHeadDim = 16;
constexpr double kScale = 0.25; // 1.0 / sqrt(16)
constexpr float kRtol = 3e-3f;
constexpr float kAtol = 3e-3f;

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
                   vector<float>& qData, vector<float>& kData, vector<float>& vData)
{
    // BNSD layout: [B, N, S, D].
    const vector<int64_t> qShape = {kBatch, kNumHeads, kQSeqLen, kHeadDim};
    const vector<int64_t> kvShape = {kBatch, kNumKvHeads, kKvSeqLen, kHeadDim};
    const vector<int64_t> pseShape = {kBatch, kNumHeads, kQSeqLen, kKvSeqLen};
    const vector<int64_t> maskShape = {kBatch, 1, kQSeqLen, kKvSeqLen};

    const int64_t qSize = GetShapeSize(qShape);
    const int64_t kvSize = GetShapeSize(kvShape);
    qData.resize(qSize);
    kData.resize(kvSize);
    vData.resize(kvSize);
    for (int64_t i = 0; i < qSize; ++i) {
        qData[i] = GenValue(static_cast<uint64_t>(i) * 7 + 1);
    }
    for (int64_t i = 0; i < kvSize; ++i) {
        kData[i] = GenValue(static_cast<uint64_t>(i) * 13 + 5);
        vData[i] = GenValue(static_cast<uint64_t>(i) * 17 + 9);
    }
    vector<uint16_t> qHalf(qData.size());
    vector<uint16_t> kHalf(kData.size());
    vector<uint16_t> vHalf(vData.size());
    for (size_t i = 0; i < qData.size(); ++i) {
        qHalf[i] = FloatToHalf(qData[i]);
    }
    for (size_t i = 0; i < kData.size(); ++i) {
        kHalf[i] = FloatToHalf(kData[i]);
        vHalf[i] = FloatToHalf(vData[i]);
    }
    // pse_shift is bound but filled with zeros so the reference stays a plain softmax attention.
    const vector<uint16_t> pseHalf(pseShape[0] * pseShape[1] * pseShape[2] * pseShape[3], 0);
    // bool mask, 0 means the position attends.
    const vector<uint8_t> maskData(maskShape[0] * maskShape[1] * maskShape[2] * maskShape[3], 0);
    // actual_seq_lengths: int64, effective sequence length of each batch.
    const vector<int64_t> seqLenData = {kQSeqLen};

    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT16, qHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT16, kHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "value", 3, kvShape, DT_FLOAT16, vHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "pse_shift", 4, pseShape, DT_FLOAT16, pseHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "atten_mask", 5, maskShape, DT_BOOL, maskData) == SUCCESS,
              return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "actual_seq_lengths", 6, {kBatch}, DT_INT64, seqLenData) == SUCCESS,
        return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "actual_seq_lengths_kv", 7, {kBatch}, DT_INT64,
                       vector<int64_t>{kKvSeqLen}) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "q_start_idx", 8, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "kv_start_idx", 9, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);

    auto node = op::FusedInferAttentionScore("fused_infer_attention_score");
    node.set_input_query(inputOps[0]);
    // key/value are dynamic inputs. create_dynamic_input_byindex_* registers one instance for each AT the position
    // declared by the IR definition (key at index 1, value at index 2); a plain create_dynamic_input_* would append
    // them after the static inputs and misalign the slot order. The input descs come from the connected Data ops.
    node.create_dynamic_input_byindex_key(1, 1);
    node.create_dynamic_input_byindex_value(1, 2);
    node.set_dynamic_input_key(0, inputOps[1]);
    node.set_dynamic_input_value(0, inputOps[2]);
    node.set_input_pse_shift(inputOps[3]);
    node.set_input_atten_mask(inputOps[4]);
    node.set_input_actual_seq_lengths(inputOps[5]);
    node.set_input_actual_seq_lengths_kv(inputOps[6]);
    node.set_input_q_start_idx(inputOps[7]);
    node.set_input_kv_start_idx(inputOps[8]);

    node.set_attr_num_heads(kNumHeads);
    node.set_attr_scale(kScale);
    node.set_attr_pre_tokens(2147483647);
    node.set_attr_next_tokens(2147483647);
    node.set_attr_input_layout("BNSD");
    node.set_attr_num_key_value_heads(kNumKvHeads);
    node.set_attr_sparse_mode(0);
    node.set_attr_inner_precise(1);
    node.set_attr_block_size(0);
    node.set_attr_antiquant_mode(0);
    node.set_attr_softmax_lse_flag(false);
    node.set_attr_key_antiquant_mode(0);
    node.set_attr_value_antiquant_mode(0);
    node.set_attr_query_quant_mode(0);
    node.set_attr_pse_type(0);
    node.set_attr_out_dtype(0);

    TensorDesc attentionOutDesc(Shape(qShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc softmaxLseDesc(Shape({0}), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_attention_out(attentionOutDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_lse(softmaxLseDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

// CPU reference: softmax(scale * Q @ K^T + pse) @ V with BNSD layout and GQA support (pse is zero here).
void ComputeReference(const vector<float>& qData, const vector<float>& kData, const vector<float>& vData,
                      vector<float>& attentionOut)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    const int64_t group = kNumHeads / kNumKvHeads;
    attentionOut.assign(GetShapeSize({kBatch, kNumHeads, kQSeqLen, kHeadDim}), 0.0f);
    for (int64_t b = 0; b < kBatch; ++b) {
        for (int64_t n = 0; n < kNumHeads; ++n) {
            const int64_t kvN = n / group;
            vector<float> score(s2);
            for (int64_t i = 0; i < s1; ++i) {
                for (int64_t j = 0; j < s2; ++j) {
                    float dot = 0.0f;
                    for (int64_t t = 0; t < d; ++t) {
                        const int64_t qIdx = ((b * kNumHeads + n) * s1 + i) * d + t;
                        const int64_t kIdx = ((b * kNumKvHeads + kvN) * s2 + j) * d + t;
                        dot += qData[qIdx] * kData[kIdx];
                    }
                    score[j] = static_cast<float>(kScale) * dot;
                }
                float maxScore = score[0];
                for (int64_t j = 1; j < s2; ++j) {
                    maxScore = std::fmax(maxScore, score[j]);
                }
                float sum = 0.0f;
                for (int64_t j = 0; j < s2; ++j) {
                    score[j] = std::exp(score[j] - maxScore);
                    sum += score[j];
                }
                for (int64_t t = 0; t < d; ++t) {
                    float y = 0.0f;
                    for (int64_t j = 0; j < s2; ++j) {
                        const int64_t vIdx = ((b * kNumKvHeads + kvN) * s2 + j) * d + t;
                        y += (score[j] / sum) * vData[vIdx];
                    }
                    const int64_t oIdx = ((b * kNumHeads + n) * s1 + i) * d + t;
                    attentionOut[oIdx] = y;
                }
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
                        const vector<float>& vData)
{
    CHECK_RET(outputs.size() == 2, LOG_PRINT("[CHECK] FAIL: expected 2 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> attentionOut;
    ComputeReference(qData, kData, vData, attentionOut);
    const vector<int64_t> qShape = {kBatch, kNumHeads, kQSeqLen, kHeadDim};
    const bool ok = CheckOutput(outputs[0], "attention_out", qShape, attentionOut);
    CHECK_RET(outputs[1].GetSize() == 0, LOG_PRINT("[CHECK] softmax_lse expected empty\n"); return FAILED);
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

    Graph graph("fused_infer_attention_score_graph");
    vector<Tensor> inputTensors;
    vector<Operator> inputOps;
    vector<Operator> outputOps;
    vector<float> qData;
    vector<float> kData;
    vector<float> vData;
    int32_t ret = BuildGraph(graph, inputTensors, inputOps, outputOps, qData, kData, vData);
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
        LOG_PRINT("FusedInferAttentionScore graph run success, output count: %zu\n", outputTensors.size());
        ret = ValidateOutputs(outputTensors, qData, kData, vData);
        delete session;
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
