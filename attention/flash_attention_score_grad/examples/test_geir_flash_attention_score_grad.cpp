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
 * \file test_geir_flash_attention_score_grad.cpp
 * \brief GE graph construction sample for FlashAttentionScoreGrad (SBH layout, fp16).
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

#include "../op_graph/flash_attention_score_grad_proto.h"

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
constexpr int64_t kQHeadNum = 1;
constexpr int64_t kKvHeadNum = 1;
constexpr int64_t kQSeqLen = 256;
constexpr int64_t kKvSeqLen = 256;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kH1 = kQHeadNum * kHeadDim;
constexpr int64_t kH2 = kKvHeadNum * kHeadDim;
constexpr double kScale = 0.08838834764831845; // 1.0 / sqrt(128)
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
        return sign; // flush tiny magnitudes to zero
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U); // clamp to infinity
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

struct ForwardStats {
    vector<float> softmaxMax;  // [S1]
    vector<float> softmaxSum;  // [S1]
    vector<float> attentionIn; // [S1, D] fp32 exact reference of the forward output
};

void ComputeForward(const vector<float>& q, const vector<float>& k, const vector<float>& v, ForwardStats& stats)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    vector<float> score(s2);
    stats.softmaxMax.assign(s1, 0.0f);
    stats.softmaxSum.assign(s1, 0.0f);
    stats.attentionIn.assign(s1 * d, 0.0f);
    for (int64_t i = 0; i < s1; ++i) {
        for (int64_t j = 0; j < s2; ++j) {
            float dot = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dot += q[i * d + t] * k[j * d + t];
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
        stats.softmaxMax[i] = maxScore;
        stats.softmaxSum[i] = sum;
        for (int64_t t = 0; t < d; ++t) {
            float y = 0.0f;
            for (int64_t j = 0; j < s2; ++j) {
                y += (score[j] / sum) * v[j * d + t];
            }
            stats.attentionIn[i * d + t] = y;
        }
    }
}

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps,
                   vector<float>& q, vector<float>& k, vector<float>& v, vector<float>& dy, vector<float>& statsMax,
                   vector<float>& statsSum, vector<float>& attentionIn)
{
    const int64_t qSize = kQSeqLen * kH1;
    const int64_t kvSize = kKvSeqLen * kH2;
    q.resize(qSize);
    k.resize(kvSize);
    v.resize(kvSize);
    dy.resize(qSize);
    for (int64_t i = 0; i < qSize; ++i) {
        q[i] = GenValue(static_cast<uint64_t>(i) * 7 + 1);
        dy[i] = GenValue(static_cast<uint64_t>(i) * 23 + 3);
    }
    for (int64_t i = 0; i < kvSize; ++i) {
        k[i] = GenValue(static_cast<uint64_t>(i) * 13 + 5);
        v[i] = GenValue(static_cast<uint64_t>(i) * 17 + 9);
    }
    ForwardStats stats;
    ComputeForward(q, k, v, stats);
    statsMax = stats.softmaxMax;
    statsSum = stats.softmaxSum;
    attentionIn = stats.attentionIn;

    const vector<int64_t> qShape = {kQSeqLen, kBatch, kH1};
    const vector<int64_t> kvShape = {kKvSeqLen, kBatch, kH2};
    const vector<int64_t> maskShape = {kQSeqLen, kKvSeqLen};
    const vector<int64_t> statShape = {kBatch, kQHeadNum, kQSeqLen, 8};

    vector<uint16_t> qHalf(q.size());
    vector<uint16_t> kHalf(k.size());
    vector<uint16_t> vHalf(v.size());
    vector<uint16_t> dyHalf(dy.size());
    vector<uint16_t> outHalf(attentionIn.size());
    for (size_t i = 0; i < q.size(); ++i) {
        qHalf[i] = FloatToHalf(q[i]);
        dyHalf[i] = FloatToHalf(dy[i]);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        kHalf[i] = FloatToHalf(k[i]);
        vHalf[i] = FloatToHalf(v[i]);
    }
    for (size_t i = 0; i < attentionIn.size(); ++i) {
        outHalf[i] = FloatToHalf(attentionIn[i]);
    }
    const vector<uint8_t> maskData(kQSeqLen * kKvSeqLen, 0);
    // softmax statistics are [B, N, S, 8]; every lane holds the same per-row value.
    vector<float> maxLanes;
    vector<float> sumLanes;
    maxLanes.reserve(statsMax.size() * 8);
    sumLanes.reserve(statsSum.size() * 8);
    for (size_t i = 0; i < statsMax.size(); ++i) {
        for (int64_t lane = 0; lane < 8; ++lane) {
            maxLanes.push_back(statsMax[i]);
            sumLanes.push_back(statsSum[i]);
        }
    }

    // Input order follows the IR definition: query, key, value, dy, atten_mask, softmax_max, softmax_sum,
    // attention_in, prefix, q_start_idx, kv_start_idx.
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT16, qHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT16, kHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "value", 3, kvShape, DT_FLOAT16, vHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "dy", 4, qShape, DT_FLOAT16, dyHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "atten_mask", 5, maskShape, DT_UINT8, maskData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "softmax_max", 6, statShape, DT_FLOAT, maxLanes) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "softmax_sum", 7, statShape, DT_FLOAT, sumLanes) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "attention_in", 8, qShape, DT_FLOAT16, outHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "prefix", 9, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "q_start_idx", 10, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "kv_start_idx", 11, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);

    auto node = op::FlashAttentionScoreGrad("flash_attention_score_grad");
    node.set_input_query(inputOps[0]);
    node.set_input_key(inputOps[1]);
    node.set_input_value(inputOps[2]);
    node.set_input_dy(inputOps[3]);
    node.set_input_atten_mask(inputOps[4]);
    node.set_input_softmax_max(inputOps[5]);
    node.set_input_softmax_sum(inputOps[6]);
    node.set_input_attention_in(inputOps[7]);
    node.set_input_prefix(inputOps[8]);
    node.set_input_q_start_idx(inputOps[9]);
    node.set_input_kv_start_idx(inputOps[10]);

    node.set_attr_scale_value(kScale);
    node.set_attr_keep_prob(1.0);
    node.set_attr_pre_tockens(65536);
    node.set_attr_next_tockens(65536);
    node.set_attr_head_num(kQHeadNum);
    node.set_attr_input_layout("SBH");
    node.set_attr_inner_precise(0);
    node.set_attr_sparse_mode(0);
    node.set_attr_pse_type(1);
    node.set_attr_out_dtype(1);
    node.set_attr_seed(0);
    node.set_attr_offset(0);

    TensorDesc dqDesc(Shape(qShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dkDesc(Shape(kvShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dvDesc(Shape(kvShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc emptyFp16Desc(Shape({0}), FORMAT_ND, DT_FLOAT16);
    TensorDesc emptyFp32Desc(Shape({0}), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_dq(dqDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dk(dkDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dv(dvDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dpse(emptyFp16Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dq_rope(emptyFp16Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dk_rope(emptyFp16Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dsink(emptyFp32Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

void ComputeBackward(const vector<float>& q, const vector<float>& k, const vector<float>& v, const vector<float>& dy,
                     const ForwardStats& stats, vector<float>& dq, vector<float>& dk, vector<float>& dv)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    dq.assign(s1 * d, 0.0f);
    dk.assign(s2 * d, 0.0f);
    dv.assign(s2 * d, 0.0f);
    for (int64_t i = 0; i < s1; ++i) {
        vector<float> prob(s2);
        for (int64_t j = 0; j < s2; ++j) {
            float dot = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dot += q[i * d + t] * k[j * d + t];
            }
            const float score = static_cast<float>(kScale) * dot;
            prob[j] = std::exp(score - stats.softmaxMax[i]) / stats.softmaxSum[i];
        }
        for (int64_t j = 0; j < s2; ++j) {
            float dP = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dP += dy[i * d + t] * v[j * d + t];
                dv[j * d + t] += prob[j] * dy[i * d + t];
            }
            float dYdotY = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dYdotY += dy[i * d + t] * stats.attentionIn[i * d + t];
            }
            const float dS = prob[j] * (dP - dYdotY);
            for (int64_t t = 0; t < d; ++t) {
                dq[i * d + t] += static_cast<float>(kScale) * dS * k[j * d + t];
                dk[j * d + t] += static_cast<float>(kScale) * dS * q[i * d + t];
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

int32_t ValidateOutputs(const vector<Tensor>& outputs, const vector<float>& q, const vector<float>& k,
                        const vector<float>& v, const vector<float>& dy, const ForwardStats& stats)
{
    CHECK_RET(outputs.size() == 7, LOG_PRINT("[CHECK] FAIL: expected 7 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> dq;
    vector<float> dk;
    vector<float> dv;
    ComputeBackward(q, k, v, dy, stats, dq, dk, dv);
    const vector<int64_t> qShape = {kQSeqLen, kBatch, kH1};
    const vector<int64_t> kvShape = {kKvSeqLen, kBatch, kH2};
    bool ok = true;
    ok = CheckOutput(outputs[0], "dq", qShape, dq) && ok;
    ok = CheckOutput(outputs[1], "dk", kvShape, dk) && ok;
    ok = CheckOutput(outputs[2], "dv", kvShape, dv) && ok;
    // dpse/dq_rope/dk_rope/dsink are optional outputs and stay empty because the matching inputs are not bound;
    // their content is not validated.
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

    Graph graph("flash_attention_score_grad_graph");
    vector<Tensor> inputTensors;
    vector<Operator> inputOps;
    vector<Operator> outputOps;
    vector<float> q;
    vector<float> k;
    vector<float> v;
    vector<float> dy;
    vector<float> statsMax;
    vector<float> statsSum;
    vector<float> attentionIn;
    int32_t ret = BuildGraph(graph, inputTensors, inputOps, outputOps, q, k, v, dy, statsMax, statsSum, attentionIn);
    if (ret == SUCCESS) {
        ForwardStats stats;
        stats.softmaxMax = statsMax;
        stats.softmaxSum = statsSum;
        stats.attentionIn = attentionIn;
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
        LOG_PRINT("FlashAttentionScoreGrad graph run success, output count: %zu\n", outputTensors.size());
        ret = ValidateOutputs(outputTensors, q, k, v, dy, stats);
        delete session;
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
