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
 * \file test_geir_flash_attention_score.cpp
 * \brief GE graph construction sample for FlashAttentionScore (SBH layout, fp32).
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

#include "../op_graph/flash_attention_score_proto.h"

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

// Deterministic pseudo-random value in [-0.5, 0.5), same value reused by the CPU reference.
float GenValue(uint64_t seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    return (static_cast<float>(seed & 0xFFFFU) / 65536.0f) - 0.5f;
}

template <typename T>
int32_t AddDataInput(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, const string& name,
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
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetTensorDesc failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = tensor.SetData(reinterpret_cast<const uint8_t*>(hostData.data()), hostData.size() * sizeof(T));
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetData failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = dataOp.update_input_desc_x(desc);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Data::update_input_desc_x failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = dataOp.update_output_desc_y(desc);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Data::update_output_desc_y failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = graph.AddOp(dataOp);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Graph::AddOp failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    inputTensors.push_back(tensor);
    inputOps.push_back(dataOp);
    return SUCCESS;
}

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps,
                   vector<float>& qData, vector<float>& kData, vector<float>& vData)
{
    auto node = op::FlashAttentionScore("flash_attention_score");
    const vector<int64_t> qShape = {kQSeqLen, kBatch, kH1};
    const vector<int64_t> kvShape = {kKvSeqLen, kBatch, kH2};
    const vector<int64_t> maskShape = {kQSeqLen, kKvSeqLen};
    const vector<int64_t> idxShape = {1};

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
    const vector<uint8_t> maskData(kQSeqLen * kKvSeqLen, 0); // 0: the position attends
    const vector<int64_t> prefixData = {0};
    const vector<int64_t> qStartIdxData = {0};
    const vector<int64_t> kvStartIdxData = {0};

    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT, qData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT, kData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "value", 3, kvShape, DT_FLOAT, vData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "atten_mask", 4, maskShape, DT_UINT8, maskData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "prefix", 5, idxShape, DT_INT64, prefixData) == SUCCESS,
              return FAILED);
    CHECK_RET(
        AddDataInput(graph, inputTensors, inputOps, "q_start_idx", 6, idxShape, DT_INT64, qStartIdxData) == SUCCESS,
        return FAILED);
    CHECK_RET(
        AddDataInput(graph, inputTensors, inputOps, "kv_start_idx", 7, idxShape, DT_INT64, kvStartIdxData) == SUCCESS,
        return FAILED);

    node.set_input_query(inputOps[0]);
    node.set_input_key(inputOps[1]);
    node.set_input_value(inputOps[2]);
    node.set_input_atten_mask(inputOps[3]);
    node.set_input_prefix(inputOps[4]);
    node.set_input_q_start_idx(inputOps[5]);
    node.set_input_kv_start_idx(inputOps[6]);
    node.set_attr_scale_value(kScale);
    node.set_attr_keep_prob(1.0);
    node.set_attr_pre_tockens(2147483647);
    node.set_attr_next_tockens(2147483647);
    node.set_attr_head_num(kQHeadNum);
    node.set_attr_input_layout("SBH");
    node.set_attr_inner_precise(0);
    node.set_attr_sparse_mode(0);
    node.set_attr_pse_type(1);
    node.set_attr_out_dtype(0);

    const vector<int64_t> statShape = {kBatch, kQHeadNum, kQSeqLen, 8};
    TensorDesc softmaxMaxDesc(Shape(statShape), FORMAT_ND, DT_FLOAT);
    TensorDesc softmaxSumDesc(Shape(statShape), FORMAT_ND, DT_FLOAT);
    TensorDesc softmaxOutDesc(Shape(qShape), FORMAT_ND, DT_FLOAT);
    TensorDesc attentionOutDesc(Shape(qShape), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_softmax_max(softmaxMaxDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_sum(softmaxSumDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_out(softmaxOutDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_attention_out(attentionOutDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

// CPU reference: softmax(scale * Q @ K^T) @ V with SBH layout, N1 == N2 == 1, B == 1.
void ComputeReference(const vector<float>& qData, const vector<float>& kData, const vector<float>& vData,
                      vector<float>& softmaxMax, vector<float>& softmaxSum, vector<float>& attentionOut)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    vector<float> score(s2);
    softmaxMax.assign(s1, 0.0f);
    softmaxSum.assign(s1, 0.0f);
    attentionOut.assign(s1 * d, 0.0f);
    for (int64_t i = 0; i < s1; ++i) {
        for (int64_t j = 0; j < s2; ++j) {
            float dot = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dot += qData[i * d + t] * kData[j * d + t];
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
        softmaxMax[i] = maxScore;
        softmaxSum[i] = sum;
        for (int64_t j = 0; j < s2; ++j) {
            const float prob = score[j] / sum;
            for (int64_t t = 0; t < d; ++t) {
                attentionOut[i * d + t] += prob * vData[j * d + t];
            }
        }
    }
}

bool CheckOutput(const Tensor& tensor, const string& name, const vector<int64_t>& shape, const vector<float>& expected,
                 float rtol, float atol)
{
    const auto desc = tensor.GetTensorDesc();
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    CHECK_RET(desc.GetShape().GetDims() == shape && desc.GetDataType() == DT_FLOAT &&
                  tensor.GetSize() == count * sizeof(float),
              LOG_PRINT("[CHECK] %s FAIL: unexpected shape, dtype, or data size\n", name.c_str());
              return false);
    vector<float> values(count);
    std::memcpy(values.data(), tensor.GetData(), count * sizeof(float));
    size_t mismatches = 0;
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float tolerance = atol + rtol * std::fabs(expected[i]);
        const float err = std::fabs(values[i] - expected[i]);
        maxAbsErr = std::fmax(maxAbsErr, err);
        if (err > tolerance) {
            if (mismatches < 4) {
                LOG_PRINT("[CHECK] %s[%zu]=%.7f expected=%.7f tolerance=%.7f\n", name.c_str(), i, values[i],
                          expected[i], tolerance);
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
    CHECK_RET(outputs.size() == 4, LOG_PRINT("[CHECK] FAIL: expected 4 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> softmaxMax;
    vector<float> softmaxSum;
    vector<float> attentionOut;
    ComputeReference(qData, kData, vData, softmaxMax, softmaxSum, attentionOut);
    // softmax_max/softmax_sum are stored as [B, N, S, 8]; all 8 lanes hold the same per-row statistic.
    const int64_t s1 = kQSeqLen;
    vector<float> maxExpanded;
    vector<float> sumExpanded;
    maxExpanded.reserve(softmaxMax.size() * 8);
    sumExpanded.reserve(softmaxSum.size() * 8);
    for (size_t i = 0; i < softmaxMax.size(); ++i) {
        for (int64_t lane = 0; lane < 8; ++lane) {
            maxExpanded.push_back(softmaxMax[i]);
            sumExpanded.push_back(softmaxSum[i]);
        }
    }
    bool ok = true;
    ok = CheckOutput(outputs[0], "softmax_max", {kBatch, kQHeadNum, kQSeqLen, 8}, maxExpanded, 1e-3f, 1e-3f) && ok;
    ok = CheckOutput(outputs[1], "softmax_sum", {kBatch, kQHeadNum, kQSeqLen, 8}, sumExpanded, 1e-3f, 1e-3f) && ok;
    ok = CheckOutput(outputs[3], "attention_out", {kQSeqLen, kBatch, kH1}, attentionOut, 1e-3f, 1e-3f) && ok;
    // softmax_out is a reserved output that the kernel does not write; its content is not validated.
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

    Graph graph("flash_attention_score_graph");
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
        LOG_PRINT("FlashAttentionScore graph run success, output count: %zu\n", outputTensors.size());
        ret = ValidateOutputs(outputTensors, qData, kData, vData);
        delete session;
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
