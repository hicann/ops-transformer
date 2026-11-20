/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ctime>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/mhc_pre_sinkhorn_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using ge::AscendString;
using ge::DataType;
using ge::FORMAT_ND;
using ge::Graph;
using ge::Operator;
using ge::Session;
using ge::Status;
using ge::Tensor;
using ge::TensorDesc;

namespace {
constexpr int64_t BATCH = 1;
constexpr int64_t SEQ_LEN = 4;
constexpr int64_t HC_MULT = 4;
constexpr int64_t HIDDEN_DIM = 4096;
constexpr int64_t HC_MIX = HC_MULT * HC_MULT + 2 * HC_MULT;
constexpr int64_t NUM_ITERS = 20;

struct TestCase {
    DataType xDtype;
    bool needBackward;
    bool hasPreMix;
    int hf32Mode; // 0: do not set enable_hf32, 1: true, 2: false
    std::string name;
};
} // namespace

std::string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return sizeof(float);
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return sizeof(uint16_t);
    }
    if (dtype == ge::DT_INT32 || dtype == ge::DT_UINT32) {
        return sizeof(uint32_t);
    }
    if (dtype == ge::DT_INT64 || dtype == ge::DT_UINT64) {
        return sizeof(uint64_t);
    }
    return sizeof(uint8_t);
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

int GenTensorData(const std::vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, DataType dtype)
{
    desc.SetRealDimCnt(shape.size());
    const int64_t elem_num = GetShapeSize(shape);
    const uint32_t dtype_size = GetDataTypeSize(dtype);
    const uint32_t byte_len = static_cast<uint32_t>(elem_num * dtype_size);
    uint8_t* data = new (std::nothrow) uint8_t[byte_len];
    if (data == nullptr) {
        return FAILED;
    }

    if (dtype == ge::DT_FLOAT) {
        auto* float_data = reinterpret_cast<float*>(data);
        for (int64_t i = 0; i < elem_num; ++i) {
            float_data[i] = 0.01f * static_cast<float>((i % 17) - 8);
        }
    } else {
        auto* half_data = reinterpret_cast<uint16_t*>(data);
        for (int64_t i = 0; i < elem_num; ++i) {
            half_data[i] = static_cast<uint16_t>(0x3f80 + (i % 7));
        }
    }

    tensor = Tensor(desc, data, byte_len);
    return SUCCESS;
}

int AddDataInput(const std::string& name, uint32_t index, DataType dtype, const std::vector<int64_t>& shape,
                 Graph& graph, std::vector<Tensor>& input_tensors, std::vector<Operator>& input_ops,
                 ge::op::MhcPreSinkhorn& op)
{
    auto data = ge::op::Data(name).set_attr_index(index);
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);

    Tensor tensor;
    if (GenTensorData(shape, tensor, desc, dtype) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input %s failed\n", GetTime().c_str(), name.c_str());
        return FAILED;
    }

    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    input_tensors.push_back(tensor);
    graph.AddOp(data);
    input_ops.push_back(data);

    if (name == "x") {
        op.set_input_x(data);
    } else if (name == "phi") {
        op.set_input_phi(data);
    } else if (name == "alpha") {
        op.set_input_alpha(data);
    } else if (name == "bias") {
        op.set_input_bias(data);
    } else if (name == "premix") {
        op.set_input_premix(data);
    }
    return SUCCESS;
}

void SetOutputDesc(ge::op::MhcPreSinkhorn& op, DataType xDtype, bool needBackward, bool hasPreMix)
{
    op.update_output_desc_hin(TensorDesc(ge::Shape({BATCH, SEQ_LEN, HIDDEN_DIM}), FORMAT_ND, xDtype));
    op.update_output_desc_h_post(TensorDesc(ge::Shape({BATCH, SEQ_LEN, HC_MULT}), FORMAT_ND, ge::DT_FLOAT));
    op.update_output_desc_h_res(TensorDesc(ge::Shape({BATCH, SEQ_LEN, HC_MULT * HC_MULT}), FORMAT_ND, ge::DT_FLOAT));

    if (needBackward || hasPreMix) {
        op.update_output_desc_h_pre(TensorDesc(ge::Shape({BATCH, SEQ_LEN, HC_MULT}), FORMAT_ND, ge::DT_FLOAT));
    } else {
        op.update_output_desc_h_pre(TensorDesc(ge::Shape({0}), FORMAT_ND, ge::DT_FLOAT));
    }
    if (needBackward) {
        op.update_output_desc_hc_before_norm(TensorDesc(ge::Shape({BATCH, SEQ_LEN, HC_MIX}), FORMAT_ND, ge::DT_FLOAT));
        op.update_output_desc_inv_rms(TensorDesc(ge::Shape({BATCH, SEQ_LEN, 1}), FORMAT_ND, ge::DT_FLOAT));
        op.update_output_desc_sum_out(
            TensorDesc(ge::Shape({NUM_ITERS * 2, BATCH, SEQ_LEN, HC_MULT}), FORMAT_ND, ge::DT_FLOAT));
        op.update_output_desc_norm_out(
            TensorDesc(ge::Shape({NUM_ITERS * 2, BATCH, SEQ_LEN, HC_MULT, HC_MULT}), FORMAT_ND, ge::DT_FLOAT));
    } else {
        op.update_output_desc_hc_before_norm(TensorDesc(ge::Shape({0}), FORMAT_ND, ge::DT_FLOAT));
        op.update_output_desc_inv_rms(TensorDesc(ge::Shape({0}), FORMAT_ND, ge::DT_FLOAT));
        op.update_output_desc_sum_out(TensorDesc(ge::Shape({0}), FORMAT_ND, ge::DT_FLOAT));
        op.update_output_desc_norm_out(TensorDesc(ge::Shape({0}), FORMAT_ND, ge::DT_FLOAT));
    }
}

int CreateGraph(std::vector<Tensor>& input_tensors, std::vector<Operator>& input_ops, std::vector<Operator>& outputs,
                Graph& graph, const TestCase& testCase)
{
    auto mhc_pre_sinkhorn = ge::op::MhcPreSinkhorn(testCase.name.c_str());

    if (AddDataInput("x", 0, testCase.xDtype, {BATCH, SEQ_LEN, HC_MULT, HIDDEN_DIM}, graph, input_tensors, input_ops,
                     mhc_pre_sinkhorn) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput("phi", 1, ge::DT_FLOAT, {HC_MIX, HC_MULT * HIDDEN_DIM}, graph, input_tensors, input_ops,
                     mhc_pre_sinkhorn) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput("alpha", 2, ge::DT_FLOAT, {3}, graph, input_tensors, input_ops, mhc_pre_sinkhorn) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput("bias", 3, ge::DT_FLOAT, {HC_MIX}, graph, input_tensors, input_ops, mhc_pre_sinkhorn) != SUCCESS) {
        return FAILED;
    }
    if (testCase.hasPreMix && AddDataInput("premix", 4, ge::DT_FLOAT, {BATCH, SEQ_LEN, HC_MULT}, graph, input_tensors,
                                           input_ops, mhc_pre_sinkhorn) != SUCCESS) {
        return FAILED;
    }

    mhc_pre_sinkhorn.set_attr_hc_mult(HC_MULT);
    mhc_pre_sinkhorn.set_attr_num_iters(NUM_ITERS);
    mhc_pre_sinkhorn.set_attr_hc_eps(1e-6f);
    mhc_pre_sinkhorn.set_attr_norm_eps(1e-6f);
    mhc_pre_sinkhorn.set_attr_need_backward(testCase.needBackward);
    if (testCase.hf32Mode == 1) {
        mhc_pre_sinkhorn.set_attr_enable_hf32(true);
    } else if (testCase.hf32Mode == 2) {
        mhc_pre_sinkhorn.set_attr_enable_hf32(false);
    }
    SetOutputDesc(mhc_pre_sinkhorn, testCase.xDtype, testCase.needBackward, testCase.hasPreMix);

    outputs.push_back(mhc_pre_sinkhorn);
    return SUCCESS;
}

bool CheckOutputs(const std::vector<Tensor>& outputs, const TestCase& testCase)
{
    if (outputs.size() != 8) {
        printf("%s - ERROR - [XIR]: output num is %zu, expected 8\n", GetTime().c_str(), outputs.size());
        return false;
    }
    const std::vector<int64_t> empty = {0};
    const std::vector<int64_t> hPre =
        (testCase.needBackward || testCase.hasPreMix) ? std::vector<int64_t>{BATCH, SEQ_LEN, HC_MULT} : empty;
    const std::vector<int64_t> backwardShape = {BATCH, SEQ_LEN, HC_MIX};
    const std::vector<int64_t> invRms = {BATCH, SEQ_LEN, 1};
    const std::vector<int64_t> sumOut = {NUM_ITERS * 2, BATCH, SEQ_LEN, HC_MULT};
    const std::vector<int64_t> normOut = {NUM_ITERS * 2, BATCH, SEQ_LEN, HC_MULT, HC_MULT};
    const std::vector<std::pair<const char*, std::vector<int64_t>>> expected = {
        {"hin", {BATCH, SEQ_LEN, HIDDEN_DIM}},
        {"h_post", {BATCH, SEQ_LEN, HC_MULT}},
        {"h_res", {BATCH, SEQ_LEN, HC_MULT * HC_MULT}},
        {"h_pre", hPre},
        {"hc_before_norm", testCase.needBackward ? backwardShape : empty},
        {"inv_rms", testCase.needBackward ? invRms : empty},
        {"sum_out", testCase.needBackward ? sumOut : empty},
        {"norm_out", testCase.needBackward ? normOut : empty},
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto shape = outputs[i].GetTensorDesc().GetShape().GetDims();
        if (shape != expected[i].second) {
            printf("%s - ERROR - [XIR]: %s shape mismatch\n", GetTime().c_str(), expected[i].first);
            return false;
        }
    }
    return true;
}

int RunCase(Session& session, uint32_t graphId, const TestCase& testCase)
{
    Graph graph(testCase.name.c_str());
    std::vector<Tensor> input_tensors;
    std::vector<Operator> input_ops;
    std::vector<Operator> output_ops;

    const char* hf32Desc = testCase.hf32Mode == 0 ? "omit" : (testCase.hf32Mode == 1 ? "true" : "false");
    printf("%s - INFO - [XIR]: Run case %s, x dtype: %d, need_backward: %d, preMix: %d, enable_hf32: %s\n",
           GetTime().c_str(), testCase.name.c_str(), static_cast<int>(testCase.xDtype), testCase.needBackward,
           testCase.hasPreMix, hf32Desc);
    Status ret = CreateGraph(input_tensors, input_ops, output_ops, graph, testCase);
    if (ret != SUCCESS) {
        return FAILED;
    }
    graph.SetInputs(input_ops).SetOutputs(output_ops);

    std::map<AscendString, AscendString> graph_options = {};
    ret = session.AddGraph(graphId, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed for case %s\n", GetTime().c_str(), testCase.name.c_str());
        return FAILED;
    }

    std::vector<Tensor> output_tensors;
    ret = session.RunGraph(graphId, input_tensors, output_tensors);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed for case %s\n", GetTime().c_str(), testCase.name.c_str());
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
        std::cout << "Error message: " << error_msg.GetString() << std::endl;
        std::cout << "Warning message: " << warning_msg.GetString() << std::endl;
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Run graph success for case %s, output num: %zu\n", GetTime().c_str(),
           testCase.name.c_str(), output_tensors.size());
    if (!CheckOutputs(output_tensors, testCase)) {
        return FAILED;
    }
    return SUCCESS;
}

int main()
{
    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> build_options = {};
    auto session = new (std::nothrow) Session(build_options);
    if (session == nullptr) {
        ge::GEFinalize();
        return FAILED;
    }

    std::vector<TestCase> testCases;
    const DataType dtypes[] = {ge::DT_BF16, ge::DT_FLOAT16};
    const char* hf32Names[] = {"omit", "true", "false"};
    for (auto dtype : dtypes) {
        const char* dtypeName = dtype == ge::DT_BF16 ? "bf16" : "fp16";
        for (int needBackward = 0; needBackward < 2; ++needBackward) {
            for (int hasPreMix = 0; hasPreMix < 2; ++hasPreMix) {
                for (int hf32Mode = 0; hf32Mode < 3; ++hf32Mode) {
                    std::string name = std::string("mhc_pre_sinkhorn_") + dtypeName + "_need_backward_" +
                                       (needBackward ? "true" : "false") + "_premix_" + (hasPreMix ? "true" : "false") +
                                       "_hf32_" + hf32Names[hf32Mode];
                    testCases.push_back({dtype, needBackward != 0, hasPreMix != 0, hf32Mode, name});
                }
            }
        }
    }

    for (size_t i = 0; i < testCases.size(); ++i) {
        ret = RunCase(*session, static_cast<uint32_t>(i), testCases[i]);
        if (ret != SUCCESS) {
            delete session;
            ge::GEFinalize();
            return FAILED;
        }
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
