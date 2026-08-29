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

#include "../op_graph/mhc_post_proto.h"

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

struct TestCase {
    DataType xDtype;
    const char *name;
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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

int GenTensorData(const std::vector<int64_t> &shape, Tensor &tensor, TensorDesc &desc, DataType dtype)
{
    desc.SetRealDimCnt(shape.size());
    const int64_t elem_num = GetShapeSize(shape);
    const uint32_t dtype_size = GetDataTypeSize(dtype);
    const uint32_t byte_len = static_cast<uint32_t>(elem_num * dtype_size);
    uint8_t *data = new (std::nothrow) uint8_t[byte_len];
    if (data == nullptr) {
        return FAILED;
    }

    if (dtype == ge::DT_FLOAT) {
        auto *float_data = reinterpret_cast<float *>(data);
        for (int64_t i = 0; i < elem_num; ++i) {
            float_data[i] = 0.01f * static_cast<float>((i % 17) - 8);
        }
    } else {
        auto *half_data = reinterpret_cast<uint16_t *>(data);
        for (int64_t i = 0; i < elem_num; ++i) {
            half_data[i] = static_cast<uint16_t>(0x3f80 + (i % 7));
        }
    }

    tensor = Tensor(desc, data, byte_len);
    return SUCCESS;
}

int AddDataInput(const std::string &name, uint32_t index, DataType dtype, const std::vector<int64_t> &shape,
                 Graph &graph, std::vector<Tensor> &input_tensors, std::vector<Operator> &input_ops,
                 ge::op::MhcPost &op)
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
    } else if (name == "h_res") {
        op.set_input_h_res(data);
    } else if (name == "h_out") {
        op.set_input_h_out(data);
    } else if (name == "h_post") {
        op.set_input_h_post(data);
    }
    return SUCCESS;
}

int CreateGraph(std::vector<Tensor> &input_tensors, std::vector<Operator> &input_ops, std::vector<Operator> &outputs,
                Graph &graph, const TestCase &testCase)
{
    auto mhc_post = ge::op::MhcPost(testCase.name);

    if (AddDataInput("x", 0, testCase.xDtype, {BATCH, SEQ_LEN, HC_MULT, HIDDEN_DIM}, graph, input_tensors, input_ops,
                     mhc_post) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput("h_res", 1, ge::DT_FLOAT, {BATCH, SEQ_LEN, HC_MULT, HC_MULT}, graph, input_tensors, input_ops,
                     mhc_post) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput("h_out", 2, testCase.xDtype, {BATCH, SEQ_LEN, HIDDEN_DIM}, graph, input_tensors, input_ops,
                     mhc_post) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput("h_post", 3, ge::DT_FLOAT, {BATCH, SEQ_LEN, HC_MULT}, graph, input_tensors, input_ops, mhc_post) !=
        SUCCESS) {
        return FAILED;
    }

    mhc_post.update_output_desc_y(
        TensorDesc(ge::Shape({BATCH, SEQ_LEN, HC_MULT, HIDDEN_DIM}), FORMAT_ND, testCase.xDtype));

    outputs.push_back(mhc_post);
    return SUCCESS;
}

int RunCase(Session &session, uint32_t graphId, const TestCase &testCase)
{
    Graph graph(testCase.name);
    std::vector<Tensor> input_tensors;
    std::vector<Operator> input_ops;
    std::vector<Operator> output_ops;

    printf("%s - INFO - [XIR]: Run case %s, x dtype: %d\n", GetTime().c_str(), testCase.name,
           static_cast<int>(testCase.xDtype));
    Status ret = CreateGraph(input_tensors, input_ops, output_ops, graph, testCase);
    if (ret != SUCCESS) {
        return FAILED;
    }
    graph.SetInputs(input_ops).SetOutputs(output_ops);

    std::map<AscendString, AscendString> graph_options = {};
    ret = session.AddGraph(graphId, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed for case %s\n", GetTime().c_str(), testCase.name);
        return FAILED;
    }

    std::vector<Tensor> output_tensors;
    ret = session.RunGraph(graphId, input_tensors, output_tensors);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed for case %s\n", GetTime().c_str(), testCase.name);
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
        std::cout << "Error message: " << error_msg.GetString() << std::endl;
        std::cout << "Warning message: " << warning_msg.GetString() << std::endl;
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Run graph success for case %s, output num: %zu\n", GetTime().c_str(), testCase.name,
           output_tensors.size());
    if (output_tensors.size() != 1) {
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

    std::vector<TestCase> testCases = {
        {ge::DT_BF16, "mhc_post_bf16"},
        {ge::DT_FLOAT16, "mhc_post_fp16"},
    };

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
