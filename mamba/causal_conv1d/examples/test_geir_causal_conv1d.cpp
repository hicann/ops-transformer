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
 * \file test_geir_causal_conv1d.cpp
 * \brief GEIR example for causal_conv1d operator.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>

#include "assert.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "ops_proto_legacy.h"
#include "../op_graph/causal_conv1d_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

const int B = 8;   // batch
const int S = 16;  // seqlen
const int D = 128; // dim
const int K = 4;   // kernel width

#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape) \
    do { \
        vector<int64_t> placeholder##intputIndex##_shape = inputShape; \
        auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0); \
        TensorDesc placeholder##intputIndex##_desc = \
            TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype); \
        placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost); \
        placeholder##intputIndex##_desc.SetFormat(FORMAT_ND); \
        Tensor tensor_placeholder##intputIndex; \
        ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex, \
                          placeholder##intputIndex##_desc, intputDtype, 2); \
        if (ret != SUCCESS) { \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str()); \
            return FAILED; \
        } \
        placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc); \
        input.push_back(tensor_placeholder##intputIndex); \
        graph.AddOp(placeholder##intputIndex); \
        causal_conv1d.set_input_##intputName(placeholder##intputIndex); \
        inputs.push_back(placeholder##intputIndex); \
    } while (0)

#define ADD_OPTIONAL_INPUT(intputIndex, intputName, intputDtype, inputShape) \
    do { \
        vector<int64_t> placeholder##intputIndex##_shape = inputShape; \
        auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0); \
        TensorDesc placeholder##intputIndex##_desc = \
            TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype); \
        placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost); \
        placeholder##intputIndex##_desc.SetFormat(FORMAT_ND); \
        Tensor tensor_placeholder##intputIndex; \
        ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex, \
                          placeholder##intputIndex##_desc, intputDtype, 2); \
        if (ret != SUCCESS) { \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str()); \
            return FAILED; \
        } \
        placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc); \
        input.push_back(tensor_placeholder##intputIndex); \
        graph.AddOp(placeholder##intputIndex); \
        causal_conv1d.set_input_##intputName(placeholder##intputIndex); \
        inputs.push_back(placeholder##intputIndex); \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        return fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        return twoByte;
    } else if (dt == ge::DT_BF16) {
        return twoByte;
    } else if (dt == ge::DT_INT16) {
        return twoByte;
    } else if (dt == ge::DT_UINT16) {
        return twoByte;
    } else if (dt == ge::DT_INT32) {
        return fourByte;
    } else if (dt == ge::DT_UINT32) {
        return fourByte;
    } else if (dt == ge::DT_INT64) {
        return eightByte;
    } else if (dt == ge::DT_UINT64) {
        return eightByte;
    } else if (dt == ge::DT_INT8) {
        return oneByte;
    }
    return oneByte;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& tensor, TensorDesc& tensor_desc, DataType data_type, int value)
{
    tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    int32_t* pData = new (std::nothrow) int32_t[data_len];
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    tensor = Tensor(tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                     Graph& graph)
{
    Status ret = SUCCESS;

    auto causal_conv1d = op::CausalConv1d("test_geir_causal_conv1d");

    // shapes
    vector<int64_t> x_shape = {B, S, D};
    vector<int64_t> weight_shape = {K, D};
    vector<int64_t> conv_states_shape = {B, K, D};
    vector<int64_t> bias_shape = {D};

    // required inputs
    ADD_INPUT(1, x, DT_BF16, x_shape);
    ADD_INPUT(2, weight, DT_BF16, weight_shape);
    ADD_INPUT(3, conv_states, DT_BF16, conv_states_shape);
    ADD_INPUT(4, bias, DT_BF16, bias_shape);

    // set outputs
    outputs.push_back(causal_conv1d);

    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: Start to add compute graph\n", GetTime().c_str());
    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    printf("%s - INFO - [XIR]: Session add graph success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    printf("%s - INFO - [XIR]: Start to run graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        std::cout << "GE error: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        printf("%s - INFO - [XIR]: Run graph failed (expected for custom op in standalone GEIR mode)\n",
               GetTime().c_str());
    } else {
        printf("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

        int input_num = input.size();
        for (int i = 0; i < input_num; i++) {
            std::cout << "input " << i << " dtype: " << input[i].GetTensorDesc().GetDataType() << std::endl;
            string input_file = "./tc_ge_irrun_test_input_" + std::to_string(i) + ".bin";
            uint8_t* input_data_i = input[i].GetData();
            int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
            std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
            uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
            WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
        }

        int output_num = output.size();
        for (int i = 0; i < output_num; i++) {
            std::cout << "output " << i << " dtype: " << output[i].GetTensorDesc().GetDataType() << std::endl;
            string output_file = "./tc_ge_irrun_test_output_" + std::to_string(i) + ".bin";
            uint8_t* output_data_i = output[i].GetData();
            int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
            std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
            uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
            WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        }
    }

    printf("%s - INFO - [XIR]: Start to finalize\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
