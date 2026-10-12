/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "ops_proto_legacy.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/inplace_partial_rotary_mul_grad_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kSeqLen = 1;
constexpr int64_t kNumHeads = 1;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kSliceStart = 0;
constexpr int64_t kSliceEnd = 128;
constexpr int64_t kRotaryMode = 1;

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
    if (dt == DT_FLOAT || dt == DT_INT32 || dt == DT_UINT32) {
        return 4;
    }
    if (dt == DT_FLOAT16 || dt == DT_BF16 || dt == DT_INT16 || dt == DT_UINT16) {
        return 2;
    }
    if (dt == DT_INT64 || dt == DT_UINT64) {
        return 8;
    }
    return 1;
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
int32_t GenTensorData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, T value)
{
    desc.SetRealDimCnt(shape.size());
    const int64_t elemNum = GetShapeSize(shape);
    const uint64_t dataSize = static_cast<uint64_t>(elemNum) * sizeof(T);
    T* data = new (std::nothrow) T[elemNum];
    if (data == nullptr) {
        return FAILED;
    }
    for (int64_t i = 0; i < elemNum; ++i) {
        data[i] = value;
    }
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(data), dataSize);
    return SUCCESS;
}

int32_t GenInputData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, DataType dataType, float value)
{
    if (dataType == DT_FLOAT) {
        return GenTensorData<float>(shape, tensor, desc, value);
    }
    if (dataType == DT_FLOAT16 || dataType == DT_BF16) {
        return GenTensorData<uint16_t>(shape, tensor, desc, static_cast<uint16_t>(value));
    }
    return FAILED;
}

int32_t WriteDataToFile(const string& binFile, uint64_t dataSize, uint8_t* inputData)
{
    FILE* fp = fopen(binFile.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), dataSize, fp);
    fclose(fp);
    return SUCCESS;
}

int32_t CreateInputTensor(const vector<int64_t>& shape, DataType dtype, float value, TensorDesc& desc, Tensor& tensor)
{
    desc = TensorDesc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);

    auto ret = GenInputData(shape, tensor, desc, dtype, value);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}

int32_t ParseDeviceId(int argc, char* argv[])
{
    // Prefer an explicit command-line device id, then fall back to ASCEND_DEVICE_ID.
    const char* deviceIdText = nullptr;
    if (argc > 1) {
        deviceIdText = argv[1];
    } else {
        deviceIdText = std::getenv("ASCEND_DEVICE_ID");
    }
    if (deviceIdText == nullptr || deviceIdText[0] == '\0') {
        return 0;
    }

    char* endPtr = nullptr;
    long deviceId = std::strtol(deviceIdText, &endPtr, 10);
    if (endPtr == deviceIdText || *endPtr != '\0' || deviceId < 0) {
        return 0;
    }
    return static_cast<int32_t>(deviceId);
}

int CreateOppInGraph(DataType inDtype, vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs,
                     Graph& graph)
{
    // Build one GEIR custom-op node. Input and output port names match inplace_partial_rotary_mul_grad_proto.h.
    auto inplacePartialRotaryMulGradOp = op::InplacePartialRotaryMulGrad("test_geir_inplace_partial_rotary_mul_grad");

    vector<int64_t> dyShape = {kBatch, kSeqLen, kNumHeads, kHeadDim};
    vector<int64_t> cosShape = {kBatch, kSeqLen, kNumHeads, kSliceEnd - kSliceStart};
    vector<int64_t> sinShape = {kBatch, kSeqLen, kNumHeads, kSliceEnd - kSliceStart};

    TensorDesc dyDesc;
    TensorDesc cosDesc;
    TensorDesc sinDesc;
    Tensor dyTensor;
    Tensor cosTensor;
    Tensor sinTensor;

    Status ret = CreateInputTensor(dyShape, inDtype, 2.0f, dyDesc, dyTensor);
    if (ret != SUCCESS) {
        return FAILED;
    }
    ret = CreateInputTensor(cosShape, inDtype, 1.0f, cosDesc, cosTensor);
    if (ret != SUCCESS) {
        return FAILED;
    }
    ret = CreateInputTensor(sinShape, inDtype, 0.0f, sinDesc, sinTensor);
    if (ret != SUCCESS) {
        return FAILED;
    }

    // Register runtime graph inputs in the same order as the input tensor vector passed to RunGraph.
    auto dyData = op::Data("dy").set_attr_index(0);
    dyData.update_input_desc_x(dyDesc);
    dyData.update_output_desc_y(dyDesc);
    input.push_back(dyTensor);
    graph.AddOp(dyData);
    inputs.push_back(dyData);

    auto cosData = op::Data("cos").set_attr_index(1);
    cosData.update_input_desc_x(cosDesc);
    cosData.update_output_desc_y(cosDesc);
    input.push_back(cosTensor);
    graph.AddOp(cosData);
    inputs.push_back(cosData);

    auto sinData = op::Data("sin").set_attr_index(2);
    sinData.update_input_desc_x(sinDesc);
    sinData.update_output_desc_y(sinDesc);
    input.push_back(sinTensor);
    graph.AddOp(sinData);
    inputs.push_back(sinData);

    inplacePartialRotaryMulGradOp.set_input_dy(dyData);
    inplacePartialRotaryMulGradOp.set_input_cos(cosData);
    inplacePartialRotaryMulGradOp.set_input_sin(sinData);
    inplacePartialRotaryMulGradOp.update_input_desc_dy(dyDesc);
    inplacePartialRotaryMulGradOp.update_input_desc_cos(cosDesc);
    inplacePartialRotaryMulGradOp.update_input_desc_sin(sinDesc);

    vector<int64_t> partialSlice = {kSliceStart, kSliceEnd};
    inplacePartialRotaryMulGradOp.set_attr_rotary_mode(kRotaryMode);
    inplacePartialRotaryMulGradOp.set_attr_partial_slice(partialSlice);

    // InplacePartialRotaryMulGrad writes the result back to dy; GEIR still exposes dy as the graph output.
    inplacePartialRotaryMulGradOp.update_output_desc_dy(dyDesc);
    outputs.push_back(inplacePartialRotaryMulGradOp);
    return SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    // bash build.sh --run_example inplace_partial_rotary_mul_grad graph
    const char* graphName = "tc_ge_irrun_test";
    Graph graph(graphName);
    vector<Tensor> input;
    int32_t deviceId = ParseDeviceId(argc, argv);
    string deviceIdStr = std::to_string(deviceId);

    printf("%s - INFO - [XIR]: Start to initialize ge on device %d using ge global options\n", GetTime().c_str(),
           deviceId);
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", deviceIdStr.c_str()},
                                                     {"ge.graphRunMode", "1"}};
    Status ret = GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed. ERROR: %d\n", GetTime().c_str(), ret);
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    vector<Operator> inputs;
    vector<Operator> outputs;
    DataType inDtype = DT_FLOAT;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create op in graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> buildOptions;
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    map<AscendString, AscendString> graphOptions;
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    string dumpPath = "./dump_inplace_partial_rotary_mul_grad";
    aclgrphDumpGraph(graph, dumpPath.c_str(), dumpPath.length());

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    vector<Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    // Dump input/output binaries for quick inspection after standalone execution.
    for (size_t i = 0; i < input.size(); ++i) {
        string inputFile = "./tc_ge_irrun_test_inplace_partial_rotary_mul_grad_input_" + std::to_string(i) + ".bin";
        uint64_t dataSize = static_cast<uint64_t>(input[i].GetTensorDesc().GetShape().GetShapeSize()) *
                            GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile(inputFile, dataSize, input[i].GetData());
    }

    for (size_t i = 0; i < output.size(); ++i) {
        string outputFile = "./tc_ge_irrun_test_inplace_partial_rotary_mul_grad_output_" + std::to_string(i) + ".bin";
        uint64_t dataSize = static_cast<uint64_t>(output[i].GetTensorDesc().GetShape().GetShapeSize()) *
                            GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(outputFile, dataSize, output[i].GetData());

        if (output[i].GetTensorDesc().GetDataType() == DT_FLOAT) {
            const float* resultData = reinterpret_cast<const float*>(output[i].GetData());
            const int64_t elemNum = output[i].GetTensorDesc().GetShape().GetShapeSize();
            for (int64_t j = 0; j < elemNum && j < 16; ++j) {
                printf("result[%ld] is: %f\n", j, resultData[j]);
            }
        }
    }

    AscendString errorMsg = GEGetErrorMsgV2();
    std::cout << "Error message: " << errorMsg.GetString() << std::endl;
    AscendString warningMsg = GEGetWarningMsgV2();
    std::cout << "Warning message: " << warningMsg.GetString() << std::endl;

    delete session;
    ret = GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
