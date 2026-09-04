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
 * \file test_geir_indexer_quant_cache.cpp
 * \brief GE IR (graph mode) RunGraph test case for IndexerQuantCache.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "nn_other.h"
#include "../op_graph/indexer_quant_cache_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// clang-format off
#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape)                                                    \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                     \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                           \
    TensorDesc placeholder##intputIndex##_desc =                                                                       \
        TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype);                               \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                  \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                              \
    Tensor tensor_placeholder##intputIndex;                                                                            \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                               \
                      placeholder##intputIndex##_desc, intputDtype, 2);                                                \
    if (ret != SUCCESS) {                                                                                              \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                                 \
        return FAILED;                                                                                                 \
    }                                                                                                                  \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                     \
    input.push_back(tensor_placeholder##intputIndex);                                                                  \
    graph.AddOp(placeholder##intputIndex);                                                                             \
    indexer_quant_cache_op.set_input_##intputName(placeholder##intputIndex);                                           \
    inputs.push_back(placeholder##intputIndex);

#define ADD_INPUT_ATTR(attrName, attrValue) indexer_quant_cache_op.set_attr_##attrName(attrValue);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                                  \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype);            \
    indexer_quant_cache_op.update_output_desc_##outputName(outputName##outputIndex##_desc);

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)
// clang-format on

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
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    } else if (dt == ge::DT_UINT8) {
        dilation = oneByte;
    } else if (dt == ge::DT_FLOAT8_E4M3FN) {
        dilation = oneByte;
    } else if (dt == ge::DT_FLOAT8_E5M2) {
        dilation = oneByte;
    } else if (dt == ge::DT_FLOAT8_E8M0) {
        dilation = oneByte;
    }
    return dilation;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t byteSizeFloat32 = 4;
    uint32_t data_len = size * byteSizeFloat32;
    float *pData = new (std::nothrow) float[size];

    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t *)pData, data_len);
    return SUCCESS;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, DataType data_type,
                    int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    int32_t *pData = new (std::nothrow) int32_t[data_len];
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(pData), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t *inputData)
{
    FILE *fp;
    fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor> &input, std::vector<Operator> &inputs,
                     std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto indexer_quant_cache_op = op::IndexerQuantCache("test_geir_indexer_quant_cache");

    // shape 定义（quant_mode=1, Normal block-wise quant）
    // x: [tokenNum, d]
    // cache: 4D [blockNum, blockSize, 1, headDim] (FLOAT8_E4M3FN), num_slots = blockNum*blockSize
    //        mode1 headDim=d=128 >= d
    // cache_scale: 4D [blockNum, blockSize, 1, scaleCol(=1)] (FLOAT)
    // slot_mapping: [tokenNum]
    const int64_t tokenNum = 1024;
    const int64_t d = 128;
    const int64_t blockNum = 128;
    const int64_t blockSizeDim = 16; // num_slots = 128*16 = 2048
    std::vector<int64_t> cacheShape = {blockNum, blockSizeDim, 1, d};
    std::vector<int64_t> cacheScaleShape = {blockNum, blockSizeDim, 1, 1};
    std::vector<int64_t> xShape = {tokenNum, d};
    std::vector<int64_t> slotMappingShape = {tokenNum};

    // 添加输入（顺序严格匹配 proto.h: cache, cache_scale, x, slot_mapping）
    ADD_INPUT(1, cache, DT_FLOAT8_E4M3FN, cacheShape);
    ADD_INPUT(2, cache_scale, DT_FLOAT, cacheScaleShape);
    ADD_INPUT(3, x, DT_FLOAT16, xShape);
    ADD_INPUT(4, slot_mapping, DT_INT32, slotMappingShape);

    // 添加输出（cache / cache_scale 原地更新）
    ADD_OUTPUT(1, cache, DT_FLOAT8_E4M3FN, cacheShape);
    ADD_OUTPUT(2, cache_scale, DT_FLOAT, cacheScaleShape);

    // 添加属性（quant_mode=1: Normal block-wise quant, scale 存为 float32）
    ADD_INPUT_ATTR(quant_mode, (int64_t)1);
    ADD_INPUT_ATTR(round_scale, true);
    ADD_INPUT_ATTR(x_scale, 1.0f);

    outputs.push_back(indexer_quant_cache_op);
    // 添加完毕
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    const char *graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT16;
    std::cout << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {

    };
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session *session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t *input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char *)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t *output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char *)output_file.c_str(), data_size, output_data_i);
        for (int64_t j = 0; j < 16 && j < (int64_t)data_size; j++) {
            LOG_PRINT("output[%d] byte[%ld] is: %u\n", i, j, (uint32_t)output_data_i[j]);
        }
    }
    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
