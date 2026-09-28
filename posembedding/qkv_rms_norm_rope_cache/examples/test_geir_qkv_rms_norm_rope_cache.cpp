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
 * \file test_geir_qkv_rms_norm_rope_cache.cpp
 * \brief QkvRmsNormRopeCache 的 GE 图模式(GEIR)调用样例
 *
 * A2/A3 侧一直没有该 example,arch35(Ascend950)补齐。
 * 走 GE 图编译 + 图执行:q_out/k_cache/v_cache 既是输入(原地 cache)也是输出,
 * index 由用户在真实场景给出;本例为通路连通性验证,数据用占位值。
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
#include "../op_graph/qkv_rms_norm_rope_cache_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// 场景:B=1,S=16,Nq=4,Nk=1,Nv=1,D=128;PA_NZ cache,BlockSize=32,BlockNum=1
const int64_t B_QKV = 1;
const int64_t S_QKV = 16;
const int64_t N_Q = 4;
const int64_t N_K = 1;
const int64_t N_V = 1;
const int64_t D_QKV = 128;
const int64_t N_QKV = N_Q + N_K + N_V;
const int64_t T_QKV = B_QKV * S_QKV;
const int64_t BLOCK_SIZE = 32;
const int64_t BLOCK_NUM = 1;
const int64_t D0 = 16; // 非量化 cache:32B / sizeof(float16)
const int64_t D1 = D_QKV / D0;

#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape) \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape; \
    auto placeholder##intputIndex = op::Data("placeholder" #intputIndex).set_attr_index(0); \
    TensorDesc placeholder##intputIndex##_desc = \
        TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype); \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost); \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND); \
    Tensor tensor_placeholder##intputIndex; \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex, \
                      placeholder##intputIndex##_desc, intputDtype, 2); \
    if (ret != SUCCESS) { \
        LOG_PRINT("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str()); \
        return FAILED; \
    } \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc); \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc); \
    input.push_back(tensor_placeholder##intputIndex); \
    graph.AddOp(placeholder##intputIndex); \
    qkv_rms_norm_rope_cache_op.set_input_##intputName(placeholder##intputIndex); \
    qkv_rms_norm_rope_cache_op.update_input_desc_##intputName(placeholder##intputIndex##_desc); \
    inputs.push_back(placeholder##intputIndex);

#define ADD_OUTPUT_ATTR(attrName, attrValue) qkv_rms_norm_rope_cache_op.set_attr_##attrName(attrValue)

#define ADD_OUTPUT(outputName, outputDtype, outputShape) \
    TensorDesc output_desc_##outputName = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    qkv_rms_norm_rope_cache_op.update_output_desc_##outputName(output_desc_##outputName);

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
    uint32_t dilation = 1;
    if (dt == ge::DT_FLOAT) {
        dilation = 4;
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        dilation = 2;
    } else if (dt == ge::DT_INT64) {
        dilation = 8;
    } else if (dt == ge::DT_INT32) {
        dilation = 4;
    } else if (dt == ge::DT_INT8) {
        dilation = 1;
    }
    return dilation;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, DataType data_type,
                    int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    // 按字节分配并零初始化:dtype 宽于 4 字节时(index 为 DT_INT64)尾段才是已初始化的
    uint8_t *pData = new (std::nothrow) uint8_t[data_len]();
    if (pData == nullptr) {
        return FAILED;
    }
    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    // Tensor 的 const uint8_t* 构造是深拷贝(带 deleter 的是另一个 SetData 重载),
    // 拷完即可释放,不必把这块 host 内存留到进程退出。
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(const char *file_path, size_t data_size, uint8_t *data)
{
    FILE *fp = fopen(file_path, "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(data, data_size, 1, fp);
    fclose(fp);
    return SUCCESS;
}

Status CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor> &input, std::vector<Operator> &inputs,
                        std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    auto qkv_rms_norm_rope_cache_op = op::QkvRmsNormRopeCache("test_geir_qkv_rms_norm_rope_cache");

    std::vector<int64_t> qkv_shape = {T_QKV, N_QKV * D_QKV};
    std::vector<int64_t> gamma_shape = {D_QKV};
    std::vector<int64_t> cosShape = {T_QKV, D_QKV};
    std::vector<int64_t> index_shape = {T_QKV};
    std::vector<int64_t> qOutShape = {T_QKV, N_Q * D_QKV};
    std::vector<int64_t> kCacheShape = {BLOCK_NUM, N_K * D1, BLOCK_SIZE, D0};
    std::vector<int64_t> vCacheShape = {BLOCK_NUM, N_V * D1, BLOCK_SIZE, D0};
    std::vector<int64_t> kProtoShape = {T_QKV, N_K * D_QKV};
    std::vector<int64_t> vProtoShape = {T_QKV, N_V * D_QKV};

    // 输入顺序严格匹配 op_graph/qkv_rms_norm_rope_cache_proto.h
    // 可选输入(k_scale/v_scale/k_offset/v_offset)在非量化档缺省,不加入图
    ADD_INPUT(1, qkv, inDtype, qkv_shape);
    ADD_INPUT(2, q_gamma, inDtype, gamma_shape);
    ADD_INPUT(3, k_gamma, inDtype, gamma_shape);
    ADD_INPUT(4, cos, inDtype, cosShape);
    ADD_INPUT(5, sin, inDtype, cosShape);
    ADD_INPUT(6, index, DT_INT64, index_shape);
    ADD_INPUT(7, q_out, inDtype, qOutShape);
    ADD_INPUT(8, k_cache, inDtype, kCacheShape);
    ADD_INPUT(9, v_cache, inDtype, vCacheShape);

    // 输出顺序严格匹配 proto:原地 cache + 3 个 before_quant
    ADD_OUTPUT(q_out, inDtype, qOutShape);
    ADD_OUTPUT(k_cache, inDtype, kCacheShape);
    ADD_OUTPUT(v_cache, inDtype, vCacheShape);
    ADD_OUTPUT(q_out_before_quant, inDtype, qOutShape);
    ADD_OUTPUT(k_out_before_quant, inDtype, kProtoShape);
    ADD_OUTPUT(v_out_before_quant, inDtype, vProtoShape);

    // 属性顺序严格匹配 proto
    std::vector<int64_t> qkvSize = {B_QKV, S_QKV, N_QKV, D_QKV};
    std::vector<int64_t> headNums = {N_Q, N_K, N_V};
    ADD_OUTPUT_ATTR(qkv_size, qkvSize);
    ADD_OUTPUT_ATTR(head_nums, headNums);
    ADD_OUTPUT_ATTR(epsilon, 1e-6f);
    ADD_OUTPUT_ATTR(cache_mode, "PA_NZ");
    ADD_OUTPUT_ATTR(is_output_qkv, true);

    outputs.push_back(qkv_rms_norm_rope_cache_op);
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    const char *graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    LOG_PRINT("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    (void)argc;
    (void)argv;

    DataType inDtype = DT_FLOAT16;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session *session = new Session(build_options);
    if (session == nullptr) {
        LOG_PRINT("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Session add ir compute graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: Session run ir compute graph success, outputs = %zu\n", GetTime().c_str(),
              output.size());

    for (size_t i = 0; i < output.size(); i++) {
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        LOG_PRINT("output %zu dtype : %d shape_size = %ld\n", i,
                  static_cast<int>(output[i].GetTensorDesc().GetDataType()), output_shape);
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    LOG_PRINT("Error message: %s\n", error_msg.GetString());
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    LOG_PRINT("Warning message: %s\n", warning_msg.GetString());

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
