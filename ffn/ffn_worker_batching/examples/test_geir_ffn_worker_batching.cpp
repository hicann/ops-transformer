/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <array>
#include <cstddef>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cstring>
#include "acl/acl.h"
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
#include "../op_graph/ffn_worker_batching_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape) \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    ffn_worker_batching_op.update_output_desc_##outputName(outputName##outputIndex##_desc);

#define ADD_INPUT_ATTR(attrName, attrValue) ffn_worker_batching_op.set_attr_##attrName(attrValue);

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
    }
    return dilation;
}

#pragma pack(push, 1)
struct ScheduleContext {
    struct CommonArea {
        uint32_t session_num;
        uint32_t micro_batch_num;
        uint32_t micro_batch_size;
        uint32_t selected_expert_num;
        uint32_t expert_num;
        uint32_t attn_to_ffn_token_size;
        uint32_t ffn_to_attn_token_size;
        int32_t schedule_mode;
        int8_t reserve0[96];
    };
    struct ControlArea {
        int32_t run_flag;
        int8_t reserve2[124];
    };
    struct FfnArea {
        uint64_t token_info_buf;
        uint64_t token_info_buf_size;
        uint64_t token_data_buf;
        uint64_t token_data_buf_size;
        uint64_t polling_index;
        int8_t reserve3[88];
        uint64_t layer_ids_buf;
        uint64_t layer_ids_buf_size;
        uint64_t session_ids_buf;
        uint64_t session_ids_buf_size;
        uint64_t micro_batch_ids_buf;
        uint64_t micro_batch_ids_buf_size;
        uint64_t expert_ids_buf;
        uint64_t expert_ids_buf_size;
        uint32_t out_num;
        int8_t reserve4[60];
    };
    struct AttentionArea {
        uint64_t token_info_buf;
        uint64_t token_info_buf_size;
        uint64_t token_data_buf;
        uint64_t token_data_buf_size;
        uint32_t micro_batch_id;
        int8_t reserve5[92];
    };
    CommonArea common;
    ControlArea control;
    AttentionArea attention;
    FfnArea ffn;
    int8_t reserve6[384];
};
static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext must occupy 1024 bytes");
static_assert(offsetof(ScheduleContext, ffn) == 384, "FfnArea must use the schedule-context offset");
#pragma pack(pop)

struct ExampleResources {
    std::array<uint8_t, sizeof(ScheduleContext)> schedule_context{};
    std::vector<void *> device_buffers;
};

int32_t CopyToDevice(const void *host_data, size_t data_size, void **device_data, ExampleResources &resources)
{
    aclError acl_ret = aclrtMalloc(device_data, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
        printf("aclrtMalloc failed. ERROR: %d\n", acl_ret);
        return FAILED;
    }
    acl_ret = aclrtMemcpy(*device_data, data_size, host_data, data_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (acl_ret != ACL_SUCCESS) {
        printf("aclrtMemcpy failed. ERROR: %d\n", acl_ret);
        aclrtFree(*device_data);
        *device_data = nullptr;
        return FAILED;
    }
    resources.device_buffers.push_back(*device_data);
    return SUCCESS;
}

void ReleaseDeviceBuffers(ExampleResources &resources)
{
    for (void *device_buffer : resources.device_buffers) {
        if (device_buffer != nullptr) {
            aclrtFree(device_buffer);
        }
    }
    resources.device_buffers.clear();
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t *inputData)
{
    FILE *fp;
    fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(std::vector<ge::Tensor> &input, std::vector<Operator> &inputs, std::vector<Operator> &outputs,
                     Graph &graph, ExampleResources &resources)
{
    auto ffn_worker_batching_op = op::FfnWorkerBatching("ffn_worker_batching");

    constexpr int64_t session_num = 1;
    constexpr int64_t micro_batch_num = 1;
    constexpr int64_t micro_batch_size = 8;
    constexpr int64_t selected_expert_num = 9;
    constexpr int64_t expert_num = 8;
    constexpr int64_t hidden_size = 4096;
    constexpr int64_t token_num = session_num * micro_batch_size * selected_expert_num;

    std::vector<int64_t> schedule_context_shape = {1024};
    std::vector<int64_t> y_shape = {token_num, hidden_size};
    std::vector<int64_t> group_list_shape = {expert_num, 2};
    std::vector<int64_t> session_ids_shape = {token_num};
    std::vector<int64_t> micro_batch_ids_shape = {token_num};
    std::vector<int64_t> token_ids_shape = {token_num};
    std::vector<int64_t> expert_offsets_shape = {token_num};
    std::vector<int64_t> dynamic_scale_shape = {token_num};
    std::vector<int64_t> actual_token_num_shape = {1};

    std::vector<uint16_t> token_data(token_num * hidden_size, 0x3c00);
    std::vector<int32_t> expert_ids(token_num);
    for (int64_t i = 0; i < token_num; ++i) {
        expert_ids[i] = static_cast<int32_t>(i % expert_num);
    }
    std::vector<int32_t> session_ids(session_num, 0);
    std::vector<int32_t> micro_batch_ids(session_num, 0);

    void *token_data_device = nullptr;
    void *expert_ids_device = nullptr;
    void *session_ids_device = nullptr;
    void *micro_batch_ids_device = nullptr;
    if (CopyToDevice(token_data.data(), token_data.size() * sizeof(uint16_t), &token_data_device, resources) !=
            SUCCESS ||
        CopyToDevice(expert_ids.data(), expert_ids.size() * sizeof(int32_t), &expert_ids_device, resources) !=
            SUCCESS ||
        CopyToDevice(session_ids.data(), session_ids.size() * sizeof(int32_t), &session_ids_device, resources) !=
            SUCCESS ||
        CopyToDevice(micro_batch_ids.data(), micro_batch_ids.size() * sizeof(int32_t), &micro_batch_ids_device,
                     resources) != SUCCESS) {
        ReleaseDeviceBuffers(resources);
        return FAILED;
    }

    ScheduleContext schedule_context = {};
    schedule_context.common.session_num = session_num;
    schedule_context.common.micro_batch_num = micro_batch_num;
    schedule_context.common.micro_batch_size = micro_batch_size;
    schedule_context.common.selected_expert_num = selected_expert_num;
    schedule_context.common.expert_num = expert_num;
    schedule_context.common.attn_to_ffn_token_size = hidden_size * sizeof(uint16_t);
    schedule_context.common.ffn_to_attn_token_size = hidden_size * sizeof(uint16_t);
    schedule_context.common.schedule_mode = 0;
    schedule_context.control.run_flag = 1;
    schedule_context.ffn.token_data_buf = reinterpret_cast<uint64_t>(token_data_device);
    schedule_context.ffn.token_data_buf_size = token_data.size() * sizeof(uint16_t);
    schedule_context.ffn.session_ids_buf = reinterpret_cast<uint64_t>(session_ids_device);
    schedule_context.ffn.session_ids_buf_size = session_ids.size() * sizeof(int32_t);
    schedule_context.ffn.micro_batch_ids_buf = reinterpret_cast<uint64_t>(micro_batch_ids_device);
    schedule_context.ffn.micro_batch_ids_buf_size = micro_batch_ids.size() * sizeof(int32_t);
    schedule_context.ffn.expert_ids_buf = reinterpret_cast<uint64_t>(expert_ids_device);
    schedule_context.ffn.expert_ids_buf_size = expert_ids.size() * sizeof(int32_t);
    schedule_context.ffn.out_num = session_num;
    std::memcpy(resources.schedule_context.data(), &schedule_context, sizeof(schedule_context));

    auto schedule_context_data = op::Data("schedule_context").set_attr_index(0);
    TensorDesc schedule_context_desc(ge::Shape(schedule_context_shape), FORMAT_ND, DT_INT8);
    schedule_context_desc.SetPlacement(ge::kPlacementHost);
    schedule_context_desc.SetFormat(FORMAT_ND);
    schedule_context_desc.SetRealDimCnt(schedule_context_shape.size());
    schedule_context_data.update_input_desc_x(schedule_context_desc);
    Tensor schedule_context_tensor(schedule_context_desc, resources.schedule_context.data(),
                                   resources.schedule_context.size());
    input.push_back(schedule_context_tensor);
    graph.AddOp(schedule_context_data);
    ffn_worker_batching_op.set_input_schedule_context(schedule_context_data);
    inputs.push_back(schedule_context_data);

    ADD_OUTPUT(1, y, DT_FLOAT16, y_shape);
    ADD_OUTPUT(2, group_list, DT_INT64, group_list_shape);
    ADD_OUTPUT(3, session_ids, DT_INT32, session_ids_shape);
    ADD_OUTPUT(4, micro_batch_ids, DT_INT32, micro_batch_ids_shape);
    ADD_OUTPUT(5, token_ids, DT_INT32, token_ids_shape);
    ADD_OUTPUT(6, expert_offsets, DT_INT32, expert_offsets_shape);
    ADD_OUTPUT(7, dynamic_scale, DT_FLOAT, dynamic_scale_shape);
    ADD_OUTPUT(8, actual_token_num, DT_INT64, actual_token_num_shape);

    std::vector<int64_t> max_out_shape_value = {session_num, micro_batch_size, selected_expert_num, hidden_size};

    ADD_INPUT_ATTR(expert_num, expert_num);
    ADD_INPUT_ATTR(max_out_shape, max_out_shape_value);
    ADD_INPUT_ATTR(token_dtype, 0);
    ADD_INPUT_ATTR(need_schedule, 0);
    ADD_INPUT_ATTR(layer_num, 1);

    outputs.push_back(ffn_worker_batching_op);
    // 添加完毕
    return SUCCESS;
}

int main()
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
    ExampleResources resources{};

    ret = CreateOppInGraph(input, inputs, outputs, graph, resources);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        ReleaseDeviceBuffers(resources);
        GEFinalize();
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
        ReleaseDeviceBuffers(resources);
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add compute graph to ir session failed\n", GetTime().c_str());
        delete session;
        ReleaseDeviceBuffers(resources);
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        std::cout << "Error message: " << error_msg.GetString() << std::endl;
        delete session;
        ReleaseDeviceBuffers(resources);
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
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ReleaseDeviceBuffers(resources);
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
