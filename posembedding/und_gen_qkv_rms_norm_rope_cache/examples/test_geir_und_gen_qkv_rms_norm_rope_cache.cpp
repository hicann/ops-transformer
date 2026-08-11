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
 * \file test_geir_und_gen_qkv_rms_norm_rope_cache.cpp
 * \brief 通过算子 IR 构图方式调用 UndGenQkvRmsNormRopeCache（仅 Ascend 950PR/950DT）。
 *
 * 用例规模：(Hq,Hk,Hv)=(8,1,1) 即 N=10、D=128，und_len=6 + gen_len=2 => T=8，
 * KV Cache = [Bn=2, Bs=8, Hk/Hv=1, D=128]（容量 16 >= T），cos_sin_cache 行数 64。
 *
 * 索引类输入必须填合法值，不能沿用「整块置零」的写法：
 *   - slot_mapping 取 0..T-1，取值须唯一且落在 [0, Bn*Bs)，重复 slot 在多核下结果不确定；
 *   - cat_indices  取 T-1..0（逆序），须落在 [0, T)，逆序可让输出 token 同时命中
 *     und 段与 gen 段，两套 RMSNorm 权重都被用到；
 *   - positions    取 (i*7+3) % max_pos，须落在 [0, max_pos)。
 * 这三者 Host 侧无法校验（见 README「约束说明」），构图时喂错只会得到脏结果或越界。
 *
 * 浮点输入填固定种子的伪随机值而非全零：权重全零会让 RMSNorm 恒输出 0，
 * 那样样例区分不出「算对了」和「什么都没算」。
 *
 * 编译（自定义算子包需已安装，A=$ASCEND_HOME_PATH）：
 *   g++ -std=c++17 -o test_geir_und_gen_qkv_rms_norm_rope_cache \
 *       test_geir_und_gen_qkv_rms_norm_rope_cache.cpp \
 *       -I$A/include -I$A/include/graph -I$A/include/ge -I$A/opp/built-in/op_graph/inc \
 *       -I$A/opp/vendors/custom_transformer/op_proto/inc \
 *       -L$A/lib64 -lgraph -lgraph_base -lge_runner -lge_compiler -lascendcl
 * 其中 libgraph_base 提供 Operator::UpdateOutputDesc、libge_compiler 提供 aclgrphDumpGraph，
 * 只链 libgraph/libge_runner 会在链接期报 undefined reference。
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
#include "../op_graph/und_gen_qkv_rms_norm_rope_cache_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

const int64_t UND_LEN = 6;
const int64_t GEN_LEN = 2;
const int64_t TOTAL_TOKENS = UND_LEN + GEN_LEN;
const int64_t NUM_HEADS_Q = 8;
const int64_t NUM_HEADS_K = 1;
const int64_t NUM_HEADS_V = 1;
const int64_t NUM_HEADS = NUM_HEADS_Q + NUM_HEADS_K + NUM_HEADS_V;
const int64_t HEAD_DIM = 128;
const int64_t BLOCK_NUM = 2;
const int64_t BLOCK_SIZE = 8;
const int64_t MAX_POS = 64;
const int64_t MROPE_AXIS_NUM = 3;

#define LOG_PRINT(message, ...)     \
    do {                            \
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
    switch (dt) {
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return 8;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2;
        default:
            return 1;
    }
}

int64_t ShapeSize(const vector<int64_t> &shapes)
{
    int64_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    return size;
}

/* 确定性伪随机序列（固定种子的 LCG），取值落在 [-1, 1)。
 * 不用 rand()/mt19937 是为了让同一份输入在任何 libstdc++ 版本上逐比特可复现，
 * 便于用回读的输入 bin 离线复算 golden 做数值比对。*/
class Lcg {
public:
    explicit Lcg(uint32_t seed) : state_(seed) {}
    float Next()
    {
        state_ = state_ * 1103515245u + 12345u;
        return static_cast<float>((state_ >> 16) & 0x7fffu) / 16383.5f - 1.0f;
    }

private:
    uint32_t state_;
};

/* float -> bf16（取高 16 位，就近偶数舍入），与 torch 的 .to(bfloat16) 一致。*/
uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

/* 浮点类输入填确定性伪随机值。全零输入会让 RMSNorm 恒输出 0，
 * 那样样例区分不出「算对了」和「什么都没算」，故此处必须填非平凡值。*/
int32_t GenFloatData(const vector<int64_t> &shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc,
    DataType data_type, uint32_t seed)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = static_cast<size_t>(ShapeSize(shapes));
    vector<uint8_t> data(size * GetDataTypeSize(data_type), 0);
    Lcg lcg(seed);
    if (data_type == DT_FLOAT) {
        float *values = reinterpret_cast<float *>(data.data());
        for (size_t i = 0; i < size; i++) {
            values[i] = lcg.Next();
        }
    } else {  // DT_BF16
        uint16_t *values = reinterpret_cast<uint16_t *>(data.data());
        for (size_t i = 0; i < size; i++) {
            values[i] = FloatToBf16(lcg.Next());
        }
    }
    input_tensor = Tensor(input_tensor_desc, data);
    return SUCCESS;
}

/* 索引类输入（DT_INT64）。缓冲区按「元素数 x 8 字节」分配并整体清零：
 * 若按元素数 x 4 字节分配（或不清零），尾半段就是未初始化堆内存，喂给算子会变成越界下标。
 *   kSlot 取 0..n-1     —— slot_mapping 要求唯一槽位，且须落在 [0, Bn*Bs)
 *   kCatReverse 取 n-1..0 —— cat_indices 逆序，使输出 token 同时命中 und 段与 gen 段
 *   kPosCycle 按 max_pos 取模 —— positions 须落在 [0, max_pos)
 * 统一走 vector<uint8_t> 重载，由 Tensor 内部拷贝，不涉及裸指针生命周期。*/
enum IndexFill { kSlot, kCatReverse, kPosCycle };

int32_t GenIndexData(const vector<int64_t> &shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc,
    IndexFill fill)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = static_cast<size_t>(ShapeSize(shapes));
    vector<uint8_t> data(size * sizeof(int64_t), 0);
    int64_t *values = reinterpret_cast<int64_t *>(data.data());
    for (size_t i = 0; i < size; i++) {
        int64_t idx = static_cast<int64_t>(i);
        if (fill == kSlot) {
            values[i] = idx;
        } else if (fill == kCatReverse) {
            values[i] = static_cast<int64_t>(size) - 1 - idx;
        } else {
            values[i] = (idx * 7 + 3) % MAX_POS;
        }
    }
    input_tensor = Tensor(input_tensor_desc, data);
    return SUCCESS;
}

int32_t WriteDataToFile(const string &bin_file, uint64_t data_size, uint8_t *inputData)
{
    FILE *fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

/* 建一个 Data 节点并接到算子的第 inputIdx 个输入上。
 * 注意 Data 的名字必须用 std::string 拼，写成 "placeholder" + idx 是指针算术不是字符串拼接。
 * fillKind：0 = 浮点伪随机（用 seed 区分各输入），1/2/3 = slot / cat 逆序 / positions 取模。*/
#define ADD_INPUT(inputIdx, inputName, inputDtype, inputShape, fillKind, seed)                       \
    do {                                                                                             \
        vector<int64_t> shape_##inputName = inputShape;                                              \
        auto data_##inputName = op::Data((string("data_") + #inputName).c_str()).set_attr_index(inputIdx); \
        TensorDesc desc_##inputName = TensorDesc(ge::Shape(shape_##inputName), FORMAT_ND, inputDtype); \
        desc_##inputName.SetPlacement(ge::kPlacementHost);                                           \
        desc_##inputName.SetFormat(FORMAT_ND);                                                       \
        Tensor tensor_##inputName;                                                                   \
        ret = ((fillKind) == 0)                                                                      \
            ? GenFloatData(shape_##inputName, tensor_##inputName, desc_##inputName, inputDtype, seed) \
            : GenIndexData(shape_##inputName, tensor_##inputName, desc_##inputName,                  \
                           static_cast<IndexFill>((fillKind) - 1));                                  \
        if (ret != SUCCESS) {                                                                        \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());           \
            return FAILED;                                                                           \
        }                                                                                            \
        data_##inputName.update_input_desc_x(desc_##inputName);                                      \
        data_##inputName.update_output_desc_y(desc_##inputName);                                     \
        input.push_back(tensor_##inputName);                                                         \
        graph.AddOp(data_##inputName);                                                               \
        op_und_gen.set_input_##inputName(data_##inputName);                                          \
        op_und_gen.update_input_desc_##inputName(desc_##inputName);                                  \
        inputs.push_back(data_##inputName);                                                          \
    } while (0)

#define ADD_OUTPUT(outputName, outputDtype, outputShape)                                             \
    do {                                                                                             \
        TensorDesc desc_out_##outputName = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
        op_und_gen.update_output_desc_##outputName(desc_out_##outputName);                           \
    } while (0)

int CreateOppInGraph(std::vector<ge::Tensor> &input, std::vector<Operator> &inputs,
    std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    auto op_und_gen = op::UndGenQkvRmsNormRopeCache("test_geir_und_gen_qkv_rms_norm_rope_cache");

    // shape 定义
    vector<int64_t> und_qkv_shape = {UND_LEN, NUM_HEADS, HEAD_DIM};
    vector<int64_t> gen_qkv_shape = {GEN_LEN, NUM_HEADS, HEAD_DIM};
    vector<int64_t> weights_shape = {HEAD_DIM};
    vector<int64_t> cos_sin_cache_shape = {MAX_POS, HEAD_DIM};
    vector<int64_t> k_cache_shape = {BLOCK_NUM, BLOCK_SIZE, NUM_HEADS_K, HEAD_DIM};
    vector<int64_t> v_cache_shape = {BLOCK_NUM, BLOCK_SIZE, NUM_HEADS_V, HEAD_DIM};
    vector<int64_t> slot_mapping_shape = {TOTAL_TOKENS};
    vector<int64_t> positions_shape = {MROPE_AXIS_NUM, TOTAL_TOKENS};
    vector<int64_t> cat_indices_shape = {TOTAL_TOKENS};
    vector<int64_t> q_shape = {TOTAL_TOKENS, NUM_HEADS_Q, HEAD_DIM};

    // 添加输入（顺序严格匹配 proto.h）
    ADD_INPUT(0, und_qkv, DT_BF16, und_qkv_shape, 0, 11);
    ADD_INPUT(1, und_weights_q, DT_BF16, weights_shape, 0, 22);
    ADD_INPUT(2, und_weights_k, DT_BF16, weights_shape, 0, 33);
    ADD_INPUT(3, cos_sin_cache, DT_FLOAT, cos_sin_cache_shape, 0, 44);
    ADD_INPUT(4, k_cache, DT_BF16, k_cache_shape, 0, 55);
    ADD_INPUT(5, v_cache, DT_BF16, v_cache_shape, 0, 66);
    ADD_INPUT(6, slot_mapping, DT_INT64, slot_mapping_shape, 1, 0);
    ADD_INPUT(7, positions, DT_INT64, positions_shape, 3, 0);
    // 以下 4 个在 IR 上是 OPTIONAL，但当前实现要求全部提供（见 README「约束说明」）
    ADD_INPUT(8, gen_qkv, DT_BF16, gen_qkv_shape, 0, 77);
    ADD_INPUT(9, gen_weights_q, DT_BF16, weights_shape, 0, 88);
    ADD_INPUT(10, gen_weights_k, DT_BF16, weights_shape, 0, 99);
    ADD_INPUT(11, cat_indices, DT_INT64, cat_indices_shape, 2, 0);

    // 添加输出（顺序严格匹配 proto.h；k_cache/v_cache 与同名输入原地复用）
    ADD_OUTPUT(q, DT_BF16, q_shape);
    ADD_OUTPUT(k_cache, DT_BF16, k_cache_shape);
    ADD_OUTPUT(v_cache, DT_BF16, v_cache_shape);

    // 添加属性（顺序严格匹配 proto.h）
    op_und_gen.set_attr_num_heads_q(NUM_HEADS_Q);
    op_und_gen.set_attr_num_heads_k(NUM_HEADS_K);
    op_und_gen.set_attr_num_heads_v(NUM_HEADS_V);
    op_und_gen.set_attr_norm_eps(1e-6f);
    op_und_gen.set_attr_mrope_section(vector<int64_t>({16, 16, 16}));

    outputs.push_back(op_und_gen);
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
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

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
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session *session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session add ir compute graph success\n", GetTime().c_str());

    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    // 输入也落盘：需要核对数值时，直接读这些 bin 喂给 tests/assets/golden.py 复算即可，
    // 不必在校验侧重复实现这里的伪随机序列
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        string input_file = "./tc_ge_irrun_test_npu_input_" + std::to_string(i) + ".bin";
        uint8_t *input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile(input_file, data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_npu_output_" + std::to_string(i) + ".bin";
        uint8_t *output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, output_data_i);
    }

    delete session;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
