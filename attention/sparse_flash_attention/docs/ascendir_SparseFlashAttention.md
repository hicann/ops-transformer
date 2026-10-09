# Ascendir_SparseFlashAttention

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 算子功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过"只计算关键部分"大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。本算子针对离散访存进行了指令缩减及搬运聚合的细致优化。

- 计算公式：

  $$
  \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
  $$

  其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。

## Ascend IR定义

Ascend IR定义所在头文件路径为[op_graph/sparse_flash_attention_proto.h](../op_graph/sparse_flash_attention_proto.h)。

```cpp
// SparseFlashAttention is also declared in the built-in ops_proto_legacy.h of the CANN package; the shared
// OPS_PROTO_DEF guard below keeps the two declarations from being compiled twice in one translation unit.
#ifndef OPS_PROTO_DEF_SPARSEFLASHATTENTION
#define OPS_PROTO_DEF_SPARSEFLASHATTENTION
REG_OP(SparseFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sparse_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(sinks, TensorType({DT_FLOAT}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(scale_value, Float)
    .ATTR(sparse_block_size, Int, 1)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .ATTR(attention_mode, Int, 0)
    .ATTR(return_softmax_lse, Bool, false)
    .OP_END_FACTORY_REG(SparseFlashAttention)
#endif
```

### 参数说明

> **说明：**<br>
> 参数维度含义：B表示Batch Size、Q_S和KV_S分别表示query和key/value的Sequence Length、Q_N和KV_N分别表示query和key/value的Head Num、Q_D和KV_D分别表示query和key/value的Head Dim、Dr表示rope的Head Dim、Q_T和KV_T分别表示query和key/value的Total Tokens、sparse_size表示一次离散选取的block数、block_num和block_size分别表示PageAttention场景下的block总数和每个block的token数。

<table style="undefined;table-layout: fixed; width: 1340px"><colgroup>
  <col style="width: 150px">
  <col style="width: 120px">
  <col style="width: 260px">
  <col style="width: 280px">
  <col style="width: 210px">
  <col style="width: 80px">
  <col style="width: 240px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query（tensor）</td>
      <td>输入</td>
      <td>attention结构的query输入。</td>
      <td>不支持空tensor和非连续。</td>
      <td>float16、bf16</td>
      <td>ND</td>
      <td>layout_query为"BSND"时，shape为[B, Q_S, Q_N, Q_D]；layout_query为"TND"时，shape为[Q_T, Q_N, Q_D]</td>
    </tr>
    <tr>
      <td>key（tensor）</td>
      <td>输入</td>
      <td>attention结构的key输入。</td>
      <td>不支持空tensor。</td>
      <td>float16、bf16</td>
      <td>ND</td>
      <td>layout_kv为"BSND"时，shape为[B, KV_S, KV_N, KV_D]；layout_kv为"TND"时，shape为[KV_T, KV_N, KV_D]；layout_kv为"PA_BSND"时，shape为[block_num, block_size, KV_N, KV_D]</td>
    </tr>
    <tr>
      <td>value（tensor）</td>
      <td>输入</td>
      <td>attention结构的value输入。</td>
      <td>不支持空tensor，shape与key的shape一致。</td>
      <td>float16、bf16</td>
      <td>ND</td>
      <td>同key</td>
    </tr>
    <tr>
      <td>sparse_indices（tensor）</td>
      <td>输入</td>
      <td>离散取kvCache的索引。</td>
      <td>不支持空tensor和非连续。需要保证每行有效值均在前半部分、无效值均在后半部分，且sparse_size大于0。</td>
      <td>int32</td>
      <td>ND</td>
      <td>layout_query为"BSND"时，shape为[B, Q_S, KV_N, sparse_size]；layout_query为"TND"时，shape为[Q_T, KV_N, sparse_size]</td>
    </tr>
    <tr>
      <td>block_table（tensor）</td>
      <td>可选输入</td>
      <td>PageAttention中kvCache存储使用的block映射表。</td>
      <td>第二维长度不小于所有batch中最大的KV_S对应的block数量，即KV_S_max / block_size向上取整。</td>
      <td>int32</td>
      <td>ND</td>
      <td>[B, KV_S_max/block_size]</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_query（tensor）</td>
      <td>可选输入</td>
      <td>表示不同Batch中query的有效token数。</td>
      <td>可传入None表示与query的Q_S长度相同。每个Batch的有效token数不超过query中的Q_S大小且不小于0。layout_query为"TND"时该入参必须传入，且以元素数量作为B值；每个元素表示当前batch与之前所有batch的token数总和（前缀和）。</td>
      <td>int32</td>
      <td>ND</td>
      <td>[B]</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_kv（tensor）</td>
      <td>可选输入</td>
      <td>表示不同Batch中key和value的有效token数。</td>
      <td>可传入None表示与key/value的KV_S长度相同。layout_kv为"TND"或"PA_BSND"时该入参必须传入；其中layout_kv为"TND"时，每个元素表示当前batch与之前所有batch的token数总和（前缀和），后一个元素的值必须大于等于前一个元素的值。</td>
      <td>int32</td>
      <td>ND</td>
      <td>[B]</td>
    </tr>
    <tr>
      <td>query_rope（tensor）</td>
      <td>可选输入</td>
      <td>MLA结构中query的rope信息。</td>
      <td>须与key_rope成对为空指针（不使用rope），或成对传入非空tensor（使用rope，Dr为64）。</td>
      <td>float16、bf16</td>
      <td>ND</td>
      <td>layout_query为"BSND"时，shape为[B, Q_S, Q_N, Dr]；layout_query为"TND"时，shape为[Q_T, Q_N, Dr]</td>
    </tr>
    <tr>
      <td>key_rope（tensor）</td>
      <td>可选输入</td>
      <td>MLA结构中key的rope信息。</td>
      <td>须与query_rope成对为空指针（不使用rope），或成对传入非空tensor（使用rope，Dr为64）。</td>
      <td>float16、bf16</td>
      <td>ND</td>
      <td>layout_kv为"BSND"时，shape为[B, KV_S, KV_N, Dr]；layout_kv为"TND"时，shape为[KV_T, KV_N, Dr]；layout_kv为"PA_BSND"时，shape为[block_num, block_size, KV_N, Dr]</td>
    </tr>
    <tr>
      <td>sinks（tensor）</td>
      <td>可选输入</td>
      <td>attention结构中的可学习的sinks信息。</td>
      <td>仅支持<term>Ascend 950PR&950DT系列产品</term>。</td>
      <td>float32</td>
      <td>ND</td>
      <td>[Q_N]</td>
    </tr>
    <tr>
      <td>scale_value（float）</td>
      <td>必要属性</td>
      <td>代表缩放系数。</td>
      <td>建议值：公式中d开根号的倒数。</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_block_size（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>代表sparse阶段的block大小。</li>
          <li>默认值为1。</li>
        </ul>
      </td>
      <td>sparse_block_size为1时，为Token-wise稀疏化场景；大于1且小于等于128时，为Block-wise稀疏化场景，块内token共享相同的稀疏化决策。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_query（string）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>标识输入query的数据排布格式。</li>
          <li>默认值为"BSND"。</li>
        </ul>
      </td>
      <td>支持BSND和TND。</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv（string）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>标识输入key的数据排布格式。</li>
          <li>默认值为"BSND"。</li>
        </ul>
      </td>
      <td>支持BSND、TND和PA_BSND，其中PA_BSND在开启PageAttention时使用。</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>表示sparse的模式。0表示全部计算；3表示rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。</li>
          <li>默认值为3。</li>
        </ul>
      </td>
      <td>支持配置值为0、3。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>用于稀疏计算，表示attention需要和前几个Token计算关联。</li>
          <li>默认值为INT64_MAX。</li>
        </ul>
      </td>
      <td>仅支持默认值2^63-1。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>用于稀疏计算，表示attention需要和后几个Token计算关联。</li>
          <li>默认值为INT64_MAX。</li>
        </ul>
      </td>
      <td>仅支持默认值2^63-1。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_mode（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>表示attention的模式。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>仅支持传入2，表示MLA-absorb模式。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_softmax_lse（bool）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>用于表示是否返回softmax_max和softmax_sum。</li>
          <li>默认值为false。</li>
        </ul>
      </td>
      <td>True表示返回，False表示不返回。该参数仅在训练且layout_kv不为PA_BSND场景支持。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_out（tensor）</td>
      <td>输出</td>
      <td>公式中的输出。</td>
      <td>不支持空tensor和非连续。</td>
      <td>float16、bf16</td>
      <td>ND</td>
      <td>layout_query为"BSND"时，shape为[B, Q_S, Q_N, Q_D]；layout_query为"TND"时，shape为[Q_T, Q_N, Q_D]</td>
    </tr>
    <tr>
      <td>softmax_max（tensor）</td>
      <td>输出</td>
      <td>Attention算法对query乘key的结果取max得到softmax_max。</td>
      <td>不支持空tensor和非连续。</td>
      <td>float32</td>
      <td>ND</td>
      <td>layout_query为"BSND"时，shape为[B, KV_N, Q_S, Q_N/KV_N]；layout_query为"TND"时，shape为[KV_N, Q_T, Q_N/KV_N]</td>
    </tr>
    <tr>
      <td>softmax_sum（tensor）</td>
      <td>输出</td>
      <td>Attention算法对query乘key的结果减去softmax_max后取exp并求sum得到softmax_sum。</td>
      <td>不支持空tensor和非连续。</td>
      <td>float32</td>
      <td>ND</td>
      <td>layout_query为"BSND"时，shape为[B, KV_N, Q_S, Q_N/KV_N]；layout_query为"TND"时，shape为[KV_N, Q_T, Q_N/KV_N]</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 确定性计算：
  - Ascendir_SparseFlashAttention默认确定性实现。
- 该接口支持推理场景下使用，支持图模式。
- KV_N仅支持1。
- block_size为一个block的token数，block_size取值为16的倍数，且最大支持1024。
- 参数query中的Q_D和key、value的KV_D值相等为512，参数query_rope中的Dr和key_rope的Dr值相等为64。
- 参数query、key、value的数据类型必须保持一致。
- query_rope与key_rope须成对为空指针或成对传入非空tensor：均为空指针时表示不使用rope，仅对512维nope部分做注意力计算；均传入非空tensor时Dr相等为64。不支持空tensor。
- sparse_block_size约束：
  - 支持sparse_block_size整除block_size。
  - <term>Ascend 950PR&950DT系列产品</term>：仅支持sparse_block_size为1。
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：支持[1,128]，且要求是2的幂次方，在PageAttention场景下要求sparse_block_size整除block_size。
- Q_N支持情况：
  - <term>Ascend 950PR&950DT系列产品</term>：Q_N支持1~128；仅在layout_kv为PA_BSND时，key、value和key_rope支持0轴非连续。
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：Q_N支持1/2/4/8/16/32/64/128；key、value和key_rope不支持非连续。
- sinks仅支持<term>Ascend 950PR&950DT系列产品</term>。

## 调用示例

示例代码如下，仅供参考。GE图模式的编译和执行过程请参考[算子调用](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式)，完整代码见[test_geir_sparse_flash_attention](../examples/test_geir_sparse_flash_attention.cpp)。

```c++
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
 * \file test_geir_sparse_flash_attention.cpp
 * \brief GE graph construction sample for SparseFlashAttention (MLA-absorb mode, TND layout, fp16).
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/sparse_flash_attention_proto.h"

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
constexpr int64_t kQTokens = 1;                 // T1
constexpr int64_t kQHeads = 16;                 // N1
constexpr int64_t kKvTokens = 2048;             // T2
constexpr int64_t kKvHeads = 1;                 // N2
constexpr int64_t kHeadDim = 512;               // D (MLA latent)
constexpr int64_t kRopeDim = 64;                // Dr
constexpr double kScale = 0.041666666666666664; // 1.0 / sqrt(512 + 64)
constexpr float kRtol = 3e-2f;
constexpr float kAtol = 3e-2f;

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
        return sign;
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
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

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps,
                   vector<float>& qData, vector<float>& kData, vector<float>& vData, vector<float>& qRopeData,
                   vector<float>& kRopeData)
{
    // TND layout.
    const vector<int64_t> qShape = {kQTokens, kQHeads, kHeadDim};
    const vector<int64_t> kvShape = {kKvTokens, kKvHeads, kHeadDim};
    const vector<int64_t> indicesShape = {kQTokens, kKvHeads, kKvTokens};
    const vector<int64_t> qRopeShape = {kQTokens, kQHeads, kRopeDim};
    const vector<int64_t> kRopeShape = {kKvTokens, kKvHeads, kRopeDim};

    qData.resize(GetShapeSize(qShape));
    kData.resize(GetShapeSize(kvShape));
    vData.resize(GetShapeSize(kvShape));
    qRopeData.resize(GetShapeSize(qRopeShape));
    kRopeData.resize(GetShapeSize(kRopeShape));
    for (int64_t i = 0; i < GetShapeSize(qShape); ++i) {
        qData[i] = GenValue(static_cast<uint64_t>(i) * 7 + 1);
    }
    for (int64_t i = 0; i < GetShapeSize(qRopeShape); ++i) {
        qRopeData[i] = GenValue(static_cast<uint64_t>(i) * 23 + 3);
    }
    for (int64_t i = 0; i < GetShapeSize(kvShape); ++i) {
        kData[i] = GenValue(static_cast<uint64_t>(i) * 13 + 5);
        vData[i] = GenValue(static_cast<uint64_t>(i) * 17 + 9);
    }
    for (int64_t i = 0; i < GetShapeSize(kRopeShape); ++i) {
        kRopeData[i] = GenValue(static_cast<uint64_t>(i) * 29 + 7);
    }
    // Identity selection: every kv token is picked in order.
    vector<int32_t> indicesData(kKvTokens);
    std::iota(indicesData.begin(), indicesData.end(), 0);
    // TND sequence lengths are prefix sums.
    const vector<int32_t> actSeqQData = {static_cast<int32_t>(kQTokens)};
    const vector<int32_t> actSeqKvData = {static_cast<int32_t>(kKvTokens)};

    vector<uint16_t> qHalf(qData.size());
    vector<uint16_t> kHalf(kData.size());
    vector<uint16_t> vHalf(vData.size());
    vector<uint16_t> qRopeHalf(qRopeData.size());
    vector<uint16_t> kRopeHalf(kRopeData.size());
    for (size_t i = 0; i < qData.size(); ++i) {
        qHalf[i] = FloatToHalf(qData[i]);
    }
    for (size_t i = 0; i < kData.size(); ++i) {
        kHalf[i] = FloatToHalf(kData[i]);
        vHalf[i] = FloatToHalf(vData[i]);
    }
    for (size_t i = 0; i < qRopeData.size(); ++i) {
        qRopeHalf[i] = FloatToHalf(qRopeData[i]);
    }
    for (size_t i = 0; i < kRopeData.size(); ++i) {
        kRopeHalf[i] = FloatToHalf(kRopeData[i]);
    }

    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT16, qHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT16, kHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "value", 3, kvShape, DT_FLOAT16, vHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "sparse_indices", 4, indicesShape, DT_INT32, indicesData) == SUCCESS,
        return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "actual_seq_lengths_query", 5, {1}, DT_INT32, actSeqQData) == SUCCESS,
        return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "actual_seq_lengths_kv", 6, {1}, DT_INT32, actSeqKvData) == SUCCESS,
        return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query_rope", 7, qRopeShape, DT_FLOAT16, qRopeHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key_rope", 8, kRopeShape, DT_FLOAT16, kRopeHalf) == SUCCESS,
              return FAILED);

    auto node = op::SparseFlashAttention("sparse_flash_attention");
    node.set_input_query(inputOps[0]);
    node.set_input_key(inputOps[1]);
    node.set_input_value(inputOps[2]);
    node.set_input_sparse_indices(inputOps[3]);
    node.set_input_actual_seq_lengths_query(inputOps[4]);
    node.set_input_actual_seq_lengths_kv(inputOps[5]);
    node.set_input_query_rope(inputOps[6]);
    node.set_input_key_rope(inputOps[7]);
    node.set_attr_scale_value(kScale);
    node.set_attr_sparse_block_size(1);
    node.set_attr_layout_query("TND");
    node.set_attr_layout_kv("TND");
    node.set_attr_sparse_mode(0);
    node.set_attr_pre_tokens(9223372036854775807);
    node.set_attr_next_tokens(9223372036854775807);
    node.set_attr_attention_mode(2);
    node.set_attr_return_softmax_lse(false);

    TensorDesc attentionOutDesc(Shape(qShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc emptyFp32Desc(Shape({0}), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_attention_out(attentionOutDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_max(emptyFp32Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_sum(emptyFp32Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

// CPU reference (MLA-absorb): score = scale * (q @ k^T + qRope @ kRope^T), out = softmax(score) @ v.
void ComputeReference(const vector<float>& qData, const vector<float>& kData, const vector<float>& vData,
                      const vector<float>& qRopeData, const vector<float>& kRopeData, vector<float>& attentionOut)
{
    const int64_t t1 = kQTokens;
    const int64_t n1 = kQHeads;
    const int64_t t2 = kKvTokens;
    const int64_t n2 = kKvHeads;
    const int64_t d = kHeadDim;
    const int64_t dr = kRopeDim;
    const int64_t group = n1 / n2;
    attentionOut.assign(t1 * n1 * d, 0.0f);
    for (int64_t i = 0; i < t1; ++i) {
        for (int64_t n = 0; n < n1; ++n) {
            const int64_t kvN = n / group;
            vector<float> score(t2);
            for (int64_t j = 0; j < t2; ++j) {
                float dot = 0.0f;
                for (int64_t t = 0; t < d; ++t) {
                    dot += qData[(i * n1 + n) * d + t] * kData[(j * n2 + kvN) * d + t];
                }
                for (int64_t t = 0; t < dr; ++t) {
                    dot += qRopeData[(i * n1 + n) * dr + t] * kRopeData[(j * n2 + kvN) * dr + t];
                }
                score[j] = static_cast<float>(kScale) * dot;
            }
            float maxScore = score[0];
            for (int64_t j = 1; j < t2; ++j) {
                maxScore = std::fmax(maxScore, score[j]);
            }
            float sum = 0.0f;
            for (int64_t j = 0; j < t2; ++j) {
                score[j] = std::exp(score[j] - maxScore);
                sum += score[j];
            }
            for (int64_t t = 0; t < d; ++t) {
                float y = 0.0f;
                for (int64_t j = 0; j < t2; ++j) {
                    y += (score[j] / sum) * vData[(j * n2 + kvN) * d + t];
                }
                attentionOut[(i * n1 + n) * d + t] = y;
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

int32_t ValidateOutputs(const vector<Tensor>& outputs, const vector<float>& qData, const vector<float>& kData,
                        const vector<float>& vData, const vector<float>& qRopeData, const vector<float>& kRopeData)
{
    CHECK_RET(outputs.size() == 3, LOG_PRINT("[CHECK] FAIL: expected 3 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> attentionOut;
    ComputeReference(qData, kData, vData, qRopeData, kRopeData, attentionOut);
    const vector<int64_t> qShape = {kQTokens, kQHeads, kHeadDim};
    const bool ok = CheckOutput(outputs[0], "attention_out", qShape, attentionOut);
    CHECK_RET(outputs[1].GetSize() == 0, LOG_PRINT("[CHECK] softmax_max expected empty\n"); return FAILED);
    CHECK_RET(outputs[2].GetSize() == 0, LOG_PRINT("[CHECK] softmax_sum expected empty\n"); return FAILED);
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

    Graph graph("sparse_flash_attention_graph");
    vector<Tensor> inputTensors;
    vector<Operator> inputOps;
    vector<Operator> outputOps;
    vector<float> qData;
    vector<float> kData;
    vector<float> vData;
    vector<float> qRopeData;
    vector<float> kRopeData;
    int32_t ret = BuildGraph(graph, inputTensors, inputOps, outputOps, qData, kData, vData, qRopeData, kRopeData);
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
        LOG_PRINT("SparseFlashAttention graph run success, output count: %zu\n", outputTensors.size());
        ret = ValidateOutputs(outputTensors, qData, kData, vData, qRopeData, kRopeData);
        delete session;
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
```
