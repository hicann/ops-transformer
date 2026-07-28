# aclnnQuantCompressor

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：QuantCompressor是推理场景下SMLA和QLI的前处理算子，是[Compressor](../../compressor/docs/aclnnCompressor.md)的量化版本。用于将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。在长序列的情况下，QuantCompressor可以有效地减少计算开销。与Compressor的区别在于：输入$x$、$W^{KV}$、$W^{Gate}$为HIFLOAT8量化数据，直接以HIFLOAT8参与Matmul运算（硬件原生支持），再对Matmul输出的FLOAT32结果乘以合并后的缩放因子进行反量化，从而降低显存占用与搬运开销。主要计算过程为：
    1. Matmul与反量化：将HIFLOAT8量化的输入$X$与$W^{KV}$做Matmul运算得到FLOAT32结果，再乘以合并缩放因子$x\_descale \cdot wkv\_descale$完成反量化得到$kv\_state$；将$X$与$W^{Gate}$做Matmul运算得到FLOAT32结果，再乘以合并缩放因子$x\_descale \cdot wgate\_descale$完成反量化得到$score\_state$。其中x_descale为per-tensor缩放（单个标量），wkv_descale与wgate_descale为per-channel缩放（通道数为coff\*D），合并后仍为per-channel缩放。$kv\_state$与$score\_state$根据输入的start_pos及cu_seqlens完成更新。
    2. 在coff为2的情况下对$kv\_state$和$score\_state$进行数据重排。
    3. 对$score\_state$按压缩比分组并与$Ape$相加，然后进行softmax运算，将softmax结果与$kv\_state$做Mul计算，后进行ReduceSum运算。

- 计算公式：

    1. 计算矩阵乘法与反量化（quant_mode=1，A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL）：

    使用HIFLOAT8输入进行Matmul，输出FLOAT32结果后乘以合并缩放因子（x_descale与wkv/wgate_descale的乘积）完成反量化：

    $$
    C4A：\left[kv\_state^a, score\_state^a\right] = (X_{hif8} @ \left[W^{aKV}_{hif8}, W^{aGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^a, wgate\_descale^a]), \left[kv\_state^b, score\_state^b\right] = (X_{hif8} @ \left[W^{bKV}_{hif8}, W^{bGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^b, wgate\_descale^b]);
    $$

    $$
    C128A：\left[kv\_state, score\_state\right] = (X_{hif8} @ \left[W^{KV}_{hif8}, W^{Gate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale, wgate\_descale])
    $$

    其中x_descale为per-tensor标量（shape为[1,]），wkv_descale、wgate_descale为per-channel向量（shape为[coff\*D,]，沿权重输出通道维度缩放），两者相乘后合并为per-channel缩放因子。

    2. 计算分组加法：

    $$
    C4A：score\_state_i^\prime = \left[score\_state_{\left[4(i-1)+1:4i,:\right]}^a; score\_state_{\left[4i+1:4(i+1),:\right]}^b\right] + Ape,~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A：score\_state_i^\prime = score\_state_{\left[128(i-1)+1:128i,:\right]} + Ape,~i=1,2,\cdots, \frac{s}{128};
    $$

    3. 计算分组Softmax：

    $$
    C4A：S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A：S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{128};
    $$

    4. 计算Hadamard乘积：

    $$
    C4A：(S_H)_i = S_i^\prime \odot \left[kv\_state^a_{\left[4(i-1)+1:4i,:\right]} ;kv\_state^b_{\left[4i+1:4(i+1),:\right]}\right],~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A：S_H = S_i^\prime \odot kv\_state;
    $$

    5. 沿着压缩轴分组求和：

    $$
    C4A：C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
     C128A：C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnQuantCompressorGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnQuantCompressor”接口执行计算。

```cpp
aclnnStatus aclnnQuantCompressorGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *wkv,
    const aclTensor *wgate,
    aclTensor       *stateCacheRef,
    const aclTensor *ape,
    const aclTensor *xDescaleOptional,
    const aclTensor *wkvDescaleOptional,
    const aclTensor *wgateDescaleOptional,
    const aclTensor *stateBlockTableOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional,
    int64_t          quantMode,
    int64_t          cmpRatio,
    int64_t          coff,
    int64_t          cacheMode,
    int64_t          stateCacheStrideDim0,
    const aclTensor *cmpKvOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

``` cpp
aclnnStatus aclnnQuantCompressor(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantCompressorGetWorkspaceSize

- **参数说明**

    | 参数名                      | 输入/输出 | 描述  |  使用说明  | 数据类型       | 数据格式   | 维度（shape） | 非连续Tensor |
    |----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|-|-|-|
    | x | 输入 | 公式中的$X$，表示原始不经压缩的数据，HIFLOAT8量化输入。 |  支持B=0,S=0,T=0的空Tensor。  | HIFLOAT8 | ND         | BS合轴：[T,H]、BS非合轴：[B,S,H]|×|
    | wkv | 输入 | 公式中的$W^{KV}$，表示kv压缩权重，HIFLOAT8量化输入。  |不支持空Tensor。| HIFLOAT8 | ND |[coff* D,H]|×|
    | wgate | 输入 | 公式中的$W^{Gate}$，表示gate压缩权重，HIFLOAT8量化输入。 |不支持空Tensor。| HIFLOAT8 | ND |[coff* D,H]|×|
    | stateCacheRef | 输入 | 公式中的$\left[kv\_state, score\_state\right]$, 表示kv\_state和score\_state的历史数据。 |不支持空Tensor| FLOAT32     | ND         |[block_num,block_size,2* coff* D]|支持0轴非连续|
    | ape | 输入 | 公式中的$Ape$，表示positional biases。 | 不支持空Tensor。|FLOAT32       | ND         |[cmp_ratio,coff* D]|×|
    | xDescale | 可选输入 | x的反量化缩放因子，per-tensor缩放。quant_mode=1时必选。 |FLOAT32       | ND         |[1,]|×|
    | wkvDescale | 可选输入 | wkv的反量化缩放因子，per-channel缩放，通道数为coff\*D。quant_mode=1时必选。 |FLOAT32       | ND         |[coff* D,]|×|
    | wgateDescale | 可选输入 | wgate的反量化缩放因子，per-channel缩放，通道数为coff\*D。quant_mode=1时必选。 |FLOAT32       | ND         |[coff* D,]|×|
    | stateBlockTable | 可选输入 | 表示state\_cache存储使用的block映射表。|当其中元素的值为0时，表示当前位置无需进行更新state\_cache操作；不支持空Tensor。| INT32 | ND         |cache_mode=1时，shape为[B,ceil(Smax/block_size)]，Smax为每个Batch中最大的Sequence Length，当x的shape为[B,S,H]时，Smax=max(start_pos)+S。当x的shape为[T,H]时，Smax=max(start_pos)+max(cu_seqlens[n+1] - cu_seqlens[n])。cache_mode=2时，shape为[B]。当其中元素的值为0时，表示当前位置无需进行更新state_cache操作|×|
    | cuSeqlens | 可选输入 | 表示不同Batch中的有效token数。  |支持B=0,S=0,T=0的空Tensor；当x的shape为[B,S,H]时，参数必须为空。| INT32          | ND         |当x的shape为[T,H]时，输入shape为[B+1,]|×|
    | seqused | 可选输入 | 表示不同Batch中实际参与压缩的token数。 |如果指定为None时，表示和每个Batch上的Sequence Length长度相同；支持B=0的空Tensor；如果指定为None时，表示和每个Batch上的Sequence Length长度相同。该入参中每个Batch的有效token数要求小于等于对应Sequence Length长度。当x的shape为[B,S,H]时，要求seqused[n] <= S，且不小于0；当x的shape为[T,H]时，要求seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n]，且不小于0。| INT32          | ND         |[B,]|×|
    | startPos | 可选输入 | 表示计算起始位置。 |支持B=0,T=0的空Tensor；当输入为None时，表示从0开始进行计算。| INT32          | ND         |[B,]|×|
    | quantMode | 输入 | 量化模式。 |取值范围为[1]，1表示A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL（HIFLOAT8输入，x按per-tensor缩放、wkv/wgate按per-channel缩放反量化）。| INT32          | -         |-|-|
    | cmpRatio | 输入 | 用于稀疏计算，表示数据压缩率。 |取值范围为[2, 4, 8, 16, 32, 64, 128]。| INT32          | -         |-|-|
    | coff | 可选输入 | 表示是否进行overlap数据重排。 |取值范围为[1, 2]。当coff=1时，无需进行overlap数据重排。当coff=2时，需要进行overlap数据重排。| INT32          | -         |-|-|
    | cacheMode | 可选输入 | 表示state_cache的存储模式。 |取值范围为[1, 2]；1表示连续buffer，2表示循环buffer。| INT32          | -         |-|-|
    | stateCacheStrideDim0 | 可选输入 | 表示state_cache的0轴stride。 |-| INT32     | -         |-|-|
    | cmpKv | 输出 | 表示压缩后的数据。 |支持B=0,S=0,T=0的空Tensor。| BFLOAT16         | ND          |BS合轴：[min(T,T//cmp_ratio+B),D]、BS非合轴：[B,ceil(S/cmp_ratio),D]|×|
    | stateCache | 输出 | 表示更新后的state_cache，与stateCacheRef为同一地址（inplace更新）。 |-| FLOAT32     | ND         |[block_num,block_size,2* coff* D]|支持0轴非连续|

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
      <thead>
        <tr>
          <th>返回值</th>
          <th>错误码</th>
          <th>描述</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>必须传入的参数（如接口核心依赖的输入/输出参数）中存在空指针。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_PARAM_INVALID</td>
          <td>161002</td>
          <td>输入参数的shape（维度/尺寸）、dtype（数据类型）不在接口支持的范围内。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_RUNTIME_ERROR</td>
          <td>361001</td>
          <td>API内存调用NPU Runtime接口时发生异常（如Runtime服务未启动、内存申请失败等）。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_INNER_TILING_ERROR</td>
          <td>561002</td>
          <td>tiling发生异常，入参的dtype类型或者shape错误。</td>
        </tr>
      </tbody>
    </table>

## aclnnQuantCompressor

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantCompressorGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnQuantCompressor默认确定性实现。
- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 输入shape限制：
    - wkv支持输入shape[coff* D,H]
    - wgate支持输入shape[coff* D,H]
    - stateCache支持输入shape[block_num,block_size,2* coff* D]，要求blockNum>0，cacheMode=2时，需要满足blockSize >= coff * cmp_ratio + S - 1。
    - ape支持输入shape[cmp_ratio,coff* D]
    - xDescale支持输入shape[1,]，per-tensor缩放。
    - wkvDescale支持输入shape[coff* D,]，per-channel缩放。
    - wgateDescale支持输入shape[coff* D,]，per-channel缩放。
    - startPos支持输入shape[B,]
    - 若x的维度采用BS合轴，即x的输入shape为[T,H]
        - cuSeqlens输入shape必须为[B+1,]。该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值，且第一位必须位0。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n]，且不小于0。
        - cacheMode=1时，state\_block\_table支持输入shape[B,ceil(Smax/block_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start\_pos)+max(cu\_seqlens[n+1] - cu\_seqlens[n])。cacheMode=2时，state\_block\_table支持输入shape[B]。
        - cmpKv，输出shape为[min(T,T//cmp_ratio+B),D]：<batch0>compressed_tokens + <batch1>compressed_tokens + ... + <batchN>compressed_tokens + pad。
    - 若x的维度不采用BS合轴，即x的输入shape为[B,S,H]
        - cuSeqlens，参数必须为空。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即要求seqused[n] <= S，且不小于0。
        - cacheMode=1时，stateBlockTable支持输入shape[B,ceil(Smax/block_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start\_pos)+S。cacheMode=2时，stateBlockTable支持输入shape[B]。
        - cmpKv，输出shape为[B,ceil(S/cmp_ratio),D]：(<batch0>compressed_tokens+pad0) + (<batch1>compressed_tokens+pad1) + ...  + (<batchN>compressed_tokens+padN)。
- 输入值域限制：
  - 该接口支持B、S泛化，且存在如下场景限制：
      - 只支持B、S为0
      - 部分长序列场景下，如果计算量过大可能会导致出现超过NPU内存的报错，注：这里计算量会受x输入shape的影响，值越大计算量越大。典型的长序列（即B、S的乘积或T较大）场景包括但不限于：
      <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 400px"><colgroup>
      <col style="width: 100px">
      <col style="width: 100px">
      </colgroup><thead>
      <tr>
      <th>B</th>
      <th>S</th>
      <th>H</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>100</td>
      <td>65525</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>25</td>
      <td>261120</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>100</td>
      <td>131072</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>100</td>
      <td>261120</td>
      <td>4096</td>
      </tr>
      </tbody>
      </table>
      </div>
- 该接口支持B、S、T取0，即shape与B、S、T值相关的入参允许传入空tensor，其余入参不支持传入空tensor。该场景下stateCache不做更新，输出cmpKv为空tensor。
- 输入属性限制：
  - quantMode取值为1（A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL），此时xDescale、wkvDescale、wgateDescale为必选输入。
  - 支持D为128/512。
  - 支持H为1K~10K，512对齐。
  - 支持blockSize为1~1024。
  - 支持cmpRatio为2/4/8/16/32/64/128。支持如下三种典型组合场景：
      - C4A: D=512, coff=2, cmp_ratio=4；
      - C4Li: D=128, coff=2, cmp_ratio=4；
      - C128A: D=512, coff=1, cmp_ratio=128。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
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
 * \file test_aclnn_quant_compressor.cpp
 * \brief QuantCompressor 算子 aclnn 调用示例
 *        场景：C4A（D=512, coff=2, cmp_ratio=4, cache_mode=1, BSH layout）
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_compressor.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

namespace {

template <typename To, typename From>
inline To BitCopy(const From& src)
{
    static_assert(sizeof(To) == sizeof(From), "size mismatch");
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

inline float Bf16ToFloat(uint16_t h)
{
    uint32_t x = static_cast<uint32_t>(h) << 16;
    return BitCopy<float>(x);
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    } else {
        *deviceAddr = nullptr;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void PrintBf16Result(const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                           *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 10; i++) {
        LOG_PRINT("cmp_kv[%ld] is: %f\n", i, Bf16ToFloat(resultData[i]));
    }
}

}  // namespace

int main()
{
    // 1. device/stream 初始化
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 场景参数: C4A (D=512, coff=2, cmp_ratio=4, cache_mode=1, BSH layout)
    int64_t B = 1;
    int64_t S = 4;
    int64_t hiddenSize = 4096;
    int64_t headDim = 512;
    int64_t coff = 2;
    int64_t cmpRatio = 4;
    int64_t cacheMode = 1;  // 1: 连续buffer (LINEAR_BUFFER)
    int64_t quantMode = 1;  // 1: A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL
    int64_t stateCacheStrideDim0 = 0;
    int64_t blockSize = 128;

    int64_t Smax = S;
    int64_t maxBlockNumPerBatch = (Smax + blockSize - 1) / blockSize;
    int64_t blockNum = B * maxBlockNumPerBatch;
    int64_t coffD = coff * headDim;

    // 2. 构造输入与输出 shape
    std::vector<int64_t> xShape = {B, S, hiddenSize};
    std::vector<int64_t> wkvShape = {coffD, hiddenSize};
    std::vector<int64_t> wgateShape = {coffD, hiddenSize};
    std::vector<int64_t> stateCacheShape = {blockNum, blockSize, 2 * coffD};
    std::vector<int64_t> apeShape = {cmpRatio, coffD};
    std::vector<int64_t> xDescaleShape = {1};
    std::vector<int64_t> wkvDescaleShape = {coffD};
    std::vector<int64_t> wgateDescaleShape = {coffD};
    std::vector<int64_t> stateBlockTableShape = {B, maxBlockNumPerBatch};
    std::vector<int64_t> startPosShape = {B};
    int64_t Sr = (S + cmpRatio - 1) / cmpRatio;
    std::vector<int64_t> cmpKvShape = {B, Sr, headDim};

    // 3. 构造 host 数据
    int64_t xSize = GetShapeSize(xShape);
    int64_t wkvSize = GetShapeSize(wkvShape);
    int64_t wgateSize = GetShapeSize(wgateShape);
    int64_t stateCacheSize = GetShapeSize(stateCacheShape);
    int64_t apeSize = GetShapeSize(apeShape);
    int64_t cmpKvSize = GetShapeSize(cmpKvShape);

    std::vector<uint8_t> xHostData(xSize, 128);
    std::vector<uint8_t> wkvHostData(wkvSize, 128);
    std::vector<uint8_t> wgateHostData(wgateSize, 128);
    std::vector<float_t> stateCacheHostData(stateCacheSize, 0.1f);
    std::vector<float_t> apeHostData(apeSize, 0.1f);
    std::vector<float_t> xDescaleHostData = {0.5f};
    std::vector<float_t> wkvDescaleHostData(coffD, 0.1f);
    std::vector<float_t> wgateDescaleHostData(coffD, 0.1f);
    std::vector<int32_t> stateBlockTableHostData;
    for (int64_t i = 0; i < B * maxBlockNumPerBatch; i++) {
        stateBlockTableHostData.push_back(static_cast<int32_t>(i + 1));
    }
    std::vector<int32_t> startPosHostData(B, 0);
    std::vector<uint16_t> cmpKvHostData(cmpKvSize, 0);

    // 4. 创建 aclTensor
    void* xDeviceAddr = nullptr;
    void* wkvDeviceAddr = nullptr;
    void* wgateDeviceAddr = nullptr;
    void* stateCacheDeviceAddr = nullptr;
    void* apeDeviceAddr = nullptr;
    void* xDescaleDeviceAddr = nullptr;
    void* wkvDescaleDeviceAddr = nullptr;
    void* wgateDescaleDeviceAddr = nullptr;
    void* stateBlockTableDeviceAddr = nullptr;
    void* startPosDeviceAddr = nullptr;
    void* cmpKvDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* wkv = nullptr;
    aclTensor* wgate = nullptr;
    aclTensor* stateCacheRef = nullptr;
    aclTensor* ape = nullptr;
    aclTensor* xDescale = nullptr;
    aclTensor* wkvDescale = nullptr;
    aclTensor* wgateDescale = nullptr;
    aclTensor* stateBlockTable = nullptr;
    aclTensor* startPos = nullptr;
    aclTensor* cmpKvOut = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_HIFLOAT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvHostData, wkvShape, &wkvDeviceAddr, aclDataType::ACL_HIFLOAT8, &wkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateHostData, wgateShape, &wgateDeviceAddr, aclDataType::ACL_HIFLOAT8, &wgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateCacheHostData, stateCacheShape, &stateCacheDeviceAddr, aclDataType::ACL_FLOAT,
                          &stateCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(apeHostData, apeShape, &apeDeviceAddr, aclDataType::ACL_FLOAT, &ape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xDescaleHostData, xDescaleShape, &xDescaleDeviceAddr, aclDataType::ACL_FLOAT, &xDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvDescaleHostData, wkvDescaleShape, &wkvDescaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &wkvDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateDescaleHostData, wgateDescaleShape, &wgateDescaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &wgateDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateBlockTableHostData, stateBlockTableShape, &stateBlockTableDeviceAddr,
                          aclDataType::ACL_INT32, &stateBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosHostData, startPosShape, &startPosDeviceAddr, aclDataType::ACL_INT32, &startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_BF16, &cmpKvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. 调用 aclnnQuantCompressorGetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnQuantCompressorGetWorkspaceSize(
        x, wkv, wgate, stateCacheRef, ape,
        xDescale, wkvDescale, wgateDescale,
        stateBlockTable, nullptr, nullptr, startPos,
        quantMode, cmpRatio, coff, cacheMode, stateCacheStrideDim0,
        cmpKvOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantCompressorGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 6. 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 7. 调用 aclnnQuantCompressor
    ret = aclnnQuantCompressor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantCompressor failed. ERROR: %d\n", ret); return ret);

    // 8. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 9. 获取输出
    LOG_PRINT("QuantCompressor execution succeeded.\n");
    PrintBf16Result(cmpKvShape, &cmpKvDeviceAddr);

    // 10. 释放资源
    aclDestroyTensor(x);
    aclDestroyTensor(wkv);
    aclDestroyTensor(wgate);
    aclDestroyTensor(stateCacheRef);
    aclDestroyTensor(ape);
    aclDestroyTensor(xDescale);
    aclDestroyTensor(wkvDescale);
    aclDestroyTensor(wgateDescale);
    aclDestroyTensor(stateBlockTable);
    aclDestroyTensor(startPos);
    aclDestroyTensor(cmpKvOut);

    aclrtFree(xDeviceAddr);
    aclrtFree(wkvDeviceAddr);
    aclrtFree(wgateDeviceAddr);
    aclrtFree(stateCacheDeviceAddr);
    aclrtFree(apeDeviceAddr);
    aclrtFree(xDescaleDeviceAddr);
    aclrtFree(wkvDescaleDeviceAddr);
    aclrtFree(wgateDescaleDeviceAddr);
    aclrtFree(stateBlockTableDeviceAddr);
    aclrtFree(startPosDeviceAddr);
    aclrtFree(cmpKvDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
