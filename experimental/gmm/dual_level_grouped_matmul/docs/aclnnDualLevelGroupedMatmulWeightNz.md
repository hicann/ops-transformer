# aclnnDualLevelGroupedMatmulWeightNz

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

当前支持根据 `groupListOptional` 沿输入矩阵的行维度分组。每组输入与对应专家的 FRACTAL_NZ 权重相乘，跨 K 段累加后加入可选偏置。

先用两侧 level1 缩放因子还原输入和权重，再计算每个 level0 K 段的部分和。

$$
P_{h,i,j} = \sum_{k \in G_h}
  \left(X_{i,k} \cdot S^{X,1}_{i,k}\right)
  \left(W_{e,k,j} \cdot S^{W,1}_{e,k,j}\right)
$$

将各段部分和乘以相应的 level0 缩放因子，以 FP32 累加后加入偏置，再转换为输出类型。

$$
Y_{i,j} = \operatorname{Cast}_{\mathrm{RINT}}\left(
  \sum_h P_{h,i,j} \cdot S^{X,0}_{i,h} \cdot S^{W,0}_{e,h,j} + B_{e,j}
\right)
$$

$i$ 和 $j$ 分别索引输入行和输出列，$e$ 是该行对应的专家。$G_h$ 是第 $h$ 个 level0 分组的 K 范围，由 `level0GroupSize` 决定。$S^{X,1}_{i,k}$ 和 $S^{W,1}_{e,k,j}$ 分别取两侧覆盖位置 $k$ 的 level1 缩放因子，分组大小由 `level1GroupSize` 决定。输出公式中的 $S^{X,0}$ 和 $S^{W,0}$ 按 K 段索引。省略偏置时，$B_{e,j}$ 取 0。RINT 表示舍入到最近可表示值，等距时取偶数。

## 函数原型

本接口采用[两段式调用](../../../../docs/zh/context/two_phase_api.md)。先调用 `aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize` 获取工作空间大小和执行器，再调用 `aclnnDualLevelGroupedMatmulWeightNz` 执行计算。

```c++
aclnnStatus aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize(
    const aclTensorList *x, const aclTensorList *weight, const aclTensorList *biasOptional,
    const aclTensorList *scaleOptional, const aclTensorList *perTokenScaleOptional,
    const aclTensor *groupListOptional, const aclTensorList *xLevel0Scale,
    const aclTensorList *weightLevel0Scale, int64_t level0GroupSize, int64_t level1GroupSize,
    int64_t splitItem, int64_t groupType, int64_t groupListType,
    aclIntArray *tuningConfigOptional, aclTensorList *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

```c++
aclnnStatus aclnnDualLevelGroupedMatmulWeightNz(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize

  - **参数说明：**

    <table style="table-layout: fixed; width: 1483px;">
    <colgroup>
    <col style="width: 220px">
    <col style="width: 90px">
    <col style="width: 240px">
    <col style="width: 370px">
    <col style="width: 150px">
    <col style="width: 110px">
    <col style="width: 190px">
    <col style="width: 113px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>使用说明</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
    <th>非连续Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>x（const aclTensorList *）</td>
    <td>输入</td>
    <td>左矩阵。</td>
    <td>按输入行分组，M 为输入行数，K 为归约长度。</td>
    <td>FLOAT4_E2M1</td>
    <td>ND</td>
    <td>(M, K)</td>
    <td>不支持</td>
    </tr>
    <tr>
    <td>weight（const aclTensorList *）</td>
    <td>输入</td>
    <td>各专家的右矩阵。</td>
    <td>按逻辑 view、stride 和物理存储描述识别转置。</td>
    <td>FLOAT4_E2M1</td>
    <td>FRACTAL_NZ</td>
    <td>逻辑 view 为 (E,K,N)，原始布局见约束说明。</td>
    <td>支持规定的转置 view</td>
    </tr>
    <tr>
    <td>biasOptional（const aclTensorList *）</td>
    <td>输入</td>
    <td>各专家的偏置。</td>
    <td>可选，传入 <code>nullptr</code> 时按 0 处理。组内各行共享该专家的偏置。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(E, N)</td>
    <td>不支持</td>
    </tr>
    <tr>
    <td>scaleOptional（const aclTensorList *）</td>
    <td>输入</td>
    <td>右矩阵 level1 量化系数。</td>
    <td>必传。每专家、每输出列、每个 level1 分组一个系数。转置布局须与 weight 一致。</td>
    <td>FLOAT8_E8M0</td>
    <td>ND</td>
    <td>逻辑 view 为 (E,ceil(K/(2×level1GroupSize)),N,2)，转置存储形状为 (E,N,ceil(K/(2×level1GroupSize)),2)。</td>
    <td>支持规定的转置 view</td>
    </tr>
    <tr>
    <td>perTokenScaleOptional（const aclTensorList *）</td>
    <td>输入</td>
    <td>左矩阵 level1 量化系数。</td>
    <td>必传。每行每个 level1 分组一个系数。</td>
    <td>FLOAT8_E8M0</td>
    <td>ND</td>
    <td>(M, ceil(K/(2×level1GroupSize)), 2)</td>
    <td>不支持</td>
    </tr>
    <tr>
    <td>groupListOptional（const aclTensor *）</td>
    <td>输入</td>
    <td>沿 M 轴的分组信息。</td>
    <td>必传。type=0/1 每专家一个元素；type=2 每行为 [expert_id, row_count]，按列表顺序划分输入行。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>type=0/1 为 (E,)，type=2 为 (E,2)。</td>
    <td>不支持</td>
    </tr>
    <tr>
    <td>xLevel0Scale（const aclTensorList *）</td>
    <td>输入</td>
    <td>左矩阵 level0 量化系数。</td>
    <td>每行每个 level0 分组一个系数。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(M, ceil(K/level0GroupSize))</td>
    <td>不支持</td>
    </tr>
    <tr>
    <td>weightLevel0Scale（const aclTensorList *）</td>
    <td>输入</td>
    <td>右矩阵 level0 量化系数。</td>
    <td>每专家、每个 level0 分组、每输出列一个系数，维度与权重是否转置无关。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(E, ceil(K/level0GroupSize), N)</td>
    <td>不支持</td>
    </tr>
    <tr>
    <td>level0GroupSize（int64_t）</td>
    <td>输入</td>
    <td>每组 FP32 level0 scale 覆盖的 K 元素数。</td>
    <td>当前版本支持 512。调用时显式传入。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>level1GroupSize（int64_t）</td>
    <td>输入</td>
    <td>每组 E8M0 level1 scale 覆盖的 K 元素数。</td>
    <td>当前版本支持 32。调用时显式传入。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>splitItem（int64_t）</td>
    <td>输入</td>
    <td>输出配置。</td>
    <td>调用时显式传入 2 或 3，示例取 3。输入和输出 TensorList 长度均为 1，两种取值等价。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupType（int64_t）</td>
    <td>输入</td>
    <td>分组轴。</td>
    <td>固定为 0，沿 M 分组。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListType（int64_t）</td>
    <td>输入</td>
    <td>分组信息的表示方式。</td>
    <td>0 表示累计结束位置，1 表示各组行数，2 表示 [专家编号, 行数] 稀疏列表。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tuningConfigOptional（aclIntArray *）</td>
    <td>输入</td>
    <td>性能调优配置。</td>
    <td>支持 <code>{0}</code>。传入 <code>nullptr</code> 时使用默认值 <code>{0}</code>。</td>
    <td>INT64</td>
    <td>-</td>
    <td>(1,)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>out（aclTensorList *）</td>
    <td>输出</td>
    <td>分组矩阵乘的输出矩阵。</td>
    <td>必传，TensorList 长度为 1。调用方预分配 Device 内存，元素类型决定输出类型。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>(M, N)</td>
    <td>不支持</td>
    </tr>
    <tr>
    <td>workspaceSize（uint64_t *）</td>
    <td>输出</td>
    <td>Device 侧所需的 workspace 大小，单位为字节。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor（aclOpExecutor **）</td>
    <td>输出</td>
    <td>包含算子计算流程的执行器。</td>
    <td>供第二段接口使用。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

    <table>
    <thead>
    <tr>
    <th style="width: 250px">返回值</th>
    <th style="width: 130px">错误码</th>
    <th style="width: 850px">描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
    <td rowspan="2">161001</td>
    <td>1. 传入的必选张量参数、<code>workspaceSize</code> 或 <code>executor</code> 为空指针。</td>
    </tr>
    <tr>
    <td>2. 已传入且长度为 1 的 TensorList 中，元素为空指针。</td>
    </tr>
    <tr>
    <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3">161002</td>
    <td>1. 已传入的 TensorList 长度不为 1。</td>
    </tr>
    <tr>
    <td>2. 张量的数据类型、秩、形状、格式、步长、存储偏移或存储范围不符合约束。</td>
    </tr>
    <tr>
    <td>3. 两级分组大小、分组模式、调优配置或其他控制参数取值不符合约束。</td>
    </tr>
    </tbody>
    </table>

## aclnnDualLevelGroupedMatmulWeightNz

  - **参数说明：**

    |参数名|输入/输出|描述|
    |---|---|---|
    |workspace|输入|在 Device 侧申请的 workspace 内存地址。workspaceSize 为 0 时可传入 nullptr。|
    |workspaceSize|输入|由第一段接口 aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize 获取的 workspace 大小。|
    |executor|输入|由第一段接口获取的算子执行器。|
    |stream|输入|指定执行任务的 Stream。|

  - **返回值：**

    返回 aclnnStatus 状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - `aclnnDualLevelGroupedMatmulWeightNz` 默认采用确定性实现。

- level1 scale 的最内层两个元素分别对应连续的两个 level1 分组。
- 矩阵、缩放因子和输出均使用长度为 1 的 TensorList，且其中的张量须有效。`biasOptional` 提供时遵循相同的 TensorList 约束。`groupListOptional` 传入单个张量。各专家共享归约和输出列维度。
- E 须大于 0，K 须为偶数，非转置权重的 N 也须为偶数。非空 Tensor 场景下，K 须大于 2，N 须大于 1。
- 支持 M 为 0 或 N 为 0 的空 Tensor 场景，此时允许 K 为 0。参数校验通过后，第一段接口返回 `ACLNN_SUCCESS`，`workspaceSize` 为 0。当 M 和 N 均不为 0 时，K 为 0 将使第一段接口返回 `ACLNN_ERR_PARAM_INVALID`。
- E、M、K、N 均不得超过 `INT32_MAX - 1`。用于张量寻址的维度乘积及权重补齐后的元素数须可用 `int64_t` 表示。
- 张量的存储偏移须为 0。ND 参数须使用实际 ND 存储，除规定的转置权重 level1 scale 视图外，ND 张量须连续。
- `out` 须为形状和存储形状均为 `(M,N)` 的 ND 张量。输出非空时，步长须为 `(N,1)`。
- 分组列表按 `groupListType` 填写。允许专家的行数为 0。
  - `0` 表示累计结束位置，各项须非负且非递减，最后一项须等于 M。
  - `1` 表示各组行数，各项须非负，总和须等于 M。
  - `2` 表示稀疏分组，形状为 `(E,2)`，每行为 `[expert_id, row_count]`。专家编号须在 `[0,E)` 内，行数须非负且总和等于 M。非零行组在前，零行组在后，两类分组内部均按专家编号递增排列。输入和输出按列表顺序分段，权重、权重缩放因子和偏置按专家编号选取。
- 调用方须保证分组列表中的值合法。无效分组值会导致输出无效，接口不会同步返回该错误。

- MXFP4 E2M1 二级量化场景支持权重转置和非转置，偏置可省略。
  - 非空参数支持的数据类型组合见下表。

    | 场景 | x、weight | xLevel0Scale、weightLevel0Scale | perTokenScaleOptional、scaleOptional | biasOptional | groupListOptional | out |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | MXFP4 E2M1 二级量化，FP16 输出 | FLOAT4_E2M1 | FLOAT32 | FLOAT8_E8M0 | FLOAT32 | INT64 | FLOAT16 |
    | MXFP4 E2M1 二级量化，BF16 输出 | FLOAT4_E2M1 | FLOAT32 | FLOAT8_E8M0 | FLOAT32 | INT64 | BFLOAT16 |

  - `x` 和 `weight` 使用 FP4 打包。每个字节存储两个相邻的物理元素，低半字节在前。ACL 张量描述符的 shape 和 stride 按逻辑元素数填写，Device 缓冲区大小按实际打包字节数分配。
  - `weight` 的逻辑视图固定为 `(E,K,N)`，按以下 NZ 存储形状分配并填充 FP4 权重。
    - 非转置时，原始形状为 `(E,K,N)`，NZ 存储形状为 `(E,ceil(N/64),ceil(K/16),16,64)`，逻辑视图步长为 `(K×N,N,1)`。
    - 转置时，原始形状为 `(E,N,K)`，NZ 存储形状为 `(E,ceil(K/64),ceil(N/16),16,64)`，逻辑视图步长为 `(K×N,1,K)`。
  - `scaleOptional` 的逻辑视图固定为 `(E,ceil(K/(2×level1GroupSize)),N,2)`，步长与权重的转置状态一致。
    - 非转置时，步长为 `(ceil(K/(2×level1GroupSize))×N×2,N×2,2,1)`。
    - 转置时，步长为 `(N×ceil(K/(2×level1GroupSize))×2,2,ceil(K/(2×level1GroupSize))×2,1)`。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。算子包安装和仓内示例命令见[示例说明](../examples/README.md#运行-aclnn-示例)。

<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  以下示例使用两个专家，构造 MXFP4 E2M1 输入、FRACTAL_NZ 权重和两级 scale，调用两段式 ACLNN 接口并校验 BF16 输出。完整源码见[仓内示例](../examples/arch35/test_aclnn_dual_level_grouped_matmul_weight_nz.cpp)。

  ```cpp
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
  #include <cstring>
  #include <iostream>
  #include <memory>
  #include <stdexcept>
  #include <string>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_dual_level_grouped_matmul_weight_nz.h"

  namespace {
  constexpr int32_t DEVICE_ID = 0;
  constexpr int64_t M = 32;
  constexpr int64_t K = 640;
  constexpr int64_t N = 128;
  constexpr int64_t EXPERTS = 2;
  constexpr int64_t GROUP_ROWS = M / EXPERTS;
  constexpr int64_t LEVEL0_GROUP_SIZE = 512;
  constexpr int64_t LEVEL1_GROUP_SIZE = 32;
  constexpr int64_t LEVEL0_GROUPS = (K + LEVEL0_GROUP_SIZE - 1) / LEVEL0_GROUP_SIZE;
  constexpr int64_t FP4_ELEMENTS_PER_BYTE = 2;
  constexpr int64_t MX_SCALE_PAIR = 2;
  constexpr int64_t LEVEL1_PAIRS = K / (MX_SCALE_PAIR * LEVEL1_GROUP_SIZE);
  constexpr float SECOND_LEVEL0_SCALE = 2.0F;
  constexpr float EXPERT_BIAS_STEP = 16.0F;
  constexpr float WEIGHT_LEVEL1_SCALE = 2.0F;
  constexpr uint32_t BF16_TO_FP32_SHIFT = 16;
  constexpr int64_t NZ_K = 16;
  constexpr int64_t NZ_N = 64;
  constexpr uint8_t FP4_ONES = 0x22;
  constexpr uint8_t E8M0_ONE = 127;
  constexpr uint8_t E8M0_TWO = 128;

  void Check(int status, const char* operation)
  {
      if (status != ACL_SUCCESS) {
          throw std::runtime_error(std::string(operation) + " failed, status=" + std::to_string(status));
      }
  }

  struct Runtime {
      aclrtStream stream = nullptr;
      bool initialized = false;
      bool deviceSet = false;

      ~Runtime()
      {
          if (stream != nullptr) {
              aclrtDestroyStream(stream);
          }
          if (deviceSet) {
              aclrtResetDevice(DEVICE_ID);
          }
          if (initialized) {
              aclFinalize();
          }
      }

      void Init()
      {
          Check(aclInit(nullptr), "aclInit");
          initialized = true;
          Check(aclrtSetDevice(DEVICE_ID), "aclrtSetDevice");
          deviceSet = true;
          Check(aclrtCreateStream(&stream), "aclrtCreateStream");
      }
  };

  struct DeviceBuffer {
      void* addr = nullptr;

      ~DeviceBuffer()
      {
          if (addr != nullptr) {
              aclrtFree(addr);
          }
      }

      void Allocate(size_t bytes)
      {
          if (bytes != 0) {
              Check(aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
          }
      }
  };

  struct Tensor {
      DeviceBuffer buffer;
      aclTensor* desc = nullptr;
      aclTensorList* list = nullptr;

      ~Tensor()
      {
          if (list != nullptr) {
              aclDestroyTensorList(list);
          } else if (desc != nullptr) {
              aclDestroyTensor(desc);
          }
      }

      template <typename T>
      void Create(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype,
                  aclFormat format = ACL_FORMAT_ND, const std::vector<int64_t>& nzShape = {})
      {
          const size_t bytes = host.size() * sizeof(T);
          buffer.Allocate(bytes);
          Check(aclrtMemcpy(buffer.addr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
          std::vector<int64_t> strides(shape.size(), 1);
          for (size_t i = shape.size() - 1; i > 0; --i) {
              strides[i - 1] = strides[i] * shape[i];
          }
          const auto& storage = nzShape.empty() ? shape : nzShape;
          desc = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, format, storage.data(),
                                 storage.size(), buffer.addr);
          if (desc == nullptr) {
              throw std::runtime_error("aclCreateTensor failed");
          }
      }

      aclTensorList* AsList()
      {
          list = aclCreateTensorList(&desc, 1);
          if (list == nullptr) {
              throw std::runtime_error("aclCreateTensorList failed");
          }
          return list;
      }
  };

  void CheckOutput(const Tensor& y)
  {
      std::vector<uint16_t> output(M * N);
      const size_t bytes = output.size() * sizeof(uint16_t);
      Check(aclrtMemcpy(output.data(), bytes, y.buffer.addr, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
      // Level1 contributes a factor of 2. The two level0 products are 1 and 4.
      constexpr float matmul =
          WEIGHT_LEVEL1_SCALE * (LEVEL0_GROUP_SIZE + (K - LEVEL0_GROUP_SIZE) * SECOND_LEVEL0_SCALE * SECOND_LEVEL0_SCALE);
      for (int64_t row = 0; row < M; ++row) {
          const float expected = matmul + EXPERT_BIAS_STEP * (row / GROUP_ROWS + 1);
          for (int64_t col = 0; col < N; ++col) {
              const uint32_t bits = static_cast<uint32_t>(output[row * N + col]) << BF16_TO_FP32_SHIFT;
              float actual;
              std::memcpy(&actual, &bits, sizeof(actual));
              if (actual != expected) {
                  throw std::runtime_error("Output mismatch at [" + std::to_string(row) + ", " + std::to_string(col) +
                                           "], expected " + std::to_string(expected) + ", got " + std::to_string(actual));
              }
          }
      }
      std::cout << "DualLevelGroupedMatmulWeightNz passed, BF16 shape (" << M << ", " << N << ")\n";
  }

  void RunExample(aclrtStream stream)
  {
      Tensor x, weight, xLevel0, xLevel1, weightLevel0, weightLevel1, groupList, bias, y;
      x.Create(std::vector<uint8_t>(M * K / FP4_ELEMENTS_PER_BYTE, FP4_ONES), {M, K}, ACL_FLOAT4_E2M1);
      // Every FP4 element is 1, so the NZ buffer can be filled directly in storage order.
      weight.Create(std::vector<uint8_t>(EXPERTS * K * N / FP4_ELEMENTS_PER_BYTE, FP4_ONES), {EXPERTS, K, N},
                    ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {EXPERTS, N / NZ_N, K / NZ_K, NZ_K, NZ_N});

      std::vector<float> xLevel0Host(M * LEVEL0_GROUPS, 1.0F);
      for (int64_t row = 0; row < M; ++row) {
          xLevel0Host[row * LEVEL0_GROUPS + 1] = SECOND_LEVEL0_SCALE;
      }
      std::vector<float> weightLevel0Host(EXPERTS * LEVEL0_GROUPS * N, 1.0F);
      std::vector<float> biasHost(EXPERTS * N, EXPERT_BIAS_STEP);
      for (int64_t expert = 0; expert < EXPERTS; ++expert) {
          for (int64_t col = 0; col < N; ++col) {
              weightLevel0Host[(expert * LEVEL0_GROUPS + 1) * N + col] = SECOND_LEVEL0_SCALE;
              biasHost[expert * N + col] = EXPERT_BIAS_STEP * (expert + 1);
          }
      }
      xLevel0.Create(xLevel0Host, {M, LEVEL0_GROUPS}, ACL_FLOAT);
      xLevel1.Create(std::vector<uint8_t>(M * LEVEL1_PAIRS * MX_SCALE_PAIR, E8M0_ONE), {M, LEVEL1_PAIRS, MX_SCALE_PAIR},
                     ACL_FLOAT8_E8M0);
      weightLevel0.Create(weightLevel0Host, {EXPERTS, LEVEL0_GROUPS, N}, ACL_FLOAT);
      weightLevel1.Create(std::vector<uint8_t>(EXPERTS * LEVEL1_PAIRS * N * MX_SCALE_PAIR, E8M0_TWO),
                          {EXPERTS, LEVEL1_PAIRS, N, MX_SCALE_PAIR}, ACL_FLOAT8_E8M0);
      groupList.Create(std::vector<int64_t>{GROUP_ROWS, GROUP_ROWS}, {EXPERTS}, ACL_INT64);
      bias.Create(biasHost, {EXPERTS, N}, ACL_FLOAT);
      y.Create(std::vector<uint16_t>(M * N, 0), {M, N}, ACL_BF16);

      constexpr int64_t splitItem = 3;
      constexpr int64_t groupType = 0;
      constexpr int64_t groupListType = 1;
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      Check(aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize(
                x.AsList(), weight.AsList(), bias.AsList(), weightLevel1.AsList(), xLevel1.AsList(), groupList.desc,
                xLevel0.AsList(), weightLevel0.AsList(), LEVEL0_GROUP_SIZE, LEVEL1_GROUP_SIZE, splitItem, groupType,
                groupListType, nullptr, y.AsList(), &workspaceSize, &executor),
            "aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize");
      std::unique_ptr<aclOpExecutor, decltype(&aclDestroyAclOpExecutor)> plan(executor, aclDestroyAclOpExecutor);
      DeviceBuffer workspace;
      workspace.Allocate(workspaceSize);
      const auto runStatus = aclnnDualLevelGroupedMatmulWeightNz(workspace.addr, workspaceSize, plan.release(), stream);
      const auto syncStatus = aclrtSynchronizeStream(stream);
      Check(runStatus, "aclnnDualLevelGroupedMatmulWeightNz");
      Check(syncStatus, "aclrtSynchronizeStream");
      CheckOutput(y);
  }
  } // namespace

  int main()
  {
      try {
          Runtime runtime;
          runtime.Init();
          RunExample(runtime.stream);
      } catch (const std::exception& error) {
          std::cerr << error.what() << '\n';
          return 1;
      }
      return 0;
  }
  ```
<!-- end id12 -->
