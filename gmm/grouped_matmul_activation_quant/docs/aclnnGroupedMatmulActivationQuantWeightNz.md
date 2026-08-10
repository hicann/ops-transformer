# aclnnGroupedMatmulActivationQuantWeightNz

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_activation_quant)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：融合GroupedMatmul、activation和quant，详细解释见计算公式。此接口为WeightNz特化版本，调用者必须传入FRACTAL_NZ格式的weight，接口按该格式解析该参数。当前版本仅支持MX量化场景，激活函数仅支持gelu_tanh。

- 计算公式：
  <details>
  <summary>MX量化场景：</summary>
  <a id="MX量化场景"></a>

    - **定义**：

      * **⋅** 表示矩阵乘法。
      * **⊙** 表示逐元素乘法。
      * gelu_tanh激活函数的数学语义为：

        $$
        \text{GELU}_{\text{tanh}}(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right)
        $$

      * 当前kernel底层实现使用gelu_sigmoid函数对gelu_tanh进行近似计算，定义为：

        $$
        \operatorname{GeluTanh}(x) =
        \frac{x}{1 + \exp\left(-1.595769121 \times \left(x + 0.044715 \times x^3\right)\right)}
        $$
      * $E$ 表示专家数，$M$ 表示总token数，$K$ 表示输入特征维度，$N$ 表示输出特征维度。
      * $blocksize$ 表示MX量化结果的存储block大小，当前仅支持64，对应2个MX量化group，每个group包含32个元素。

    - **输入**：

      * $X$：激活矩阵。
      * $groupList$：分组索引列表，按groupListType解释为cumsum或count。
      * $weight$：分组weight矩阵。
      * $weightScale$：weight矩阵量化因子。
      * $bias$：MXFP8场景下必须为空，支持nullptr、空tensorlist或长度为1且元素shape为(0)的空tensorlist。
      * $xScaleOptional$：左矩阵量化因子，对应公式中的$xScale$。

    - **输出**：

      * $Y$：激活并量化后的输出矩阵，数据类型由输出Tensor y指定。
      * $YScale$：输出量化因子。

    - **计算过程**

      - 1.根据groupList[i]确定当前分组的token范围，$i \in [0, Len(groupList))$。

      - 2.根据分组确定的入参进行GroupedMatmul和反量化计算，中间GroupedMatmul结果默认为FLOAT32类型：

        $C_i = (X_i \cdot weight_i) \odot xScale_{i\ Broadcast} \odot weightScale_{i\ Broadcast}$

      - 3.执行gelu_tanh激活：

        $S_i = GeluTanh(C_i)$

        注：当前kernel底层实现使用gelu_sigmoid函数近似计算gelu_tanh，近似公式见定义。

      - 4.对激活结果进行MX量化，目标数据类型DType由输出Tensor y的数据类型指定：

        - 场景1，当scaleAlg为0时，表示OCP实现，将激活结果$S_i$在N轴按$group\_size=32$分组，一组$group\_size$个数$\{V_j\}_{j=1}^{group\_size}$动态量化为$\{YScale, \{P_j\}_{j=1}^{group\_size}\}$。

          $$
          shared\_exp = floor(log_2(max_j(|V_j|))) - emax
          $$

          $$
          YScale = 2 ^ {shared\_exp}
          $$

          $$
          P_j = cast\_to\_dst\_type(V_j / YScale, roundMode), \space j \space from \space 1 \space to \space group\_size
          $$

          量化后的$P_j$按对应$V_j$的位置组成输出$Y$，$YScale$按对应N轴的量化group组成输出$YScale$。

          - $emax$：对应数据类型的最大正则数的指数位。

            | DataType | emax |
            | :---: | :---: |
            | FLOAT8_E4M3FN | 8 |
            | FLOAT8_E5M2 | 15 |

        - 场景2，当scaleAlg为1时，表示cuBLAS实现，只涉及FP8类型。将激活结果$S_i$在N轴按存储$blocksize=64$分组，每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标FP8类型。如果最后一块不足$blocksize$个元素，缺失值视为0并按完整块处理。

          找到该块中数值的最大绝对值：

          $$
          Amax(D_{fp32}^b)=max(\{|d_j|\}_{j=1}^{k})
          $$

          将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值：

          $$
          S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
          $$

          将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。先从$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$，再为避免量化溢出对指数向上取整：

          $$
          E_{int}^b = \begin{cases}
          E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\
          E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\
          E_{int}^b, & \text{否则}
          \end{cases}
          $$

          $$
          S_{ue8m0}^b = 2 ^ {E_{int}^b}
          $$

          $$
          R_{fp32}^b = \frac{1}{fp32(S_{ue8m0}^b)}
          $$

          对块内每个元素执行量化：

          $$
          d^j = DType(d_{fp32}^j \cdot R_{fp32}^b), \space j \space from \space 1 \space to \space blocksize
          $$

          最终输出的量化结果为$\left(S^b, [d^j]_{j=1}^{k}\right)$，其中$S^b$表示块缩放因子$S_{ue8m0}^b$，$[d^j]_{j=1}^{k}$表示块内量化后的数据。

  </details>

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupedMatmulActivationQuantWeightNz”接口执行计算。

```cpp
aclnnStatus aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
  const aclTensor     *x,
  const aclTensor     *groupList,
  const aclTensorList *weight,
  const aclTensorList *weightScale,
  const aclTensorList *bias,
  const aclTensor     *xScaleOptional,
  const char          *activationType,
  int64_t              groupListType,
  const aclIntArray   *tuningConfig,
  const char          *quantMode,
  const char          *roundMode,
  int64_t              scaleAlg,
  float                dstTypeMax,
  aclTensor           *y,
  aclTensor           *yScale,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulActivationQuantWeightNz(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize

- **参数说明**
  <table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th style="white-space: nowrap">输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th style="white-space: nowrap">维度(shape)</th>
      <th>非连续的Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x（const aclTensor *）</td>
      <td rowspan="1">输入</td>
      <td>表示左矩阵，对应公式中的X。</td>
      <td>必选参数。</td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupList（const aclTensor *）</td>
      <td rowspan="1">输入</td>
      <td>表示分组信息，对应公式中的groupList。</td>
      <td>必选参数。根据groupListType输入不同格式数据。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight（const aclTensorList *）</td>
      <td rowspan="1">输入</td>
      <td>表示weight矩阵，对应公式中的weight。</td>
      <td>tensorList长度当前仅支持1。调用者必须传入FRACTAL_NZ格式的weight，接口按该格式解析该参数；viewShape要求为3维，storageShape要求为5维。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FRACTAL_NZ</td>
      <td>3（storageShape为5）</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightScale（const aclTensorList *）</td>
      <td rowspan="1">输入</td>
      <td>表示weight矩阵的量化因子，对应公式中的weightScale。</td>
      <td>tensorList长度当前仅支持1。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias（const aclTensorList *）</td>
      <td rowspan="1">可选输入</td>
      <td>表示偏置。</td>
      <td>MXFP8场景下必须为空，支持nullptr、空tensorlist或长度为1且元素shape为(0)的空tensorlist。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xScaleOptional（const aclTensor *）</td>
      <td rowspan="1">输入</td>
      <td>表示左矩阵的量化因子，对应公式中的xScale。</td>
      <td>当前MXFP8量化场景下必选，不能传nullptr。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>activationType（const char *）</td>
      <td>输入</td>
      <td>表示激活函数类型。</td>
      <td>必选属性。当前仅支持"gelu_tanh"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType（int64_t）</td>
      <td>输入</td>
      <td>表示groupList的解释方式。</td>
      <td>支持取值0、1；当groupListType为0时，groupList必须为非负单调非递减数列；当groupListType为1时，groupList必须为非负数列。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tuningConfig（const aclIntArray *）</td>
      <td>可选输入</td>
      <td>表示调优参数，预留参数。</td>
      <td>支持传入nullptr，当前暂不使用。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（const char *）</td>
      <td>可选输入</td>
      <td>表示量化模式。</td>
      <td>支持传入nullptr或空字符串。当前仅支持"mx"；传入nullptr或空字符串时，若x的数据类型为FLOAT8_E4M3FN或FLOAT8_E5M2且xScaleOptional的数据类型为FLOAT8_E8M0，则推导为"mx"。若显式传入非"mx"，则报错。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode（const char *）</td>
      <td>可选输入</td>
      <td>表示量化舍入模式。</td>
      <td>支持传入nullptr或空字符串，传入nullptr或空字符串时按"rint"处理，当前仅支持"rint"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg（int64_t）</td>
      <td>输入</td>
      <td>表示量化因子计算算法。</td>
      <td>当前MX量化场景下支持0或1，其中0表示OCP实现，1表示cuBLAS实现。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax（float）</td>
      <td>输入</td>
      <td>表示maxType的取值，对应公式中的Amax(DType)。</td>
      <td>当前MXFP8场景仅支持0.0，表示Amax(DType)为量化结果数据类型的最大值。6.0-12.0为后续数据类型为FLOAT4_E2M1且blocksize为32的场景预留。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor *）</td>
      <td>输出</td>
      <td>表示激活并量化后的输出矩阵。</td>
      <td>必选参数。</td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yScale（aclTensor *）</td>
      <td>输出</td>
      <td>表示输出量化因子。</td>
      <td>必选参数。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t *）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor **）</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - x仅支持非转置输入，weight支持非转置和转置输入，接口会根据weight和weightScale的形状及转置关系推导weight是否转置。
    - weightScale转置属性需要与weight保持一致。
    - 空Tensor处理规则：
      - weight和weightScale作为必选输入，不支持空tensorlist，tensorlist中的元素不能为nullptr。
      - 支持M为0或N为0的空Tensor场景：x和xScaleOptional支持M为0，weight和weightScale支持N为0；该场景下允许K为0，第一段接口返回ACLNN_SUCCESS，workspaceSize为0；当M和N均不为0时，不支持K为0。
      - 支持groupList为空的空Tensor场景：当groupList为空，且输出y或yScale为空时，第一段接口返回ACLNN_SUCCESS，workspaceSize为0。

  <!-- end id7 -->

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>必选参数为空指针，例如x、groupList、xScaleOptional、weight、weightScale、activationType、y、yScale、workspaceSize或executor为空。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>activationType不是"gelu_tanh"。</td>
    </tr>
    <tr>
      <td>bias非空。</td>
    </tr>
    <tr>
      <td>输入数据类型或数据格式不支持。</td>
    </tr>
    <tr>
      <td>shape维度不匹配。</td>
    </tr>
    <tr>
      <td>属性值不在支持范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulActivationQuantWeightNz

- **参数说明**

  <table style="undefined;table-layout: fixed;width: 795px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 505px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>acl stream流。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
    - aclnnGroupedMatmulActivationQuantWeightNz默认确定性实现。
- 非空Tensor场景下，groupList第1维最小为1，最大为1024。
- MXFP8量化场景下需满足以下约束条件：
    - 数据类型需要满足下表：
      <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 180px">
      <col style="width: 160px">
      <col style="width: 160px">
      <col style="width: 160px">
      <col style="width: 180px">
      <col style="width: 160px">
      </colgroup>
      <thead>
        <tr>
          <th>MX量化场景</th>
          <th>激活函数</th>
          <th>x</th>
          <th>weight</th>
          <th>weightScale</th>
          <th>xScaleOptional</th>
          <th>y</th>
          <th>yScale</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>MXFP8</td>
          <td>gelu_tanh</td>
          <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
          <td>FLOAT8_E4M3FN</td>
          <td>FLOAT8_E8M0</td>
          <td>FLOAT8_E8M0</td>
          <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
          <td>FLOAT8_E8M0</td>
        </tr>
      </tbody>
      </table>

    - shape约束需要满足下表：
      <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 280px">
      <col style="width: 320px">
      <col style="width: 180px">
      <col style="width: 110px">
      <col style="width: 160px">
      </colgroup>
      <thead>
        <tr>
          <th>MX量化场景</th>
          <th>激活函数</th>
          <th>x</th>
          <th>weight（storageShape）</th>
          <th>weightScale</th>
          <th>xScale</th>
          <th>output</th>
          <th>outputScale</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>MXFP8</td>
          <td>gelu_tanh</td>
          <td>(M, K)</td>
          <td><ul>
          <li>非转置shape形如{(E, ceil(N / 32), ceil(K / 16), 16, 32)}</li>
          <li>转置shape形如{(E, ceil(K / 32), ceil(N / 16), 16, 32)}</li></ul></td>
          <td><ul>
          <li>非转置shape形如{(E, ceil(K / 64), N, 2)}</li>
          <li>转置shape形如{(E, N, ceil(K / 64), 2)}</li></ul></td>
          <td>(M, ceil(K / 64), 2)</td>
          <td>(M, N)</td>
          <td>(M, ceil(N / 64), 2)</td>
        </tr>
      </tbody>
      </table>

    - 表中xScale和outputScale的shape第三维为2，表示每个64元素的存储block中包含2个MX量化group，每个group覆盖32个元素。

    - N必须为64整数倍。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdint>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_activation_quant_weight_nz.h"

/*!
 * \file test_aclnn_grouped_matmul_activation_quant_weight_nz.cpp
 * \brief GroupedMatmulActivationQuant WeightNz MXFP8 aclnn example for Ascend 950PR/Ascend 950DT.
 */

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

template <typename T1, typename T2>
auto CeilDiv(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclFormat formatType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType, shape.data(),
        shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostData,
    const std::vector<std::vector<int64_t>> &shapes, void **deviceAddr, aclDataType dataType, aclFormat formatType,
    aclTensorList **tensorList)
{
    int size = static_cast<int>(shapes.size());
    std::vector<aclTensor *> tensors(size, nullptr);
    for (int i = 0; i < size; ++i) {
        int ret = CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i, dataType, formatType, &tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensorList = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorListWithStorageShape(const std::vector<std::vector<T>> &hostData,
    const std::vector<std::vector<int64_t>> &viewShapes, const std::vector<std::vector<int64_t>> &storageShapes,
    const std::vector<std::vector<int64_t>> &strides, void **deviceAddr, aclDataType dataType, aclFormat formatType,
    aclTensorList **tensorList)
{
    int size = static_cast<int>(storageShapes.size());
    std::vector<aclTensor *> tensors(size, nullptr);
    for (int i = 0; i < size; ++i) {
        auto storageSize = GetShapeSize(storageShapes[i]) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr + i, storageSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

        ret = aclrtMemcpy(*(deviceAddr + i), storageSize, hostData[i].data(), storageSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        tensors[i] = aclCreateTensor(viewShapes[i].data(), viewShapes[i].size(), dataType, strides[i].data(), 0,
            formatType, storageShapes[i].data(), storageShapes[i].size(), *(deviceAddr + i));
    }
    *tensorList = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

int RunGroupedMatmulActivationQuantWeightNz(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t e = 2;
    int64_t m = 4;
    int64_t k = 64;
    int64_t n = 128;
    int64_t kScaleBlock = CeilDiv(k, 64);
    int64_t nScaleBlock = CeilDiv(n, 64);

    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> groupListShape = {e};
    std::vector<std::vector<int64_t>> weightViewShapes = {{e, k, n}};
    std::vector<std::vector<int64_t>> weightStorageShapes = {{e, CeilDiv(n, 32), CeilDiv(k, 16), 16, 32}};
    std::vector<std::vector<int64_t>> weightStrides = {{k * n, n, 1}};
    std::vector<std::vector<int64_t>> weightScaleShapes = {{e, kScaleBlock, n, 2}};
    std::vector<int64_t> xScaleShape = {m, kScaleBlock, 2};
    std::vector<int64_t> yShape = {m, n};
    std::vector<int64_t> yScaleShape = {m, nScaleBlock, 2};

    std::vector<uint8_t> xHostData(GetShapeSize(xShape), 1);
    std::vector<int64_t> groupListHostData = {2, 2};
    std::vector<uint8_t> weightHostData(GetShapeSize(weightStorageShapes[0]), 1);
    std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleShapes[0]), 1);
    std::vector<uint8_t> xScaleHostData(GetShapeSize(xScaleShape), 1);
    std::vector<uint8_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<uint8_t> yScaleHostData(GetShapeSize(yScaleShape), 0);

    void *xDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *weightScaleDeviceAddr = nullptr;
    void *xScaleDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *yScaleDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *groupList = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *weightScale = nullptr;
    aclTensorList *bias = nullptr;
    aclTensor *xScale = nullptr;
    aclTensor *y = nullptr;
    aclTensor *yScale = nullptr;

    ret = CreateAclTensor<uint8_t>(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
        aclFormat::ACL_FORMAT_ND, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
        aclFormat::ACL_FORMAT_ND, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<std::vector<uint8_t>> weightHostDataList = {weightHostData};
    ret = CreateAclTensorListWithStorageShape<uint8_t>(weightHostDataList, weightViewShapes, weightStorageShapes,
        weightStrides, &weightDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorListPtr(
        weight, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<std::vector<uint8_t>> weightScaleHostDataList = {weightScaleHostData};
    ret = CreateAclTensorList<uint8_t>(weightScaleHostDataList, weightScaleShapes, &weightScaleDeviceAddr,
        aclDataType::ACL_FLOAT8_E8M0, aclFormat::ACL_FORMAT_ND, &weightScale);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightScaleTensorListPtr(
        weightScale, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
        aclFormat::ACL_FORMAT_ND, &xScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xScaleTensorPtr(xScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
        aclFormat::ACL_FORMAT_ND, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
        aclFormat::ACL_FORMAT_ND, &yScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yScaleTensorPtr(yScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> yScaleDeviceAddrPtr(yScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const char *activationType = "gelu_tanh";
    int64_t groupListType = 1;
    aclIntArray *tuningConfig = nullptr;
    const char *quantMode = "mx";
    const char *roundMode = "rint";
    int64_t scaleAlg = 0;
    float dstTypeMax = 0.0F;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    ret = aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(x, groupList, weight, weightScale, bias, xScale,
        activationType, groupListType, tuningConfig, quantMode, roundMode, scaleAlg, dstTypeMax, y, yScale,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    ret = aclnnGroupedMatmulActivationQuantWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulActivationQuantWeightNz failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("y shape is: [%ld, %ld]\n", yShape[0], yShape[1]);
    LOG_PRINT("yScale shape is: [%ld, %ld, %ld]\n", yScaleShape[0], yScaleShape[1], yScaleShape[2]);
    LOG_PRINT("GroupedMatmulActivationQuantWeightNz example execute success.\n");
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = RunGroupedMatmulActivationQuantWeightNz(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
        LOG_PRINT("RunGroupedMatmulActivationQuantWeightNz failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
