# GroupedMatmulFinalizeRouting

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作

## 参数说明

> 数据类型列中的角标说明：<sup>1</sup> 表示仅 <term>Ascend 950PR/Ascend 950DT</term> 支持的数据类型；<sup>2</sup> 表示仅 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 支持的数据类型。无角标表示全系列产品均支持。各产品详细的参数约束请参见对应的 [aclnn 接口文档](#调用说明)。

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 120px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>输入x(左矩阵)。</td>
      <td>INT8、FLOAT8_E5M2<sup>1</sup>、FLOAT8_E4M3FN<sup>1</sup>、HIFLOAT8<sup>1</sup>、FLOAT4_E2M1<sup>1</sup></td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>输入weight(右矩阵)</td>
      <td>INT4<sup>2</sup>、INT8、INT32<sup>2</sup>、FLOAT8_E5M2<sup>1</sup>、FLOAT8_E4M3FN<sup>1</sup>、HIFLOAT8<sup>1</sup>、FLOAT4_E2M1<sup>1</sup></td>
      <td>ND、NZ</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>输入</td>
      <td>量化参数中的缩放因子，perchannel量化参数</td>
      <td>INT64<sup>2</sup>、BFLOAT16、FLOAT32、FLOAT8_E8M0<sup>1</sup></td>
      <td>ND</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>矩阵的偏移</td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>输入</td>
      <td>非对称量化的偏移量</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional</td>
      <td>输入</td>
      <td>伪量化的缩放因子</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>输入</td>
      <td>伪量化的偏移量</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pertokenScaleOptional</td>
      <td>输入</td>
      <td>矩阵计算的反量化参数</td>
      <td>FLOAT32、FLOAT8_E8M0<sup>1</sup></td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupListOptional</td>
      <td>输入</td>
      <td>输入和输出分组轴方向的matmul大小分布</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sharedInputOptional</td>
      <td>输入</td>
      <td>moe计算中共享专家的输出，需要与moe专家的输出进行combine操作</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logitOptional</td>
      <td>输入</td>
      <td>moe专家对各个token的logit大小</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rowIndexOptional</td>
      <td>输入</td>
      <td>moe专家输出按照该rowIndex进行combine，其中的值即为combine做scatter add的索引</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>计算的输出类型：0：FLOAT32；1：FLOAT16；2：BFLOAT16。目前仅支持0。</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>sharedInputWeight</td>
      <td>属性</td>
      <td>共享专家与moe专家进行combine的系数，sharedInput先与该参数乘，然后在和moe专家结果累加。</td>
      <td>FLOAT</td>
      <td></td>
    </tr>
    <tr>
      <td>sharedInputOffset</td>
      <td>属性</td>
      <td>共享专家输出的在总输出中的偏移。</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>transposeX</td>
      <td>属性</td>
      <td>左矩阵是否转置，仅支持false。</td>
      <td>BOOL</td>
      <td></td>
    </tr>
    <tr>
      <td>transposeW</td>
      <td>属性</td>
      <td>右矩阵是否转置。<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>仅支持false；<term>Ascend 950PR/Ascend 950DT</term>支持true或false。</td>
      <td>BOOL</td>
      <td></td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>属性</td>
      <td>分组模式：配置为0：cumsum模式，即为前缀和；配置为1：count模式。</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>tuningConfigOptional</td>
      <td>属性</td>
      <td>数组中的第一个元素表示各个专家处理的token数的预期值，算子tiling时会按照数组的第一个元素合理进行tiling切分，性能更优。从第二个元素开始预留，用户无须填写。未来会进行扩展。兼容历史版本，用户如不使用该参数，不传入（即为nullptr）即可。</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出结果。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## 约束说明

<details>
<summary><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></summary>

输入和输出支持以下数据类型组合：

| x1   | x2         | scaleOptional | biasOptional | offsetOptional | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupListOptional | sharedInputOptional | logitOptional | rowIndexOptional | out     |
| ---- | ---------- | ------------- | ------------ | -------------- | ---------------------- | ----------------------- | --------------------- | ----------------- | ------------------- | ------------- | ---------------- | ------- |
| INT8 | INT4       | INT64         | FLOAT32      | FLOAT32        | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT32 |
| INT8 | INT4       | INT64         | FLOAT32      | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT32 |
| INT8 | INT8（NZ） | FLOAT32       | null         | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |
| INT8 | INT8（NZ） | FLOAT32       | null         | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |
| INT8 | INT4（NZ） | INT64         | FLOAT32      | FLOAT32        | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |
| INT8 | INT4（NZ） | INT64         | FLOAT32      | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |

</details>

<details>
<summary><term>Ascend 950PR/Ascend 950DT</term></summary>

**ND格式（aclnnGroupedMatmulFinalizeRoutingV3）支持的数据类型组合：**

仅支持MX全量化场景，相关信息参考[量化介绍](../../docs/zh/context/quant_mode_introduction.md)。offsetOptional、antiquantScaleOptional、antiquantOffsetOptional必须设置为空。

| MX量化场景 | x1                        | x2                         | scaleOptional | biasOptional     | pertokenScaleOptional | groupListOptional | sharedInputOptional | logitOptional | rowIndexOptional | out     |
| ---------- | ------------------------- | -------------------------- | ------------- | ---------------- | --------------------- | ----------------- | ------------------- | ------------- | ---------------- | ------- |
| MXFP8      | FLOAT8_E4M3FN / FLOAT8_E5M2 | FLOAT8_E4M3FN / FLOAT8_E5M2 | FLOAT8_E8M0   | BFLOAT16 / null      | FLOAT8_E8M0           | INT64             | BFLOAT16 / null         | FLOAT32       | INT64            | FLOAT32 |
| MXFP4      | FLOAT4_E2M1               | FLOAT4_E2M1                | FLOAT8_E8M0   | BFLOAT16 / null      | FLOAT8_E8M0           | INT64             | BFLOAT16 / null         | FLOAT32       | INT64            | FLOAT32 |

- x2维度为(e,k,n)，转置情况下维度为(e,n,k)，e取值范围[1,1024]。
- scaleOptional数据类型为FLOAT8_E8M0，转置属性必须和x2保持一致。
- pertokenScaleOptional数据类型为FLOAT8_E8M0。

**NZ格式（aclnnGroupedMatmulFinalizeRoutingWeightNzV2）支持的数据类型组合：**

| 量化场景 | x1                | x2                | scale           | bias          | pertokenScaleOptional | groupList | sharedInput | logit   | rowIndex | out   |
| ------- | ----------------- | ----------------- | --------------- | ------------- | --------------------- | --------- | ----------- | ------- | -------- | ----- |
| 全量化   | INT8              | INT8              | FLOAT/BFLOAT16      | BFLOAT16/null     | FLOAT/null            | INT64     | BFLOAT16        | FLOAT   | INT64/INT32 | FLOAT |
| 全量化   | FLOAT8_E4M3FN     | FLOAT8_E4M3FN     | FLOAT/BFLOAT16      | BFLOAT16/null     | FLOAT/null            | INT64     | BFLOAT16        | FLOAT   | INT64     | FLOAT |
| 全量化   | HIFLOAT8          | HIFLOAT8          | FLOAT/BFLOAT16      | BFLOAT16/null     | FLOAT/null            | INT64     | BFLOAT16        | FLOAT   | INT64     | FLOAT |
| 伪量化   | FLOAT8_E4M3FN     | FLOAT4_E2M1       | FLOAT8_E8M0     | BFLOAT16/null     | FLOAT8_E8M0           | INT64     | BFLOAT16        | FLOAT   | INT64     | FLOAT |

- x2的e取值范围[1,1024]，支持转置属性为true或false。
- 全量化场景中scale的shape为(e, 1, n)，pertokenScaleOptional的shape为(m)。
- 伪量化场景中x2的format为NZ_C0_32，transposeX2固定为true，k必须满足k%32==0，n必须满足n%32==0。

</details>

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_matmul_finalize_routing](examples/test_aclnn_grouped_matmul_finalize_routing.cpp) | 通过[aclnnGroupedMatmulFinalizeRoutingV3](docs/aclnnGroupedMatmulFinalizeRoutingV3.md)接口方式调用GroupedMatmulFinalizeRouting算子。 |
| aclnn调用 | [test_aclnn_grouped_matmul_finalize_routing_weight_nz](examples/arch35/test_aclnn_grouped_matmul_finalize_routing_weightnz.cpp) | 通过[aclnnGroupedMatmulFinalizeRoutingWeightNzV2](docs/aclnnGroupedMatmulFinalizeRoutingWeightNzV2.md)接口方式调用GroupedMatmulFinalizeRoutingWeightNz算子。 |
| aclnn调用 | [test_aclnn_grouped_matmul_finalize_routing_mx](examples/arch35/test_aclnn_grouped_matmul_finalize_routing_mx.cpp) | <term>Ascend 950PR/Ascend 950DT</term>下通过[aclnnGroupedMatmulFinalizeRoutingV3](docs/aclnnGroupedMatmulFinalizeRoutingV3.md)接口方式调用GroupedMatmulFinalizeRouting算子（MX量化场景）。 |
| aclnn调用 | [test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2_mxa8w4](examples/arch35/test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2_mxa8w4.cpp) | <term>Ascend 950PR/Ascend 950DT</term>下通过[aclnnGroupedMatmulFinalizeRoutingWeightNzV2](docs/aclnnGroupedMatmulFinalizeRoutingWeightNzV2.md)接口方式调用GroupedMatmulFinalizeRoutingWeightNz算子（MxA8W4量化场景）。 |
