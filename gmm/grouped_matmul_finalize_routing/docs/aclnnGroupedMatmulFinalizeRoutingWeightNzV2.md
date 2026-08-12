# aclnnGroupedMatmulFinalizeRoutingWeightNzV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_finalize_routing)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作，支持w为AI处理器亲和数据排布格式(NZ)。

本接口相较于aclnnGroupedMatmulFinalizeRoutingWeightNz，此接口新增：

- 新增入参offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、tuningConfigOptional，其中前三个参数当前为预留参数，暂不生效，传入空指针即可。

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：新增对INT4类型weight矩阵的支持，支持tuningConfigOptional调优参数，数组中的第一个值表示各个专家处理的token数的预期值，算子tiling时会按照该预期值合理进行tiling切分，性能更优。请根据实际情况选择合适的接口。
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR/Ascend 950DT</term>：新增Pertoken-perchannel、静态pertensor-perchannel和MxA8W4量化场景，相关信息参考[量化介绍](../../../docs/zh/context/quant_mode_introduction.md)。

<!-- end id8 -->

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnGroupedMatmulFinalizeRoutingWeightNzV2"接口执行计算。

```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
    const aclTensor   *x1,
    const aclTensor   *x2,
    const aclTensor   *scale,
    const aclTensor   *bias,
    const aclTensor   *offsetOptional,
    const aclTensor   *antiquantScaleOptional,
    const aclTensor   *antiquantOffsetOptional,
    const aclTensor   *pertokenScaleOptional,
    const aclTensor   *groupList,
    const aclTensor   *sharedInput,
    const aclTensor   *logit,
    const aclTensor   *rowIndex,
    int64_t            dtype,
    float              sharedInputWeight,
    int64_t            sharedInputOffset,
    bool               transposeX1,
    bool               transposeX2,
    int64_t            groupListType,
    const aclIntArray *tuningConfigOptional,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 400px">
  <col style="width: 230px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 190px">
  <col style="width: 145px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>输入x（左矩阵）。</td>
      <td>-</td>
      <td>INT8、FLOAT8_E4M3FN<sup>1</sup>、HIFLOAT8<sup>1</sup></td>
      <td>ND</td>
      <td>(m, k)</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>输入weight（右矩阵）。</td>
      <td>支持昇腾亲和数据排布格式(NZ)。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
      <td>INT8、INT4<sup>2</sup>、INT32<sup>2</sup>、FLOAT8_E4M3FN<sup>1</sup>、HIFLOAT8<sup>1</sup>、FLOAT4_E2M1<sup>1</sup></td>
      <td>NZ</td>
      <td>支持三维</td>
      <td>仅转置场景支持非连续</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>量化参数中的缩放因子，per-channel或Mx量化参数。</td>
      <td>-</td>
      <td>FLOAT、BF16、INT64<sup>2</sup>、FLOAT8_E8M0<sup>1</sup></td>
      <td>ND</td>
      <td>shape支持3-4维</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>矩阵的偏移。</td>
      <td>-</td>
      <td>BF16、FLOAT</td>
      <td>ND</td>
      <td>shape支持二维，维度为(e, n)，e、n和x2的e、n一致</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>输入</td>
      <td>非对称量化的偏移量。</td>
      <td>目前暂未启用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional</td>
      <td>输入</td>
      <td>伪量化的缩放因子。</td>
      <td>目前暂未启用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>输入</td>
      <td>伪量化的偏移量。</td>
      <td>目前暂未启用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>pertokenScaleOptional</td>
      <td>输入</td>
      <td>矩阵计算的反量化参数。</td>
      <td>-</td>
      <td>FLOAT、FLOAT8_E8M0<sup>1</sup></td>
      <td>ND</td>
      <td>shape支持1维和3维，维度为(m)，m和x1的m一致；MxA8W4场景为三维(m, ceil(k/64), 2)</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td>输入</td>
      <td>输入和输出分组轴方向的matmul大小分布。</td>
      <td>根据groupListType输入不同格式数据。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>shape支持一维，维度为(e)，e和x2的e一致</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>sharedInput</td>
      <td>输入</td>
      <td>moe计算中共享专家的输出，需要与moe专家的输出进行combine操作。</td>
      <td>-</td>
      <td>BF16</td>
      <td>ND</td>
      <td>支持二维，维度为(bsdp, n)，bsdp必须小于等于batchSize/e，n和x2的n一致。</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>logit</td>
      <td>输入</td>
      <td>moe专家对各个token的logit大小。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>shape支持一维，维度为(m)，m和x1的m一致</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>rowIndex</td>
      <td>输入</td>
      <td>moe专家输出按照该rowIndex进行combine，其中的值即为combine做scatter add的索引。</td>
      <td>-</td>
      <td>INT64、INT32</td>
      <td>ND</td>
      <td>shape支持一维，维度为(m)，m和x1的m一致</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>输入</td>
      <td>计算的输出类型：0：FLOAT；1：FLOAT16；2：BFLOAT16。目前仅支持0。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInputWeight</td>
      <td>输入</td>
      <td>共享专家与moe专家进行combine的系数，sharedInput先与该参数乘，然后在和moe专家结果累加。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInputOffset</td>
      <td>输入</td>
      <td>共享专家输出的在总输出中的偏移。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>输入</td>
      <td>左矩阵是否转置，仅支持false。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>输入</td>
      <td>右矩阵是否转置。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>输入</td>
      <td>分组模式：配置为0：cumsum模式，即为前缀和；配置为1：count模式。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tuningConfigOptional</td>
      <td>输入</td>
      <td>数组中的第一个元素表示各个专家处理的token数的预期值，算子tiling时会按照数组的第一个元素合理进行tiling切分，性能更优。数组中的第二个元素设置为1，则算子tiling时会根据实际输入尝试使用更适合的算法，当k<=2048的时候，性能可能更优。从第三个元素开始预留，用户无须填写。未来会进行扩展。兼容历史版本，用户如不使用该参数，不传入（即为nullptr）即可。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出结果。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch, n)</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 上表数据类型列中的角标"1"代表该系列不支持的数据类型。
    - x1仅支持INT8。维度m的取值范围为[1,16\*1024\*8]，k支持2048。
    - x2支持INT8、INT4以及INT32。当输入为INT32时维度为(e, k, n / 8)，输入转为INT4时维度为(e, k, n)，e取值范围[1,256]，k支持2048，n支持7168。只支持转置属性为false。
    - offsetOptional的shape支持三维，维度为(e, 1, n)，e、n和weight的e、n一致。
    - scale支持INT64、FLOAT、BF16。
    - rowIndex支持INT64、INT32。
    - x1、x2、groupList是必选参数，scale、pertokenScaleOptional、logit、rowIndex、bias、sharedInput是可选参数。
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 上表数据类型列中的角标"2"代表该系列不支持的数据类型。
    - rowIndex在x1以及x2数据类型为INT8时，数据类型支持INT64、INT32；在x1以及x2数据类型为FLOAT8_E4M3FN、HIFLOAT8或MxA8W4量化模式时，数据类型仅支持INT64。
    - x1、x2、scale、groupList、logit、rowIndex是必选参数，pertokenScaleOptional、sharedInput、bias是可选参数。MxA8W4场景下pertokenScaleOptional为必选参数。目前暂不支持offsetOptional参数，必须为nullptr。
    - 当groupListType为0时，groupList须为非负单调非递减数列（累积和），且最后一个值不大于x1中tensor的第一维；当groupListType为1时，groupList须为非负数组（各组大小），且数值的总和不大于x1中tensor的第一维。
    - out的第一维batch、sharedInputOffset必须大于等于0，且小于等于m。
    - 当sharedInput不为空时，其第一维长度为shareInputLen。sharedInputOffset、shareInputLen以及两者之和均需小于等于out的第一维batch。
    - x1支持M为0，x2支持N为0；其他空Tensor场景不支持。

  <!-- end id10 -->

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 700px">
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
      <td>传入参数是必选输入、输出或者必须属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>x1、x2、scale、bias、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、pertokenScaleOptional、groupList、sharedInput、logit、rowIndex、sharedInputWeight、sharedInputOffset、transposeX1、transposeX2、或out的数据类型或数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>x1、x2、scale、bias、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、pertokenScaleOptional、groupList、sharedInput、logit、rowIndex或out的shape不满足校验条件。</td>
    </tr>
    <tr>
      <td>除x1的M为0或x2的N为0外，传入了其他空Tensor。</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatmulFinalizeRoutingWeightNzV2

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
    <col style="width: 173px">
    <col style="width: 112px">
    <col style="width: 668px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：

  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：aclnnGroupedMatmulFinalizeRoutingWeightNzV2默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：aclnnGroupedMatmulFinalizeRoutingWeightNzV2默认非确定性实现，仅支持在输入x1和x2都是int8类型时，通过aclrtCtxSetSysParamOpt开启确定性。

  <!-- end id12 -->

<!-- npu="A3,910b" id13 -->
<details>
<summary><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></summary>

  - 公共约束
    - groupList：当groupListType为0时，groupList必须为非负单调非递减数列；当groupListType为1时，groupList必须为非负数列。
    - x1维度m的取值范围为 $[1, 16*1024*8]$，k支持2048。
    - x2的e取值范围$[1, 256]$，k支持2048，n支持7168。只支持转置属性为false。
    - offsetOptional的shape支持三维，维度为(e, 1, n)，e、n和x2的e、n一致。

  - 输入和输出支持以下数据类型组合

    | x1    | x2    | scale   | bias    | offsetOptional  | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupList | sharedInput | logit   |   rowIndex | out   | tuningConfigOptional |
    |------|------|---------|---------|---------|----------------|-----------------|---------------|-----------|-------------|---------|----------|-------| ----------------------|
    | INT8 | INT8 | FLOAT | null    | null    | null           | null            | FLOAT       | INT64     | BFLOAT16    | FLOAT | INT64    | FLOAT |   IntArray             |
    | INT8 | INT8 | FLOAT | null    | null    | null           | null            | FLOAT       | INT64     | BFLOAT16    | FLOAT | INT64    | FLOAT |   IntArray             |
    | INT8 | INT4 | INT64   | FLOAT | FLOAT | null           | null            | FLOAT       | INT64     | BFLOAT16    | FLOAT | INT64    | FLOAT |   IntArray             |
    | INT8 | INT4 | INT64   | FLOAT | null    | null           | null            | FLOAT       | INT64     | BFLOAT16    | FLOAT | INT64    | FLOAT |   IntArray             |

</details>
<!-- end id13 -->

<!-- npu="950" id14 -->
<details>
<summary><term>Ascend 950PR/Ascend 950DT</term></summary>

  - 公共约束
    - groupList：当groupListType为0时，groupList必须为非负单调非递减数列；当groupListType为1时，groupList必须为非负数列。
    - x2的e取值范围[1, 1024]。
    - sharedInput的n和x2的n一致。
    - x1支持M为0，x2支持N为0；其他空Tensor场景不支持。

  - 全量化场景支持的数据类型为：

    | x1    | x2    | scale   | bias    | offsetOptional | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupList | sharedInput | logit | rowIndex | out   | tuningConfigOptional |
    |------|------|---------|---------|---------|----------|-------|------|------|------|-------|--------|-------|------|
    | INT8 | INT8 | FLOAT/BFLOAT16 | BFLOAT16/null | null | null | null | FLOAT/null | INT64 | BFLOAT16 | FLOAT | INT64/INT32 | FLOAT | null |
    | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT/BFLOAT16 | BFLOAT16/null | null | null | null | FLOAT/null | INT64 | BFLOAT16 | FLOAT | INT64 | FLOAT | null |
    | HIFLOAT8 | HIFLOAT8 | FLOAT/BFLOAT16 | BFLOAT16/null | null | null | null | FLOAT/null | INT64 | BFLOAT16 | FLOAT | INT64 | FLOAT | null |

    - x2支持转置属性为true或者false，非转置下维度为(e, k, n)，转置下维度为(e, n, k)。
    - scale的shape为(e, 1, n)，e、n和x2的e、n一致。
    - pertokenScaleOptional的shape为(m)，m和x1的m一致。

  - 伪量化场景支持的数据类型为：

    | x1    | x2    | scale   | bias    | offsetOptional | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupList | sharedInput | logit | rowIndex | out   | tuningConfigOptional |
    |------|------|---------|---------|---------|----------|-------|------|------|------|-------|--------|-------|------|
    | FLOAT8_E4M3FN | FLOAT4_E2M1 | FLOAT8_E8M0 | BFLOAT16/null | null | null | null | FLOAT8_E8M0 | INT64 | BFLOAT16 | FLOAT | INT64 | FLOAT | null |

    - x2的format为NZ_C0_32，shape为(e, n, k)，对应NZ格式的shape为(e, ceil(k/32), ceil(n/16), 16, 32)。
    - transposeX2固定为true。
    - pertokenScaleOptional为必选参数，shape为(m, ceil(k/64), 2)。
    - scale的shape为(e, n, ceil(k/64), 2)。
    - k必须满足k % 32 == 0。
    - n必须满足n % 32 == 0。

</details>
<!-- end id14 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<!-- npu="A3,910b" id15 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    ```Cpp
    #include <iostream>
    #include <memory>
    #include <vector>

    #include "acl/acl.h"
    #include "aclnnop/aclnn_permute.h"
    #include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
    #include "aclnnop/aclnn_trans_matmul_weight.h"

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
        for (auto i : shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    int Init(int32_t deviceId, aclrtStream *stream)
    {
        // 固定写法，资源初始化
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSetDevice(deviceId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return 0;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        // 调用aclrtMalloc申请device侧内存
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // 计算连续tensor的strides
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        // 调用aclCreateTensor接口创建aclTensor
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    template <typename T>
    int CreateAclTensorWeight(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                          aclDataType dataType, aclTensor **tensor)
    {
        auto size = static_cast<uint64_t>(GetShapeSize(shape));

        const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
        auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                  return ret);
        size *= sizeof(T);

        // 调用aclrtMalloc申请device侧内存
        ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // 计算连续tensor的strides
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        std::vector<int64_t> storageShape;
        storageShape.push_back(GetShapeSize(shape));

        // 调用aclCreateTensor接口创建aclTensor
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  storageShape.data(), storageShape.size(), *deviceAddr);
        return 0;
    }

      int main() {
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = Init(deviceId, &stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        int64_t m = 192;
        int64_t k = 2048;
        int64_t n = 7168;
        int64_t e = 4;
        int64_t batch = 24;
        int64_t bsdp = 8;
        int64_t dtype = 0;
        float shareInputWeight = 1.0;
        int64_t sharedInputOffset = 0;
        bool transposeX = false;
        bool transposeW = false;
        int64_t groupListType = 1;

        std::vector<int64_t> xShape = {m, k};
        std::vector<int64_t> wShape = {e, k, n};
        std::vector<int64_t> scaleShape = {e, n};
        std::vector<int64_t> pertokenScaleShape = {m};
        std::vector<int64_t> groupListShape = {e};
        std::vector<int64_t> sharedInputShape = {bsdp, n};
        std::vector<int64_t> logitShape = {m};
        std::vector<int64_t> rowIndexShape = {m};
        std::vector<int64_t> outShape = {batch, n};
        std::vector<int64_t> tuningConfigVal = {1};

        void *xDeviceAddr = nullptr;
        void *wDeviceAddr = nullptr;
        void *scaleDeviceAddr = nullptr;
        void *pertokenScaleDeviceAddr = nullptr;
        void *groupListDeviceAddr = nullptr;
        void *sharedInputDeviceAddr = nullptr;
        void *logitDeviceAddr = nullptr;
        void *rowIndexDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *tuningConfigDeviceAddr = nullptr;

        aclTensor* x = nullptr;
        aclTensor* w = nullptr;
        aclTensor* bias = nullptr;
        aclTensor* groupList = nullptr;
        aclTensor* scale = nullptr;
        aclTensor* pertokenScale = nullptr;
        aclTensor* sharedInput = nullptr;
        aclTensor* logit = nullptr;
        aclTensor* rowIndex = nullptr;
        aclTensor* out = nullptr;

        std::vector<int8_t> xHostData(GetShapeSize(xShape));
        std::vector<int8_t> wHostData(GetShapeSize(wShape));
        std::vector<float> scaleHostData(GetShapeSize(scaleShape));
        std::vector<float> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
        std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
        groupListHostData[0] = 7;
        groupListHostData[1] = 32;
        groupListHostData[2] = 40;
        groupListHostData[3] = 64;

        std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
        std::vector<int64_t> logitHostData(GetShapeSize(logitShape));
        std::vector<float> rowIndexHostData(GetShapeSize(rowIndexShape));
        std::vector<float> outHostData(GetShapeSize(outShape));

        // 创建x aclTensor
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建AI处理器亲和数据排布格式的w aclTensor
        ret = CreateAclTensorWeight(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_INT8, &w);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建scale aclTensor
        ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建pertokenScale aclTensor
        ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &pertokenScale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建groupList aclTensor
        ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建sharedInput aclTensor
        ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16, &sharedInput);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建logit aclTensor
        ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建rowIndex aclTensor
        ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建out aclTensor
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建tuningConfig aclIntArray
        aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
        std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> tuningConfigIntArrayPtr(tuningConfig, aclDestroyIntArray);
        std::unique_ptr<void, aclError (*)(void *)> tuningConfigDeviceAddrPtr(tuningConfigDeviceAddr, aclrtFree);
        CHECK_RET(tuningConfig != nullptr, -1);
        // 3. 调用CANN算子库API，需要修改为具体的Api名称
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        // 调用aclnnTransMatmulWeight第一段接口
        ret = aclnnTransMatmulWeightGetWorkspaceSize(w, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用aclnnTransMatmulWeight第二段接口
        ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

        // 调用aclnnGroupedMatmulFinalizeRoutingWeightNzV2第一段接口
        workspaceSize = 0;
        ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(x, w, scale, nullptr, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype, shareInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, tuningConfig, out, &workspaceSize, &executor);

        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用aclnnGroupedMatmulFinalizeRoutingWeightNzV2第二段接口
        ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2 failed. ERROR: %d\n", ret); return ret);

        // 4.（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
        auto size = GetShapeSize(outShape);
        std::vector<float> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("result[%lld] is: %f\n", i, resultData[i]);
        }

        // 6. 释放aclTensor资源，需要根据具体API的接口定义修改
        aclDestroyTensor(x);
        aclDestroyTensor(w);
        aclDestroyTensor(scale);
        aclDestroyTensor(pertokenScale);
        aclDestroyTensor(groupList);
        aclDestroyTensor(sharedInput);
        aclDestroyTensor(logit);
        aclDestroyTensor(rowIndex);
        aclDestroyTensor(out);

        // 7.释放device资源，需要根据具体API的接口定义修改
        aclrtFree(xDeviceAddr);
        aclrtFree(wDeviceAddr);
        aclrtFree(scaleDeviceAddr);
        aclrtFree(pertokenScaleDeviceAddr);
        aclrtFree(groupListDeviceAddr);
        aclrtFree(sharedInputDeviceAddr);
        aclrtFree(logitDeviceAddr);
        aclrtFree(rowIndexDeviceAddr);
        aclrtFree(outDeviceAddr);
        aclDestroyIntArray(tuningConfig);

        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }
    ```

<!-- end id15 -->
<!-- npu="950" id16 -->
- <term>Ascend 950PR/Ascend 950DT</term>：

  - Pertoken量化数据流示例：

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"

  #define CHECK_RET(cond, return_expr)                                                                                   \
      do {                                                                                                               \
          if (!(cond)) {                                                                                                 \
              return_expr;                                                                                               \
          }                                                                                                              \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr)                                                                              \
      do {                                                                                                               \
          if (!(cond)) {                                                                                                 \
              Finalize(deviceId, stream);                                                                                \
              return_expr;                                                                                               \
          }                                                                                                              \
      } while (0)

  #define LOG_PRINT(message, ...)                                                                                        \
      do {                                                                                                               \
          printf(message, ##__VA_ARGS__);                                                                                \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      // 固定写法，资源初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWeightNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                              aclDataType dataType, aclTensor **tensor)
  {
      // 计算NZ格式的storageShape
      int64_t e = shape[0];
      int64_t k = shape[1];
      int64_t n = shape[2];
      std::vector<int64_t> shapeNz = {e, n / 64, k / 16, 16L, 64L};

      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建NZ格式的aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ, shapeNz.data(), shapeNz.size(), *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1.（固定写法）device/stream初始化，参考AscendCL对外接口列表
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t m = 192;
      int64_t k = 2048;
      int64_t n = 7168;
      int64_t e = 4;
      int64_t batch = 24;
      int64_t bsdp = 8;
      int64_t dtype = 0;
      float shareInputWeight = 1.0;
      int64_t sharedInputOffset = 0;
      bool transposeX = false;
      bool transposeW = false;
      int64_t groupListType = 1;

      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> wShape = {e, k, n};
      std::vector<int64_t> scaleShape = {e, 1, n};
      std::vector<int64_t> pertokenScaleShape = {m};
      std::vector<int64_t> groupListShape = {e};
      std::vector<int64_t> sharedInputShape = {bsdp, n};
      std::vector<int64_t> logitShape = {m};
      std::vector<int64_t> rowIndexShape = {m};
      std::vector<int64_t> outShape = {batch, n};

      void *xDeviceAddr = nullptr;
      void *wDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *pertokenScaleDeviceAddr = nullptr;
      void *groupListDeviceAddr = nullptr;
      void *sharedInputDeviceAddr = nullptr;
      void *logitDeviceAddr = nullptr;
      void *rowIndexDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;

      aclTensor *x = nullptr;
      aclTensor *w = nullptr;
      aclTensor *groupList = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *pertokenScale = nullptr;
      aclTensor *sharedInput = nullptr;
      aclTensor *logit = nullptr;
      aclTensor *rowIndex = nullptr;
      aclTensor *out = nullptr;

      std::vector<int8_t> xHostData(GetShapeSize(xShape));
      std::vector<int8_t> wHostData(GetShapeSize(wShape));
      std::vector<float> scaleHostData(GetShapeSize(scaleShape));
      std::vector<float> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
      std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
      // 对groupList赋值
      groupListHostData[0] = 64;

      std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
      std::vector<int64_t> logitHostData(GetShapeSize(logitShape));
      std::vector<float> rowIndexHostData(GetShapeSize(rowIndexShape));
      std::vector<float> outHostData(GetShapeSize(outShape));

      // 创建x aclTensor
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_HIFLOAT8, &x);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // 创建AI处理器亲和数据排布格式(NZ格式)的w aclTensor
      ret = CreateAclTensorWeightNz(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_HIFLOAT8, &w);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // 创建scale aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // 创建pertokenScale aclTensor
      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, aclDataType::ACL_FLOAT,
                            &pertokenScale);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // 创建groupList aclTensor
      ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // 创建sharedInput aclTensor
      ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16,
                            &sharedInput);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // 创建logit aclTensor
      ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // 创建rowIndex aclTensor
      ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // 调用aclnnGroupedMatmulFinalizeRoutingWeightNzV2第一段接口
      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
          x, w, scale, nullptr, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype,
          shareInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, nullptr, out, &workspaceSize,
          &executor);
      CHECK_FREE_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                    return ret);

      // 根据第一段接口计算出的workspaceSize申请device内存
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // 调用aclnnGroupedMatmulFinalizeRoutingWeightNzV2第二段接口
      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2 failed. ERROR: %d\n", ret); return ret);

      // 4.（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                    return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%lld] is: %f\n", static_cast<long long>(i), resultData[i]);
      }

      // 6. 释放aclTensor资源，需要根据具体API的接口定义修改
      aclDestroyTensor(x);
      aclDestroyTensor(w);
      aclDestroyTensor(scale);
      aclDestroyTensor(pertokenScale);
      aclDestroyTensor(groupList);
      aclDestroyTensor(sharedInput);
      aclDestroyTensor(logit);
      aclDestroyTensor(rowIndex);
      aclDestroyTensor(out);

      // 7.释放device资源，需要根据具体API的接口定义修改
      aclrtFree(xDeviceAddr);
      aclrtFree(wDeviceAddr);
      aclrtFree(scaleDeviceAddr);
      aclrtFree(pertokenScaleDeviceAddr);
      aclrtFree(groupListDeviceAddr);
      aclrtFree(sharedInputDeviceAddr);
      aclrtFree(logitDeviceAddr);
      aclrtFree(rowIndexDeviceAddr);
      aclrtFree(outDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```

  - MxA8W4数据流示例：

  ``` Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include <cmath>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

  #define CHECK_RET(cond, return_expr)                                                                                   \
      do {                                                                                                               \
          if (!(cond)) {                                                                                                 \
              return_expr;                                                                                               \
          }                                                                                                              \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr)                                                                              \
      do {                                                                                                               \
          if (!(cond)) {                                                                                                 \
              Finalize(deviceId, stream);                                                                                \
              return_expr;                                                                                               \
          }                                                                                                              \
      } while (0)

  #define LOG_PRINT(message, ...)                                                                                        \
      do {                                                                                                               \
          printf(message, ##__VA_ARGS__);                                                                                \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int64_t CeilDiv(int64_t a, int64_t b)
  {
      if (b == 0) return 0;
      if (a <= 0) return 0;
      return (a - 1) / b + 1;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides(shape.size(), 1L);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(const std::vector<T> &hostData, const std::vector<int64_t> &shape, int64_t *storageShape,
                                uint64_t storageShapeSize, void **deviceAddr, aclDataType dataType, aclTensor **tensor,
                                aclFormat format)
  {
      auto size = hostData.size() * sizeof(T);
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, storageShape,
                                storageShapeSize, *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor, aclrtStream &stream)
  {
      void *srcDeviceAddr = nullptr;
      aclTensor *srcTensor = nullptr;
      auto size = hostData.size() * sizeof(T);

      auto ret = aclrtMalloc(&srcDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(srcDeviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides(shape.size(), 1L);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      srcTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), srcDeviceAddr);

      int64_t *dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, 29, aclFormat::ACL_FORMAT_ND, &dstShape, &dstShapeSize,
                                                     &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
                return ret);

      aclTensor *dstTensor = nullptr;
      void *dstDeviceAddr = nullptr;

      uint64_t tensorSize = 1;
      for (uint64_t i = 0; i < dstShapeSize; i++) {
          tensorSize *= dstShape[i];
      }
      std::vector<T> dstTensorHostData(tensorSize, 0);

      ret = CreateAclTensorWithFormat(dstTensorHostData, shape, dstShape, dstShapeSize, &dstDeviceAddr, dataType,
                                      &dstTensor, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      *tensor = dstTensor;
      *deviceAddr = dstDeviceAddr;

      aclDestroyTensor(srcTensor);
      aclrtFree(srcDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }

      return ACL_SUCCESS;
  }

  int RunExample(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

      int64_t m = 192;
      int64_t k = 2048;
      int64_t n = 7168;
      int64_t e = 4;
      int64_t batch = 24;
      int64_t bsdp = 8;
      int64_t dtype = 0;
      float shareInputWeight = 1.0;
      int64_t sharedInputOffset = 0;
      bool transposeX = false;
      bool transposeW = true;
      int64_t groupListType = 1;
      int64_t ceil_k_64 = CeilDiv(k, 64);

      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> wShape = {e, n, k};
      std::vector<int64_t> scaleShape = {e, n, ceil_k_64, 2};
      std::vector<int64_t> pertokenScaleShape = {m, ceil_k_64, 2};
      std::vector<int64_t> groupListShape = {e};
      std::vector<int64_t> sharedInputShape = {bsdp, n};
      std::vector<int64_t> logitShape = {m};
      std::vector<int64_t> rowIndexShape = {m};
      std::vector<int64_t> outShape = {batch, n};

      void *xDeviceAddr = nullptr;
      void *wDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *pertokenScaleDeviceAddr = nullptr;
      void *groupListDeviceAddr = nullptr;
      void *sharedInputDeviceAddr = nullptr;
      void *logitDeviceAddr = nullptr;
      void *rowIndexDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;

      aclTensor *x = nullptr;
      aclTensor *w = nullptr;
      aclTensor *groupList = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *pertokenScale = nullptr;
      aclTensor *sharedInput = nullptr;
      aclTensor *logit = nullptr;
      aclTensor *rowIndex = nullptr;
      aclTensor *out = nullptr;

      std::vector<uint8_t> xHostData(GetShapeSize(xShape));
      std::vector<uint8_t> wHostData(GetShapeSize(wShape));
      std::vector<uint8_t> scaleHostData(GetShapeSize(scaleShape));
      std::vector<uint8_t> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
      std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape), m/e);

      std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
      std::vector<float> logitHostData(GetShapeSize(logitShape));
      std::vector<float> rowIndexHostData(GetShapeSize(rowIndexShape));
      std::vector<float> outHostData(GetShapeSize(outShape));

      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);

      ret = CreateAclTensorNz(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &w, stream);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);

      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);

      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                            aclDataType::ACL_FLOAT8_E8M0, &pertokenScale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale,
                                                                                            aclDestroyTensor);

      ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);

      ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16,
                            &sharedInput);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);

      ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);

      ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);

      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(workspaceAddr, aclrtFree);

      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
          x, w, scale, nullptr, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype,
          shareInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, nullptr, out, &workspaceSize,
          &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }

      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2 failed. ERROR: %d\n", ret);
                return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      auto size = GetShapeSize(outShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < std::min((int64_t)10, size); i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }

      return ACL_SUCCESS;
  }

  int main()
  {
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = RunExample(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunExample failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- end id16 -->
