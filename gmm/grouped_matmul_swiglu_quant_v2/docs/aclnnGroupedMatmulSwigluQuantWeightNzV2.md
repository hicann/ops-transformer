# aclnnGroupedMatmulSwigluQuantWeightNzV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_swiglu_quant_v2)

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

- 接口功能：融合GroupedMatmul、dequant、swiglu和quant，详细解释见计算公式。[aclnnGroupedMatmulSwigluQuantV2](./aclnnGroupedMatmulSwigluQuantV2.md)接口的weightNZ特化版本，此接口与aclnnGroupedMatmulSwigluQuantV2的区别在于：weight参数在该场景下强制视为FRACTAL_NZ格式。

- 计算公式：

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    <details>
    <summary>量化场景A8W8（A指激活矩阵，W指权重矩阵，8指INT8数据类型）：</summary>
    <a id="量化场景A8W8"></a>

      - **定义**：

        * **⋅** 表示矩阵乘法。
        * **⊙** 表示逐元素乘法。
        * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
        * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **输入**：

        * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
        * $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        * $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，N是输出维度。
        * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数。
        * $grouplist∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
      - **输出**：

        * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
        * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。

      - **计算过程**

        - 1.根据groupList[i]确定当前分组的token ，$i \in [0,Len(groupList)]$。

          >例子：假设groupList=[3,4,4,6]、groupListType=cumsum或groupList=[3,1,0,2]、groupListType=count。
          >
          >注：以上两种不同的分组方式，实际为相同的分组结果。
          >
          >第0个右矩阵`W[0,:,:]`，对应索引位置[0,3)的token`x[0:3]`（共3-0=3个token），对应`x_scale[0:3]`、`w_scale[0]`、`bias[0]`、`offset[0]`、`Q[0:3]`、`Q_scale[0:3]`、`Q_offset[0:3]`；
          >
          >第1个右矩阵`W[1,:,:]`，对应索引位置[3,4)的token`x[3:4]`（共4-3=1个token），对应`x_scale[3:4]`、`w_scale[1]`、`bias[1]`、`offset[1]`、`Q[3:4]`、`Q_scale[3:4]`、`Q_offset[3:4]`；
          >
          >第2个右矩阵`W[2,:,:]`，对应索引位置[4,4)的token`x[4:4]`（共4-4=0个token），对应`x_scale[4:4]`、`w_scale[2]`、`bias[2]`、`offset[2]`、`Q[4:4]`、`Q_scale[4:4]`、`Q_offset[4:4]`；
          >
          >第3个右矩阵`W[3,:,:]`，对应索引位置[4,6)的token`x[4:6]`（共6-4=2个token），对应`x_scale[4:6]`、`w_scale[3]`、`bias[3]`、`offset[3]`、`Q[4:6]`、`Q_scale[4:6]`、`Q_offset[4:6]`；
          >
          >注：grouplist中未指定的部分将不会参与更新。
          >例如当groupList=[12,14,18]、GroupListType=cumsum，X的shape为[30，:]时。
          >
          >则第一个输出Q的shape为[30,:]，其中Q[18:,：]的部分不会进行更新和初始化，其中数据为显存空间申请时的原数据。
          >
          >同理，第二个输出Q_scale的shape为[30]，其中Q\_scale[18:]的部分不会进行更新或初始化，其中数据为显存空间申请时的原数据。
          >
          >即输出的Q[:grouplist[-1],:]和Q\_scale[:grouplist[-1]]为有效数据部分。

        - 2.根据分组确定的入参进行如下计算：

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$

        - 3.量化输出结果

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>
    <details>
    <summary>MSD场景A8W4（A指激活矩阵，W指权重矩阵，8指INT8数据类型，4指INT4数据类型）：</summary>
    <a id="MSD场景A8W4"></a>

      - **定义**：
        * **⋅** 表示矩阵乘法。
        * **⊙** 表示逐元素乘法。
        * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
        * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
        * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **输入**：
        * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
        * $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        * $weightAssistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（生成辅助矩阵的计算过程见下文）。
        * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，K\_group\_num是在K轴维度上的分组数，N是输出维度。
        * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数。
        * $grouplist∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
      - **输出**：
        * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
        * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
      - **计算过程**
        - 1.根据groupList[i]确定当前分组的token，$i \in [0,Len(groupList)]$。
          - 分组逻辑与A8W8相同。
        - 2.生成辅助矩阵（weightAssistMatrix）的计算过程（请注意weightAssistMatrix部分计算为离线生成作为输入，并非算子内部完成）：
          - 当为per-channel量化（$w\_scale$为2维）：

            $weightAssistMatrix_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$

          - 当为per-group量化（$w\_scale$为3维）：

            $weightAssistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$

            注：$num\_per\_group = K // K\_group\_num$

        - 3.根据分组确定的入参进行如下计算：

          - 3.1.将左矩阵$\mathbb{Z_8}$，转变为高低位两部分的$\mathbb{Z_4}$
            $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
            $X\_low\_4bits_{i} = X_{i} \& 0x0f - 8$
          - 3.2.做矩阵乘时，开启per-channel或per-group量化
            per-channel：

            $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            per-group：

            $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

            $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

          - 3.3.将高低位的矩阵乘结果还原为整体的结果

            $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$

            $C_{i,act}, gate_{i} = split(C_{i})$

            $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$

        - 3.量化输出结果

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>
    <details>
    <summary>量化场景A4W4（A指激活矩阵，W指权重矩阵，4指INT4数据类型）：</summary>
    <a id="量化场景A4W4"></a>

      - **定义**：

        * **⋅** 表示矩阵乘法。
        * **⊙** 表示逐元素乘法。
        * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
        * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
        * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **输入**：

        * $X∈\mathbb{Z_4}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
        * $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        * $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，N是输出维度。
        * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数。
        * $smoothScale∈\mathbb{R}^{E \times N/2}(逐channel)或\mathbb{R}^{E}(逐tensor)$：平滑缩放因子，E是专家个数，N是输出维度。
        * $grouplist∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
      - **输出**：

        * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
        * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。

      - **计算过程**

        - 1.根据groupList[i]确定当前分组的token ，$i \in [0,Len(groupList)]$。
          - 分组逻辑与A8W8相同。

        - 2.根据分组确定的入参进行如下计算：

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$

          $S_{i} = S_{i} \odot smoothScale_{i\ Broadcast}$

        - 3.量化输出结果

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    <details>
    <summary>MX量化场景：</summary>

      - **定义**：

        * **⋅** 表示矩阵乘法。
        * **⊙** 表示逐元素乘法。
      - **输入**：

        * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
        * $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        * $w\_scale∈\mathbb{R}^{E \times ceil(K / 64) \times N \times 2}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，K是特征维度， N是输出维度。
        * $x\_scale∈\mathbb{R}^{M \times ceil(K / 64) \times 2}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数，K是特征维度。
        * $grouplist∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
      - **输出**：

        * $Q∈\mathbb{F}_{8E4M3FN}^{M \times N / 2}$或$Q∈\mathbb{F}_{4E1M2}^{M \times N / 2}$或$Q∈\mathbb{F}_{4E2M1}^{M \times N / 2}$：量化后的输出矩阵，数据类型支持FLOAT8_E4M3FN、FLOAT4_E1M2、FLOAT4_E2M1。
        * $Q\_scale∈\mathbb{R}^{M \times ceil((N / 2) / 64) \times 2}$：量化缩放因子。
      - **计算过程**
        - 1.根据groupList[i]确定当前分组的token ，$i \in [0,Len(groupList)]$

        - 2.根据分组确定的入参进行如下计算：

          $C_{i} = (X_{i}\cdot W_{i} )\odot xScale_{i\ Broadcast} \odot wScale_{i\ Broadcast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$，其中$Swish(x)=\frac{x}{1+e^{-x}}$

        - 3.量化输出结果

          $shared\_exp = \left\lfloor \log_2(max_i(|S_i|)) \right\rceil - emax$

          $QScale = 2 ^ {shared\_exp}$

          $Q_i = quantize\_to\_element\_format(S_i/Qscale), \space i\space from\space 1\space to\space blocksize$
          - $emax$: 对应数据类型的最大正则数的指数位。

            |   DataType    | emax |
            | :-----------: | :--: |
            | FLOAT8_E4M3FN |  8   |
            | FLOAT4_E1M2   |  1   |
            | FLOAT4_E2M1   |  2   |

          - $blocksize$：指每次量化的元素个数，仅支持32。
    </details>

  <!-- end id8 -->

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnGroupedMatmulSwigluQuantWeightNzV2"接口执行计算。

```Cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
    const aclTensor     *x,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensorList *weightAssistMatrix,
    const aclTensor     *bias,
    const aclTensor     *xScale,
    const aclTensor     *smoothScale,
    const aclTensor     *groupList,
    int64_t              dequantMode,
    int64_t              dequantDtype,
    int64_t              quantMode,
    int64_t              groupListType,
    const aclIntArray   *tuningConfig,
    aclTensor           *output,
    aclTensor           *outputScale,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize

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
        <td>x</td>
        <td rowspan="1">输入</td>
        <td>表示左矩阵，对应公式中的X。</td>
        <td>-</td>
        <td>INT8<sup>2</sup>、INT4<sup>2</sup>、INT32<sup>2</sup>、FLOAT8_E4M3FN<sup>1</sup>、FLOAT4_E2M1<sup>1</sup>、FLOAT4_E1M2<sup>1</sup></td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td rowspan="1">输入</td>
        <td>表示权重矩阵，对应公式中的W。</td>
        <td>此接口weight强制视为FRACTAL_NZ格式。</td>
        <td>INT8<sup>2</sup>、INT4<sup>2</sup>、INT32<sup>2</sup>、FLOAT8_E4M3FN<sup>1</sup>、FLOAT4_E2M1<sup>1</sup>、FLOAT4_E1M2<sup>1</sup></td>
        <td>FRACTAL_NZ</td>
        <td>4、5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightScale</td>
        <td rowspan="1">输入</td>
        <td>表示右矩阵的量化因子，公式中的wScale。</td>
        <td>首轴长度需与weight的首轴维度相等，尾轴长度需要与weight还原为ND格式的尾轴相同。</td>
        <td>UINT64<sup>2</sup>、FLOAT<sup>2</sup>、FLOAT16<sup>2</sup>、BFLOAT16<sup>2</sup>、FLOAT8_E8M0<sup>1</sup></td>
        <td>ND</td>
        <td>1、2、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightAssistMatrix</td>
        <td rowspan="1">可选输入</td>
        <td>表示计算矩阵乘时的辅助矩阵，公式中的weightAssistMatrix。</td>
        <td><ul>
          <li>仅A8W4场景生效，其他场景需传空指针。</li>
          <li>首轴长度需与weight的首轴维度相等，尾轴长度需要与weight还原为ND格式的尾轴相同。</li>
        </ul></td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1、2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td rowspan="1">可选输入</td>
        <td>表示矩阵乘计算的偏移值。</td>
        <td>预留输入，暂不支持，需要传空指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>xScale</td>
        <td rowspan="1">输入</td>
        <td>表示左矩阵的量化因子，公式中的xScale。</td>
        <td>-</td>
        <td>FLOAT<sup>2</sup>、FLOAT8_E8M0<sup>1</sup></td>
        <td>ND</td>
        <td>1、3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>smoothScale</td>
        <td rowspan="1">可选输入</td>
        <td>表示平滑缩放因子。</td>
        <td><ul>
        <li>在A4W4场景下可选，其他场景需传空指针。</li>
        <li>A4W4场景下首轴长度需与weight的首轴维度相等。</li>
        <li>A4W4场景下支持空指针或两种形状：(E, N / 2)或(E,)。</li>
        </ul></td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1、2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>groupList</td>
        <td rowspan="1">输入</td>
        <td>表示每个分组参与计算的Token个数，公式中的grouplist。</td>
        <td><ul>
          <li>长度需与weight的首轴维度相等。</li>
          <li>当groupListType为0时，最后一个值不大于x中tensor的第一维，当groupListType为1时，数值的总和不大于x中tensor的第一维。</li>
          <li>groupList中的值约束了输出数据的有效部分，groupList未指定的部分将不会参与更新。</li>
        </ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dequantMode</td>
        <td rowspan="1">输入</td>
        <td>表示反量化计算类型，用于确定激活矩阵与权重矩阵的反量化方式。</td>
        <td><ul>
          <li>0表示激活矩阵per-token，权重矩阵per-channel。</li>
          <li>1表示激活矩阵per-token，权重矩阵per-group。</li>
          <li>2表示MX量化。</li>
        </ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dequantDtype</td>
        <td rowspan="1">输入</td>
        <td>表示中间GroupedMatmul的结果数据类型。</td>
        <td><ul>
          <li>0表示FLOAT。</li>
          <li>1表示FLOAT16。</li>
          <li>27表示BFLOAT16。</li>
          <li>28表示UNDEFINED。</li>
        </ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td rowspan="1">输入</td>
        <td>表示量化计算类型，用于确定swiglu结果的量化模式。</td>
        <td><ul>
          <li>0表示per-token。</li>
          <li>2表示MX量化。</li>
        </ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupListType</td>
        <td rowspan="1">输入</td>
        <td>表示分组的解释方式，用于确定groupList的语义。</td>
        <td><ul><li>0表示cumsum模式，groupList中的每个元素代表当前分组的累计长度。</li><li>1表示count模式，groupList中的每个元素代表该分组包含多少元素。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>tuningConfig</td>
        <td rowspan="1">可选输入</td>
        <td>用于算子预估M/E的大小，走不同的算子模板，以适配不同场景性能要求。</td>
        <td>数组，传入的第一个数字表示各个专家处理的token数的预期值，用于优化tiling，A4W4 右矩阵NZ输入时开启，其他输入请传入空指针。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>output</td>
        <td rowspan="1">输出</td>
        <td>表示输出的量化结果，公式中的Q。</td>
        <td>-</td>
        <td>INT8<sup>2</sup>、FLOAT8_E4M3FN<sup>1</sup>、FLOAT4_E1M2<sup>1</sup>、FLOAT4_E2M1<sup>1</sup></td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputScale</td>
        <td rowspan="1">输出</td>
        <td>表示输出的量化因子，公式中的QScale。</td>
        <td>-</td>
        <td>FLOAT<sup>2</sup>、FLOAT8_E8M0<sup>1</sup></td>
        <td>ND</td>
        <td>1、3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td rowspan="1">输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td rowspan="1">输出</td>
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
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
      - <strong>weight强制视为FRACTAL_NZ格式。</strong>
      - 上表数据类型列中的角标“1”代表该系列不支持的数据类型
      - weight在A4W4下支持转置，其他输入仅支持非转置，INT32为A8W4和A4W4场景下的适配用途，实际1个INT32会被解释为8个INT4数据，A8W8场景不支持ND数据格式。
      - 支持dequantMode参数：A8W4场景和A4W4场景支持取值0和1，A8W8场景仅支持取值0。
      - 不支持dequantDtype和quantMode参数。
      - x和weight不支持空Tensor。
      - weight NZ转置输入时，仅支持单Tensor模式
      - weight、weightScale和weightAssistMatrix支持单Tensor场景（tensorlist长度为1）和多Tensor场景（tensorlist长度大于1）。
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：
      - <strong>weight强制视为FRACTAL_NZ格式。</strong>
      - 上表数据类型列中的角标“2”代表该系列不支持的数据类型
      - 支持dequantMode参数：MX量化场景支持取值2。
      - 支持dequantDtype参数：MX量化场景支持取值0。
      - 支持quantMode参数：MX量化场景支持取值2。
      - 仅支持dequantMode和quantMode相同取值。
      - x和xScale支持M为0的空Tensor。
      - weight和weightScale支持N为0的空Tensor。
      - MXFP4和MXFP8场景下，weight和weightScale既支持单Tensor场景（tensorlist长度必须为1）也支持多Tensor场景（tensorlist长度大于等于1）。多Tensor场景下，tensorlist长度需等于E，E可以为1，weight、weightScale的shape需要按照E的维度展平，例如{(E, K, N)}需要变成{E个(K, N)}。
      - MXA8W4场景下weight和weightScale支持多Tensor场景（tensorlist长度大于等于1）。
      - MXFP4场景支持weight NZ格式，x和weight为FLOAT4_E2M1或FLOAT4_E1M2（支持交叉），output支持FLOAT8_E4M3FN、FLOAT4_E1M2或FLOAT4_E2M1。

    <!-- end id10 -->

  - **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：
    <table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
    <col style="width: 167px">
    <col style="width: 123px">
    <col style="width: 860px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>参数x、weight、weightScale、xScale、groupList、output、outputScale是空指针。</td>
      </tr>
      <tr>
        <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="10">161002</td>
        <td>传入的x、weight、weightScale、xScale、groupList、output、outputScale的数据维度不满足约束。</td>
      </tr>
      <tr>
        <td>传入的x、weight、weightScale、xScale、groupList、output、outputScale数据的shape不满足约束条件。</td>
      </tr>
      <tr>
        <td>传入的x、weight、weightScale、xScale、groupList、output、outputScale数据的format不满足约束条件。</td>
      </tr>
      <tr>
        <td>在不支持多Tensor的场景下，传入的weight、weightScale的tensor list长度不为1；或在多Tensor场景下，weight和weightScale的tensor list长度不一致。</td>
      </tr>
      <tr>
        <td>传入的x、xScale为空tensor，传入的weight、weightScale为空tensorList。</td>
      </tr>
      <tr>
        <td>groupList的元素个数大于weight的首轴长度。</td>
      </tr>
      <tr>
        <td>传入的dequantMode、quantMode、dequantDtype不满足约束条件。</td>
      </tr>
      <tr>
        <td>传入的bias、weightAssistMatrix、smoothScale、tuningConfig不满足约束条件。</td>
      </tr>
    </tbody>
    </table>

## aclnnGroupedMatmulSwigluQuantWeightNzV2

- **参数说明**
  <table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
    <col style="width: 167px">
    <col style="width: 123px">
    <col style="width: 860px">
    </colgroup>
    <thead>
      <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
      <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize获取。</td></tr>
      <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
      <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody>
  </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  - 确定性计算：
      - aclnnGroupedMatmulSwigluQuantWeightNzV2默认为确定性实现。

  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - A8W8/A8W4/A4W4量化场景下需满足以下约束条件：
        - 数据类型需要满足下表：
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 100px">
          <col style="width: 100px">
          <col style="width: 300px">
          <col style="width: 300px">
          <col style="width: 130px">
          <col style="width: 80px">
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 130px">
          </colgroup>
          <thead>
            <tr>
              <th>量化场景</th>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>weightAssistMatrix</th>
              <th>xScale</th>
              <th>smoothScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>A8W8</td>
              <td>INT8</td>
              <td>INT8</td>
              <td>FLOAT、FLOAT16、BFLOAT16</td>
              <td>nullptr</td>
              <td>FLOAT</td>
              <td>nullptr</td>
              <td>INT8</td>
              <td>FLOAT</td>
            </tr>
            <tr>
              <td>A8W4</td>
              <td>INT8</td>
              <td>INT4、INT32</td>
              <td>UINT64</td>
              <td>FLOAT</td>
              <td>FLOAT</td>
              <td>nullptr</td>
              <td>INT8</td>
              <td>FLOAT</td>
            </tr>
            <tr>
              <td>A4W4</td>
              <td>INT4、INT32</td>
              <td>INT4、INT32</td>
              <td>UINT64</td>
              <td>nullptr</td>
              <td>FLOAT</td>
              <td>nullptr/FLOAT</td>
              <td>INT8</td>
              <td>FLOAT</td>
            </tr>
          </tbody>
          </table>

        - shape约束需要满足下表：
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 100px">
          <col style="width: 100px">
          <col style="width: 300px">
          <col style="width: 300px">
          <col style="width: 130px">
          <col style="width: 80px">
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 130px">
          </colgroup>
          <thead>
            <tr>
              <th>量化场景</th>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>weightAssistMatrix</th>
              <th>xScale</th>
              <th>smoothScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>A8W8</td>
              <td>(M, K)</td>
              <td>NZ格式shape形如{(E, N / 32, K / 16, 16, 32)}</td>
              <td>{(E, N)}</td>
              <td>nullptr</td>
              <td>(M,)</td>
              <td>nullptr</td>
              <td>(M, N / 2)</td>
              <td>(M,)</td>
            </tr>
              <tr>
              <td>A8W4</td>
              <td>(M, K)</td>
              <td><ul>
              <li>ND格式shape形如{(E, K, N)}</li>
              <li>NZ格式且INT4时shape形如{(E, N / 64, K / 16, 16, 64)}</li>
              <li>NZ格式且INT32时shape形如{(E, N / 64, K / 16, 16, 8)}</li></ul></td>
              <td><ul>
              <li>per-channel场景shape形如{(E, N)}</li>
              <li>per-group场景shape形如{(E, K_group_num, N)}</li></ul></td>
              <td>{(E, N)}</td>
              <td>(M,)</td>
              <td>nullptr</td>
              <td>(M, N / 2)</td>
              <td>(M,)</td>
            </tr>
            <tr>
              <td>A4W4</td>
              <td>(M, K)</td>
              <td><ul>
              <li>ND格式shape形如{(E, K, N)}</li>
              <li>A4W4支持非转置和转置NZ</li>
              <li>NZ非转置格式且INT4时shape形如{(E, N / 64, K / 16, 16, 64)}</li>
              <li>NZ非转置格式且INT32时shape形如{(E, N / 64, K / 16, 16, 8)}</li>
              <li>NZ转置格式且INT4时原始shape形如{(E, K / 64, N / 16, 16, 64)}，并调用transpose(-1,-2)后传入</li>
              <li>NZ转置格式且INT32时原始shape形如{(E, K / 64, N / 16, 16, 8)}，并调用transpose(-1,-2)后传入</li>
             <li>NZ转置输入时，per-group的K/K_group_num请按照64对齐</li></ul>
              </td>
              <td><ul>
              <li>per-channel场景shape形如{(E, N)}</li>
              <li>per-group场景shape形如{(E, K_group_num, N)}</li></ul>
              </td>
              <td>nullptr</td>
              <td>(M,)</td>
              <td><ul>
              <li>nullptr</li>
              <li>(E,)</li>
              <li>(E, N / 2)</li>
              </ul></td>
              <td>(M, N / 2)</td>
              <td>(M,)</td>
            </tr>
          </tbody>
          </table>

        - A8W8场景下，不支持N轴长度超过10240，不支持x的尾轴长度大于等于65536。
        - A8W4场景下，不支持N轴长度超过10240，不支持x的尾轴长度大于等于20000。
        - A4W4场景下，不支持N轴长度超过10240，不支持x的尾轴长度大于等于20000。
        - 多tensor场景下，即tensorlist长度大于1时，weight、weightScale和weightAssistMatrix的shape需要按照E的维度展平，例如{(E, K, N)}需要变成{E个(K, N)}。
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - groupList第1维最大支持1024，即最多支持1024个group。
    - MX量化场景下需满足以下约束条件：
        - 数据类型需要满足下表：
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 300px">
          <col style="width: 300px">
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 130px">
          </colgroup>
          <thead>
            <tr>
              <th>MX量化场景</th>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>xScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>MXFP8</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E8M0</td>
              <td>FLOAT8_E8M0</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E8M0</td>
            </tr>
          <tr>
            <td>MXFP4</td>
            <td>FLOAT4_E2M1、FLOAT4_E1M2</td>
            <td>FLOAT4_E2M1、FLOAT4_E1M2</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT4_E1M2</td>
            <td>FLOAT8_E8M0</td>
          </tr>
            <tr>
              <td>MXA8W4</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT4_E2M1</td>
              <td>FLOAT8_E8M0</td>
              <td>FLOAT8_E8M0</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E8M0</td>
            </tr>
          </tbody>
          </table>

        - shape约束需要满足下表：
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 250px">
          <col style="width: 320px">
          <col style="width: 180px">
          <col style="width: 250px">
          <col style="width: 160px">
          <col style="width: 160px">
          </colgroup>
          <thead>
            <tr>
              <th>MX量化场景</th>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>xScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>MXFP8</td>
              <td>(M, K)</td>
              <td><ul>
              <li>非转置shape形如{(E, ceil(N / 32), ceil(K / 16), 16, 32)}</li>
              <li>多Tensor非转置shape形如{E个(ceil(N / 32), ceil(K / 16), 16, 32)}，viewShape形如{E个(K, N)}</li>
              <li>转置shape形如{(E, ceil(K / 32), ceil(N / 16), 16, 32)}</li></ul></td>
              <td><ul>
              <li>非转置shape形如{(E, ceil(K / 64), N, 2)}</li>
              <li>多Tensor非转置shape形如{E个(ceil(K / 64), N, 2)}</li>
              <li>转置shape形如{(E, N, ceil(K / 64), 2)}</li></ul></td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
            <tr>
              <td>MXFP4</td>
              <td>(M, K)</td>
              <td><ul>
              <li>非转置shape形如{(E, ceil(N / 64), ceil(K / 16), 16, 64)}</li>
              <li>多Tensor非转置shape形如{E个(ceil(N / 64), ceil(K / 16), 16, 64)}，viewShape形如{E个(K, N)}</li>
              <li>转置shape形如{(E, ceil(K / 64), ceil(N / 16), 16, 64)}</li></ul></td>
              <td><ul>
              <li>非转置shape形如{(E, ceil(K / 64), N, 2)}</li>
              <li>多Tensor非转置shape形如{E个(ceil(K / 64), N, 2)}</li>
              <li>转置shape形如{(E, N, ceil(K / 64), 2)}</li></ul></td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
            <tr>
              <td>MXA8W4</td>
              <td>(M, K)</td>
              <td><ul>
              <li>转置shape形如{(E, K/32, N/16, 16, 32)}</li></ul></td>
              <td><ul>
              <li>转置shape形如{(E, N, ceil(K / 64), 2)}</li></ul></td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
            <tr>
              <td>MXA8W4（多tensor）</td>
              <td>(M, K)</td>
              <td><ul>
              <li>转置view shape形如{E个(N, K)}，storage shape形如{E个(N/32, K/16, 16, 32)}</li></ul></td>
              <td><ul>
              <li>转置view shape形如{E个(N, ceil(K / 64), 2)}</li></ul></td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
          </tbody>
          </table>

        - weightScale转置属性需要与weight保持一致。
        - MX量化场景下，当输入为fp8数据类型时，需满足N为64对齐；当输入为fp4数据类型时，需满足N为128对齐。
        - MXFP4和MXFP8多Tensor场景，weight和weightScale的tensorlist长度需等于E。
        - MXFP4场景下，K需大于2。
        - MXFP4场景下，左右矩阵内轴均需为偶数。
        - MXFP4场景下，weight后两轴均不能为1。
        - MXA8W4场景下需满足以下约束条件：
          - K % 32 == 0，即K轴需32对齐。
          - K >= 32。
          - N >= 128。
          - weight仅支持转置。用户需构造转置的NZ格式weight数据，调用前需transpose成非转置状态。
          - 多tensor场景下（tensorlist长度大于等于1），weight和weightScale的tensor数量需一致。
          - 多tensor场景下，weight中每个tensor的K和N需相等。
          - 多tensor场景下，weight中每个tensor的view shape为2D，weightScale中每个tensor的view shape为3D；单tensor场景下，weight中每个tensor的view shape为3D，weightScale中每个tensor的view shape为4D。
          - 多tensor场景下，所有weightScale tensor的shape需与对应weight tensor匹配。
          - 多tensor场景下，groupList的长度需等于weight/weightScale的tensor数量。

  <!-- end id12 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

  <!-- npu="A3,910b" id13 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    ```cpp
    #include <iostream>
    #include <vector>
    #include "acl/acl.h"
    #include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"

    #define CHECK_RET(cond, return_expr)                                                                                   \
        do {                                                                                                               \
            if (!(cond)) {                                                                                                 \
                return_expr;                                                                                               \
            }                                                                                                              \
        } while (0)

    #define LOG_PRINT(message, ...)                                                                                        \
        do {                                                                                                               \
            printf(message, ##__VA_ARGS__);                                                                                \
        } while (0)

    int64_t GetShapeSize(const std::vector<int64_t>& shape) {
        int64_t shapeSize = 1;
        for (auto i : shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    int Init(int32_t deviceId, aclrtStream* stream) {
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
    int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                        void** deviceAddr, aclDataType dataType, aclFormat formatType, aclTensor** tensor) {
        auto size = GetShapeSize(shape) * sizeof(T);
        // 调用aclrtMalloc申请device侧内存
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // 调用aclrtMemcpy将host侧数据复制到device侧内存上
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // 计算连续tensor的strides
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
        }

        // 调用aclCreateTensor接口创建aclTensor
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                                shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    template <typename T>
    int CreateAclTensorList(const std::vector<T> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                            void **deviceAddr, aclDataType dataType, aclFormat formatType, aclTensorList **tensor) {
        int size = shapes.size();
        aclTensor* tensors[size];
        for (int i = 0; i < size; i++) {
            int ret = CreateAclTensor<T>(hostData, shapes[i], deviceAddr + i, dataType, formatType, tensors + i);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
        }
        *tensor = aclCreateTensorList(tensors, size);
        return ACL_SUCCESS;
    }

    int main() {
        // 1.（固定写法）device/stream初始化，参考acl API手册
        // 根据自己的实际device填写deviceId
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = Init(deviceId, &stream);
        // check根据自己的需要处理
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        int64_t E = 4;
        int64_t M = 8192;
        int64_t N = 4096;
        int64_t K = 7168;
        std::vector<int64_t> xShape = {M, K};
        std::vector<std::vector<int64_t>> weightShape = {{E, N / 32 , K / 16, 16, 32}};
        std::vector<std::vector<int64_t>> weightScaleShape = {{E, N}};
        std::vector<int64_t> xScaleShape = {M};
        std::vector<int64_t> groupListShape = {E};
        std::vector<int64_t> outputShape = {M, N / 2};
        std::vector<int64_t> outputScaleShape = {M};

        void* xDeviceAddr = nullptr;
        void* weightDeviceAddr[1];
        void* weightScaleDeviceAddr[1];
        void* xScaleDeviceAddr = nullptr;
        void* groupListDeviceAddr = nullptr;
        void* outputDeviceAddr = nullptr;
        void* outputScaleDeviceAddr = nullptr;

        aclTensor* x = nullptr;
        aclTensorList* weight = nullptr;
        aclTensorList* weightScale = nullptr;
        aclTensor* xScale = nullptr;
        aclTensor* groupList = nullptr;
        aclTensor* output = nullptr;
        aclTensor* outputScale = nullptr;

        std::vector<int8_t> xHostData(M * K, 1);
        std::vector<int8_t> weightHostData(E * N * K, 1);
        std::vector<float> weightScaleHostData(E * N, 0.5f);
        std::vector<float> xScaleHostData(M, 0.0314f);
        std::vector<int64_t> groupListHostData = {1, 2, 2, 3};
        std::vector<int8_t> outputHostData(M * N / 2, 0);
        std::vector<float> outputScaleHostData(M, 0);

        // 创建x aclTensor
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建weight aclTensorList
        ret = CreateAclTensorList(weightHostData, weightShape, weightDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建weightScale aclTensorList
        ret = CreateAclTensorList(weightScaleHostData, weightScaleShape, weightScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &weightScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建xScale aclTensor
        ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &xScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建groupList aclTensor
        ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建output aclTensor
        ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &output);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建outputScale aclTensor
        ret = CreateAclTensor(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &outputScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // 新增V2参数
        aclTensorList* weightAssistMatrix = nullptr;
        aclTensor* bias = nullptr;
        aclTensor* smoothScale = nullptr;
        int64_t dequantMode = 0;
        int64_t dequantDtype = 28;
        int64_t quantMode = 0;
        int64_t groupListType = 0;

        std::vector<int64_t> tuningConfigData = {};
        aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;

        // 3. 调用CANN算子库API
        // 调用aclnnGroupedMatmulSwigluQuantWeightNzV2第一段接口
        ret = aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
            x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode, dequantDtype,
            quantMode, groupListType, tuningConfig, output, outputScale, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用aclnnGroupedMatmulSwigluQuantWeightNzV2第二段接口
        ret = aclnnGroupedMatmulSwigluQuantWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2 failed. ERROR: %d\n", ret); return ret);

        // 4.（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
        auto size = 10;
        std::vector<int8_t> out1Data(size, 0);
        ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), outputDeviceAddr,
                            size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, out1Data[j]);
        }
        std::vector<float> out2Data(size, 0);
        ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), outputScaleDeviceAddr,
                            size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, out2Data[j]);
        }
        // 6. 释放aclTensor、aclTensorList和aclScalar，需要根据具体API的接口定义修改
        aclDestroyTensor(x);
        aclDestroyTensorList(weight);
        aclDestroyTensorList(weightScale);
        aclDestroyTensor(xScale);
        aclDestroyTensor(groupList);
        aclDestroyTensor(output);
        aclDestroyTensor(outputScale);

        aclDestroyIntArray(tuningConfig);

        // 7. 释放device资源，需要根据具体API的接口定义修改
        aclrtFree(xDeviceAddr);
        for (int64_t i = 0; i < 1; i++) {
            aclrtFree(weightDeviceAddr[i]);
            aclrtFree(weightScaleDeviceAddr[i]);
        }
        aclrtFree(weightDeviceAddr);
        aclrtFree(weightScaleDeviceAddr);
        aclrtFree(xScaleDeviceAddr);
        aclrtFree(groupListDeviceAddr);
        aclrtFree(outputDeviceAddr);
        aclrtFree(outputScaleDeviceAddr);
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }
    ```

  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：

    ```cpp
    #include <iostream>
    #include <memory>
    #include <vector>

    #include "acl/acl.h"
    #include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"

    #define CHECK_RET(cond, return_expr) \
        do { \
            if (!(cond)) { \
                return_expr; \
            } \
        } while (0)

    #define CHECK_FREE_RET(cond, return_expr) \
        do { \
            if (!(cond)) { \
                Finalize(deviceId, stream); \
                return_expr; \
            } \
        } while (0)

    #define LOG_PRINT(message, ...) \
        do { \
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
                        aclDataType dataType, aclFormat formatType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                                  shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    template <typename T>
    int CreateAclTensorList(const std::vector<std::vector<T>> &hostData,
                            const std::vector<std::vector<int64_t>> &shapes,
                            void **deviceAddr, aclDataType dataType, aclFormat formatType, aclTensorList **tensor)
    {
        int size = shapes.size();
        aclTensor *tensors[size];
        for (int i = 0; i < size; i++) {
            int ret = CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i, dataType, formatType, tensors + i);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
        }
        *tensor = aclCreateTensorList(tensors, size);
        return ACL_SUCCESS;
    }

    template <typename T>
    int CreateAclTensorListNZTranspose(const std::vector<std::vector<T>> &hostData,
                                       const std::vector<std::vector<int64_t>> &viewShapes,
                                       const std::vector<std::vector<int64_t>> &storageShapes,
                                       const std::vector<std::vector<int64_t>> &transposeStrides,
                                       void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
    {
        int size = storageShapes.size();
        aclTensor *tensors[size];
        for (int i = 0; i < size; i++) {
            auto storageSize = GetShapeSize(storageShapes[i]) * sizeof(T);
            auto ret = aclrtMalloc(deviceAddr + i, storageSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

            ret = aclrtMemcpy(*(deviceAddr + i), storageSize, hostData[i].data(), storageSize, ACL_MEMCPY_HOST_TO_DEVICE);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

            tensors[i] = aclCreateTensor(viewShapes[i].data(), viewShapes[i].size(), dataType,
                                         transposeStrides[i].data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                         storageShapes[i].data(), storageShapes[i].size(), *(deviceAddr + i));
        }
        *tensor = aclCreateTensorList(tensors, size);
        return ACL_SUCCESS;
    }

    template <typename T>
    int CreateAclTensorListTranspose(const std::vector<std::vector<T>> &hostData,
                                     const std::vector<std::vector<int64_t>> &viewShapes,
                                     const std::vector<std::vector<int64_t>> &transposeStrides,
                                     void **deviceAddr, aclDataType dataType, aclFormat formatType, aclTensorList **tensor)
    {
        int size = viewShapes.size();
        aclTensor *tensors[size];
        for (int i = 0; i < size; i++) {
            auto storageSize = GetShapeSize(viewShapes[i]) * sizeof(T);
            auto ret = aclrtMalloc(deviceAddr + i, storageSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

            ret = aclrtMemcpy(*(deviceAddr + i), storageSize, hostData[i].data(), storageSize, ACL_MEMCPY_HOST_TO_DEVICE);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

            tensors[i] = aclCreateTensor(viewShapes[i].data(), viewShapes[i].size(), dataType,
                                         transposeStrides[i].data(), 0, formatType, viewShapes[i].data(),
                                         viewShapes[i].size(), *(deviceAddr + i));
        }
        *tensor = aclCreateTensorList(tensors, size);
        return ACL_SUCCESS;
    }

    template <typename T1, typename T2>
    auto CeilDiv(T1 a, T2 b) -> T1
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    }

    void Finalize(int32_t deviceId, aclrtStream stream)
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }

    int aclnnGroupedMatmulSwigluQuantWeightNzV2MxA8W4Test(int32_t deviceId, aclrtStream &stream)
    {
        auto ret = Init(deviceId, &stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        int64_t E = 8;
        int64_t M = 2048;
        int64_t N = 4096;
        int64_t K = 4096;

        int64_t kScaleBlock = CeilDiv(K, 64);

        std::vector<int64_t> xShape = {M, K};
        std::vector<std::vector<int64_t>> weightViewShapes = {{E, K, N}};
        std::vector<std::vector<int64_t>> weightStorageShapes = {{E, K / 32, N / 16, 16, 32}};
        std::vector<std::vector<int64_t>> weightTransposeStrides = {{K * N, 1, K}};
        std::vector<std::vector<int64_t>> weightScaleViewShapes = {{E, kScaleBlock, N, 2}};
        std::vector<std::vector<int64_t>> weightScaleTransposeStrides = {{kScaleBlock * N * 2, 2, kScaleBlock * 2, 1}};
        std::vector<int64_t> xScaleShape = {M, CeilDiv(K, 64), 2};
        std::vector<int64_t> groupListShape = {E};
        std::vector<int64_t> outputShape = {M, N / 2};
        std::vector<int64_t> outputScaleShape = {M, CeilDiv((N / 2), 64), 2};

        void *xDeviceAddr = nullptr;
        void *weightDeviceAddr = nullptr;
        void *weightScaleDeviceAddr = nullptr;
        void *xScaleDeviceAddr = nullptr;
        void *groupListDeviceAddr = nullptr;
        void *outputDeviceAddr = nullptr;
        void *outputScaleDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensorList *weight = nullptr;
        aclTensorList *weightScale = nullptr;
        aclTensor *xScale = nullptr;
        aclTensor *groupList = nullptr;
        aclTensor *output = nullptr;
        aclTensor *outputScale = nullptr;

        std::vector<uint8_t> xHostData(M * K, 1);
        std::vector<uint8_t> weightHostData(E * (K / 32) * (N / 16) * 16 * 32, 1);
        std::vector<uint8_t> weightScaleHostData(E * kScaleBlock * N * 2, 1);
        std::vector<uint8_t> xScaleHostData(M * CeilDiv(K, 64) * 2, 1);
        std::vector<int64_t> groupListHostData(E, 256);
        std::vector<uint8_t> outputHostData(M * N / 2, 1);
        std::vector<uint8_t> outputScaleHostData(M * CeilDiv((N / 2), 64) * 2, 1);

        aclIntArray *tuningConfig = nullptr;

        int64_t quantMode = 2;
        int64_t dequantMode = 2;
        int64_t dequantDtype = 0;
        int64_t groupListType = 1;

        ret = CreateAclTensor<uint8_t>(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                                       aclFormat::ACL_FORMAT_ND, &x);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<std::vector<uint8_t>> weightHostDataList = {weightHostData};
        ret = CreateAclTensorListNZTranspose<uint8_t>(weightHostDataList, weightViewShapes, weightStorageShapes,
                                                      weightTransposeStrides, &weightDeviceAddr,
                                                      aclDataType::ACL_FLOAT4_E2M1, &weight);
        std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorListPtr(weight,
                                                                                                   aclDestroyTensorList);
        std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<std::vector<uint8_t>> weightScaleHostDataList = {weightScaleHostData};
        ret = CreateAclTensorListTranspose<uint8_t>(weightScaleHostDataList, weightScaleViewShapes,
                                                    weightScaleTransposeStrides, &weightScaleDeviceAddr,
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

        ret = CreateAclTensor<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                       aclFormat::ACL_FORMAT_ND, &groupList);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor<uint8_t>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                                       aclFormat::ACL_FORMAT_ND, &output);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(output, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> outputDeviceAddrPtr(outputDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor<uint8_t>(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr,
                                       aclDataType::ACL_FLOAT8_E8M0, aclFormat::ACL_FORMAT_ND, &outputScale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputScaleTensorPtr(outputScale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> outputScaleDeviceAddrPtr(outputScaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        ret = aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
            x, weight, weightScale, nullptr, nullptr, xScale, nullptr, groupList, dequantMode, dequantDtype, quantMode,
            groupListType, tuningConfig, output, outputScale, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        ret = aclnnGroupedMatmulSwigluQuantWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2 failed. ERROR: %d\n", ret);
                  return ret);

        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        auto size = GetShapeSize(outputShape);
        std::vector<uint8_t> outputData(size, 0);
        ret = aclrtMemcpy(outputData.data(), size * sizeof(outputData[0]), outputDeviceAddr, size * sizeof(outputData[0]),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputData from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, outputData[j]);
        }

        size = GetShapeSize(outputScaleShape);
        std::vector<uint8_t> outputScaleData(size, 0);
        ret = aclrtMemcpy(outputScaleData.data(), size * sizeof(outputScaleData[0]), outputScaleDeviceAddr,
                          size * sizeof(outputScaleData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputScaleData from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, outputScaleData[j]);
        }

        return ACL_SUCCESS;
    }

    int main()
    {
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = aclnnGroupedMatmulSwigluQuantWeightNzV2MxA8W4Test(deviceId, stream);
        CHECK_FREE_RET(ret == ACL_SUCCESS,
                       LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2MxA8W4Test failed. ERROR: %d\n", ret);
                       return ret);

        Finalize(deviceId, stream);
        return 0;
    }
    ```

  <!-- end id14 -->
