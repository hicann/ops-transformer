# GroupedMatmulSwigluQuant

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      √     |
|<term>Atlas A3系列产品</term>|      √     |
|<term>Atlas A2系列产品</term>|      √     |
|<term>Atlas 200I/500 A2推理产品</term>|      ×     |
|<term>Atlas推理系列产品</term>|      ×     |
|<term>Atlas训练系列产品</term>|      ×     |
|<term>Kirin X90处理器系列产品</term> | √ |
|<term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 接口功能：融合GroupedMatmul 、dquant、swiglu和quant，详细解释见计算公式。
- 计算公式：

  - **定义**：

    * **⋅** 表示矩阵乘法。
    * **⊙** 表示逐元素乘法。
    * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
    * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
    * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
  - **输入**：

    * $X∈\mathbb{Z_8}^{M \times K}$：输入矩阵（左矩阵），M是总token数，K是特征维度。
    * $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
    * $bias∈\mathbb{R}^{E \times N}$：A8W4 场景的 FLOAT32 辅助矩阵（`weightAssistMatrix`），E是专家个数，N是输出维度；A8W8 场景需传空指针。
    * $offset∈\mathbb{R}^{E  \times N}$：per-channel非对称反量化的偏移，E是专家个数，N是输出维度。
    * $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，N是输出维度。
    * $x\_scale∈\mathbb{R}^{M}$：输入矩阵（左矩阵）的逐token缩放因子，M是总token数。
    * $groupList∈\mathbb{N}^{E}$：前缀和的分组索引列表。
  - **输出**：

    * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
    * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
  - **计算过程**

    - 1.根据groupList[i]确定当前分组的token ，$i \in [0,Len(groupList)]$。

      >例子：假设groupList=[3,4,4,6]，从0开始计数。
      >
      >第0个右矩阵`W[0,:,:]`，对应索引位置[0,3)的token`x[0:3]`（共3-0=3个token），对应`x_scale[0:3]`、`w_scale[0]`、`bias[0]`、`offset[0]`、`Q[0:3]`、`Q_scale[0:3]`；
      >
      >第1个右矩阵`W[1,:,:]`，对应索引位置[3,4)的token`x[3:4]`（共4-3=1个token），对应`x_scale[3:4]`、`w_scale[1]`、`bias[1]`、`offset[1]`、`Q[3:4]`、`Q_scale[3:4]`；
      >
      >第2个右矩阵`W[2,:,:]`，对应索引位置[4,4)的token`x[4:4]`（共4-4=0个token），对应`x_scale[4:4]`、`w_scale[2]`、`bias[2]`、`offset[2]`、`Q[4:4]`、`Q_scale[4:4]`；
      >
      >第3个右矩阵`W[3,:,:]`，对应索引位置[4,6)的token`x[4:6]`（共6-4=2个token），对应`x_scale[4:6]`、`w_scale[3]`、`bias[3]`、`offset[3]`、`Q[4:6]`、`Q_scale[4:6]`；
      >
      >请注意：groupList中未指定的部分将不会参与更新。
      >例如groupList=[12,14,18]，X的shape为[30，:]。
      >
      >则第一个输出Q的shape为[30，:]，其中Q[18:，：]的部分不会进行更新和初始化，其中数据为显存空间申请时的原数据。
      >
      >同理，第二个输出Q的shape为[30]，其中Q\_scale[18:]的部分不会进行更新或初始化，其中数据为显存空间申请时的原数据。
      >
      >即输出的Q[:groupList[-1],:]和Q\_scale[:groupList[-1]]为有效数据部分。

    - 2.根据分组确定的入参进行如下计算：

      $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

      $C_{i,act}, gate_{i} = split(C_{i})$

      $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$

      >注：`bias` 输入的作用和传入要求随场景不同：A8W4 场景必须传入 FLOAT32 辅助矩阵（`weightAssistMatrix`），A8W8 场景需传空指针；它不是 INT32 矩阵乘偏置。当前版本不支持$offset_{i}$，非对称量化暂不支持。

    - 3.确定量化方式

      - 当量化方式为对称量化时：

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left \lfloor \frac{S_{i}}{Q\_scale_{i}}\right \rceil $

      - 当量化方式为非对称量化时：(暂不支持)

        $Q\_scale_{i} = \frac{max(S_{i})-min(S_{i})}{255}$

        $Q\_offset_{i} = -128 - \left \lfloor \frac{min(S_{i})}{Q\_scale_{i}}\right \rceil$

        $Q_{i} = \left \lfloor \frac{S_{i}}{ Q\_scale_{i} } + Q\_offset_{i}\right \rceil $

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|:---:|------|----------|:-----------:|
| x | 输入 | 公式中的左矩阵 $X$ | INT8 | ND |
| weight | 输入 | 公式中的分组权重矩阵 $W$。INT32为INT4打包类型，每个INT32存放8个INT4元素 | INT8、INT4<sup>1</sup>、INT32<sup>1</sup> | ND/NZ |
| bias | 可选输入 | A8W4场景的辅助矩阵（`weightAssistMatrix`），shape为 `[E, N]`，该场景必须传入。A8W8场景传入 nullptr | FLOAT | ND |
| offset | 可选输入 | perchannel非对称反量化的偏移量（预留参数），传入 nullptr | FLOAT | ND |
| weightScale | 输入 | 公式中的 $w\_scale$，代表右矩阵的量化缩放因子 | UINT64<sup>1</sup>、FLOAT、FLOAT16、BFLOAT16<sup>2</sup> | ND |
| xScale | 输入 | 公式中的 $x\_scale$，代表左矩阵的量化缩放因子 | FLOAT | ND |
| <abbr title="长度为专家数，元素为各组token数的累积和">groupList</abbr> | 输入 | 指定各专家对应的token范围 | INT64 | ND |
| output | 输出 | 量化结果 $Q$ | INT8 | ND |
| outputScale | 输出 | 每个token的量化缩放因子 $Q\_scale$ | FLOAT | ND |
| outputOffset | 输出 | 预留输出。Ascend 950的WeightNZ路径忽略该参数，其余场景传入 nullptr | - | - |

- <term>Ascend 950PR&950DT系列产品</term>：
  - 上表数据类型列中的角标 <sup>1</sup> 代表该系列不支持的数据类型。
  - weight仅支持NZ格式。
  - bias和offset均须传入 nullptr。
- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：
  - 上表数据类型列中的角标 <sup>2</sup> 代表该系列不支持的数据类型。

---

## 约束说明

- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
   - N轴长度不能超过10240。
   - K轴长度不能超过65536。
- <term>Ascend 950PR&950DT系列产品</term>：
  - 仅支持A8W8 pertoken perchannel量化场景。
   - N 为正数且为64的整数倍。

> aclnnGroupedMatmulSwigluQuant接口详细约束请参见 [aclnnGroupedMatmulSwigluQuant约束说明](docs/aclnnGroupedMatmulSwigluQuant.md#约束说明)。
>
> WeightNZ接口详细约束请参见 [aclnnGroupedMatmulSwigluQuantWeightNZ约束说明](docs/aclnnGroupedMatmulSwigluQuantWeightNZ.md#约束说明)。

---

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| aclnn API | [test_aclnn_grouped_matmul_swiglu_quant](examples/test_aclnn_grouped_matmul_swiglu_quant.cpp) | 通过[aclnnGroupedMatmulSwigluQuant](docs/aclnnGroupedMatmulSwigluQuant.md)两段式接口调用GroupedMatmulSwigluQuant算子。 |
| aclnn API | [test_aclnn_grouped_matmul_swiglu_quant_weight_nz](examples/arch35/test_aclnn_grouped_matmul_swiglu_quant_weight_nz.cpp) | 通过[aclnnGroupedMatmulSwigluQuantWeightNZ](docs/aclnnGroupedMatmulSwigluQuantWeightNZ.md)两段式接口调用GroupedMatmulSwigluQuantWeightNZ算子 |
