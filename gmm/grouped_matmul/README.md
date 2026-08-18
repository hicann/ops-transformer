# GroupedMatmul

<style>
abbr {
  border-bottom: 1px dotted #666;
  cursor: help;
}
</style>

## 简介

GroupedMatmul是MoE网络中的重要算子，核心功能是将已经按专家分组的token打包，一次完成所有分组的矩阵乘计算。支持按M轴或K轴分组，涵盖非量化、伪量化和全量化三种计算场景。在MoE推理中，GroupedMatmul通常被调用两次：第一次对输入token与各专家的上投影权重做分组矩阵乘，得到各专家的中间结果；第二次将中间结果与各专家的下投影权重做分组矩阵乘，得到各专家的输出。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff'}}}%%
graph LR
    B["moe_gating_top_k"] --> C["moe_init_routing"]
    C --> D["grouped_matmul(gate_up_proj)"]
    D --> E["swiglu"]
    E --> F["grouped_matmul(down_proj)"]
    F --> G["moe_finalize_routing"]

    classDef gmm fill:#E8ECFF,stroke:none,color:#333;
    classDef other fill:#DCEFFA,stroke:none,color:#333;

    class D,F gmm;
    class B,C,E,G other;
```

---

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      √     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Kirin X90 处理器系列产品</term> | √ |
|<term>Kirin 9030 处理器系列产品</term> | √ |

---

## 专业术语

| 术语 | 简要说明 |
|------|----------|
| 非量化 | 左矩阵和右矩阵均为浮点类型直接计算 |
| 全量化 | 左矩阵和右矩阵均为低精度计算，通过scale还原，详见[《量化介绍》](../../docs/zh/context/quant_mode_introduction.md) |
| 伪量化 | 对右矩阵权重进行量化，包括perchannel量化模式 |
| pertensor | 简称T量化，每个Tensor共用一个相同的量化参数 |
| perchannel | 简称C量化，量化对象是右矩阵，每个channel分别使用独立的量化参数 |
| pertoken | 简称K量化，量化对象是左矩阵，每个token分别使用独立的量化参数 |
| pergroup | 简称G量化，在reduce轴上对数据分组，每组使用独立的量化参数 |
| perblock | 简称B量化，在所有轴上对数据分块，每块使用独立的量化参数 |
| T-C | 左矩阵pertensor，右矩阵perchannel |
| T-T | 左矩阵pertensor，右矩阵pertensor |
| K-C | 左矩阵pertoken，右矩阵perchannel |
| K-T | 左矩阵pertoken，右矩阵pertensor |
| G-B | 左矩阵pergroup，右矩阵perblock |
| MX | pergroup-pergroup（G-G）量化模式，量化参数类型为FLOAT8_E8M0，group size为32 的特例 |
| ND | 常规连续排布 |
| NZ | [FRACTAL_NZ](../../docs/zh/context/data_format.md)格式（亲和排布） |

## 功能说明

接口功能：实现分组矩阵乘计算。公式为 $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中 $g$ 为分组数。

- **M轴分组（groupType=0）**：$k_i$、$n_i$各组相同，$m_i$可以不相同。
- **K轴分组（groupType=2）**：$m_i$、$n_i$各组相同，$k_i$可以不相同。

基础计算公式：

$$
y_i=x_i\times weight_i + bias_i
$$

详细计算公式（按场景分类：非量化、伪量化、全量化）及版本演进请参见 [aclnnGroupedMatmulV5功能说明](docs/aclnnGroupedMatmulV5.md#3-功能说明)。

---

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|:---:|------|----------|:-----------:|
| x | 输入 | 公式中的输入 $x$ | FLOAT、FLOAT16、INT16<sup>1</sup>、INT8、INT4<sup>1</sup>、BFLOAT16、FLOAT8_E5M2<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup>、FLOAT4_E2M1<sup>2</sup> | ND |
| weight | 输入 | 公式中的 $weight$ | FLOAT、FLOAT16、INT16<sup>1</sup>、INT8、INT4、BFLOAT16、FLOAT8_E5M2<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup>、FLOAT4_E2M1<sup>2</sup> | ND/NZ |
| biasOptional | 可选输入 | 公式中的 $bias$ | FLOAT、FLOAT16、INT32、BFLOAT16<sup>2</sup> | ND |
| scaleOptional | 可选输入 | 公式中的 $scale$，代表量化参数中的缩放因子 | FLOAT、UINT64、BFLOAT16、FLOAT8_E8M0<sup>2</sup>、INT64<sup>2</sup> | ND |
| offsetOptional | 可选输入 | 公式中的 $offset$，代表量化参数中的偏移量 | FLOAT | ND |
| antiquantScaleOptional | 可选输入 | 公式中的 $antiquant\_scale$，代表伪量化参数中的缩放因子 | FLOAT16、BFLOAT16 | ND |
| antiquantOffsetOptional | 可选输入 | 公式中的 $antiquant\_offset$，代表伪量化参数中的偏移量 | FLOAT16、BFLOAT16 | ND |
| perTokenScaleOptional | 可选输入 | 公式中的 $per\_token\_scale$，代表量化参数中由 x量化引入的缩放因子 | FLOAT、FLOAT8_E8M0<sup>2</sup> | ND |
| groupListOptional | 可选输入 | 代表输入和输出分组轴方向的 matmul大小分布 | INT64 | ND |
| activationInputOptional | 可选输入 | 代表激活函数的反向输入，当前只支持传入 nullptr | - | - |
| activationQuantScaleOptional | 可选输入 | 激活函数量化缩放因子（预留参数），当前只支持传入 nullptr | - | - |
| activationQuantOffsetOptional | 可选输入 | 激活函数量化偏移量（预留参数），当前只支持传入 nullptr | - | - |
| <abbr title="代表输出是否要做 Tensor切分：0/1=多Tensor 输出；2/3=单Tensor 输出">splitItem</abbr> | 属性 | 代表输出是否要做 Tensor切分 | INT64 | - |
| <abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr> | 属性 | 代表需要分组的轴 | INT64 | - |
| <abbr title="代表 groupList输入的分组方式：0=累积和；1=各组大小；2=[组索引, 组大小] 对">groupListType</abbr> | 属性 | 代表 groupList输入的分组方式 | INT64 | - |
| <abbr title="代表激活函数类型：0=NONE/1=ReLU/2=GELU_TANH/4=FAST_GELU/5=SILU">actType</abbr> | 属性 | 代表激活函数类型 | INT64 | - |
| tuningConfigOptional | 可选输入 | 第一个数代表各个专家处理的 token数的预期值；第二个数置1时，A8W4可选开启离线按`[E,N,K]`排布并转换为NZ的特殊weight格式（不表示`transposeWeight=true`）；第三个数代表允许额外使用的内存空间 | INT64 | - |
| out | 输出 | 公式中的输出 $y$ | FLOAT、FLOAT16、INT32、INT8、BFLOAT16 | ND |
| activationFeatureOutOptional | 输出 | 激活函数的输入数据，当前只支持传入 nullptr | - | - |
| dynQuantScaleOutOptional | 输出 | 动态量化缩放因子，当前只支持传入 nullptr | - | - |

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 上表数据类型列中的角标 <sup>1</sup> 代表该系列不支持的数据类型。
  - 输入参数 x、weight均不支持INT16 类型，且 x 不支持INT4 类型。
  - 输入参数 x、weight，输出参数 out在非量化场景支持最多 1024个Tensor，在伪量化支持最多 128个Tensor，在全量化场景最多支持 1个Tensor。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 上表数据类型列中的角标 <sup>2</sup> 代表该系列不支持的数据类型。
  - 不支持FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT8_E8M0类型。
  - 输入参数 biasOptional不支持BFLOAT16。
  - 输入参数 scaleOptional不支持INT64 类型。
  - 输入参数 x、weight，输出参数 out支持最多 128个Tensor。
- <term>Kirin X90/Kirin 9030 处理器系列产品</term>：
  - 不支持BFLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、HIFLOAT8类型。

---

## 约束说明

### 公共约束

- x/weight转置时对应 Tensor必须 [非连续](../../docs/zh/context/non_contiguous_tensor.md)。
- weight为 [FRACTAL_NZ](../../docs/zh/context/data_format.md) 格式时，shape 须满足 NZ格式要求。
- perTokenScaleOptional：通常仅1维，长度与 x的M一致。仅支持 x/weight/out均为单Tensor场景。
- groupListOptional：out中 TensorList长度为1时，groupList约束输出数据有效范围，未指定部分不参与更新。
- groupListType：
  - groupListType=0：须为非负单调非递减数列（累积和）。
  - groupListType=1：须为非负数列（各组大小）。
  - groupListType=2：须为非负数列，shape `[E, 2]`（[组索引, 组大小]），非零组前置。

> 各场景详细约束请参见 [aclnnGroupedMatmulV5约束说明](docs/aclnnGroupedMatmulV5.md#7-约束说明)。

---

## 接口调用

| 调用方式 | 接口文档 | 调用样例 | 说明 |
|----------|----------|----------|------|
| aclnn调用 | [aclnnGroupedMatmulV5](docs/aclnnGroupedMatmulV5.md) | 见接口文档 [示例代码索引](docs/aclnnGroupedMatmulV5.md#调用示例) | 两段式接口：先调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取workspace大小和executor，再调用 `aclnnGroupedMatmulV5` 执行计算 |
