# GroupedMatmulActivationQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：融合GroupedMatmul、activation和quant计算。当前版本支持WeightNz路径下的MXFP8和MXFP4输入场景，激活函数仅支持gelu_tanh，量化输出支持FP8和FP4。

- 计算公式：
  - <term>Ascend 950PR/Ascend 950DT</term>：

    <details>
    <summary>MX量化场景：</summary>

    - 定义：
      - $E$ 表示专家数，$M$ 表示总token数，$K$ 表示输入特征维度，$N$ 表示输出特征维度。
      - $blocksize$ 表示MX量化时共享指数的分组大小，当前仅支持64。
      - **·** 表示矩阵乘法，**⊙** 表示逐元素乘法。

    - 根据group_list确定每个group对应的token范围。

    - 对每个group执行GroupedMatmul和反量化计算，中间结果默认为FLOAT32类型：

      $$
      C_i = (X_i \cdot weight_i) \odot x\_scale_{i\ Broadcast} \odot weight\_scale_{i\ Broadcast}
      $$

    - 执行gelu_tanh激活：

      $$
      S_i = GeluTanh(C_i)
      $$

      当前kernel底层实现使用gelu_sigmoid函数对gelu_tanh进行近似计算：

      $$
      \operatorname{GeluTanh}(x) =
      \frac{x}{1 + \exp\left(-1.595769121 \times \left(x + 0.044715 \times x^3\right)\right)}
      $$

    - 对激活结果在N轴按$blocksize=32$分组执行MX动态量化，输出量化结果$y$和量化因子$y\_scale$。

    </details>

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| ------ | -------------- | ---- | -------- | -------- | ----------- |
| x | 输入 | 左矩阵，对应公式中的$X$。 | FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1、FLOAT4_E1M2 | ND | 2维，形如`(M, K)` |
| group_list | 输入 | 分组信息，对应公式中的`group_list`。 | INT64 | ND | 1维，形如`(E,)` |
| weight | 动态输入 | 右矩阵dynamic tensorList，长度仅支持1。MXFP8时为FLOAT8_E4M3FN；MXFP4时可为任一FLOAT4类型，且x与weight必须同为FLOAT4。 | FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2 | FRACTAL_NZ | viewShape为3维，storageShape为5维 |
| weight_scale | 动态输入 | `weight`的MX量化因子，dynamic tensorList，长度仅支持1。 | FLOAT8_E8M0 | ND | 4维 |
| bias | 动态输入 | bias dynamic tensorList。当前MX场景必须为空。 | FLOAT | ND | - |
| x_scale | 可选输入 | `x`的MX量化因子。当前MX场景必须传入。 | FLOAT8_E8M0 | ND | 3维，形如`(M, ceil(K / 64), 2)` |
| y | 输出 | 激活并量化后的输出矩阵。FP4输出仅支持MXFP4输入场景。 | FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1、FLOAT4_E1M2 | ND | 2维，形如`(M, N)` |
| y_scale | 输出 | 输出`y`的MX量化因子。 | FLOAT8_E8M0 | ND | 3维，形如`(M, ceil(N / 64), 2)` |
| activation_type | 必选属性 | 激活函数类型，当前仅支持`"gelu_tanh"`。 | STRING | - | - |
| transpose_weight | 可选属性 | 表示`weight`是否转置，默认false。 | BOOL | - | - |
| group_list_type | 可选属性 | 表示`group_list`输入的分组方式，支持0和1，默认0。 | INT64 | - | - |
| tuning_config | 可选属性 | 预留调优参数，默认`[0]`，当前暂不使用。 | LIST_INT | - | - |
| quant_mode | 可选属性 | 量化模式，当前仅支持`"mx"`。 | STRING | - | - |
| y_dtype | 可选属性 | 输出`y`的数据类型，默认DT_UNDEFINED，此时输出类型跟随`x`；支持35（FLOAT8_E5M2）、36（FLOAT8_E4M3FN）、40（FLOAT4_E2M1）、41（FLOAT4_E1M2）。 | INT64 | - | - |
| round_mode | 可选属性 | 量化舍入模式，默认`"rint"`，当前仅支持`"rint"`。 | STRING | - | - |
| scale_alg | 可选属性 | MX量化scale算法，默认0；支持0、1、2。0支持全部输出类型；1仅支持FP8输出；2仅支持FLOAT4_E2M1输出。 | INT64 | - | - |
| dst_type_max | 可选属性 | 表示maxType的取值，对应公式中的Amax(DType)，默认0.0。scale_alg为2时支持0.0或[6.0, 12.0]；其他scale_alg仅支持0.0。 | FLOAT | - | - |

## 约束说明

- `x`仅支持非转置输入；`weight`支持非转置和转置输入，`weight_scale`转置属性需要与`weight`保持一致。
- `N`必须为64的整数倍，`group_list`第一维取值范围为[1, 1024]。
- MXFP4场景下`K`必须为偶数且不能为2，weight的NZ storageShape尾维为64；MXFP8场景尾维为32。
- `scale_alg=1`仅支持FLOAT8输出；`scale_alg=2`仅支持FLOAT4_E2M1输出；FLOAT4_E1M2输出仅支持`scale_alg=0`。
- `bias`为空时，支持nullptr、空tensorList或长度为1且元素shape为(0)的空tensorList。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_grouped_matmul_activation_quant_weight_nz](examples/arch35/test_aclnn_grouped_matmul_activation_quant_weight_nz.cpp) | 通过[aclnnGroupedMatmulActivationQuantWeightNz](docs/aclnnGroupedMatmulActivationQuantWeightNz.md)方式调用GroupedMatmulActivationQuant算子。 |
