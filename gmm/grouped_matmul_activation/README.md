# GroupedMatmulActivation

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：融合按M轴分组的矩阵乘（Grouped Matrix Multiplication，GMM）和Swish门控线性单元（Swish-Gated Linear Unit，SwiGLU）激活，用于混合专家（Mixture of Experts，MoE）网络的专家前向计算。输入和权重采用MXFP4 E2M1，权重为FRACTAL_NZ格式，输出为BFLOAT16，不进行输出量化。
- 计算公式：对每个专家的输入和权重应用对应的E8M0缩放因子，再执行矩阵乘。

  $$
  C_i = \operatorname{DequantMX}(X_i,S_i^X) \times \operatorname{DequantMX}(W_i,S_i^W)
  $$

  每个E8M0缩放因子作用于K轴上连续的32个元素。`swiglu_mode=2`时，将矩阵乘结果沿N轴前后分半，前半为激活分支，后半为线性分支：

  $$
  u_i = \min(C_i[..., :N/2], clamp\_limit)
  $$

  $$
  v_i = \operatorname{clip}(C_i[..., N/2:],-clamp\_limit,clamp\_limit)+glu\_bias
  $$

  $$
  y_i = \operatorname{Cast}_{BF16}\left(\frac{u_i}{1+\exp(-glu\_alpha\times u_i)}\times v_i\right)
  $$

  `M`为输入token总数，`K`为输入特征维度，`N`为矩阵乘输出维度，`E`为专家个数，`i`为专家索引。

## 参数说明

下表使用算子定义的参数名和顺序。ACLNN和PyTorch的参数映射分别参见对应接口文档。
表中weight和weight_scale的转置shape是通路归一化后传给算子IR的shape，不是ACLNN调用方直接传入的视图shape；两者的区别见[ACLNN约束说明](docs/aclnnGroupedMatmulActivationWeightNz.md#约束说明)。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :---: | :--- | :--- | :---: |
| x | 输入 | 激活矩阵，逻辑shape为`(M,K)`。 | FLOAT4_E2M1 | ND |
| x_scale | 输入 | 激活的缩放因子，shape为`(M,K/64,2)`。 | FLOAT8_E8M0 | ND |
| group_list | 输入 | 专家分组信息，shape为`(E)`。 | INT64 | ND |
| weight | 输入 | 动态TensorList，长度为1，包含全部专家的权重。非转置时逻辑shape为`(E,K,N)`，转置时为`(E,N,K)`。 | FLOAT4_E2M1 | FRACTAL_NZ |
| weight_scale | 输入 | 动态TensorList，长度为1。非转置时shape为`(E,K/64,N,2)`，转置时为`(E,N,K/64,2)`。 | FLOAT8_E8M0 | ND |
| bias | 输入 | 动态TensorList，预留的矩阵乘偏置，当前必须为空。 | BFLOAT16 | ND |
| y | 输出 | SwiGLU结果，shape为`(M,N/2)`。 | BFLOAT16 | ND |
| dequant_mode | 可选属性 | 反量化模式，取值为2，表示MX反量化，默认值为2。 | INT | - |
| dequant_dtype | 可选属性 | 矩阵乘中间结果类型，取值为0，表示FLOAT32，默认值为0。 | INT | - |
| act_type | 可选属性 | 激活类型，取值为`"swiglu"`，默认值为`"swiglu"`。 | STRING | - |
| group_list_type | 可选属性 | 取值为0或1，0表示累计结束位置，1表示各组token数，默认值为0。 | INT | - |
| tuning_config | 可选属性 | 预留的调优参数，当前必须为空列表，默认值为`[]`。 | LIST_INT | - |
| swiglu_mode | 可选属性 | 默认值为2，表示前后分半、先裁剪再激活并应用alpha/bias的SwiGLU；取值0仅供旧WeightNzV3内部兼容调用。公开接口仅支持2。 | INT | - |
| clamp_limit | 可选属性 | 裁剪上限，须为有限正数，默认值为7.0。 | FLOAT | - |
| glu_alpha | 可选属性 | sigmoid输入缩放系数，须为有限数，默认值为1.702。 | FLOAT | - |
| glu_bias | 可选属性 | 线性分支偏置，须为有限数，默认值为1.0。 | FLOAT | - |
| transpose_weight | 可选属性 | 默认值为false。false表示权重采用`(E,K,N)`视图，true表示采用`(E,N,K)`视图；weight_scale须使用对应布局。ACLNN根据视图推导该属性。 | BOOL | - |

## 约束说明

- 仅沿M轴分组。M须大于0，K、N须为正数且满足128对齐，专家数E的取值范围为`[1,1024]`。
- `weight`和`weight_scale`分别通过一个Tensor传入全部E个专家的数据，不能按专家拆成多个TensorList元素。
- `group_list_type=0`时，分组列表须非负、单调非递减，最后一个值不大于M；`group_list_type=1`时，各组token数须非负且总和不超过M。
- PyTorch接口使用`activation_type`，不暴露`dequant_mode`、`dequant_dtype`和`transpose_weight`。CPU分组列表及输入存储的约束以[PyTorch接口文档](docs/torchapi_grouped_matmul_activation.md)为准。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn API | [可运行示例](examples/arch35/test_aclnn_grouped_matmul_activation_weight_nz.cpp) | 通过[aclnnGroupedMatmulActivationWeightNz](docs/aclnnGroupedMatmulActivationWeightNz.md)两段式接口调用；[编译运行说明](examples/README.md)。 |
| PyTorch API | [调用示例](docs/torchapi_grouped_matmul_activation.md#调用示例) | 通过[grouped_matmul_activation](docs/torchapi_grouped_matmul_activation.md)接口调用。 |

构建算子包的方法参见[源码构建](../../docs/zh/install/quick_install.md)。
