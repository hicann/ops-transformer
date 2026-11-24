# aclnnGroupedMatmulSwigluQuantWeightNzV3

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

- 接口功能：融合GroupedMatmul、dequant、SwiGLU和quant。该接口为weight NZ特化版本，支持Ascend 950PR&950DT系列产品的单Tensor MX A8W8、MX A8W4及A4W4 BF16输出场景。`weight`必须为三维FRACTAL_NZ格式，A8W8和A4W4支持转置和非转置weightNZ；MX A8W4仅支持转置weightNZ。`x`仅支持非转置输入。

- 相比V2，V3增加SwiGLU计算参数`swigluMode`、`clampLimit`、`gluAlpha`和`gluBias`，以及MX量化参数`roundMode`、`scaleAlg`和`dstTypeMax`。量化输出路径的`swigluMode`仅支持2；A4W4 BF16输出路径支持0或2，且`outputScale`必须为nullptr。

- 计算公式：

  1. 根据`groupList`和`groupListType`将`x`划分为不同的分组，并为每个分组选择对应的权重切片。
  2. 执行grouped matmul，并结合`xScale`和`weightScale`完成MX分块反量化，得到待进行SwiGLU计算的矩阵`x_i`：

     $$x_i[m,n] = \sum_{k=0}^{K-1} \left(X_i[m,k] \cdot s_i^X[m,\lfloor k/32\rfloor]\right) \left(W_i[k,n] \cdot s_i^W[\lfloor k/32\rfloor,n]\right)$$

     其中`W_i`表示逻辑上的`[K,N]`权重，$s_i^X$、$s_i^W$是解码后的逻辑缩放因子，按MX block规则在K方向每32个元素一组广播，每64个元素存储两个E8M0 scale。缩放发生在K方向求和之前，不能将随K变化的scale移到矩阵乘法结果之后相乘。

  3. 沿最后一维前后分半，对前半部分执行带clamp和alpha的SiLU，对后半部分执行带clamp和bias的线性分支，再逐元素相乘，得到`swigluOut_i`。
  4. MX A8W4传入`smoothScale`时，先在FP32精度将`swigluOut_i[m,j]`乘以`smoothScale[i,j]`；未传入时跳过该乘法。随后按既有BF16最近偶数舍入及MXFP8动态量化规则，输出`output`和E8M0格式的`outputScale`：

     $$V_i = \operatorname{BF16}_{\mathrm{rint}}(swigluOut_i \cdot smoothScale[i,:]),\qquad output, outputScale = DynamicMxQuant(V_i)$$

     `smoothScale[i,:]`在该expert的token行上广播；缺省时按全1处理。乘法必须发生在BF16舍入之前，不得先将SwiGLU结果舍入为BF16再乘smooth。

  - `swigluMode`取值和计算公式：

  - `swigluMode=2`：沿最后一维前后分半拆分，前半部分为`x_glu`，后半部分为`x_linear`。本接口固定采用前半部分作为激活分支。

    $$x\_glu = x\_glu.clamp(min=None, max=clampLimit)$$

    $$x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)$$

    $$out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)$$

    $$swigluOut_i = out\_glu * (x\_linear + gluBias)$$

  本接口当前固定沿最后一维拆分，具体约束见“约束说明”。

  - MXFP8输出场景先对`smoothScale`处理后的SwiGLU结果（未传入时为`swigluOut_i`）按最近偶数舍入转为BF16，再沿输出的N轴每32个元素组成一个量化组；每64个元素在`outputScale`中存储两个E8M0 scale。设该组的BF16元素为$V_j$，最大绝对值为$a=\max_j |V_j|$。输出类型固定为FLOAT8_E4M3FN，其最大有限值为448。对处于正常可表示范围的非零有限$a$，两种`scaleAlg`的计算公式分别为：

    - `scaleAlg=0`（OCP）：共享指数不向上进位。

      $$e_{\mathrm{OCP}}=\lfloor\log_2 a\rfloor-8,\qquad YScale=2^{e_{\mathrm{OCP}}}$$

      在正常范围内，设$a$的BF16指数位为$b$（含偏置127），则对应的E8M0编码为$s=\max(b-8,0)$。

    - `scaleAlg=1`（cuBLAS）：先用目标类型的最大有限值计算块缩放比，再将其指数向上取整，避免组内最大值在量化时溢出。

      $$r=\frac{a}{448},\qquad e_{\mathrm{cuBLAS}}=\lceil\log_2 r\rceil,\qquad YScale=2^{e_{\mathrm{cuBLAS}}}$$

      本接口在BF16上实现该规则：设$a$的指数位为$b$、7位尾数为$t$，E8M0编码为$s=\max(b+\mathbf{1}_{t>96}-8,0)$。阈值$t=96$对应$448=1.75\times 2^8$，等于阈值时不进位。

    - 对组内各元素，两种算法均使用对应的共享scale量化：

      $$P_j=\operatorname{cast}_{\mathrm{FLOAT8\_E4M3FN},\mathrm{rint}}\!\left(V_j\times\operatorname{BF16}\!\left(\frac{1}{YScale}\right)\right)$$

      $P_j$按原位置组成`output`。有限scale的E8M0编码$s$解码为$YScale=2^{s-127}$。`dstTypeMax=0.0`表示采用E4M3FN的默认最大值448，并非以0为除数；当前不支持非零自定义值。

  非转置和转置的`weight`均使用`ACL_FORMAT_FRACTAL_NZ`格式，通过view stride和物理存储排列区分；`weightScale`的转置属性必须与`weight`一致。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize`获取计算所需的workspace大小以及包含算子计算流程的执行器，再调用`aclnnGroupedMatmulSwigluQuantWeightNzV3`执行计算。

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
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
    const aclIntArray   *tuningConfigOptional,
    int64_t              swigluMode,
    double               clampLimit,
    double               gluAlpha,
    double               gluBias,
    const char          *roundMode,
    int64_t              scaleAlg,
    double               dstTypeMax,
    aclTensor           *output,
    aclTensor           *outputScale,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize

该接口完成输入参数校验、输出shape校验、算子执行流程构造，并返回执行所需的workspace大小和executor。

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 | 非连续的Tensor |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | x（const aclTensor *） | 输入 | 激活矩阵。 | A8W8/A8W4为FP8；A4W4为FP4。 | FLOAT8_E4M3FN、FLOAT4_E2M1 | ND | 2维，(M, K) | √ |
  | weight（const aclTensorList *） | 输入 | 分组权重矩阵。 | 支持转置和非转置weightNZ；具体view和storage shape见约束说明。 | FLOAT8_E4M3FN、FLOAT4_E2M1 | FRACTAL_NZ | 3维，(E, K, N) | 仅支持约束说明中的转置weight view |
  | weightScale（const aclTensorList *） | 输入 | 权重量化因子。 | 与weight对应，具体view和storage shape见约束说明。 | FLOAT8_E8M0 | ND | 4维，(E, ceil(K / 64), N, 2) | √ |
  | weightAssistMatrix（const aclTensorList *） | 输入 | 权重辅助矩阵。 | 当前不支持有效数据；可传nullptr、长度为0的TensorList，或长度为1且唯一元素为nullptr/shape (0)的TensorList。 | - | - | - | - |
  | bias（const aclTensor *） | 输入 | 矩阵乘偏置。 | 当前不支持，必须传nullptr。 | - | - | - | - |
  | xScale（const aclTensor *） | 输入 | 激活量化因子。 | 按K方向每64个元素对应两个E8M0 scale。 | FLOAT8_E8M0 | ND | 3维，(M, ceil(K / 64), 2) | √ |
  | smoothScale（const aclTensor *） | 输入 | 按expert的平滑量化因子。 | 仅MX A8W4可选，其他场景必须传nullptr。FP32 SwiGLU结果乘smooth后再进行BF16舍入和MX量化。 | FLOAT | ND | 2维，(E, N / 2) | √，入口连续化 |
  | groupList（const aclTensor *） | 输入 | 分组信息。 | 长度必须为E；其语义由groupListType决定。 | INT64 | ND | 1维，(E) | √ |
  | dequantMode（int64_t） | 输入 | 反量化模式。 | 当前仅支持2，表示MX反量化。 | - | - | - | - |
  | dequantDtype（int64_t） | 输入 | GroupedMatmul中间结果类型。 | 当前仅支持ACL_FLOAT（枚举值为0）。 | - | - | - | - |
  | quantMode（int64_t） | 输入 | 输出量化模式。 | 当前仅支持2，表示MX量化。 | - | - | - | - |
  | groupListType（int64_t） | 输入 | 分组列表解释方式。 | 0表示cumsum，1表示count。 | - | - | - | - |
  | tuningConfigOptional（const aclIntArray *） | 输入 | tiling调优配置。 | 当前不支持，必须传nullptr或空数组。 | INT64 | - | - | - |
  | swigluMode（int64_t） | 输入 | SwiGLU计算模式。 | 量化输出仅支持2：沿最后一维前后分半，前半为激活分支，后半为线性分支。A4W4 BF16输出还支持0。 | - | - | - | - |
  | clampLimit（double） | 输入 | 表示SwiGLU中间结果的裁剪上限。 | 激活分支裁剪上限为clampLimit，线性分支裁剪范围为[-clampLimit, clampLimit]；swigluMode=2时必须为有限、可由FLOAT表示且转换为FLOAT后仍大于0的值；A4W4的swigluMode=0不使用该参数。 | - | - | - | - |
  | gluAlpha（double） | 输入 | 表示Sigmoid输入的缩放系数。 | swigluMode=2时必须为有限且可由FLOAT表示的值；A4W4的swigluMode=0不使用该参数。 | - | - | - | - |
  | gluBias（double） | 输入 | 表示线性分支的偏置。 | swigluMode=2时必须为有限且可由FLOAT表示的值；A4W4的swigluMode=0不使用该参数。 | - | - | - | - |
  | roundMode（const char *） | 输入 | MX量化舍入模式。 | 传入nullptr或空字符串时按`"rint"`处理；当前仅支持`"rint"`。 | - | - | - | - |
  | scaleAlg（int64_t） | 输入 | MX量化scale算法。 | 当前接口支持0（OCP）和1（cuBLAS）；其他值会报错。 | - | - | - | - |
  | dstTypeMax（double） | 输入 | MX量化目标类型最大值。 | 当前MXFP8输出仅支持0.0。 | - | - | - | - |
  | output（aclTensor *） | 输出 | 计算结果。 | A8W8/A8W4为FP8量化结果；A4W4为BF16。 | FLOAT8_E4M3FN、BFLOAT16 | ND | 2维，(M, N / 2) | √ |
  | outputScale（aclTensor *） | 可选输出 | MXFP8输出scale。 | A8W8/A8W4必须传入；A4W4 BF16输出必须传nullptr。 | FLOAT8_E8M0 | ND | 3维，(M, ceil((N / 2) / 64), 2) | √ |
  | workspaceSize（uint64_t *） | 输出 | Device侧workspace大小。 | 返回值。 | - | - | - | - |
  | executor（aclOpExecutor **） | 输出 | 算子执行器。 | 返回值，传给第二段接口。 | - | - | - | - |

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 810px">
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
      <td>x、weight、weightScale、xScale、groupList、output、workspaceSize或executor为空指针；A8W8/A8W4的outputScale为空指针，或者weight或weightScale的TensorList中包含空Tensor。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>输入或输出的数据类型不在支持范围内；A8W8要求x、weight和output为FLOAT8_E4M3FN，A8W4的weight为FLOAT4_E2M1且output仍为FLOAT8_E4M3FN；两者的xScale、weightScale和outputScale均为FLOAT8_E8M0，groupList为INT64。</td>
    </tr>
    <tr>
      <td>输入或输出的参数维度、shape或format不满足约束，例如weight不是三维FRACTAL_NZ格式，或输出shape不匹配。</td>
    </tr>
    <tr>
      <td>当前设备不是支持的产品类型，或者weight、weightScale的TensorList长度不为1。</td>
    </tr>
    <tr>
      <td>dequantMode、quantMode、dequantDtype不满足2、2、ACL_FLOAT的约束。</td>
    </tr>
    <tr>
      <td>weightAssistMatrix、bias、smoothScale或tuningConfigOptional传入了当前场景不支持的内容。</td>
    </tr>
    <tr>
      <td>A8W8/A8W4的swigluMode不等于2，或A4W4的swigluMode不为0或2；swigluMode=2时，clampLimit、gluAlpha、gluBias不满足有限性及float32可表示性约束，其中clampLimit转换为float32后还必须大于0。</td>
    </tr>
    <tr>
      <td>roundMode、scaleAlg、dstTypeMax不满足MXFP8场景约束。</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulSwigluQuantWeightNzV3

该接口使用第一段接口返回的executor，在指定stream上执行算子计算。

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 |
  |--------|:---:|------|
  | workspace | 输入 | 在Device侧申请的workspace内存地址 |
  | workspaceSize | 输入 | 在Device侧申请的workspace大小，由第一段接口`aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize`获取 |
  | executor | 输入 | op执行器，包含了算子计算流程 |
  | stream | 输入 | 指定执行任务的Stream |

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

### 确定性计算

aclnnGroupedMatmulSwigluQuantWeightNzV3默认为确定性实现。

### MX A8W4

- `x`固定为FLOAT8_E4M3FN，`weight`固定为FLOAT4_E2M1 FRACTAL_NZ；不接受FLOAT8_E5M2、FLOAT4_E1M2或ND权重。`xScale`、`weightScale`及`outputScale`均为FLOAT8_E8M0，`output`固定为FLOAT8_E4M3FN。
- 当前仅支持单Tensor承载全部expert，仅支持`transposeWeight=true`，由weight和weightScale的view stride识别，不额外暴露转置参数。
- API入口的weight view均为`(E, K, N)`。转置stride为`(K*N, 1, K)`，物理storage为`(E, ceil(K/64), ceil(N/16), 16, 64)`。FP4的C0为64，不能沿用FP8的32；storage维度以FP4元素计数，每字节打包两个FP4元素。
- weightScale的入口view均为`(E, ceil(K/64), N, 2)`。转置stride为`(N*ceil(K/64)*2, 2, ceil(K/64)*2, 1)`，对应storage为`(E, N, ceil(K/64), 2)`。weight与weightScale的转置属性必须一致。入口复用既有转置归一化和scale连续化函数。
- `E`范围为`[1,1024]`，`M`为正整数，`K`、`N`为正的128整数倍；`xScale`、输出及groupList的shape和分组语义与下方A8W8规则一致。
- `smoothScale`可为nullptr；非空时必须严格为FLOAT、ND、`(E,N/2)`，不接受`(E)`、`(N/2)`或其他广播shape。支持非连续输入，入口检查存储边界后进行连续化，并传给既有执行函数。
- 仅支持`swigluMode=2`、`dequantMode=2`、`dequantDtype=ACL_FLOAT`、`quantMode=2`、`roundMode="rint"`（nullptr或空字符串取默认值）、`scaleAlg=0/1`、`dstTypeMax=0.0`。

### MX A4W4 BF16输出

- `x`、`weight`均为FLOAT4_E2M1，`xScale`、`weightScale`为FLOAT8_E8M0；`output`为BFLOAT16、ND、`(M,N/2)`，`outputScale`必须为nullptr，不执行MXFP8输出量化。
- `x`为非转置ND `(M,K)`；weight TensorList长度为1，view为`(E,K,N)`，支持转置和非转置FRACTAL_NZ。转置storage为`(E,K/64,N/16,16,64)`；非转置storage为`(E,N/64,K/16,16,64)`，C0均为64个逻辑FP4元素。weightScale的view为`(E,K/64,N,2)`，转置属性须与weight一致。
- `E`为`[1,1024]`内的整数，`M>0`，`K`和`N`均为128的正整数倍；`xScale`为`(M,K/64,2)`。groupList为Device INT64 ND `(E)`，分组语义与下方说明一致。
- `smoothScale`、`bias`必须为nullptr；weightAssistMatrix及tuningConfigOptional沿用参数表的空值约束。
- `swigluMode`支持0和2。模式0不使用clampLimit、gluAlpha、gluBias；模式2使用上述clamp计算公式和标量校验规则。
- `dequantMode=2`、`dequantDtype=ACL_FLOAT`、`quantMode=2`仍须传入；roundMode、scaleAlg、dstTypeMax仍按参数表校验，不能因BF16输出而传入任意值。

### 原有MX A8W8

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
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
              <td>转置时view：(E, K, N)，stride：(K * N, 1, K)，storage：(E, ceil(K / 32), ceil(N / 16), 16, 32)<br>
              非转置时view：(E, K, N)，stride：(K * N, N, 1)，storage：(E, ceil(N / 32), ceil(K / 16), 16, 32)</td>
              <td>转置时view：(E, ceil(K / 64), N, 2)，stride：(N * ceil(K / 64) * 2, 2, ceil(K / 64) * 2, 1)，storage：(E, N, ceil(K / 64), 2)<br>
              非转置时view：(E, ceil(K / 64), N, 2)，stride：(ceil(K / 64) * N * 2, N * 2, 2, 1)，storage：(E, ceil(K / 64), N, 2)</td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
          </tbody>
          </table>

        - 转置和非转置的weight、weightScale具有相同的view shape，但view stride和storage布局不同，转置属性通过view stride区分。
        - `E`表示expert数量，取值为`[1, 1024]`范围内的整数。
        - `M`表示x的行数，必须为非负整数。
        - `K`表示矩阵乘归约维度，必须为正整数。
        - `N`表示GroupedMatmul结果的列数，必须为正整数且为64的整数倍。
        - `M=0`且其他参数合法时返回空输出，workspace大小为0。

        - weight约束：TensorList长度必须为1，唯一Tensor的首维为E，承载全部expert，不支持按expert拆成多个Tensor。仅支持上表列出的非转置或转置布局，不支持任意非连续权重view。
        - weightScale约束：TensorList长度必须为1，与weight的唯一Tensor对应；最后一维固定为2，转置属性必须与weight一致。
        - groupList约束：长度为E，`groupListType`仅支持0或1。
            - `groupListType=0`：groupList须为非负单调非递减数列（累积和），最后一个值不大于x中tensor的第一维。
            - `groupListType=1`：groupList须为非负数列（各组大小），数值总和不大于x中tensor的第一维。
            - groupList未指定的output区域不参与计算，其内容不作保证。MXFP8场景下，对应的outputScale填充编码0。MX A8W4场景下，对应的outputScale内容不作保证。
  <!-- end id7 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  ```cpp
  #include <algorithm>
  #include <cstdint>
  #include <cstdio>
  #include <exception>
  #include <fstream>
  #include <memory>
  #include <string>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"

  namespace {
  constexpr int64_t SHAPE_PRODUCT_IDENTITY = 1;
  constexpr int64_t ELEMENT_STRIDE = 1;
  constexpr int64_t LAST_DIM_OFFSET = 1;
  constexpr int64_t PENULTIMATE_DIM_OFFSET = 2;
  constexpr int64_t DEFAULT_PRINT_ELEMENTS = 32;
  constexpr int64_t OUTPUT_PRINT_ELEMENTS = 64;
  constexpr int64_t NZ_K0 = 16;
  constexpr int64_t NZ_C0 = 32;
  constexpr int64_t MX_GROUP_SIZE = 64;
  constexpr int64_t SCALE_PAIR_SIZE = 2;
  constexpr int64_t SWIGLU_SPLIT_FACTOR = 2;
  constexpr size_t SINGLE_TENSOR_COUNT = 1;
  constexpr uint8_t E8M0_ONE = 127;
  constexpr size_t X_PATTERN_MULTIPLIER = 7;
  constexpr size_t X_PATTERN_OFFSET = 3;
  constexpr size_t WEIGHT_PATTERN_MULTIPLIER = 5;
  constexpr size_t WEIGHT_PATTERN_OFFSET = 1;
  constexpr int ARG_TRANSPOSE = 1;
  constexpr int ARG_DUMP_PREFIX = 2;
  constexpr int ARG_SCALE_ALG = 3;
  constexpr int MAX_ARGUMENT_COUNT = 4;
  constexpr int INVALID_ARGUMENT_EXIT = 2;
  constexpr int64_t SCALE_ALG_CUBLAS = 1;
  } // namespace

  #define CHECK_RET(cond, return_expr) \
      do { \
          if (!(cond)) { \
              return_expr; \
          } \
      } while (0)

  #define LOG_PRINT(message, ...) \
      do { \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = SHAPE_PRODUCT_IDENTITY;
      for (auto v : shape) {
          shapeSize *= v;
      }
      return shapeSize;
  }

  template <typename T1, typename T2>
  auto Ceil(T1 a, T2 b) -> T1
  {
      if (b == 0) {
          return a;
      }
      return (a + b - LAST_DIM_OFFSET) / b;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ret=%d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ret=%d\n", ret); aclFinalize(); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ret=%d\n", ret); aclrtResetDevice(deviceId);
                aclFinalize(); return ret);
      return ACL_SUCCESS;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &logicalShape,
                      const std::vector<int64_t> &storageShape, void **deviceAddr, aclDataType dataType,
                      aclFormat formatType, aclTensor **tensor, const std::vector<int64_t> *customStrides = nullptr)
  {
      uint64_t size = GetShapeSize(storageShape) * sizeof(T);
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ret=%d\n", ret); return ret);
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ret=%d\n", ret); return ret);
      std::vector<int64_t> strides(logicalShape.size(), ELEMENT_STRIDE);
      for (int64_t i = logicalShape.size() - PENULTIMATE_DIM_OFFSET; i >= 0; --i) {
          strides[i] = logicalShape[i + LAST_DIM_OFFSET] * strides[i + LAST_DIM_OFFSET];
      }
      if (customStrides != nullptr) {
          strides = *customStrides;
      }
      *tensor = aclCreateTensor(logicalShape.data(), logicalShape.size(), dataType, strides.data(), 0, formatType,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);
      return ACL_SUCCESS;
  }

  template <typename T>
  int CreateAclTensorND(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      return CreateAclTensor(hostData, shape, shape, deviceAddr, dataType, ACL_FORMAT_ND, tensor);
  }

  template <typename T>
  void PrintVector(const std::vector<T> &data, const std::string &name, int64_t printNum = DEFAULT_PRINT_ELEMENTS)
  {
      LOG_PRINT("======== %s ========\n", name.c_str());
      int64_t size = static_cast<int64_t>(data.size());
      int64_t limit = std::min(size, printNum);
      for (int64_t i = 0; i < limit; ++i) {
          LOG_PRINT("%s[%ld] = %d\n", name.c_str(), i, static_cast<int32_t>(data[i]));
      }
      LOG_PRINT("\n");
  }

  bool DumpFile(const std::string &path, const std::vector<uint8_t> &data)
  {
      std::ofstream outputFile(path, std::ios::binary);
      if (!outputFile.is_open()) {
          LOG_PRINT("failed to open dump file: %s\n", path.c_str());
          return false;
      }
      outputFile.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
      outputFile.flush();
      if (!outputFile.good()) {
          LOG_PRINT("failed to write or flush dump file: %s\n", path.c_str());
          return false;
      }
      return true;
  }

  int RunExample(bool transposeWeight, const std::string &dumpPrefix, int64_t scaleAlg, aclrtStream stream)
  {
      // 2. 构造输入和输出 Tensor；权重转置时同步调整 weightScale 的 view。
      constexpr int64_t dequantMode = 2;
      const int64_t dequantDtype = ACL_FLOAT;
      constexpr int64_t quantMode = 2;
      const int64_t groupListType = 0;
      constexpr int64_t swigluMode = 2;
      constexpr double clampLimit = 7.0;
      constexpr double gluAlpha = 1.702;
      constexpr double gluBias = 1.0;
      const char *roundMode = "rint";
      const double dstTypeMax = 0.0;
      const aclTensorList *weightAssistMatrix = nullptr;
      const aclTensor *bias = nullptr;
      const aclTensor *smoothScale = nullptr;
      const aclIntArray *tuningConfigOptional = nullptr;
      constexpr int64_t E = 1;
      constexpr int64_t M = 2048;
      constexpr int64_t K = 64;
      constexpr int64_t N = 128;

      std::vector<int64_t> xShape = {M, K};
      // The API entry view is always [E, K, N]. A transposed (ZN) weight is
      // distinguished by its view stride and physical FRACTAL_NZ storage.
      std::vector<int64_t> weightLogicalShape = {E, K, N};
      std::vector<int64_t> weightStorageShape =
          transposeWeight ? std::vector<int64_t>{E, Ceil(K, NZ_C0), Ceil(N, NZ_K0), NZ_K0, NZ_C0} :
                            std::vector<int64_t>{E, Ceil(N, NZ_C0), Ceil(K, NZ_K0), NZ_K0, NZ_C0};
      const int64_t scaleK = Ceil(K, MX_GROUP_SIZE);
      std::vector<int64_t> weightScaleShape = {E, scaleK, N, SCALE_PAIR_SIZE};
      std::vector<int64_t> weightScaleStorageShape =
          transposeWeight ? std::vector<int64_t>{E, N, scaleK, SCALE_PAIR_SIZE} : weightScaleShape;
      std::vector<int64_t> xScaleShape = {M, Ceil(K, MX_GROUP_SIZE), SCALE_PAIR_SIZE};
      std::vector<int64_t> groupListShape = {E};
      std::vector<int64_t> outputShape = {M, N / SWIGLU_SPLIT_FACTOR};
      std::vector<int64_t> outputScaleShape = {M, Ceil(N / SWIGLU_SPLIT_FACTOR, MX_GROUP_SIZE), SCALE_PAIR_SIZE};

      constexpr uint8_t fp8Pattern[] = {0x38, 0xb8, 0x40, 0x30, 0xc0, 0x28, 0x3c, 0xbc, 0x48, 0xc8, 0x50, 0x20};
      std::vector<uint8_t> xHostData(GetShapeSize(xShape));
      std::vector<uint8_t> weightHostData(GetShapeSize(weightStorageShape));
      for (size_t i = 0; i < xHostData.size(); ++i) {
          xHostData[i] =
              fp8Pattern[(i * X_PATTERN_MULTIPLIER + X_PATTERN_OFFSET) % (sizeof(fp8Pattern) / sizeof(fp8Pattern[0]))];
      }
      for (size_t i = 0; i < weightHostData.size(); ++i) {
          weightHostData[i] = fp8Pattern[(i * WEIGHT_PATTERN_MULTIPLIER + WEIGHT_PATTERN_OFFSET) %
                                         (sizeof(fp8Pattern) / sizeof(fp8Pattern[0]))];
      }
      std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleShape), E8M0_ONE);
      std::vector<uint8_t> xScaleHostData(GetShapeSize(xScaleShape), E8M0_ONE);
      std::vector<int64_t> groupListHostData = {M};
      std::vector<uint8_t> outputHostData(GetShapeSize(outputShape), 0);
      std::vector<uint8_t> outputScaleHostData(GetShapeSize(outputScaleShape), 0);

      void *xDeviceAddr = nullptr;
      void *weightDeviceAddr = nullptr;
      void *weightScaleDeviceAddr = nullptr;
      void *xScaleDeviceAddr = nullptr;
      void *groupListDeviceAddr = nullptr;
      void *outputDeviceAddr = nullptr;
      void *outputScaleDeviceAddr = nullptr;

      aclTensor *x = nullptr;
      aclTensor *weightTensor = nullptr;
      aclTensor *weightScaleTensor = nullptr;
      aclTensor *xScale = nullptr;
      aclTensor *groupList = nullptr;
      aclTensor *output = nullptr;
      aclTensor *outputScale = nullptr;

      aclTensorList *weight = nullptr;
      aclTensorList *weightScale = nullptr;

      auto ret = CreateAclTensorND<uint8_t>(xHostData, xShape, &xDeviceAddr, ACL_FLOAT8_E4M3FN, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      std::vector<int64_t> weightViewStrides = transposeWeight ? std::vector<int64_t>{K * N, ELEMENT_STRIDE, K} :
                                                                 std::vector<int64_t>{K * N, N, ELEMENT_STRIDE};
      ret = CreateAclTensor<uint8_t>(weightHostData, weightLogicalShape, weightStorageShape, &weightDeviceAddr,
                                     ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ, &weightTensor, &weightViewStrides);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weightTensor, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      aclTensor *weightArray[] = {weightTensor};
      weight = aclCreateTensorList(weightArray, SINGLE_TENSOR_COUNT);
      CHECK_RET(weight != nullptr, LOG_PRINT("aclCreateTensorList(weight) failed\n"); return ACL_ERROR_FAILURE);
      std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorListPtr(weight,
                                                                                                 aclDestroyTensorList);
      // aclDestroyTensorList also destroys its member tensors.
      weightTensorPtr.release();
      // The ZN scale view is obtained by transposing an [E, N, scaleK, 2]
      // source. Its API entry view remains [E, scaleK, N, 2].
      std::vector<int64_t> weightScaleViewStrides =
          transposeWeight ?
              std::vector<int64_t>{N * scaleK * SCALE_PAIR_SIZE, SCALE_PAIR_SIZE, scaleK * SCALE_PAIR_SIZE,
                                   ELEMENT_STRIDE} :
              std::vector<int64_t>{scaleK * N * SCALE_PAIR_SIZE, N * SCALE_PAIR_SIZE, SCALE_PAIR_SIZE, ELEMENT_STRIDE};
      ret =
          CreateAclTensor<uint8_t>(weightScaleHostData, weightScaleShape, weightScaleStorageShape, &weightScaleDeviceAddr,
                                   ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &weightScaleTensor, &weightScaleViewStrides);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightScaleTensorPtr(weightScaleTensor,
                                                                                          aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      aclTensor *weightScaleArray[] = {weightScaleTensor};
      weightScale = aclCreateTensorList(weightScaleArray, SINGLE_TENSOR_COUNT);
      CHECK_RET(weightScale != nullptr, LOG_PRINT("aclCreateTensorList(weightScale) failed\n"); return ACL_ERROR_FAILURE);
      std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightScaleTensorListPtr(
          weightScale, aclDestroyTensorList);
      weightScaleTensorPtr.release();

      ret = CreateAclTensorND<uint8_t>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, ACL_FLOAT8_E8M0, &xScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xScaleTensorPtr(xScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensorND<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, ACL_INT64, &groupList);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensorND<uint8_t>(outputHostData, outputShape, &outputDeviceAddr, ACL_FLOAT8_E4M3FN, &output);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(output, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outputDeviceAddrPtr(outputDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensorND<uint8_t>(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, ACL_FLOAT8_E8M0,
                                       &outputScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputScaleTensorPtr(outputScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outputScaleDeviceAddrPtr(outputScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 第一段接口完成校验并获取 workspace 大小和 executor。
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
          x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode, dequantDtype,
          quantMode, groupListType, tuningConfigOptional, swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg,
          dstTypeMax, output, outputScale, &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ret=%d\n", ret); return ret);
      LOG_PRINT("workspaceSize = %lu\n", workspaceSize);
      void *workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("workspace malloc failed\n"); return ret);
      }
      std::unique_ptr<void, aclError (*)(void *)> workspacePtr(workspaceAddr, aclrtFree);

      // 4. 第二段接口执行算子，随后同步并读取输出。
      ret = aclnnGroupedMatmulSwigluQuantWeightNzV3(workspaceAddr, workspaceSize, executor, stream);
      // Synchronize even if launch reports an error, before releasing device buffers.
      const auto syncRet = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("run op failed. ret=%d\n", ret); return ret);
      CHECK_RET(syncRet == ACL_SUCCESS, LOG_PRINT("sync stream failed. ret=%d\n", syncRet); return syncRet);

      LOG_PRINT("run success\n");

      ret = aclrtMemcpy(outputHostData.data(), outputHostData.size() * sizeof(uint8_t), outputDeviceAddr,
                        outputHostData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy output failed. ret=%d\n", ret); return ret);

      ret = aclrtMemcpy(outputScaleHostData.data(), outputScaleHostData.size() * sizeof(uint8_t), outputScaleDeviceAddr,
                        outputScaleHostData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputScale failed. ret=%d\n", ret); return ret);

      PrintVector<uint8_t>(outputHostData, "output", MX_GROUP_SIZE);

      PrintVector<uint8_t>(outputScaleHostData, "outputScale", MX_GROUP_SIZE);

      if (!dumpPrefix.empty()) {
          CHECK_RET(DumpFile(dumpPrefix + "_output.bin", outputHostData), return ACL_ERROR_FAILURE);
          CHECK_RET(DumpFile(dumpPrefix + "_output_scale.bin", outputScaleHostData), return ACL_ERROR_FAILURE);
          LOG_PRINT("dumpPrefix=%s\n", dumpPrefix.c_str());
      }

      // Tensor lists, tensors, workspace and device buffers are released by RAII
      // before main destroys the stream and resets the device.
      return ACL_SUCCESS;
  }

  int main(int argc, char **argv)
  {
      bool transposeWeight = false;
      int64_t scaleAlg = 0;
      const std::string dumpPrefix = argc > ARG_DUMP_PREFIX ? argv[ARG_DUMP_PREFIX] : "";
      try {
          const std::string transposeArg = argc > ARG_TRANSPOSE ? argv[ARG_TRANSPOSE] : "0";
          const std::string scaleAlgArg = argc > ARG_SCALE_ALG ? argv[ARG_SCALE_ALG] : "0";
          if (argc > MAX_ARGUMENT_COUNT || (transposeArg != "0" && transposeArg != "1") ||
              (scaleAlgArg != "0" && scaleAlgArg != "1")) {
              LOG_PRINT("usage: %s [transposeWeight: 0|1] [dumpPrefix] [scaleAlg: 0|1]\n", argv[0]);
              return INVALID_ARGUMENT_EXIT;
          }
          transposeWeight = transposeArg == "1";
          scaleAlg = scaleAlgArg == "1" ? SCALE_ALG_CUBLAS : 0;
      } catch (const std::exception &error) {
          LOG_PRINT("invalid arguments: %s\n", error.what());
          return INVALID_ARGUMENT_EXIT;
      }

      // 1. 初始化 Device 和 Stream。
      const int32_t deviceId = 0;
      aclrtStream stream = nullptr;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      try {
          ret = RunExample(transposeWeight, dumpPrefix, scaleAlg, stream);
      } catch (const std::exception &error) {
          LOG_PRINT("example failed: %s\n", error.what());
          ret = ACL_ERROR_FAILURE;
      }

      // 5. 释放 Device 和 Stream；Tensor、TensorList 与 workspace 在 RunExample 中释放。
      const auto destroyRet = aclrtDestroyStream(stream);
      const auto resetRet = aclrtResetDevice(deviceId);
      const auto finalizeRet = aclFinalize();
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      CHECK_RET(destroyRet == ACL_SUCCESS, LOG_PRINT("aclrtDestroyStream failed. ret=%d\n", destroyRet);
                return destroyRet);
      CHECK_RET(resetRet == ACL_SUCCESS, LOG_PRINT("aclrtResetDevice failed. ret=%d\n", resetRet); return resetRet);
      CHECK_RET(finalizeRet == ACL_SUCCESS, LOG_PRINT("aclFinalize failed. ret=%d\n", finalizeRet); return finalizeRet);
      LOG_PRINT("resource cleanup success\n");
      return ACL_SUCCESS;
  }
  ```
<!-- end id8 -->
