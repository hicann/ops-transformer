# aclnnGroupedMatmulSwigluQuantWeightNzV3

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_swiglu_quant_v2)

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

- 接口功能：融合GroupedMatmul、dequant、SwiGLU和quant。该接口为weight NZ特化版本，调用者必须传入FRACTAL_NZ格式的`weight`。当前仅支持Ascend 950PR/Ascend 950DT的单Tensor MXFP8场景。

- 与V2接口的主要区别：增加SwiGLU计算参数`swigluMode`、`clampLimit`、`gluAlpha`和`gluBias`，以及MX量化参数`roundMode`、`scaleAlg`和`dstTypeMax`。当前`swigluMode`仅支持2。

- 计算公式：

  1. 根据 `groupList` 和 `groupListType` 将 `x` 划分为不同的分组，并为每个分组选择对应的权重切片。
  2. 执行 grouped matmul，并结合 `xScale` 和 `weightScale` 完成 MXFP8 反量化，得到待进行 SwiGLU 计算的矩阵 `x_i`：

     $$x_i[m,n] = \sum_{k=0}^{K-1} \left(X_i[m,k] \cdot s_i^X[m,\lfloor k/32\rfloor]\right) \left(W_i[k,n] \cdot s_i^W[\lfloor k/32\rfloor,n]\right)$$

     其中 `W_i` 表示逻辑上的 `[K,N]` 权重，$s_i^X$、$s_i^W$ 是解码后的逻辑缩放因子，按MX block规则在K方向每32个元素一组广播，每64个元素存储两个E8M0 scale。缩放发生在K方向求和之前，不能将随K变化的scale移到矩阵乘法结果之后相乘。

  3. 沿最后一维前后分半，对前半部分执行带clamp和alpha的SiLU，对后半部分执行带clamp和bias的线性分支，再逐元素相乘，得到`swigluOut_i`。
  4. 对 `swigluOut_i` 执行 MXFP8 动态量化，输出量化结果 `output` 和 E8M0 格式的 `outputScale`：

     $$output, outputScale = DynamicMxQuant(swigluOut_i)$$

  - `swigluMode`取值和计算公式：

  - `swigluMode=2`：沿最后一维前后分半拆分，前半部分为 `x_glu`，后半部分为 `x_linear`。本接口固定采用前半部分作为激活分支。

    $$x\_glu = x\_glu.clamp(min=None, max=clampLimit)$$

    $$x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)$$

    $$out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)$$

    $$swigluOut_i = out\_glu * (x\_linear + gluBias)$$

  本接口当前固定沿最后一维拆分，具体约束见“约束说明”。

  - MX 量化先将 `swigluOut_i` 按最近偶数舍入转为 BF16，再沿输出的 N 轴每 32 个元素组成一个量化组；每 64 个元素在 `outputScale` 中存储两个 E8M0 scale。设该组的 BF16 元素为 $V_j$，最大绝对值为 $a=\max_j |V_j|$。输出类型固定为 FLOAT8_E4M3FN，其最大有限值为 448。对处于正常可表示范围的非零有限 $a$，两种 `scaleAlg` 的计算公式分别为：

    - `scaleAlg=0`（OCP）：共享指数不向上进位。

      $$e_{\mathrm{OCP}}=\lfloor\log_2 a\rfloor-8,\qquad YScale=2^{e_{\mathrm{OCP}}}$$

      在正常范围内，设 $a$ 的 BF16 指数位为 $b$（含偏置 127），则对应的 E8M0 编码为 $s=\max(b-8,0)$。

    - `scaleAlg=1`（cuBLAS）：先用目标类型的最大有限值计算块缩放比，再将其指数向上取整，避免组内最大值在量化时溢出。

      $$r=\frac{a}{448},\qquad e_{\mathrm{cuBLAS}}=\lceil\log_2 r\rceil,\qquad YScale=2^{e_{\mathrm{cuBLAS}}}$$

      本接口在 BF16 上实现该规则：设 $a$ 的指数位为 $b$、7 位尾数为 $t$，E8M0 编码为 $s=\max(b+\mathbf{1}_{t>96}-8,0)$。阈值 $t=96$ 对应 $448=1.75\times 2^8$，等于阈值时不进位。

    - 对组内各元素，两种算法均使用对应的共享 scale 量化：

      $$P_j=\operatorname{cast}_{\mathrm{FLOAT8\_E4M3FN},\mathrm{rint}}\!\left(V_j\times\operatorname{BF16}\!\left(\frac{1}{YScale}\right)\right)$$

      $P_j$ 按原位置组成 `output`。有限 scale 的 E8M0 编码 $s$ 解码为 $YScale=2^{s-127}$；超出下界时编码截断到 0。全零组使用编码 0（解码值为 $2^{-127}$），含 Inf/NaN 的组使用编码 255。OCP 对最大指数位为 0 的组（全零或仅包含 BF16 次正规数）将倒数置 0；cuBLAS 仅对全零组将倒数置 0。实现使用 BF16 倒数乘法，不能以无限精度除法替代。`dstTypeMax=0.0` 表示采用 E4M3FN 的默认最大值 448，并非以 0 为除数；当前不支持非零自定义值。

  非转置和转置的 `weight` 均使用 `ACL_FORMAT_FRACTAL_NZ` 格式，通过 view stride 和物理存储排列区分。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize` 获取计算所需的 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnGroupedMatmulSwigluQuantWeightNzV3` 执行计算。

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
    const aclTensor *x,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensorList *weightAssistMatrix,
    const aclTensor *bias,
    const aclTensor *xScale,
    const aclTensor *smoothScale,
    const aclTensor *groupList,
    int64_t dequantMode,
    int64_t dequantDtype,
    int64_t quantMode,
    int64_t groupListType,
    const aclIntArray *tuningConfigOptional,
    int64_t swigluMode,
    double clampLimit,
    double gluAlpha,
    double gluBias,
    const char *roundMode,
    int64_t scaleAlg,
    double dstTypeMax,
    aclTensor *output,
    aclTensor *outputScale,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize

该接口完成输入参数校验、输出 shape 校验、算子执行流程构造，并返回执行所需的 workspace 大小和 executor。

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
      <tr><td>x</td><td>输入</td><td>激活矩阵。</td><td>MXFP8 输入；当前仅支持 FLOAT8_E4M3FN。</td><td>FLOAT8_E4M3FN</td><td>ND</td><td>2 维，(M, K)</td><td>√</td></tr>
      <tr><td>weight</td><td>输入</td><td>分组权重矩阵。</td><td>TensorList长度必须为1；非转置和转置的API入口view shape均为(E, K, N)。支持下文规定的非转置和转置weight的view、stride和storage shape，包括转置形成的非连续view；接口保留其FRACTAL_NZ物理存储，不执行任意非连续weight的转连续操作。</td><td>FLOAT8_E4M3FN</td><td>FRACTAL_NZ</td><td>3维，(E, K, N)</td><td>仅支持约束说明中的转置weight view</td></tr>
      <tr><td>weightScale</td><td>输入</td><td>权重量化因子。</td><td>TensorList 长度必须为 1，与 weight 的唯一 Tensor 对应；非转置和转置的API入口view shape均为(E, ceil(K / 64), N, 2)，转置属性必须与weight一致。</td><td>FLOAT8_E8M0</td><td>ND</td><td>4 维，(E, ceil(K / 64), N, 2)</td><td>√</td></tr>
      <tr><td>weightAssistMatrix</td><td>可选输入</td><td>权重辅助矩阵。</td><td>当前不支持有效数据；可传nullptr、长度为0的TensorList，或长度为1且唯一元素为nullptr/shape (0)的TensorList。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>bias</td><td>可选输入</td><td>矩阵乘偏置。</td><td>当前不支持，必须传 nullptr。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>xScale</td><td>输入</td><td>激活量化因子。</td><td>按 K 方向每 64 个元素对应两个 E8M0 scale。</td><td>FLOAT8_E8M0</td><td>ND</td><td>3 维，(M, ceil(K / 64), 2)</td><td>√</td></tr>
      <tr><td>smoothScale</td><td>可选输入</td><td>平滑量化因子。</td><td>当前不支持，必须传 nullptr。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>groupList</td><td>输入</td><td>分组信息。</td><td>长度必须为 E；其语义由 groupListType 决定。</td><td>INT64</td><td>ND</td><td>1 维，(E)</td><td>√</td></tr>
      <tr><td>dequantMode</td><td>输入</td><td>反量化模式。</td><td>当前仅支持 2，表示 MX 量化反量化。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>dequantDtype</td><td>输入</td><td>GroupedMatmul 中间结果类型。</td><td>当前仅支持 0，即 DT_FLOAT。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>quantMode</td><td>输入</td><td>输出量化模式。</td><td>当前仅支持 2，表示 MX 量化。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>groupListType</td><td>输入</td><td>分组列表解释方式。</td><td>0 表示 cumsum，1 表示 count。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>tuningConfigOptional</td><td>可选输入</td><td>tiling 调优配置。</td><td>当前不支持，必须传 nullptr 或空数组。</td><td>aclIntArray</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>swigluMode</td><td>输入</td><td>SwiGLU 计算模式。</td><td>当前仅支持 2：沿最后一维前后分半，前半为激活分支，后半为线性分支。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>clampLimit</td><td>输入</td><td>表示SwiGLU中间结果的裁剪上限。</td><td>必选参数。激活分支裁剪上限为clampLimit，线性分支裁剪范围为[-clampLimit, clampLimit]；必须为有限、可由FLOAT表示且大于0的值。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>gluAlpha</td><td>输入</td><td>表示Sigmoid输入的缩放系数。</td><td>必选参数，必须为有限且可由FLOAT表示的值。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>gluBias</td><td>输入</td><td>表示线性分支的偏置。</td><td>必选参数，必须为有限且可由FLOAT表示的值。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>roundMode</td><td>输入</td><td>MX量化舍入模式。</td><td>可选参数，传入nullptr或空字符串时按`"rint"`处理；当前仅支持`"rint"`。</td><td>const char *</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>scaleAlg</td><td>输入</td><td>MX 量化 scale 算法。</td><td>当前接口支持 0（OCP）和 1（cuBLAS）；其他值会报错。</td><td>INT64</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>dstTypeMax</td><td>输入</td><td>MX 量化目标类型最大值。</td><td>当前 MXFP8 输出仅支持 0.0。</td><td>DOUBLE</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>output</td><td>输出</td><td>MXFP8 量化结果。</td><td>调用方创建并传入，当前仅支持 FLOAT8_E4M3FN。</td><td>FLOAT8_E4M3FN</td><td>ND</td><td>2 维，(M, N / 2)</td><td>√</td></tr>
      <tr><td>outputScale</td><td>输出</td><td>MXFP8 输出 scale。</td><td>调用方创建并传入。</td><td>FLOAT8_E8M0</td><td>ND</td><td>3 维，(M, ceil((N / 2) / 64), 2)</td><td>√</td></tr>
      <tr><td>workspaceSize</td><td>输出</td><td>Device 侧 workspace 大小。</td><td>返回值。</td><td>uint64_t *</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>executor</td><td>输出</td><td>算子执行器。</td><td>返回值，传给第二段接口。</td><td>aclOpExecutor **</td><td>-</td><td>-</td><td>-</td></tr>
    </tbody>
    </table>

- **返回值**

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
    <td>x、weight、weightScale、xScale、groupList、output、outputScale、workspaceSize或executor为空指针，或者weight或weightScale的TensorList中包含空Tensor。</td>
  </tr>
  <tr>
    <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="7">161002</td>
    <td>输入或输出的数据类型不在支持范围内；当前 MXFP8 weightNZ 场景要求 x、weight 和 output 为 FLOAT8_E4M3FN，xScale、weightScale 和 outputScale 为 FLOAT8_E8M0，groupList 为 INT64。</td>
  </tr>
  <tr>
    <td>输入或输出的参数维度、shape或format不满足约束，例如weight不是三维FRACTAL_NZ格式，或输出shape不匹配。</td>
  </tr>
  <tr>
    <td>当前设备不是 Ascend 950，或者 weight、weightScale 的 TensorList 长度不为 1。</td>
  </tr>
  <tr>
    <td>dequantMode、quantMode、dequantDtype 不满足 2、2、0 的约束。</td>
  </tr>
  <tr>
    <td>weightAssistMatrix、bias、smoothScale 或 tuningConfigOptional 传入了当前场景不支持的内容。</td>
  </tr>
  <tr>
    <td>swigluMode 不等于 2，或 clampLimit、gluAlpha、gluBias 不满足有限性及 float32 可表示性约束；其中 clampLimit 还必须大于 0。</td>
  </tr>
  <tr>
    <td>roundMode、scaleAlg、dstTypeMax 不满足 MXFP8 场景约束。</td>
  </tr>
</tbody>
</table>

## aclnnGroupedMatmulSwigluQuantWeightNzV3

该接口使用第一段接口返回的 executor，在指定 stream 上执行算子计算。

- **参数说明**

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `workspace` | 输入 | Device 侧申请的 workspace 内存地址；`workspaceSize` 大于 0 时不能为空，等于 0 时可传 `nullptr`。 |
| `workspaceSize` | 输入 | Device 侧 workspace 大小，必须使用第一段接口返回的值。 |
| `executor` | 输入 | 第一段接口返回的 op executor，不能为空。 |
| `stream` | 输入 | 指定算子执行的 ACL stream。 |

- **返回值**

返回 `aclnnStatus` 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

当 `executor` 为空，或 `workspaceSize` 大于 0 且 `workspace` 为空时，返回 `ACLNN_ERR_PARAM_NULLPTR`（161001）。

## 约束说明

  - 确定性计算：
      - aclnnGroupedMatmulSwigluQuantWeightNzV3默认为确定性实现。

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - 支持范围：仅支持下述 MXFP8 weightNZ 场景，不支持 per-token、MXFP4、A8W4、A4W4 或其他量化场景；`weight` 和 `weightScale` 的 TensorList 长度必须均为 1，每个唯一 Tensor 承载全部 E 个 expert，不支持按 expert 拆成多个 Tensor；`weight` 必须为三维 FRACTAL_NZ 格式，支持非转置和转置输入，`x` 仅支持非转置输入。
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
              <td>API入口view：(E, K, N)<br>
              非转置storage：(E, ceil(N / 32), ceil(K / 16), 16, 32)<br>
              转置storage：(E, ceil(K / 32), ceil(N / 16), 16, 32)</td>
              <td>API入口view：(E, ceil(K / 64), N, 2)<br>
              非转置source/storage与入口view相同<br>
              转置source/storage：(E, N, ceil(K / 64), 2)</td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
          </tbody>
          </table>

        - 维度与布局：`E` 是 expert 数量，范围为 [1, 1024]；`weight` 的逻辑首维及 `groupList` 长度均为 E，单 Tensor 可承载多个 expert。`M`（token 数）和 `K`（输入特征数）均大于 0；SwiGLU 拆分前的 `N` 为 64 的正整数倍，输出宽度为 `N / 2`。`weightScale` 最后一维固定为 2。`weight` 和 `weightScale` 的转置属性必须一致；仅支持上表列出的非转置或转置布局，不支持任意非连续权重 view；其他输入输出 Tensor 支持符合上述 shape 和布局约束的非连续 view。
        - 分组与可选输入：`groupListType` 仅支持 0（累计 token 数，非负、非降且不大于 `M`）和 1（各组 token 数，非负且总和不大于 `M`）；未指定的输出区域不更新。`bias`、`smoothScale` 必须为 `nullptr`；`weightAssistMatrix` 仅可为 `nullptr`、空 TensorList，或包含唯一 `nullptr`/shape (0) Tensor 的 TensorList；`tuningConfigOptional` 仅可为 `nullptr` 或空数组。
        - SwiGLU 与量化参数：`swigluMode` 仅支持 2；`clampLimit`、`gluAlpha` 和 `gluBias` 必须为有限且可表示为 `float32` 的值，`clampLimit` 还必须大于 0。`dequantMode=quantMode=2`，`dequantDtype=0`（中间计算使用 `DT_FLOAT`）；`roundMode` 仅支持 `"rint"`，`nullptr` 或空字符串等效于 `"rint"`；`scaleAlg` 仅支持 0（OCP）和 1（cuBLAS）；`dstTypeMax` 只能为有限的 0.0。
  <!-- end id7 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  以下示例与[示例源码](../examples/arch35/test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.cpp)一致，依次演示资源初始化、输入输出构造、两段式接口调用及资源释放。FP8 输入使用 E4M3FN 编码，权重按 FRACTAL_NZ 格式构造。

  示例默认使用非转置权重及 `scaleAlg=0`。运行前加载 CANN 环境并安装算子包；可通过参数 `0/1` 指定是否转置权重，并通过第三个参数指定 `scaleAlg`，例如 `./test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3 1 result 1`。示例设备号为 0，请按实际环境调整。

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
      const int64_t dequantDtype = 0;
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
