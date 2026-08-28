# aclnnMhcPreBackward

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_pre_backward)

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

- 接口功能：MhcPreBackward是MhcPre的反向算子，MhcPre算子基于一系列计算得到mHC（Manifold-Constrained Hyper-Connections）架构中的$H^{res}$和$H^{post}$投影矩阵以及Atten或MLP层的输入矩阵$h^{in}$。

- 计算公式：

  - 输出组合梯度计算：

    - 正向公式：

      $$
      H\_in = \sum_{i=1}^{N} x[{B,S,i,:}] · H\_pre_n[B,S,i]
      $$

    - 反向计算：

      $$
      \begin{aligned}
      H\_pre\_grad &= \text{Reduce}\left(H\_in\_grad.\text{unsqueeze}(-2) \odot x, \text{dim}=-1\right) \quad ([B,S,N]) \\
      x\_grad\_vec3 &= H\_in\_grad \times H\_pre \quad ([B,S,N,D])
      \end{aligned}
      $$

  - Sigmoid门控反向（H_pre）：
    - 正向公式：

      $$
      H\_pre = \text{Sigmoid}(\alpha\_pre * H\_pre\_1 + bias\_pre) + hc\_eps
      $$

    - 反向计算：

      $$
      \begin{aligned}
      s &= H\_pre - hc\_eps \\
      H\_pre\_2\_grad &= H\_pre\_grad \odot s \odot (1 - s) \\
      H\_pre\_1\_grad &= H\_pre\_2\_grad \cdot \alpha\_pre \\
      \alpha\_pre\_grad &= \sum_{b,s,n}^{B,S,N} \left(H\_pre\_2\_grad \cdot H\_pre\_1\right) \\
      bias\_pre\_grad &= \sum_{b,s}^{B,S} H\_pre\_2\_grad \quad ([N])
      \end{aligned}
      $$

  - Sigmoid门控反向（H_post）：
    - 正向公式：

      $$
      H\_post = \text{Sigmoid}(\alpha\_post * H\_post\_1 + bias\_post) * 2
      $$

    - 反向计算：

      $$
      \begin{aligned}
      H\_post\_2\_grad &= H\_post\_grad \odot \left(H\_post \cdot \left(1 - \frac{H\_post}{2}\right)\right) \\
      H\_post\_1\_grad &= H\_post\_2\_grad \cdot \alpha_{post} \\
      \alpha_{post\_grad} &= \sum_{b,s,n}^{B,S,N} \left(H\_post\_2\_grad \cdot H_{post\_1}\right) \\
      bias\_post\_grad &= \sum_{b,s}^{B,S} H\_post\_2\_grad \quad ([N])
      \end{aligned}
      $$

  - 残差连接反向（H_res）：
    - 正向公式：

      $$
      H\_res = \alpha\_res * H\_res\_1 + bias\_res
      $$

    - 反向计算：

      $$
      \begin{aligned}
      H\_res\_2\_grad &= H\_res\_grad \cdot \alpha_{res} \quad ([B,S,N,N]) \\
      \alpha\_res\_grad &= \sum_{b,s,i,j}^{B,S,N,N} \left(H\_res\_grad \cdot H\_res\_2\right) \\
      bias\_res\_grad &= \sum_{b,s}^{B,S} H\_res\_grad \quad ([N,N]) \\
      H\_res\_1\_grad &= \text{Reshape}(H\_res\_2\_grad) \quad ([B,S,N^2])
      \end{aligned}
      $$

  - RMSNorm Fusion反向：
    - 正向公式：

      $$
      H\_mix\_tmp = H\_mix * inv\_rms
      $$

    - 反向计算：

      $$
      \begin{aligned}
      H\_mix\_tmp\_grad &= \text{Concat}(H\_pre\_1\_grad, H\_post\_1\_grad, H\_res\_1\_grad) \quad ([B,S,2N+N^2]) \\
      H\_mix\_grad &= H\_mix\_tmp\_grad \cdot inv\_rms \\
      inv\_rms_{grad} &= \sum_{\text{last\_dim}} \left(H\_mix\_tmp\_grad \cdot H\_mix\right) \quad ([B,S,1])
      \end{aligned}
      $$

  - 矩阵乘法反向：
    - 正向公式：

      $$
      H\_mix = x\_rs @ phi^T
      $$

      $$
      x\_rs = x * gamma
      $$

    - 反向计算：

      $$
      \begin{aligned}
      x\_rs\_grad &= H\_mix\_grad @ phi \quad ([B,S,ND]) \\
      X &= \text{Reshape}(x\_rs, [B\cdot S, ND]) \\
      G &= \text{Reshape}(H\_mix\_grad, [B\cdot S, 2N+N^2]) \\
      phi_{grad} &= G^T @ X \quad ([2N+N^2, ND])
      \end{aligned}
      $$

  - 特征缩放反向：
    - 正向公式：

      $$
      x\_rs = x * gamma
      $$

    - 反向计算：

      $$
      \begin{aligned}
      x\_grad\_mm &= x\_rs\_grad * gamma \\
      gamma\_grad &= \sum_{b=1}^{B}\sum_{s=1}^{S} (x * x\_rs\_grad)\quad ([N,D])
      \end{aligned}
      $$

  - RMS归一化梯度计算：
    - 正向公式：

      $$
      inv\_rms = \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + eps}}, \quad其中\ n = N * D
      $$

    - 反向计算：

      $$
      \begin{aligned}
      x\_rs\_grad\_inv &= - \left(\frac{inv\_rms\_grad \cdot {inv\_rms}^3}{N*D}\right) \cdot x\_rs \\
      x\_rs\_grad &= x\_grad\_mm + x\_rs\_grad\_inv \\
      x\_grad\_vec1 &= \text{Reshape}(x\_rs\_grad, [B,S,N,D]) \\
      x\_grad &= x\_grad\_vec3 + x\_grad\_vec1
      \end{aligned}
      $$

  - 融合mhc_post的grad_x相加操作：

    $$
    \begin{aligned}
    x\_grad &= x\_grad + grad\_x\_post
    \end{aligned}
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMhcPreBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMhcPreBackward”接口执行计算。

```Cpp
aclnnStatus aclnnMhcPreBackwardGetWorkspaceSize(
    const aclTensor     *x,
    const aclTensor     *phi,
    const aclTensor     *alpha,
    const aclTensor     *gradHIn,
    const aclTensor     *gradHPost,
    const aclTensor     *gradHRes,
    const aclTensor     *invRms,
    const aclTensor     *hMix,
    const aclTensor     *hPre,
    const aclTensor     *hPost,
    const aclTensor     *gamma,
    const aclTensor     *gradXPostOptional,
    float               hcEps,
    const aclTensor     *gradX,
    const aclTensor     *gradPhi,
    const aclTensor     *gradAlpha,
    const aclTensor     *gradBias,
    const aclTensor     *gradGamma,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnMhcPreBackward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnMhcPreBackwardGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">
            <col style="width: 400px">
            <col style="width: 212px">
            <col style="width: 100px">
            <col style="width: 290px">
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
            <td>x（aclTensor*）</td>
            <td>输入</td>
            <td>待计算数据，表示网络中mHC层的输入数据。</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(B, S, N, D)、(T, N, D)。其中，B：支持泛化；S：支持泛化；T=B*S。</li></ul></td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>3-4</td>
            <td>√</td>
        </tr>
        <tr>
            <td>phi（aclTensor*）</td>
            <td>输入</td>
            <td>mHC的参数矩阵。</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(2N+N*N, N*D)或者(2N+N!, N*D)。其中，N与`x`的N保持一致；D与`x`的D保持一致。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>alpha（aclTensor*）</td>
            <td>输入</td>
            <td>mHC的缩放参数alpha。</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(3)。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>1</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradHIn（aclTensor*）</td>
            <td>输入</td>
            <td>hIn作为Atten/MLP层的输入。正向输出hIn对应的梯度。</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(B, S, D)、(T, D)。其中，B与`x`的B保持一致；S与`x`的S保持一致；T=B*S；D与`x`的D保持一致。</li></ul></td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>2-3</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradHPost（aclTensor*）</td>
            <td>输入</td>
            <td>正向输出hPost对应的梯度。</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(B,S,N)、(T,N)。其中，B与`x`的B保持一致；S与`x`的S保持一致；T=B*S；N与`x`的N保持一致。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2-3</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradHRes（aclTensor*）</td>
            <td>输入</td>
            <td>正向输出hRes对应的梯度。</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(B, S, N, N)、(T, N, N)、(B, S, N!)或(T, N!)。其中，B与`x`的B保持一致；S与`x`的S保持一致；T=B*S；N与`x`的N保持一致。N!表示N的阶乘排列数（如N=4时，N!=24）。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2-4</td>
            <td>√</td>
        </tr>
        <tr>
            <td>invRms（aclTensor*）</td>
            <td>输入</td>
            <td>正向RmsNorm计算的invRms。</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(B, S)、(T)。其中，B与`x`的B保持一致；S与`x`的S保持一致；T=B*S。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>1-2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>hMix（aclTensor*）</td>
            <td>输入</td>
            <td>正向计算流x@phi的结果</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(B, S, 2N+N*N)、(B, S, 2N+N!)、(T, 2N+N*N)、(T, 2N+N!)。其中，B与`x`的B保持一致；S与`x`的S保持一致；T=B*S；N与`x`的N保持一致。最后一维需与`phi`的第0维（fusionSize）保持一致。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2-3</td>
            <td>√</td>
        </tr>
        <tr>
            <td>hPre（aclTensor*）</td>
            <td>输入</td>
            <td>正向sigmoid计算之后的hPre矩阵</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(B, S, N)、(T, N)。其中，B与`x`的B保持一致；S与`x`的S保持一致；T=B*S；N与`x`的N保持一致。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2-3</td>
            <td>√</td>
        </tr>
        <tr>
            <td>hPost（aclTensor*）</td>
            <td>输入</td>
            <td>正向的hPost输出</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(B, S, N)、(T, N)。其中，B与`x`的B保持一致；S与`x`的S保持一致；T=B*S；N与`x`的N保持一致。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2-3</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gamma（aclTensor*）</td>
            <td>可选输入</td>
            <td>RmsNorm的缩放系数gamma</td>
            <td><ul><li>不支持空Tensor。</li><li>如果传入nullptr，则表示全1的tensor。</li><li>shape为(N ,D)。其中，N与`x`的N保持一致；D与`x`的D保持一致。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradXPostOptional（aclTensor*）</td>
            <td>可选输入</td>
            <td>post反向输出的gradX</td>
            <td><ul><li>不支持空Tensor。</li><li>如果传入nullptr，则表示全0的tensor。</li><li>shape为(B, S, N, D)、(T, N, D)。其中，B与`x`的B保持一致；S与`x`的S保持一致；T=B*S；N与`x`的N保持一致；D与`x`的D保持一致。</li></ul></td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>3-4</td>
            <td>√</td>
        </tr>
        <tr>
            <td>hcEps（float）</td>
            <td>可选输入</td>
            <td>HPre的sigmoid后的eps参数</td>
            <td>
           <li>建议值为1e-6</li>
            </td>
            <td>FLOAT32</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>gradX（aclTensor*）</td>
            <td>输出</td>
            <td>x对应的梯度。</td>
            <td><ul><li>不支持空Tensor。</li><li>与输入x的维度、数据类型保持一致</li><li>shape为(B, S, N, D)、(T, N, D)。其中，B与`x`的B保持一致；S与`x`的S保持一致；T=B*S；N与`x`的N保持一致；D与`x`的D保持一致。</li></ul></td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>3-4</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradPhi（aclTensor*）</td>
            <td>输出</td>
            <td>phi对应的梯度。</td>
            <td><ul><li>不支持空Tensor。</li><li>与输入phi的维度保持一致</li><li>shape为(2N+N*N, N*D)或(2N+N!, N*D)。其中，N与`x`的N保持一致；D与`x`的D保持一致。与`phi`的shape保持一致。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradAlpha（aclTensor*）</td>
            <td>输出</td>
            <td>alpha对应的梯度。</td>
            <td><ul><li>不支持空Tensor。</li><li>与输入alpha维度保持一致。</li><li>shape为(3)。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>1</td>
            <td>√</td>
        </tr>
          <tr>
            <td>gradBias（aclTensor*）</td>
            <td>输出</td>
            <td>bias对应的梯度。</td>
            <td><ul><li>不支持空Tensor。</li><li>shape为(2N+N*N)或(2N+N!)。与`phi`的第0维（fusionSize）保持一致。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>1</td>
            <td>√</td>
        </tr>
        <tr>
            <td>gradGamma（aclTensor*）</td>
            <td>可选输出</td>
            <td>gamma对应的梯度。</td>
            <td><ul><li>不支持空Tensor。</li><li>当输入gamma不为nullptr时，此变量才会输出。</li><li>与输入gamma的Shape维度保持一致。</li><li>shape为(N, D)。其中，N与`x`的N保持一致；D与`x`的D保持一致。</li></ul></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>workspaceSize（uint64_t*）</td>
            <td>输出</td>
            <td>返回需要在Device侧申请的workspace大小。</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>executor（aclOpExecutor**）</td>
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

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 参数`phi`的shape仅支持(2N+N\*N, N\*D)。
    - 参数`gradHRes`的shape仅支持(B, S, N, N)、(T, N, N)。
    - 参数`hMix`的shape仅支持(B, S, 2N+N\*N)、(T, 2N+N\*N)。
    - 参数`gradPhi`的shape仅支持(2N+N\*N, N\*D)。
    - 参数`gradBias`的shape仅支持(2N+N\*N)。
  <!-- end id7 -->

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>必选参数或者输出是空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>输入变量，x、phi、gamma、alpha的数据类型和数据格式不在支持的范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内部调用npu runtime的接口异常。</td>
            </tr>
        </tbody>
    </table>

## aclnnMhcPreBackward

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPreBackwardGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream流。</td>
        </tr>
    </tbody>
    </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnMhcPreBackward默认采用确定性实现。

- 规格约束
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
        <col style="width: 100px">
        <col style="width: 500px">
        </colgroup>
        <thead>
            <tr>
                <th>格式</th>
                <th>gradHRes shape</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>BSNN</td>
            <td>(B, S, N, N)</td>
        </tr>
        <tr>
            <td>TNN</td>
            <td>(T, N, N)</td>
        </tr>
        </tbody>
    </table>

    <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
        <col style="width: 100px">
        <col style="width: 500px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>N</td>
            <td>N值目前支持4、6、8。</td>
        </tr>
        <tr>
            <td>D</td>
            <td>D支持1~16384范围以内，要求64元素对齐。</td>
        </tr>
        <tr>
            <td>fusionSize</td>
            <td>仅支持N*N+2N。</td>
        </tr>
        </tbody>
    </table>
  <!-- end id7 -->

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>格式</th>
                <th>gradHRes shape</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>BSNN</td>
            <td>(B, S, N, N)</td>
        </tr>
        <tr>
            <td>BSN!</td>
            <td>(B, S, N!)</td>
        </tr>
        <tr>
            <td>TNN</td>
            <td>(T, N, N)</td>
        </tr>
        <tr>
            <td>TN!</td>
            <td>(T, N!)</td>
        </tr>
        </tbody>
    </table>

    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>N</td>
            <td>N值目前仅支持4。</td>
        </tr>
        <tr>
            <td>D</td>
            <td>D支持1~16384范围以内，要求128元素对齐。</td>
        </tr>
        <tr>
            <td>fusionSize</td>
            <td>支持N!+2N和N*N+2N两种fusionSize。</td>
        </tr>
        </tbody>
    </table>
  <!-- end id8 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_backward.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr, const char *name, size_t elemSize = sizeof(float))
{
    auto size = GetShapeSize(shape);
    size_t copyBytes = size * elemSize;
    if (elemSize == 2) {
        // BF16: read raw bytes, convert to float for display
        std::vector<uint16_t> rawData(size, 0);
        auto ret = aclrtMemcpy(rawData.data(), copyBytes, *deviceAddr, copyBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
        LOG_PRINT("%s result (first 10 elements):\n", name);
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            union {
                uint32_t i;
                float f;
            } u;
            u.i = (uint32_t)rawData[i] << 16;
            LOG_PRINT("  [%ld] = %f\n", i, u.f);
        }
    } else {
        // float32
        std::vector<float> resultData(size, 0);
        auto ret = aclrtMemcpy(resultData.data(), copyBytes, *deviceAddr, copyBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
        LOG_PRINT("%s result (first 10 elements):\n", name);
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  [%ld] = %f\n", i, resultData[i]);
        }
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
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

int main()
{
    // 1.（固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {1024, 4, 512};      // T, N, D
    std::vector<int64_t> phiShape = {24, 2048};        // 2 * N + N * N, N * D
    std::vector<int64_t> alphaShape = {3};             // 固定大小
    std::vector<int64_t> gradHInShape = {1024, 512};   // T, D
    std::vector<int64_t> gradHPostShape = {1024, 4};   // T, N
    std::vector<int64_t> gradHResShape = {1024, 4, 4}; // T, N, N
    std::vector<int64_t> invRmsShape = {1024};         // T
    std::vector<int64_t> hMixShape = {1024, 24};       // T, 2 * N + N * N
    std::vector<int64_t> hPreShape = {1024, 4};        // T, N
    std::vector<int64_t> hPostShape = {1024, 4};       // T, N
    std::vector<int64_t> gammaShape = {4, 512};        // N, D
    std::vector<int64_t> gradXShape = {1024, 4, 512};  // T, N, D
    std::vector<int64_t> gradPhiShape = {24, 2048};    // 2 * N + N * N, N * D
    std::vector<int64_t> gradAlphaShape = {3};         // 固定大小
    std::vector<int64_t> gradGammaShape = {4, 512};    // N, D
    std::vector<int64_t> gradBiasShape = {24};         // 2*N+N*N
    std::vector<int64_t> gradXPostOptionalShape = {1024, 4, 512};

    void *xDeviceAddr = nullptr;
    void *phiDeviceAddr = nullptr;
    void *alphaDeviceAddr = nullptr;
    void *gradHInDeviceAddr = nullptr;
    void *gradHPostDeviceAddr = nullptr;
    void *gradHResDeviceAddr = nullptr;
    void *invRmsDeviceAddr = nullptr;
    void *hMixDeviceAddr = nullptr;
    void *hPreDeviceAddr = nullptr;
    void *hPostDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *gradXDeviceAddr = nullptr;
    void *gradPhiDeviceAddr = nullptr;
    void *gradAlphaDeviceAddr = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    void *gradGammaDeviceAddr = nullptr;
    void *gradXPostOptionalDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *phi = nullptr;
    aclTensor *alpha = nullptr;
    aclTensor *gradHIn = nullptr;
    aclTensor *gradHPost = nullptr;
    aclTensor *gradHRes = nullptr;
    aclTensor *invRms = nullptr;
    aclTensor *hMix = nullptr;
    aclTensor *hPre = nullptr;
    aclTensor *hPost = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *gradX = nullptr;
    aclTensor *gradPhi = nullptr;
    aclTensor *gradAlpha = nullptr;
    aclTensor *gradBias = nullptr;
    aclTensor *gradGamma = nullptr;
    aclTensor *gradXPostOptional = nullptr;

    std::vector<short> xHostData(1024 * 4 * 512, 1.0);
    std::vector<float> phiHostData(24 * 2048, 1.0);
    std::vector<float> alphaHostData(3, 1.0);
    std::vector<short> gradHInHostData(1024 * 512, 1.0);
    std::vector<float> gradHPostHostData(1024 * 4, 1.0);
    std::vector<float> gradHResHostData(1024 * 4 * 4, 1.0);
    std::vector<float> invRmsHostData(1024, 1.0);
    std::vector<float> hMixHostData(1024 * 24, 1.0);
    std::vector<float> hPreHostData(1024 * 4, 1.0);
    std::vector<float> hPostHostData(1024 * 4, 1.0);
    std::vector<float> gammaHostData(4 * 512, 1.0);
    std::vector<short> gradXHostData(1024 * 4 * 512, 0);
    std::vector<float> gradPhiHostData(24 * 2048, 0);
    std::vector<float> gradAlphaHostData(3, 0);
    std::vector<float> gradBiasHostData(24, 0);
    std::vector<float> gradGammaHostData(4 * 512, 0);
    std::vector<short> gradXPostOptionalHostData(1024 * 4 * 512, 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(phiHostData, phiShape, &phiDeviceAddr, aclDataType::ACL_FLOAT, &phi);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(alphaHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alpha);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHInHostData, gradHInShape, &gradHInDeviceAddr, aclDataType::ACL_BF16, &gradHIn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHPostHostData, gradHPostShape, &gradHPostDeviceAddr, aclDataType::ACL_FLOAT, &gradHPost);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHResHostData, gradHResShape, &gradHResDeviceAddr, aclDataType::ACL_FLOAT, &gradHRes);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(invRmsHostData, invRmsShape, &invRmsDeviceAddr, aclDataType::ACL_FLOAT, &invRms);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hMixHostData, hMixShape, &hMixDeviceAddr, aclDataType::ACL_FLOAT, &hMix);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hPreHostData, hPreShape, &hPreDeviceAddr, aclDataType::ACL_FLOAT, &hPre);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hPostHostData, hPostShape, &hPostDeviceAddr, aclDataType::ACL_FLOAT, &hPost);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_BF16, &gradX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradPhiHostData, gradPhiShape, &gradPhiDeviceAddr, aclDataType::ACL_FLOAT, &gradPhi);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradAlphaHostData, gradAlphaShape, &gradAlphaDeviceAddr, aclDataType::ACL_FLOAT, &gradAlpha);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradGammaHostData, gradGammaShape, &gradGammaDeviceAddr, aclDataType::ACL_FLOAT, &gradGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXPostOptionalHostData, gradXPostOptionalShape, &gradXPostOptionalDeviceAddr,
                          aclDataType::ACL_BF16, &gradXPostOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    float hc_eps = 1e-6;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 80 * 1024 * 1024;
    aclOpExecutor *executor;

    // 调用aclnnMhcPreBackward第一段接口
    ret = aclnnMhcPreBackwardGetWorkspaceSize(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost,
                                              gamma, gradXPostOptional, hc_eps, gradX, gradPhi, gradAlpha, gradBias,
                                              gradGamma, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnMhcPreBackward第二段接口
    ret = aclnnMhcPreBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreBackward failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(gradXShape, &gradXDeviceAddr, "gradX", sizeof(short));
    PrintOutResult(gradPhiShape, &gradPhiDeviceAddr, "gradPhi");
    PrintOutResult(gradAlphaShape, &gradAlphaDeviceAddr, "gradAlpha");
    PrintOutResult(gradBiasShape, &gradBiasDeviceAddr, "gradBias");
    PrintOutResult(gradGammaShape, &gradGammaDeviceAddr, "gradGamma");

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(phi);
    aclDestroyTensor(alpha);
    aclDestroyTensor(gradHIn);
    aclDestroyTensor(gradHPost);
    aclDestroyTensor(gradHRes);
    aclDestroyTensor(invRms);
    aclDestroyTensor(hMix);
    aclDestroyTensor(hPre);
    aclDestroyTensor(hPost);
    aclDestroyTensor(gamma);
    aclDestroyTensor(gradX);
    aclDestroyTensor(gradPhi);
    aclDestroyTensor(gradAlpha);
    aclDestroyTensor(gradBias);
    aclDestroyTensor(gradGamma);
    aclDestroyTensor(gradXPostOptional);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(phiDeviceAddr);
    aclrtFree(alphaDeviceAddr);
    aclrtFree(gradHInDeviceAddr);
    aclrtFree(gradHPostDeviceAddr);
    aclrtFree(gradHResDeviceAddr);
    aclrtFree(invRmsDeviceAddr);
    aclrtFree(hMixDeviceAddr);
    aclrtFree(hPreDeviceAddr);
    aclrtFree(hPostDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    aclrtFree(gradPhiDeviceAddr);
    aclrtFree(gradAlphaDeviceAddr);
    aclrtFree(gradBiasDeviceAddr);
    aclrtFree(gradGammaDeviceAddr);
    aclrtFree(gradXPostOptionalDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
