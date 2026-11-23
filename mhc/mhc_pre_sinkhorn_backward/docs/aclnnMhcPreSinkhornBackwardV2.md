# aclnnMhcPreSinkhornBackwardV2

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

- **接口功能**：aclnnMhcPreSinkhornBackwardV2是[aclnnMhcPreSinkhornV2](../../mhc_pre_sinkhorn/docs/aclnnMhcPreSinkhornV2.md)的反向接口。计算对应的梯度反向传播。

  - **Cube计算模式**：该接口在Cube中默认使用HF32计算模式（与premix能力合入前一致）。
    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：可通过可选属性`enableHf32`（V2接口传入，默认true）控制，传入false时切换为FP32计算模式。
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT系列</term>：支持通过可选属性`enableHf32`切换Cube计算模式，默认值为`true`，使用HF32计算。传入`false`时使用FP32计算模式。
    <!-- end id8 -->

- **计算公式**：

  - **第一部分计算H_res_grad**：设当前梯度为$\mathbf{{grad}_{\text{curr}} = {gradHRes}}$（`gradHRes`为输入参数，即正向输出`hRes`的梯度），共进行$numIters$（Sinkhorn迭代次数，对应`sk_iter_count`，当前仅支持20）次迭代。

    - **前${numIters} - 1$次迭代**：对迭代编号$i = {numIters} - 1, {numIters} - 2, \dots, 1$，依次执行，其中$sumOut$（对应输入参数`sumOut`）为Sinkhorn变换正向计算保存的中间sum结果，$normOut$（对应输入参数`normOut`）为Sinkhorn变换正向计算保存的中间norm结果，${dotProd}_k$为第$k$次点积计算结果（中间变量）：

     $$
      \begin{aligned}
        {dotProd}_{2i+1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \left( {grad}_{\text{curr}} \cdot {normOut}_{2i+1} \right) \\
              \\
        {grad}_{\text{curr}} &\gets \frac{{grad}_{\text{curr}} - {dotProd}_{2i+1}}{{sumOut}_{2i+1}}, \\
        \\
        {dotProd}_{2i} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \left( {grad}_{\text{curr}} \cdot {normOut}_{2i} \right), \\
        \\
        {grad}_{\text{curr}} &\gets \frac{{grad}_{\text{curr}} - {dotProd}_{2i}}{{sumOut}_{2i}}, \\
      \end{aligned}
      $$

    - **最后一次迭代**：

      $$
      \begin{aligned}
        {dotProd}_{1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \left( {grad}_{\text{curr}} \cdot {normOut}_{1} \right), \\
        \\
        {grad}_{\text{curr}} &\gets \frac{{grad}_{\text{curr}} - {dotProd}_{1}}{{sumOut}_{1}}, \\
        \\
        {dotProd}_{0} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \left( {grad}_{\text{curr}} \cdot {normOut}_{0} \right), \\
        \\
        {HResGrad} &\gets \left( {grad}_{\text{curr}} - {dotProd}_{0} \right) \cdot {normOut}_{0}, \\
      \end{aligned}
      $$

  - **输入**：${gradHin} \in [B, S, C]$（`gradHin`为输入参数，即正向输出`hIn`的梯度）

  - **输出组合梯度计算**：

    - **正向计算公式**：其中$x$为输入参数（前向输入x），$N$为输入Tensor中N维度的大小。按可选输入preMix是否传入分为两套：

      - **preMix未传入**：加权系数使用前向保存的中间结果h_pre（$hPre$为输入参数）：

        $$
        {HIn} = \sum_{i=1}^{N} {x}[B, S, i, :] \cdot {hPre}[B, S, i]
        $$

      - **preMix传入**：加权系数使用可选输入preMix（$preMix$为输入参数，通常填上一轮输出的hPre），$hPre$等中间变量仍按前向公式正常计算输出：

        $$
        {HIn} = \sum_{i=1}^{N} {x}[B, S, i, :] \cdot {preMix}[B, S, i]
        $$

    - **反向计算公式**：按preMix是否传入分为两套：

      - **preMix未传入**：$HPreGrad$即pre分支的梯度$dPre$：

        $$
        \begin{aligned}
        {HPreGrad} &= \text{Reduce}\left({gradHin}.\text{unsqueeze}(-2) \odot {x}, \text{dim}=-1\right) \quad ([B,S,N]) \\
        {dPre} &= {HPreGrad} \\
        {xGradVec3} &= {gradHin} \times {hPre} \quad ([B,S,N,C])
        \end{aligned}
        $$

      - **preMix传入**：$dPre$改由可选输入$gradPre$提供，$HPreGrad$的计算结果作为可选输出$gradPreMix$（preMix的梯度）：

        $$
        \begin{aligned}
        {gradPreMix} &= \text{Reduce}\left({gradHin}.\text{unsqueeze}(-2) \odot {x}, \text{dim}=-1\right) \quad ([B,S,N]) \\
        {dPre} &= {gradPre} \\
        {xGradVec3} &= {gradHin} \times {preMix} \quad ([B,S,N,C])
        \end{aligned}
        $$

      其中$dPre$进入后续sigmoid反传链。

  - **门控激活层梯度计算**：

    - **Sigmoid门控反向（H_pre）**：

      - 正向公式：

        $$
        {hPre} = \text{Sigmoid}({alphaPre} \cdot {hPre1} + {biasPre}) + {hcEps}
        $$

        其中$alphaPre$为输入参数`alpha`的第一个元素（对应H_pre的缩放系数），$biasPre$为输入参数`bias`的前N个元素，$hcEps$为输入参数`hcEps`（数值稳定性参数）。

      - 反向计算：

        $$
        \begin{aligned}
        {s} &= {hPre} - {hcEps} \\
        {HPre2Grad} &= {dPre} \odot {s} \odot (1 - {s}) \\
        {HPre1Grad} &= {HPre2Grad} \cdot {alphaPre} \\
        {alphaPreGrad} &= \sum_{b,s,n}^{B,S,N} \left({HPre2Grad} \cdot {hPre1}\right) \\
        {biasPreGrad} &= \sum_{b,s}^{B,S} {HPre2Grad} \quad ([N])
        \end{aligned}
        $$

    - **Sigmoid门控反向（H_post）**：

      - 正向公式：

        $$
        {hPost} = \text{Sigmoid}({alphaPost} \cdot {hPost1} + {biasPost}) \times 2
        $$

        其中$alphaPost$为输入参数`alpha`的第二个元素，$biasPost$为输入参数`bias`的中间N个元素。

      - 反向计算：

        $$
        \begin{aligned}
        {HPost2Grad} &= {gradHPost} \odot \left({hPost} \cdot \left(1 - \frac{{hPost}}{2}\right)\right) \\
        {HPost1Grad} &= {HPost2Grad} \cdot {alphaPost} \\
        {alphaPostGrad} &= \sum_{b,s,n}^{B,S,N} \left({HPost2Grad} \cdot {hPost1}\right) \\
        {biasPostGrad} &= \sum_{b,s}^{B,S} {HPost2Grad} \quad ([N])
        \end{aligned}
        $$

        其中$gradHPost$为输入参数（输出`h_post`的梯度）。

    - **残差连接反向（H_res）**：

      - 正向公式：

        $$
        {hRes} = {alphaRes} \cdot {hRes1} + {biasRes}
        $$

        其中$alphaRes$为输入参数`alpha`的第三个元素，$biasRes$为输入参数`bias`的后$N^2$个元素。

      - 反向计算：

        $$
        \begin{aligned}
        {HRes2Grad} &= {HResGrad} \cdot {alphaRes} \quad ([B,S,N,N]) \\
        {alphaResGrad} &= \sum_{b,s,i,j}^{B,S,N,N} \left({HResGrad} \cdot {hRes2}\right) \\
        {biasResGrad} &= \sum_{b,s}^{B,S} {HResGrad} \quad ([N,N]) \\
        {HRes1Grad} &= \text{Reshape}({HRes2Grad}) \quad ([B,S,N^2])
        \end{aligned}
        $$

  - **RMS归一化融合反向**：

    - **RMSNorm Fusion反向**：

      - 正向公式：

        $$
        {hMixTmp} = {hMix} \cdot {invRms}
        $$

        其中$hMix$为线性投影层输出，$invRms$为输入参数（前向保存的中间结果inv_rms）。

      - 反向计算：

        $$
        \begin{aligned}
        {hMixTmpGrad} &= \text{Concat}({HPre1Grad}, {HPost1Grad}, {HRes1Grad}) \quad ([B,S,2N+N^2]) \\
        {hMixGrad} &= {hMixTmpGrad} \cdot {invRms} \\
        {invRmsGrad} &= \sum_{\text{last\_dim}} \left({hMixTmpGrad} \cdot {hMix}\right) \quad ([B,S,1])
        \end{aligned}
        $$

  - **线性投影层梯度计算**：

    - **矩阵乘法反向**：

      - 正向公式：

        $$
        {hMix} = {xRs} @ {phi}^T
        $$

        $$
        {xRs} = {x} \cdot {gamma}
        $$

        其中$phi$为输入参数（前向参数phi），$gamma$为RMS归一化的缩放参数（由输入$x$计算得到）。

      - 反向计算：

        $$
        \begin{aligned}
        {xRsGrad} &= {hMixGrad} @ {phi} \quad ([B,S,N \cdot C]) \\
        {X} &= \text{Reshape}({xRs}, [B \cdot S, N \cdot C]) \\
        {G} &= \text{Reshape}({hMixGrad}, [B \cdot S, 2N+N^2]) \\
        {phiGrad} &= {G}^T @ {X} \quad ([2N+N^2, N \cdot C])
        \end{aligned}
        $$

    - **特征缩放反向**：

      - 正向公式：

        $$
        {xRs} = {x} \cdot {gamma}
        $$

      - 反向计算：

        $$
        \begin{aligned}
        {xGradMm} &= {xRsGrad} \cdot {gamma} \\
        {gammaGrad} &= \sum_{b=1}^{B}\sum_{s=1}^{S} ({x} \cdot {xRsGrad}) \quad ([N,C])
        \end{aligned}
        $$

  - **RMS归一化梯度计算**：

    - 正向公式：

      $$
      {invRms} = \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}{x}_i^2 + {eps}}}, \quad其中\ n = N \times C
      $$

      其中$eps$为RMS归一化的数值稳定性参数。

    - 反向计算：

      $$
      \begin{aligned}
      {xRsGradInv} &= - \left(\frac{{invRmsGrad} \cdot {invRms}^3}{N \times C}\right) \cdot {xRs} \\
      {xRsGrad} &= {xGradMm} + {xRsGradInv} \\
      {xGradVec1} &= \text{Reshape}({xRsGrad}, [B,S,N,C]) \\
      {xGrad} &= {xGradVec3} + {xGradVec1}
      \end{aligned}
      $$

  - **符号说明**：

    | 符号 | 含义 |
    |------|------|
    | $\mathbf{{grad}_{\text{curr}}}$ | 当前梯度（迭代中间变量） |
    | ${sumOut}_k$ | 第$k$次迭代的sum结果（对应输入参数`sumOut`） |
    | ${normOut}_k$ | 第$k$次迭代的norm结果（对应输入参数`normOut`） |
    | ${dotProd}_k$ | 第$k$次点积计算结果（中间变量） |
    | $gradHin$ | 输出`h_in`的梯度（输入参数） |
    | $gradHPost$ | 输出`h_post`的梯度（输入参数） |
    | $gradHRes$ | 输出`h_res`的梯度（输入参数） |
    | $x$ | 前向输入x（输入参数） |
    | $hPre$ | 前向保存的中间结果h_pre（输入参数） |
    | $hMix$ | 线性投影层输出（中间变量） |
    | $hMixTmp$ | RMS归一化后的输出（中间变量） |
    | $invRms$ | 前向保存的中间结果inv_rms（输入参数） |
    | $phi$ | 前向参数phi（输入参数） |
    | $alpha$ | 前向参数alpha（输入参数，包含3个元素分别对应H_pre、H_post、H_res的缩放系数） |
    | $bias$ | 前向参数bias（输入参数，包含2N+N^2个元素） |
    | $hcEps$ | 数值稳定性参数（输入参数） |
    | $B, S, N, C$ | 分别为batch size、sequence length、专家数、特征维度；TND（3维）格式下 B×S 合并为一维 T |
    | $\cdot$ | 点积（内积）或逐元素乘法 |
    | $\odot$ | 逐元素乘法 |
    | $\sum_{\dim=d,\text{keepdim}=\text{True}}$ | 在指定维度上求和并保持维度 |
    | $preMix$ | 可选输入，premix加权系数（通常填上一轮输出的hPre） |
    | $gradPre$ | 可选输入，反向链路上pre的梯度 |
    | $gradPreMix$ | 可选输出，preMix的梯度 |
    | $dPre$ | pre分支的梯度（中间变量），v1下为$HPreGrad$，v2下为$gradPre$ |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMhcPreSinkhornBackwardV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMhcPreSinkhornBackwardV2”接口执行计算。

```cpp
aclnnStatus aclnnMhcPreSinkhornBackwardV2GetWorkspaceSize(
    const aclTensor  *gradHin,
    const aclTensor  *gradHPost,
    const aclTensor  *gradHRes,
    const aclTensor  *x,
    const aclTensor  *phi,
    const aclTensor  *alpha,
    const aclTensor  *bias,
    const aclTensor  *hPre,
    const aclTensor  *hcBeforeNorm,
    const aclTensor  *invRms,
    const aclTensor  *sumOut,
    const aclTensor  *normOut,
    const aclTensor  *preMix,
    const aclTensor  *gradPre,
    double            hcEps,
    bool              enableHf32,
    const aclTensor  *gradX,
    const aclTensor  *gradPhi,
    const aclTensor  *gradAlpha,
    const aclTensor  *gradBias,
    const aclTensor  *gradPreMix,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMhcPreSinkhornBackwardV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnMhcPreSinkhornBackwardV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px">
    <colgroup>
    <col style="width: 265px">
    <col style="width: 120px">
    <col style="width: 223px">
    <col style="width: 391px">
    <col style="width: 181px">
    <col style="width: 111px">
    <col style="width: 126px">
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
        <td>gradHin（const aclTensor *）</td>
        <td>输入</td>
        <td>正向输出hin的梯度。</td>
        <td><ul><il>支持空Tensor。</il><il>shape支持(B,S,C)、(T,C)。</il></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gradHPost（const aclTensor *）</td>
        <td>输入</td>
        <td>正向输出hPost的梯度。</td>
        <td><ul><il>支持空Tensor。</il><il>shape支持(B,S,N)、(T,N)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gradHRes（const aclTensor *）</td>
        <td>输入</td>
        <td>正向输出hRes的梯度。</td>
        <td><ul><il>支持空Tensor。</il><il>shape支持(B,S,N*N)、(B,S,N,N)、(T,N*N)、(T,N,N)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>2-4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>x（const aclTensor *）</td>
        <td>输入</td>
        <td>前向输入x。</td>
        <td><ul><il>支持空Tensor。</il><il>shape支持(B,S,N,C)、(T,N,C)。</il></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>phi（const aclTensor *）</td>
        <td>输入</td>
        <td>前向参数phi。</td>
        <td><ul><il>不支持空Tensor。</il><il>shape支持(2N+N^2, N*C)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>alpha（const aclTensor *）</td>
        <td>输入</td>
        <td>前向参数alpha。</td>
        <td><ul><il>不支持空Tensor。</il><il>shape支持(3)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>bias（const aclTensor *）</td>
        <td>输入</td>
        <td>前向参数bias。</td>
        <td><ul><il>不支持空Tensor。</il><il>shape支持(2N+N^2)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>hPre（const aclTensor *）</td>
        <td>输入</td>
        <td>前向保存的中间结果hPre。</td>
        <td><ul><il>支持空Tensor。</il><il>shape支持(B,S,N)、(T,N)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>hcBeforeNorm（const aclTensor *）</td>
        <td>输入</td>
        <td>前向保存的中间结果hcBeforeNorm。</td>
        <td><ul><il>支持空Tensor。</il><il>shape支持(B,S,N*N+2*N)、(T,N*N+2*N)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>invRms（const aclTensor *）</td>
        <td>输入</td>
        <td>前向保存的中间结果invRms。</td>
        <td><ul><il>支持空Tensor。</il><il>shape支持(B,S,1)、(T,1)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sumOut（const aclTensor *）</td>
        <td>输入</td>
        <td>Sinkhorn变换正向计算保存的中间sum结果。</td>
        <td><ul><il>支持空Tensor。</il><il>shape支持(2*sk_iter_count,B,S,N)、(2*sk_iter_count,T,N)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>normOut（const aclTensor *）</td>
        <td>输入</td>
        <td>Sinkhorn变换正向计算保存的中间norm结果。</td>
        <td><ul><il>支持空Tensor。</il><il>shape支持(2*sk_iter_count,B,S,N,N)、(2*sk_iter_count,T,N,N)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>4-5</td>
        <td>√</td>
    </tr>
    <tr>
        <td>preMix（const aclTensor *）</td>
        <td>输入</td>
        <td>可选输入，premix加权系数（通常填上一轮输出的hPre）。传入时反向计算合并premix梯度分支。</td>
        <td><ul><il>可为nullptr（不传），与gradPre必须同时传入或同时缺省。</il><il>shape支持(B,S,N)、(T,N)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gradPre（const aclTensor *）</td>
        <td>输入</td>
        <td>可选输入，反向链路上pre的梯度gradPre。</td>
        <td><ul><il>可为nullptr（不传），与preMix必须同时传入或同时缺省。</il><il>shape支持(B,S,N)、(T,N)。</il></ul></td>
        <td>FLOAT32 </td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>hcEps（double）</td>
        <td>输入</td>
        <td>数值稳定性参数。</td>
        <td>建议值1e-6。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>enableHf32（bool）</td>
        <td>输入</td>
        <td>可选属性（仅V2接口），表示Cube是否启用HF32计算模式。</td>
        <td>可为true（默认，与V1接口一致）。传入false切换为FP32。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gradX（aclTensor *）</td>
        <td>输出</td>
        <td>输入x的梯度。</td>
        <td><ul><il>不支持空Tensor。</il><il>shape支持(B,S,N,C)、(T,N,C)。</il></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gradPhi（aclTensor *）</td>
        <td>输出</td>
        <td>参数phi的梯度。</td>
        <td><ul><il>不支持空Tensor。</il><il>shape支持(2N+N^2, N*C)。</il></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gradAlpha（aclTensor *）</td>
        <td>输出</td>
        <td>参数alpha的梯度。</td>
        <td><ul><il>不支持空Tensor。</il><il>shape支持(3)。</il></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gradBias（aclTensor *）</td>
        <td>输出</td>
        <td>参数bias的梯度。</td>
        <td><ul><il>不支持空Tensor。</il><il>shape支持(2N+N^2)。</il></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gradPreMix（aclTensor *）</td>
        <td>输出</td>
        <td>可选输出，preMix的梯度，为∑<sub>c</sub>(gradHin · x)。</td>
        <td><ul><il>不支持空Tensor。</il><il>仅preMix与gradPre同时传入时为真实值，否则无效。</il><il>shape支持(B,S,N)、(T,N)。</il></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t *）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor **）</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1180px"><colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
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
                <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="5">161002</td>
                <td>输入或输出tensor的维度个数(dim)与参数说明不符。</td>
            </tr>
            <tr>
                <td>输入或输出tensor的维度(shape)与参数说明不符。</td>
            </tr>
            <tr>
                <td>输入变量的数据类型不在支持的范围内。</td>
            </tr>
            <tr>
                <td>N取值不符合约束要求。</td>
            </tr>
            <tr>
                <td>C不符合大于0小于100000且可以被128整除的要求。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_INNER_NULLPTR</td>
                <td>561103</td>
                <td>sk_iter_count不等于20。</td>
            </tr>
        </tbody>
    </table>

## aclnnMhcPreSinkhornBackwardV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1180px"><colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPreSinkhornBackwardV2GetWorkspaceSize获取。</td>
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

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 维度格式约束：

  输入支持两种维度格式：BSND（4维）和TND（3维）。其中T = B × S，表示所有Batch序列长度的累加和。所有带B/S维度的输入需保持维度格式一致（同为4维BSND或同为3维TND）。

- 规格约束：

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：N维度支持大于0且不超过8。
  <!-- end id9 -->

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：N维度仅支持4。
  <!-- end id10 -->

  - C：大于0、小于100000且可以被128整除。
  - sk_iter_count：Sinkhorn迭代次数当前仅支持20。

- 确定性计算：

  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：aclnnMhcPreSinkhornBackwardV2默认确定性实现。
  <!-- end id11 -->

  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：aclnnMhcPreSinkhornBackwardV2默认非确定性实现，支持通过`aclrtSetSysParamOpt`开启确定性。
  <!-- end id12 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_sinkhorn_backward.h"
// V2 接口声明与 V1 同头文件

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

void PrintTensorDataFloat(const std::vector<int64_t>& shape, void* device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_data.data(), size * sizeof(float), device_addr, size * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

void PrintTensorDataBfloat16(const std::vector<int64_t>& shape, void* device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_bf16(size);

    aclError ret = aclrtMemcpy(host_bf16.data(), size * sizeof(uint16_t), device_addr, size * sizeof(uint16_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements, BF16 hex): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("0x%04x ", host_bf16[i]);
    }
    LOG_PRINT("\n");
}

int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed, error: %d\n", ret); return -1);

    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); return -1);

    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); return -1);

    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); return -1);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); return -1);

    return 0;
}

int CreateAclTensorBfloat16(const std::vector<float>& host_data, const std::vector<int64_t>& shape, void*& device_addr,
                            aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_data_bf16(size);
    for (int64_t i = 0; i < size; ++i) {
        host_data_bf16[i] = static_cast<uint16_t>(static_cast<int16_t>(host_data[i] * 32767));
    }

    int64_t byte_size = size * sizeof(uint16_t);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, byte_size, host_data_bf16.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorFloat32(const std::vector<float>& host_data, const std::vector<int64_t>& shape, void*& device_addr,
                           aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorBfloat16Output(const std::vector<int64_t>& shape, void*& device_addr, aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(uint16_t);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorFloat32Output(const std::vector<int64_t>& shape, void*& device_addr, aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

struct Tensors {
    void *grad_hin_addr = nullptr, *grad_h_post_addr = nullptr, *grad_h_res_addr = nullptr;
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr;
    void *h_pre_addr = nullptr, *hc_before_norm_addr = nullptr, *inv_rms_addr = nullptr;
    void *sum_out_addr = nullptr, *norm_out_addr = nullptr;
    void *pre_mix_addr = nullptr, *grad_pre_addr = nullptr;
    void *grad_x_addr = nullptr, *grad_phi_addr = nullptr, *grad_alpha_addr = nullptr, *grad_bias_addr = nullptr;
    void* grad_pre_mix_addr = nullptr;
    aclTensor *grad_hin = nullptr, *grad_h_post = nullptr, *grad_h_res = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr;
    aclTensor *h_pre = nullptr, *hc_before_norm = nullptr, *inv_rms = nullptr;
    aclTensor *sum_out = nullptr, *norm_out = nullptr;
    aclTensor *pre_mix = nullptr, *grad_pre = nullptr;
    aclTensor *grad_x = nullptr, *grad_phi = nullptr, *grad_alpha = nullptr, *grad_bias = nullptr;
    aclTensor* grad_pre_mix = nullptr;
};

int CreateInputTensors(const std::vector<int64_t>& grad_hin_shape, const std::vector<int64_t>& grad_h_post_shape,
                       const std::vector<int64_t>& grad_h_res_shape, const std::vector<int64_t>& x_shape,
                       const std::vector<int64_t>& phi_shape, const std::vector<int64_t>& alpha_shape,
                       const std::vector<int64_t>& bias_shape, const std::vector<int64_t>& h_pre_shape,
                       const std::vector<int64_t>& hc_before_norm_shape, const std::vector<int64_t>& inv_rms_shape,
                       const std::vector<int64_t>& sum_out_shape, const std::vector<int64_t>& norm_out_shape,
                       const std::vector<int64_t>& pre_mix_shape, const std::vector<int64_t>& grad_pre_shape,
                       Tensors& tensors)
{
    std::vector<float> grad_hin_host_data(GetShapeSize(grad_hin_shape), 1.0f);
    std::vector<float> grad_h_post_host_data(GetShapeSize(grad_h_post_shape), 1.0f);
    std::vector<float> grad_h_res_host_data(GetShapeSize(grad_h_res_shape), 1.0f);
    std::vector<float> x_host_data(GetShapeSize(x_shape), 1.0f);
    std::vector<float> phi_host_data(GetShapeSize(phi_shape), 1.0f);
    std::vector<float> alpha_host_data(GetShapeSize(alpha_shape), 1.0f);
    std::vector<float> bias_host_data(GetShapeSize(bias_shape), 1.0f);
    std::vector<float> h_pre_host_data(GetShapeSize(h_pre_shape), 1.0f);
    std::vector<float> hc_before_norm_host_data(GetShapeSize(hc_before_norm_shape), 1.0f);
    std::vector<float> inv_rms_host_data(GetShapeSize(inv_rms_shape), 1.0f);
    std::vector<float> sum_out_host_data(GetShapeSize(sum_out_shape), 1.0f);
    std::vector<float> norm_out_host_data(GetShapeSize(norm_out_shape), 1.0f);
    std::vector<float> pre_mix_host_data(GetShapeSize(pre_mix_shape), 1.0f);
    std::vector<float> grad_pre_host_data(GetShapeSize(grad_pre_shape), 0.0f);

    int ret = CreateAclTensorBfloat16(grad_hin_host_data, grad_hin_shape, tensors.grad_hin_addr, tensors.grad_hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_hin_tensor failed\n"); return -1);
    ret =
        CreateAclTensorFloat32(grad_h_post_host_data, grad_h_post_shape, tensors.grad_h_post_addr, tensors.grad_h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(grad_h_res_host_data, grad_h_res_shape, tensors.grad_h_res_addr, tensors.grad_h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorBfloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(h_pre_host_data, h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(hc_before_norm_host_data, hc_before_norm_shape, tensors.hc_before_norm_addr,
                                 tensors.hc_before_norm);
    CHECK_RET(ret == 0, LOG_PRINT("Create hc_before_norm_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(inv_rms_host_data, inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(sum_out_host_data, sum_out_shape, tensors.sum_out_addr, tensors.sum_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_out_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(norm_out_host_data, norm_out_shape, tensors.norm_out_addr, tensors.norm_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_out_tensor failed\n"); return -1);
    // V2 可选输入: pre_mix 与 grad_pre 必须成对(外部链路对 hPre 的梯度, 未消费时置零)
    ret = CreateAclTensorFloat32(pre_mix_host_data, pre_mix_shape, tensors.pre_mix_addr, tensors.pre_mix);
    CHECK_RET(ret == 0, LOG_PRINT("Create pre_mix_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(grad_pre_host_data, grad_pre_shape, tensors.grad_pre_addr, tensors.grad_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_pre_tensor failed\n"); return -1);

    return 0;
}

int CreateOutputTensors(const std::vector<int64_t>& grad_x_shape, const std::vector<int64_t>& grad_phi_shape,
                        const std::vector<int64_t>& grad_alpha_shape, const std::vector<int64_t>& grad_bias_shape,
                        const std::vector<int64_t>& grad_pre_mix_shape, Tensors& tensors)
{
    int ret = CreateAclTensorBfloat16Output(grad_x_shape, tensors.grad_x_addr, tensors.grad_x);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_phi_shape, tensors.grad_phi_addr, tensors.grad_phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_alpha_shape, tensors.grad_alpha_addr, tensors.grad_alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_bias_shape, tensors.grad_bias_addr, tensors.grad_bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_bias_tensor failed\n"); return -1);
    // V2 第 5 输出: grad_pre_mix(premix 的梯度), 与 pre_mix 同 shape; pre_mix 未传时为全零占位
    ret = CreateAclTensorFloat32Output(grad_pre_mix_shape, tensors.grad_pre_mix_addr, tensors.grad_pre_mix);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_pre_mix_tensor failed\n"); return -1);

    return 0;
}

void DestroyTensors(Tensors& tensors)
{
    aclDestroyTensor(tensors.grad_hin);
    aclDestroyTensor(tensors.grad_h_post);
    aclDestroyTensor(tensors.grad_h_res);
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.h_pre);
    aclDestroyTensor(tensors.hc_before_norm);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.sum_out);
    aclDestroyTensor(tensors.norm_out);
    aclDestroyTensor(tensors.pre_mix);
    aclDestroyTensor(tensors.grad_pre);
    aclDestroyTensor(tensors.grad_x);
    aclDestroyTensor(tensors.grad_phi);
    aclDestroyTensor(tensors.grad_alpha);
    aclDestroyTensor(tensors.grad_bias);
    aclDestroyTensor(tensors.grad_pre_mix);
}

void FreeDeviceMemory(Tensors& tensors)
{
    aclrtFree(tensors.grad_hin_addr);
    aclrtFree(tensors.grad_h_post_addr);
    aclrtFree(tensors.grad_h_res_addr);
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.h_pre_addr);
    aclrtFree(tensors.hc_before_norm_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.sum_out_addr);
    aclrtFree(tensors.norm_out_addr);
    aclrtFree(tensors.grad_x_addr);
    aclrtFree(tensors.grad_phi_addr);
    aclrtFree(tensors.grad_alpha_addr);
    aclrtFree(tensors.grad_bias_addr);
    aclrtFree(tensors.pre_mix_addr);
    aclrtFree(tensors.grad_pre_addr);
    aclrtFree(tensors.grad_pre_mix_addr);
}

int main()
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    Tensors tensors;

    int64_t bs = 2, seq_len = 128, n = 4, c = 256;
    int64_t hc_mult = n;
    int64_t hc_mix = n * n + 2 * n;
    int64_t num_iters = 20;
    double hc_eps = 1e-6;

    std::vector<int64_t> grad_hin_shape = {bs, seq_len, c};
    std::vector<int64_t> grad_h_post_shape = {bs, seq_len, n};
    std::vector<int64_t> grad_h_res_shape = {bs, seq_len, n, n};
    std::vector<int64_t> x_shape = {bs, seq_len, n, c};
    std::vector<int64_t> phi_shape = {hc_mix, n * c};
    std::vector<int64_t> alpha_shape = {3};
    std::vector<int64_t> bias_shape = {hc_mix};
    std::vector<int64_t> h_pre_shape = {bs, seq_len, n};
    std::vector<int64_t> hc_before_norm_shape = {bs, seq_len, hc_mix};
    std::vector<int64_t> inv_rms_shape = {bs, seq_len, 1};
    std::vector<int64_t> sum_out_shape = {num_iters * 2, bs, seq_len, n};
    std::vector<int64_t> norm_out_shape = {num_iters * 2, bs, seq_len, n, n};

    std::vector<int64_t> grad_x_shape = {bs, seq_len, n, c};
    std::vector<int64_t> grad_phi_shape = {hc_mix, n * c};
    std::vector<int64_t> grad_alpha_shape = {3};
    std::vector<int64_t> grad_bias_shape = {hc_mix};
    std::vector<int64_t> pre_mix_shape = {bs, seq_len, n};
    std::vector<int64_t> grad_pre_shape = {bs, seq_len, n};
    std::vector<int64_t> grad_pre_mix_shape = {bs, seq_len, n};
    bool enable_hf32 = true;

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);

    ret = CreateInputTensors(grad_hin_shape, grad_h_post_shape, grad_h_res_shape, x_shape, phi_shape, alpha_shape,
                             bias_shape, h_pre_shape, hc_before_norm_shape, inv_rms_shape, sum_out_shape,
                             norm_out_shape, pre_mix_shape, grad_pre_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    ret = CreateOutputTensors(grad_x_shape, grad_phi_shape, grad_alpha_shape, grad_bias_shape, grad_pre_mix_shape,
                              tensors);
    CHECK_RET(ret == 0, return -1);

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // V2 接口: 14 张量(含可选 pre_mix/grad_pre, 须成对) + hc_eps + enable_hf32, 5 输出。
    // 第 5 输出 grad_pre_mix 为 premix 的梯度(hin 链式项 sum(grad_hin*x) 由 kernel 内部计入);
    // pre_mix/grad_pre 传 nullptr 时等价 V1 行为, 本例展示 premix 场景用法。
    aclnnStatus aclnn_ret = aclnnMhcPreSinkhornBackwardV2GetWorkspaceSize(
        tensors.grad_hin, tensors.grad_h_post, tensors.grad_h_res, tensors.x, tensors.phi, tensors.alpha, tensors.bias,
        tensors.h_pre, tensors.hc_before_norm, tensors.inv_rms, tensors.sum_out, tensors.norm_out, tensors.pre_mix,
        tensors.grad_pre, hc_eps, enable_hf32, tensors.grad_x, tensors.grad_phi, tensors.grad_alpha, tensors.grad_bias,
        tensors.grad_pre_mix, &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcPreSinkhornBackwardV2GetWorkspaceSize failed, error: %d\n", aclnn_ret);
              return -1);

    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); return -1);
    }

    aclnn_ret = aclnnMhcPreSinkhornBackwardV2(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSinkhornBackwardV2 failed, error: %d\n", aclnn_ret);
              return -1);

    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed\n"); return -1);

    LOG_PRINT("MhcPreSinkhornBackwardV2 compute success!\n");
    LOG_PRINT("Output tensor grad_x (first 10 elements):\n");
    PrintTensorDataBfloat16(grad_x_shape, tensors.grad_x_addr);

    DestroyTensors(tensors);
    FreeDeviceMemory(tensors);
    if (workspace_size > 0)
        aclrtFree(workspace_addr);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();
    LOG_PRINT("All resources released successfully!\n");
    return 0;
}
```
