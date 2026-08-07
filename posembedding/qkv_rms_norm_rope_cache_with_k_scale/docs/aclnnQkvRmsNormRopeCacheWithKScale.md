# aclnnQkvRmsNormRopeCacheWithKScale

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- 接口功能：`aclnnQkvRmsNormRopeCacheWithKScale`在一次调用中完成融合QKV拆分、Q/K RMSNorm、位置编码、共享矩阵乘、Q/K量化以及PagedAttention KV Cache更新。Q分支根据接口约束写入`qOut`和`qScaleOptional`；K、V分支根据`slotMapping`原地更新`kCacheRef`、`vCacheRef`和`kScaleCacheRef`。V分支不执行RMSNorm、位置编码或共享矩阵乘。

- 计算公式：

  以下公式统一按逻辑`[T,N,D]`布局描述，`layoutQkv`和`layoutQOut`只决定Tensor中T轴与N轴的物理顺序，不改变逐元素计算。$T$表示token总数，$N_q/N_k/N_v$表示Q/K/V head数，$D=128$表示head维度。

  计算过程按QKV拆分、归一化、位置编码、共享矩阵乘、输出量化和Cache写回六步展开。

  **1. Q/K/V拆分与Q/K RMSNorm**

  将`qkv`的逻辑视图记为$\boldsymbol X$，则

  $$
  \begin{aligned}
  \boldsymbol X
  &=[\,\boldsymbol Q\mid\boldsymbol K\mid\boldsymbol V\,]_{N},\\
  \boldsymbol Q&\in\mathbb R^{T\times N_q\times D},\qquad
  \boldsymbol K\in\mathbb R^{T\times N_k\times D},\qquad
  \boldsymbol V\in\mathbb R^{T\times N_v\times D}.
  \end{aligned}
  $$

  沿逻辑head轴的切分边界为

  $$
  \boldsymbol X_{u,n,:}=
  \begin{cases}
  \boldsymbol Q_{u,n,:}, & 0\leq n<N_q,\\
  \boldsymbol K_{u,n-N_q,:}, & N_q\leq n<N_q+N_k,\\
  \boldsymbol V_{u,n-N_q-N_k,:}, & N_q+N_k\leq n<N_q+N_k+N_v.
  \end{cases}
  $$

  对$\boldsymbol A\in\{\boldsymbol Q,\boldsymbol K\}$，令$\boldsymbol\gamma^{Q}$和$\boldsymbol\gamma^{K}$分别对应`qGamma`和`kGamma`，RMSNorm结果为

  $$
  Y^{A}_{u,n,d}
  =
  \frac{A_{u,n,d}\gamma^{A}_{d}}
  {\sqrt{\epsilon+\dfrac{1}{D}\displaystyle\sum_{j=0}^{D-1}A_{u,n,j}^{2}}},
  \qquad 0\leq d<D.
  $$

  **2. 位置参数解析与cos/sin选取**

  **RoPE位置输入**

  对batch $b$，当前调用包含的token数为$L_b$。对该batch内第$i$个token，其全局token下标$u$和RoPE位置$p_u$为

  $$
  \begin{aligned}
  L_b
  &=\mathrm{queryStartLocOptional}_{b+1}
    -\mathrm{queryStartLocOptional}_{b},\\
  u
  &=\mathrm{queryStartLocOptional}_{b}+i,\qquad 0\leq i<L_b,\\
  p_u
  &=\mathrm{seqLensOptional}_{b}-L_b+i.
  \end{aligned}
  $$

  令$D_{\mathrm{half}}=D/2$，则token $u$使用的cos和sin为

  $$
  c_{u,\ell}
  =\mathrm{cosSin}_{p_u,\ell},
  \qquad
  s_{u,\ell}
  =\mathrm{cosSin}_{p_u,D_{\mathrm{half}}+\ell},
  \qquad 0\leq\ell<D_{\mathrm{half}}.
  $$

  **M-RoPE位置输入**

  `mropePositionOptional`的每一行对应一个token，三列依次给出T/H/W三路`cosSin`行下标。定义列映射

  $$
  \iota(\mathrm T)=0,\qquad
  \iota(\mathrm H)=1,\qquad
  \iota(\mathrm W)=2,
  $$

  以及三路原始位置编码

  $$
  R_{a,u,d}
  =
  \mathrm{cosSin}_{
    \mathrm{mropePositionOptional}_{u,\iota(a)},d},
  \qquad
  a\in\{\mathrm T,\mathrm H,\mathrm W\}.
  $$

  将`mropeSectionOptional=[t,h,w]`记为

  $$
  \boldsymbol{s}
  =(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})
  =(t,h,w).
  $$

  $s_{\mathrm T}$不参与lane选源，也不做独立lane容量上限校验；它仍须非负并参与三项总和校验。T路是half-lane的默认来源，H/W在各自覆盖范围内替换对应lane。对$0\leq\ell<D_{\mathrm{half}}$，定义

  $$
  r(\ell)=\left\lfloor\frac{\ell}{3}\right\rfloor,\qquad
  \rho(\ell)=\ell\bmod 3,
  $$

  $$
  \sigma(\ell)=
  \begin{cases}
  \mathrm H, & \rho(\ell)=1\ \land\ r(\ell)<s_{\mathrm H},\\
  \mathrm W, & \rho(\ell)=2\ \land\ r(\ell)<s_{\mathrm W},\\
  \mathrm T, & \text{其他情况}.
  \end{cases}
  $$

  M-RoPE最终使用的cos和sin为

  $$
  c_{u,\ell}=R_{\sigma(\ell),u,\ell},
  \qquad
  s_{u,\ell}=R_{\sigma(\ell),u,D_{\mathrm{half}}+\ell}.
  $$

  M-RoPE只改变cos/sin的取值方式，Q/K输出的head维度仍为$D$，不会扩展为$3D$。

  **3. Q/K位置编码**

  对$\boldsymbol A\in\{\boldsymbol Q,\boldsymbol K\}$，将RMSNorm结果$\boldsymbol Y^A$按head维前后两半拆分，half-split位置编码为

  $$
  \begin{aligned}
  Z^{A,\mathrm{low}}_{u,n,\ell}
  &=Y^{A,\mathrm{low}}_{u,n,\ell}c_{u,\ell}
    -Y^{A,\mathrm{high}}_{u,n,\ell}s_{u,\ell},\\
  Z^{A,\mathrm{high}}_{u,n,\ell}
  &=Y^{A,\mathrm{high}}_{u,n,\ell}c_{u,\ell}
    +Y^{A,\mathrm{low}}_{u,n,\ell}s_{u,\ell}.
  \end{aligned}
  $$

  **4. Q/K共享矩阵乘**

  令$\boldsymbol W$表示`rotationOptional`，位置编码结果转换为BF16后右乘该共享矩阵，得到FP32结果$\boldsymbol H^Q$和$\boldsymbol H^K$：

  $$
  \boldsymbol H^A_{u,n,:}
  =
  \operatorname{BF16}\!\left(\boldsymbol Z^A_{u,n,:}\right)
  \boldsymbol W,
  \qquad A\in\{Q,K\}.
  $$

  **5. Q/K输出与动态量化**

  对经过共享矩阵乘得到的$\boldsymbol H^A$，输出类型和是否执行动态量化由“约束说明”确定。未执行动态量化时，直接转换到约束规定的输出类型$\tau_A$：

  $$
  O^A_{u,n,d}=\operatorname{cast}_{\tau_A}\!\left(H^A_{u,n,d}\right),
  \qquad A\in\{Q,K\}.
  $$

  对执行动态量化的分支，令$M_{\tau_A}$表示目标类型对应的正向量化上限。统一的逐token、逐head量化公式为

  $$
  M_{\tau_A}=
  \begin{cases}
  448, & \tau_A=\mathrm{FP8\ E4M3FN},\\
  127, & \tau_A=\mathrm{INT8},
  \end{cases}
  $$

  $$
  \begin{aligned}
  m^A_{u,n}
  &=\max_{0\leq d<D}\left|H^A_{u,n,d}\right|,\\
  \alpha^A_{u,n}
  &=\frac{m^A_{u,n}}{M_{\tau_A}},\\
  \widehat H^A_{u,n,d}
  &=\operatorname{cast}_{\tau_A}
    \!\left(\frac{H^A_{u,n,d}}{\alpha^A_{u,n}}\right).
  \end{aligned}
  $$

  其中`cast`表示按目标类型完成转换；当目标类型为INT8时，转换包含舍入和$[-127,127]$范围饱和。启用动态量化时，Q分支的$\boldsymbol\alpha^Q$写入`qScaleOptional`、$\widehat{\boldsymbol H}^Q$写入`qOut`，K分支的$\boldsymbol\alpha^K$写入`kScaleCacheRef`、$\widehat{\boldsymbol H}^K$写入`kCacheRef`。未启用动态量化的分支只产生直接转换结果，不产生scale。

  **6. V缩放、量化与Cache写回**

  将`vScaleOptional`按其输入shape广播到V的逻辑shape，记为$\boldsymbol\beta$。V分支统一在FP8 E4M3FN转换前逐元素乘以该缩放因子：

  $$
  \widehat V_{u,n,d}
  =
  \operatorname{cast}_{\mathrm{FP8\ E4M3FN}}
  \!\left(V_{u,n,d}\beta_{n,d}\right).
  $$

  对token $u$，令$z_u=\mathrm{slotMapping}_u$，$B_{\mathrm{cache}}$表示Cache的`BlockSize`，则写回block和block内偏移为

  $$
  b_u=\left\lfloor\frac{z_u}{B_{\mathrm{cache}}}\right\rfloor,
  \qquad
  o_u=z_u\bmod B_{\mathrm{cache}}.
  $$

  仅更新以下位置：

  $$
  \begin{aligned}
  \mathrm{kCacheRef}'_{b_u,n,o_u,d}
  &=\widehat H^K_{u,n,d},\\
  \mathrm{vCacheRef}'_{b_u,n,o_u,d}
  &=\widehat V_{u,n,d},\\
  \mathrm{kScaleCacheRef}'_{b_u,n,o_u,0}
  &=\alpha^K_{u,n}.
  \end{aligned}
  $$

  其中带撇号的Tensor表示接口执行后的原地更新结果；未被`slotMapping`选中的Cache位置保持不变。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnQkvRmsNormRopeCacheWithKScale”接口执行计算。

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor   *qkv,
    const aclTensor   *qGamma,
    const aclTensor   *kGamma,
    const aclTensor   *cosSin,
    const aclTensor   *slotMapping,
    aclTensor         *kCacheRef,
    aclTensor         *vCacheRef,
    aclTensor         *kScaleCacheRef,
    const aclTensor   *queryStartLocOptional,
    const aclTensor   *seqLensOptional,
    const aclTensor   *rotationOptional,
    const aclTensor   *vScaleOptional,
    const aclTensor   *mropePositionOptional,
    const aclIntArray *headNums,
    const char        *layoutQkv,
    const char        *layoutQOut,
    float              epsilon,
    const aclIntArray *mropeSectionOptional,
    const char        *qQuantMode,
    aclTensor         *qOut,
    aclTensor         *qScaleOptional,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1312px"><colgroup>
  <col style="width: 158px">
  <col style="width: 120px">
  <col style="width: 333px">
  <col style="width: 137px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 107px">
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
      <th>维度（shape）</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">qkv（const aclTensor*）</td>
      <td>输入</td>
      <td>Q/K/V融合输入，对应公式中的<code>qkv</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td><code>[T,Nq+Nk+Nv,D]</code>或<code>[Nq+Nk+Nv,T,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qGamma（const aclTensor*）</td>
      <td>输入</td>
      <td>Q分支RMSNorm权重，对应Q分支公式中的<code>gamma</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kGamma（const aclTensor*）</td>
      <td>输入</td>
      <td>K分支RMSNorm权重，对应K分支公式中的<code>gamma</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cosSin（const aclTensor*）</td>
      <td>输入</td>
      <td>RoPE/M-RoPE位置编码表，对应公式中的<code>cosSin</code>、<code>cos</code>和<code>sin</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>前<code>D/2</code>列为cos，后<code>D/2</code>列为sin。</li><li>第一维需覆盖本次调用会访问的RoPE位置。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[MaxSeqLen,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">slotMapping（const aclTensor*）</td>
      <td>输入</td>
      <td>每个token写入cache的slot索引，对应公式中的<code>slotMapping</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>取值范围应为<code>[0,BlockNum*BlockSize-1]</code>。</li><li>同一次调用内多个token写入同一slot时，最终写入顺序和结果未定义。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[T]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kCacheRef（aclTensor*）</td>
      <td>输出</td>
      <td>KCache写回Tensor，对应公式中的<code>kCacheRef</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>接口基于传入Tensor原地更新。</li><li>数据类型及其与K量化结果的对应关系见“约束说明”。</li><li>支持非连续Tensor，需满足“约束说明”中的stride限制。</li></ul></td>
      <td>FLOAT8_E4M3FN、INT8</td>
      <td>ND</td>
      <td><code>[BlockNum,Nk,BlockSize,D]</code></td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vCacheRef（aclTensor*）</td>
      <td>输出</td>
      <td>VCache写回Tensor，对应公式中的<code>vCacheRef</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>接口基于传入Tensor原地更新。</li><li>支持非连续Tensor，需满足“约束说明”中的stride限制。</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td><code>[BlockNum,Nv,BlockSize,D]</code></td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kScaleCacheRef（aclTensor*）</td>
      <td>输出</td>
      <td>K动态量化scale cache写回Tensor，对应公式中的<code>kScaleCacheRef</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>接口基于传入Tensor原地更新。</li><li>支持非连续Tensor，需满足“约束说明”中的stride限制。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[BlockNum,Nk,BlockSize,1]</code></td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">queryStartLocOptional（const aclTensor*）</td>
      <td>可选输入</td>
      <td>当前调用内各batch token数的前缀和，对应公式中的<code>queryStartLoc</code>。</td>
      <td><ul><li>传入时不支持空指针或空Tensor；是否传入及其与其他位置输入的组合见“约束说明”。</li><li>长度需大于等于2。</li><li><code>queryStartLocOptional[0]</code>应为0，<code>queryStartLocOptional[Batch]</code>应为<code>T</code>。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[Batch+1]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">seqLensOptional（const aclTensor*）</td>
      <td>可选输入</td>
      <td>每个batch追加本次token后的实际序列长度，对应公式中的<code>seqLens</code>。</td>
      <td><ul><li>传入时不支持空指针或空Tensor；是否传入及其与其他位置输入的组合见“约束说明”。</li><li>长度需等于<code>queryStartLocOptional.shape[0]-1</code>。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[Batch]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">rotationOptional（const aclTensor*）</td>
      <td>可选输入</td>
      <td>Q/K共享矩阵乘权重，对应公式中的<code>rotationOptional</code>。</td>
      <td><ul><li>接口合同要求非空Tensor。</li><li>用于Q/K共享矩阵乘。</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td><code>[D,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vScaleOptional（const aclTensor*）</td>
      <td>可选输入</td>
      <td>V分支量化缩放因子，对应公式中的<code>vScaleOptional</code>。</td>
      <td><ul><li>接口合同要求非空Tensor。</li><li>用于V分支FP8量化前缩放。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>RoPE：<code>[Nv]</code><br>M-RoPE：<code>[Nv,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">mropePositionOptional（const aclTensor*）</td>
      <td>可选输入</td>
      <td>M-RoPE场景的T/H/W三路位置索引。每个token占一行，三列依次为T、H、W。</td>
      <td><ul><li>传入时不支持空指针或空Tensor；是否传入及其与其他位置输入的组合见“约束说明”。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[T,3]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">headNums（const aclIntArray*）</td>
      <td>输入</td>
      <td>Q/K/V头数数组，依次映射为公式中的<code>Nq</code>、<code>Nk</code>、<code>Nv</code>。</td>
      <td><ul><li>不支持空指针。</li><li>必须包含3个正整数。</li><li>RoPE和M-RoPE均要求<code>Nq&lt;=64</code>、<code>Nq=8*Nk</code>、<code>Nk=Nv</code>。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>长度为3</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layoutQkv（const char*）</td>
      <td>输入</td>
      <td><code>qkv</code>的N/T轴布局标识，对应公式中<code>T</code>、<code>Nq</code>、<code>Nk</code>、<code>Nv</code>所在轴。</td>
      <td><ul><li>默认值为<code>"TND"</code>。</li><li>传入空指针或空字符串时按默认值处理。</li><li>大小写敏感，仅支持<code>"TND"</code>和<code>"NTD"</code>。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layoutQOut（const char*）</td>
      <td>输入</td>
      <td><code>qOut</code>和<code>qScaleOptional</code>的N/T轴布局标识。</td>
      <td><ul><li>默认值为<code>"NTD"</code>。</li><li>传入空指针或空字符串时按默认值处理。</li><li>大小写敏感，仅支持<code>"TND"</code>和<code>"NTD"</code>。</li><li>当前不支持<code>layoutQkv="NTD"</code>、<code>layoutQOut="TND"</code>。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">epsilon（float）</td>
      <td>输入</td>
      <td>RMSNorm防除零参数，对应公式中的<code>epsilon</code>。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">mropeSectionOptional（const aclIntArray*）</td>
      <td>可选输入</td>
      <td>M-RoPE场景的T/H/W lane容量参数，语义顺序为<code>(T,H,W)</code>，按<code>[t,h,w]</code>传入。</td>
      <td><ul><li>是否传入及其与<code>mropePositionOptional</code>的组合见“约束说明”。</li><li>空数组与未传入等价；非空时长度必须为3，值域和总和约束见“约束说明”。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>长度为3（<code>[t,h,w]</code>）</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qQuantMode（const char*）</td>
      <td>输入</td>
      <td>Q分支量化模式。</td>
      <td><ul><li>传入空指针或空字符串时按默认值<code>"PerTokenPerHead"</code>处理。</li><li>支持值及其与输出类型、位置输入的组合见“约束说明”。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qOut（aclTensor*）</td>
      <td>输出</td>
      <td>Q分支输出，对应计算公式中的直接转换结果或动态量化结果。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>接口直接读取<code>qOut</code>的数据类型，不单独传入dtype属性。</li><li>输出类型及其与量化模式、位置输入的组合见“约束说明”。</li></ul></td>
      <td>FLOAT8_E4M3FN、BFLOAT16</td>
      <td>ND</td>
      <td><code>[T,Nq,D]</code>或<code>[Nq,T,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qScaleOptional（aclTensor*）</td>
      <td>可选输出</td>
      <td>Q分支每个token/head对应的动态量化scale，对应Q分支动态量化公式中的<code>scale</code>。</td>
      <td><ul><li>是否需要传入及其与Q输出的组合见“约束说明”。</li><li>传入时不支持空Tensor。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[T,Nq]</code>或<code>[Nq,T]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>不支持空指针。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>不支持空指针。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td style="white-space: nowrap">ACLNN_ERR_PARAM_NULLPTR</td>
      <td style="white-space: nowrap">161001</td>
      <td><code>qkv</code>、<code>qGamma</code>、<code>kGamma</code>、<code>cosSin</code>、<code>slotMapping</code>、<code>kCacheRef</code>、<code>vCacheRef</code>、<code>kScaleCacheRef</code>、<code>rotationOptional</code>、<code>vScaleOptional</code>、<code>headNums</code>、<code>qOut</code>、<code>workspaceSize</code>或<code>executor</code>为空指针，或支持场景要求非空的<code>queryStartLocOptional</code>、<code>seqLensOptional</code>或<code>qScaleOptional</code>为空指针。</td>
    </tr>
    <tr>
      <td rowspan="6" style="white-space: nowrap">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6" style="white-space: nowrap">161002</td>
      <td>Tensor为空Tensor。</td>
    </tr>
    <tr>
      <td>Tensor数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>Tensor数据格式为私有格式，或不满足ND格式要求。</td>
    </tr>
    <tr>
      <td>Tensor shape、<code>headNums</code>、<code>layoutQkv</code>或<code>layoutQOut</code>不满足接口约束。</td>
    </tr>
    <tr>
      <td><code>kCacheRef</code>、<code>vCacheRef</code>或<code>kScaleCacheRef</code>非连续Tensor的stride不满足约束。</td>
    </tr>
    <tr>
      <td><code>mropePositionOptional</code>与非空<code>mropeSectionOptional</code>未同时缺席或同时存在，<code>mropeSectionOptional</code>长度或单项值域不满足约束，或位置参数、<code>qQuantMode</code>、<code>qOut</code> dtype、<code>qScaleOptional</code>的组合与所选场景不匹配。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td style="white-space: nowrap">561101</td>
      <td>创建<code>aclOpExecutor</code>失败。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_INNER_NULLPTR</td>
      <td style="white-space: nowrap">561103</td>
      <td>输入Contiguous处理、Cache视图创建、算子任务构图或构图输出检查失败。</td>
    </tr>
  </tbody></table>

## aclnnQkvRmsNormRopeCacheWithKScale

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。当第一段接口返回的<code>workspaceSize</code>为0时，可传入<code>nullptr</code>。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize</code>获取。</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 支持场景：

  | 场景 | 位置输入 | Q分支约束 | K分支约束 | 支持的N/T布局 |
  |:---|:---|:---|:---|:---|
  | RoPE场景 | 传入`queryStartLocOptional`和`seqLensOptional`；不传`mropePositionOptional`；`mropeSectionOptional`不传或传空数组 | `qQuantMode="PerTokenPerHead"`；`qOut`为FP8 E4M3FN；`qScaleOptional`为FP32 | K量化为FP8 E4M3FN并写入`kCacheRef`；K scale写入FP32 `kScaleCacheRef` | `NTD -> NTD`、`TND -> TND`、`TND -> NTD` |
  | M-RoPE场景 | 不传`queryStartLocOptional`和`seqLensOptional`；传入`mropePositionOptional[T,3]`和`mropeSectionOptional=[t,h,w]` | `qQuantMode="NoQuant"`；`qOut`为BF16；`qScaleOptional`为`nullptr` | K量化为INT8并写入`kCacheRef`；K scale写入FP32 `kScaleCacheRef` | 仅`TND -> TND` |

  本接口不提供独立的场景属性，而是根据两组位置输入的有效presence确定场景。空`mropeSectionOptional`与未传入等价，只有非空数组才表示M-RoPE属性存在。每组位置输入必须同时存在或同时缺席，两组不能同时存在或同时缺席。`qOut`的数据类型由调用方构造的`qOut` Tensor决定，ACLNN接口不单独接收输出dtype参数。

  两个场景均要求`D=128`、`0<Nq<=64`、`Nq=8*Nk`且`Nk=Nv`。`rotationOptional`和`vScaleOptional`在接口合同中均须传入：Q/K在位置编码后共享右乘`rotationOptional[D,D]`；V在RoPE场景按head乘`vScaleOptional[Nv]`，在M-RoPE场景按head和通道乘`vScaleOptional[Nv,D]`，随后转换为FP8 E4M3FN。三个Cache Tensor均为输入输出别名，只更新`slotMapping`指定的slot，其余位置保持原值。

- 确定性说明：aclnnQkvRmsNormRopeCacheWithKScale默认确定性实现。
- 输入shape限制：
  - 仅支持`D=128`。
  - `headNums=[Nq,Nk,Nv]`必须满足`0<Nq<=64`、`Nq=8*Nk`、`Nk=Nv`。
  - `vScaleOptional`在RoPE场景的shape必须为`[Nv]`，在M-RoPE场景的shape必须为`[Nv,D]`。
  - `layoutQkv`控制`qkv`的N/T轴布局，默认值为`"TND"`；`layoutQOut`控制`qOut`和`qScaleOptional`的N/T轴布局，默认值为`"NTD"`：
    - `layoutQkv="TND"`，`layoutQOut="TND"`：`qkv=[T, Nq+Nk+Nv, D]`，`qOut=[T, Nq, D]`，`qScaleOptional=[T, Nq]`。
    - `layoutQkv="TND"`，`layoutQOut="NTD"`：`qkv=[T, Nq+Nk+Nv, D]`，`qOut=[Nq, T, D]`，`qScaleOptional=[Nq, T]`。
    - `layoutQkv="NTD"`，`layoutQOut="NTD"`：`qkv=[Nq+Nk+Nv, T, D]`，`qOut=[Nq, T, D]`，`qScaleOptional=[Nq, T]`。
  - `kCacheRef`、`vCacheRef`和`kScaleCacheRef`的`BlockNum`和`BlockSize`必须一致。
  - `kCacheRef`和`vCacheRef`均为4维正stride，最后一维stride为1，head维和token维stride均不小于`D=128`；`kScaleCacheRef`为4维正stride且最后一维stride为1；`kCacheRef`和`vCacheRef`前三维stride必须一致。
  - M-RoPE的`mropePositionOptional`逻辑shape固定为`[T,3]`，位置索引按`P[u,0]`、`P[u,1]`、`P[u,2]`分别读取token `u`的T/H/W坐标。
- 输入值域限制：
  - `seqLensOptional[b]`必须满足`seqLensOptional[b] >= queryStartLocOptional[b+1] - queryStartLocOptional[b]`。若`seqLensOptional[b]`小于该batch本次调用的token数，行为未定义。
  - M-RoPE场景下`mropeSectionOptional=[t,h,w]`的长度必须为3。令$\boldsymbol{s}=(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})$，其中$i=0,1,2$依次对应T/H/W，则

    $$
    s_{\mathrm T}\geq 0,\qquad
    0\leq s_i\leq C_i=\left\lfloor\frac{D/2+2-i}{3}\right\rfloor\ (i\in\{1,2\}),
    \qquad s_{\mathrm T}+s_{\mathrm H}+s_{\mathrm W}\leq D/2.
    $$

    当前`D=128`时，H/W上限均为21；$s_{\mathrm T}$没有独立lane上限，三项之和不得超过64。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<!-- npu="950" id7 -->
- <term>Ascend 950PR/Ascend 950DT</term>：

  ```c++
  #include <cstdint>
  #include <cstdio>
  #include <cstring>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_qkv_rms_norm_rope_cache_with_k_scale.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  struct TensorResource {
      aclTensor *tensor = nullptr;
      void *deviceAddr = nullptr;
  };

  struct AclResource {
      int32_t deviceId = 0;
      aclrtStream stream = nullptr;
      bool aclInited = false;
      bool deviceSet = false;
      std::vector<TensorResource *> tensors;
      aclIntArray *headNums = nullptr;
      aclOpExecutor *executor = nullptr;
      void *workspaceAddr = nullptr;
  };

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto dim : shape) {
          shapeSize *= dim;
      }
      return shapeSize;
  }

  std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t> &shape)
  {
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      return strides;
  }

  uint16_t FloatToBf16(float value)
  {
      uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      return static_cast<uint16_t>(bits >> 16);
  }

  int Init(int32_t deviceId, AclResource &resource)
  {
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      resource.aclInited = true;

      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      resource.deviceId = deviceId;
      resource.deviceSet = true;

      ret = aclrtCreateStream(&resource.stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return ACL_SUCCESS;
  }

  void FreeResource(AclResource &resource)
  {
      for (auto *tensorResource : resource.tensors) {
          if (tensorResource != nullptr && tensorResource->tensor != nullptr) {
              aclDestroyTensor(tensorResource->tensor);
              tensorResource->tensor = nullptr;
          }
      }
      if (resource.headNums != nullptr) {
          aclDestroyIntArray(resource.headNums);
          resource.headNums = nullptr;
      }
      for (auto *tensorResource : resource.tensors) {
          if (tensorResource != nullptr && tensorResource->deviceAddr != nullptr) {
              aclrtFree(tensorResource->deviceAddr);
              tensorResource->deviceAddr = nullptr;
          }
      }
      if (resource.workspaceAddr != nullptr) {
          aclrtFree(resource.workspaceAddr);
          resource.workspaceAddr = nullptr;
      }
      if (resource.stream != nullptr) {
          aclrtDestroyStream(resource.stream);
          resource.stream = nullptr;
      }
      if (resource.deviceSet) {
          aclrtResetDevice(resource.deviceId);
          resource.deviceSet = false;
      }
      if (resource.aclInited) {
          aclFinalize();
          resource.aclInited = false;
      }
  }

  int ReturnAfterCleanup(int ret, AclResource &resource)
  {
      FreeResource(resource);
      return ret;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                      TensorResource &resource)
  {
      const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
      auto ret = aclrtMalloc(&resource.deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(resource.deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides = GetContiguousStrides(shape);
      resource.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), resource.deviceAddr);
      CHECK_RET(resource.tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return ACL_SUCCESS;
  }

  int main()
  {
      AclResource resource;
      auto ret = Init(0, resource);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
                return ReturnAfterCleanup(ret, resource));

      constexpr int64_t T = 4;
      constexpr int64_t Nq = 16;
      constexpr int64_t Nk = 2;
      constexpr int64_t Nv = 2;
      constexpr int64_t D = 128;
      constexpr int64_t Batch = 1;
      constexpr int64_t MaxSeqLen = 16;
      constexpr int64_t BlockNum = 1;
      constexpr int64_t BlockSize = 16;

      std::vector<int64_t> qkvShape = {T, Nq + Nk + Nv, D};
      std::vector<int64_t> qGammaShape = {D};
      std::vector<int64_t> kGammaShape = {D};
      std::vector<int64_t> cosSinShape = {MaxSeqLen, D};
      std::vector<int64_t> slotMappingShape = {T};
      std::vector<int64_t> kCacheShape = {BlockNum, Nk, BlockSize, D};
      std::vector<int64_t> vCacheShape = {BlockNum, Nv, BlockSize, D};
      std::vector<int64_t> kScaleCacheShape = {BlockNum, Nk, BlockSize, 1};
      std::vector<int64_t> queryStartLocShape = {Batch + 1};
      std::vector<int64_t> seqLensShape = {Batch};
      std::vector<int64_t> rotationShape = {D, D};
      std::vector<int64_t> vScaleShape = {Nv};
      std::vector<int64_t> qOutShape = {T, Nq, D};
      std::vector<int64_t> qScaleShape = {T, Nq};

      std::vector<uint16_t> qkvHostData(GetShapeSize(qkvShape), FloatToBf16(0.125f));
      std::vector<float> qGammaHostData(GetShapeSize(qGammaShape), 1.0f);
      std::vector<float> kGammaHostData(GetShapeSize(kGammaShape), 1.0f);
      std::vector<float> cosSinHostData(GetShapeSize(cosSinShape), 0.0f);
      for (int64_t row = 0; row < MaxSeqLen; ++row) {
          for (int64_t col = 0; col < D / 2; ++col) {
              cosSinHostData[row * D + col] = 1.0f;
          }
      }
      std::vector<int32_t> slotMappingHostData = {0, 1, 2, 3};
      std::vector<uint8_t> kCacheHostData(GetShapeSize(kCacheShape), 0);
      std::vector<uint8_t> vCacheHostData(GetShapeSize(vCacheShape), 0);
      std::vector<float> kScaleCacheHostData(GetShapeSize(kScaleCacheShape), 0.0f);
      std::vector<int32_t> queryStartLocHostData = {0, T};
      std::vector<int32_t> seqLensHostData = {T};
      std::vector<uint16_t> rotationHostData(GetShapeSize(rotationShape), FloatToBf16(0.0f));
      for (int64_t i = 0; i < D; ++i) {
          rotationHostData[i * D + i] = FloatToBf16(1.0f);
      }
      std::vector<float> vScaleHostData(GetShapeSize(vScaleShape), 1.0f);
      std::vector<uint8_t> qOutHostData(GetShapeSize(qOutShape), 0);
      std::vector<float> qScaleHostData(GetShapeSize(qScaleShape), 0.0f);

      TensorResource qkv;
      TensorResource qGamma;
      TensorResource kGamma;
      TensorResource cosSin;
      TensorResource slotMapping;
      TensorResource kCache;
      TensorResource vCache;
      TensorResource kScaleCache;
      TensorResource queryStartLoc;
      TensorResource seqLens;
      TensorResource rotation;
      TensorResource vScale;
      TensorResource qOut;
      TensorResource qScale;
      resource.tensors = {&qkv,         &qGamma,       &kGamma, &cosSin, &slotMapping, &kCache, &vCache,
                          &kScaleCache, &queryStartLoc, &seqLens, &rotation, &vScale,  &qOut,   &qScale};

      ret = CreateAclTensor(qkvHostData, qkvShape, ACL_BF16, qkv);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(qGammaHostData, qGammaShape, ACL_FLOAT, qGamma);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(kGammaHostData, kGammaShape, ACL_FLOAT, kGamma);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(cosSinHostData, cosSinShape, ACL_FLOAT, cosSin);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(slotMappingHostData, slotMappingShape, ACL_INT32, slotMapping);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(kCacheHostData, kCacheShape, ACL_FLOAT8_E4M3FN, kCache);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(vCacheHostData, vCacheShape, ACL_FLOAT8_E4M3FN, vCache);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(kScaleCacheHostData, kScaleCacheShape, ACL_FLOAT, kScaleCache);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(queryStartLocHostData, queryStartLocShape, ACL_INT32, queryStartLoc);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(seqLensHostData, seqLensShape, ACL_INT32, seqLens);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(rotationHostData, rotationShape, ACL_BF16, rotation);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(vScaleHostData, vScaleShape, ACL_FLOAT, vScale);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(qOutHostData, qOutShape, ACL_FLOAT8_E4M3FN, qOut);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(qScaleHostData, qScaleShape, ACL_FLOAT, qScale);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));

      std::vector<int64_t> headNumsVec = {Nq, Nk, Nv};
      resource.headNums = aclCreateIntArray(headNumsVec.data(), headNumsVec.size());
      CHECK_RET(resource.headNums != nullptr,
                LOG_PRINT("aclCreateIntArray failed.\n"); return ReturnAfterCleanup(ACL_ERROR_INVALID_PARAM, resource));

      const char *layoutQkv = "TND";
      const char *layoutQOut = "TND";
      float epsilon = 1e-6f;
      uint64_t workspaceSize = 0;
      aclnnStatus status = aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
          qkv.tensor, qGamma.tensor, kGamma.tensor, cosSin.tensor, slotMapping.tensor, kCache.tensor, vCache.tensor,
          kScaleCache.tensor, queryStartLoc.tensor, seqLens.tensor, rotation.tensor, vScale.tensor, nullptr,
          resource.headNums, layoutQkv, layoutQOut, epsilon, nullptr, "PerTokenPerHead", qOut.tensor, qScale.tensor,
          &workspaceSize, &resource.executor);
      CHECK_RET(status == ACL_SUCCESS,
                LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize failed. ERROR: %d\n", status);
                return ReturnAfterCleanup(static_cast<int>(status), resource));

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&resource.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                    return ReturnAfterCleanup(ret, resource));
      }

      status = aclnnQkvRmsNormRopeCacheWithKScale(resource.workspaceAddr, workspaceSize, resource.executor,
                                                  resource.stream);
      CHECK_RET(status == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScale failed. ERROR: %d\n", status);
                return ReturnAfterCleanup(static_cast<int>(status), resource));

      ret = aclrtSynchronizeStream(resource.stream);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                return ReturnAfterCleanup(ret, resource));
      LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScale execute success.\n");
      FreeResource(resource);
      return 0;
  }
  ```

<!-- end id7 -->
