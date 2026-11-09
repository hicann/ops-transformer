# aclnnMegaMhc

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

- 接口功能：MegaMhc融合多阶段RmsNorm、MhcPost和MhcPre计算，支持可选的输入RmsNorm和中间RmsNorm，更新多路残差状态与Post/Res混合系数，并将各路残差加权合并、归一化为下一子层的输入。传入外部Pre混合系数时，还输出本次计算的Pre系数。

- 计算公式：

  以下公式从融合算子中RmsNorm、MhcPost和MhcPre各小算子的计算功能与执行顺序出发，分阶段说明算子功能。$B$为批大小，$S$为序列长度，$T=B\times S$为展平后的token数，$N$为残差head数，$D$为每个head的隐藏维度。使用BS布局时按下式计算；T布局等价于将$B$、$S$折叠为$T$。$\operatorname{FP32}$表示转换为float32，$\operatorname{Cast}_{dtype(x)}$表示转换为输入x的数据类型。

  1. 可选执行RmsNorm算子功能，对输入x进行均方根归一化，简称rms_norm_0阶段。仅当`rms_norm_0_weight`非空时执行下式；为空时跳过本阶段，直接令$x\_1=x$。

     $$
     \begin{aligned}
     x\_fp32_{b,s,d} &= \operatorname{FP32}(x_{b,s,d}), \\
     inv\_rms\_0_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}x\_fp32_{b,s,i}^{2}+rms\_norm\_eps\right), \\
     x\_1_{b,s,d} &= \operatorname{Cast}_{dtype(x)}\left(x\_fp32_{b,s,d}\cdot inv\_rms\_0_{b,s}\cdot rms\_norm\_0\_weight_d\right).
     \end{aligned}
     $$

  2. 执行MhcPost算子功能，将上一阶段的`x_1`与映射后的残差融合，简称mhc_post阶段。

     $$
     \begin{aligned}
     term1_{b,s,n,d} &= \operatorname{FP32}(post\_mix_{b,s,n,0})\cdot \operatorname{FP32}(x\_1_{b,s,d}), \\
     term2_{b,s,n,d} &= \sum_{j=0}^{N-1}\operatorname{FP32}(comb\_res\_mix_{b,s,j,n})\cdot \operatorname{FP32}(residual_{b,s,j,d}), \\
     x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(x)}\left(term1_{b,s,n,d}+term2_{b,s,n,d}\right).
     \end{aligned}
     $$

     $j$为归约head，$n$为输出head。当前版本要求`comb_res_mix`非空。

  3. 可选执行RmsNorm算子功能，对融合后的完整残差行进行均方根归一化，简称rms_norm_mid阶段。

     当`rms_norm_mid_weight`非空时：

     $$
     \begin{aligned}
     x\_2\_flat &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
     inv\_rms\_mid_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_flat_{b,s,k}^{2}+rms\_norm\_eps\right), \\
     x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(x)}\left(\operatorname{reshape}\left(x\_2\_flat\cdot inv\_rms\_mid\cdot rms\_norm\_mid\_weight,[B,S,N,D]\right)_{b,s,n,d}\right).
     \end{aligned}
     $$

     当`rms_norm_mid_weight`为空时，跳过本阶段，`x_2`保持MhcPost算子的输出融合阶段的输出。无论是否执行本阶段，均保存：

     $$
     new\_residual=x\_2.
     $$

  4. 执行MhcPre算子功能，计算混合系数并对各路残差加权合并，简称mhc_pre阶段。

     $$
     \begin{aligned}
     x\_2\_rs &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
     inv\_rms\_pre_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_rs_{b,s,k}^{2}+hc\_norm\_eps\right), \\
     proj\_w &= fn^{T}, \\
     x\_hat &= (x\_2\_rs\cdot proj\_w)\cdot inv\_rms\_pre.
     \end{aligned}
     $$

     先执行矩阵乘$x\_2\_rs\cdot proj\_w$，再乘$inv\_rms\_pre$，用于生成本次混合系数；该RMS缩放不改变后续加权合并使用的`x_2`。按照`[N,N,N*N]`切分：

     $$
     (x\_pre,x\_post,x\_comb)=\operatorname{split}(x\_hat,[N,N,N^2]),
     $$

     $$
     (bias\_pre,bias\_post,bias\_res)=\operatorname{split}(mix\_bases,[N,N,N^2]).
     $$

     MhcPre算子的Pre、Post和Res三个分支的系数生成阶段计算如下：

     $$
     \begin{aligned}
     pre\_mix_{b,s,n,0} &= \operatorname{sigmoid}(mix\_scales_0\cdot x\_pre_{b,s,n}+bias\_pre_n)+hc\_pre\_eps, \\
     new\_post\_mix_{b,s,n,0} &= hc\_post\_scale\cdot\operatorname{sigmoid}(mix\_scales_1\cdot x\_post_{b,s,n}+bias\_post_n), \\
     x\_comb &= \operatorname{reshape}(x\_comb,[B,S,N,N]), \\
     bias\_res &= \operatorname{reshape}(bias\_res,[N,N]), \\
     comb\_logits_{b,s,i,j} &= mix\_scales_2\cdot x\_comb_{b,s,i,j}+bias\_res_{i,j}.
     \end{aligned}
     $$

     `pre_mix`是本次MhcPre计算的混合系数。`shifted_prev_mix`非空时，通过`new_prev_mix`返回本次计算的`pre_mix`；为空时不输出`new_prev_mix`。该输出不回传外部传入的`shifted_prev_mix`。公式中的new_post_mix是MhcPre算子的Post分支系数生成阶段的新系数，comb_logits是MhcPre算子的Res分支系数生成阶段的新logits，均不覆盖输入post_mix、comb_res_mix。bias_res由长度为N²的一维偏置按行优先顺序还原为[N,N]，再与MhcPre算子的Res分支投影阶段的结果逐元素相加。

     `shifted_prev_mix`为空时使用本次计算的`pre_mix`，非空时使用传入值替代`pre_mix`。将选中的系数与`x_2`相乘，并沿head维求和：

     $$
     \begin{aligned}
     w &= \begin{cases}pre\_mix, & shifted\_prev\_mix=\mathrm{nullptr},\\shifted\_prev\_mix, & shifted\_prev\_mix\neq\mathrm{nullptr},\end{cases} \\
     h\_in_{b,s,d} &= \operatorname{Cast}_{dtype(x)}\left(\sum_{n=0}^{N-1}w_{b,s,n,0}\cdot\operatorname{FP32}(x\_2_{b,s,n,d})\right).
     \end{aligned}
     $$

     传入的`shifted_prev_mix`不再经过上述Pre分支的Sigmoid和`hc_pre_eps`处理；它仅替换加权合并所用的`pre_mix`，不改变Post、Res分支系数的生成。合并结果`h_in`仍在第5阶段执行`rms_norm_1`。本次`pre_mix`与外部`shifted_prev_mix`使用独立UB区域；外部系数可提前载入，本次系数写回后，其UB区域方可再次生成新值。

     MhcPre算子的Res分支Sinkhorn归一化阶段（预留功能）

     令$L=num\_sinkhorn\_iters$。当前版本仅支持$L=0$，直接返回MhcPre算子的Res分支系数生成阶段的logits，不执行指数变换或Sinkhorn归一化：

     $$
     new\_comb\_res\_mix=comb\_logits.
     $$

     以下$L>0$的计算为预留功能，当前版本不支持。对每个token的$N\times N$矩阵独立计算，第一次迭代先沿最后一维执行softmax，再执行列归一化：

     $$
     \begin{aligned}
     A^{(0)}_{b,s,i,j} &= \operatorname{softmax}_{j}(comb\_logits_{b,s,i,j})+sinkhorn\_eps, \\
     H^{(0)}_{b,s,i,j} &= \frac{A^{(0)}_{b,s,i,j}}{\displaystyle\sum_{p=0}^{N-1}A^{(0)}_{b,s,p,j}+sinkhorn\_eps}.
     \end{aligned}
     $$

     随后对$r=1,\ldots,L-1$依次执行行归一化和列归一化：

     $$
     \begin{aligned}
     R^{(r)}_{b,s,i,j} &= \frac{H^{(r-1)}_{b,s,i,j}}{\displaystyle\sum_{q=0}^{N-1}H^{(r-1)}_{b,s,i,q}+sinkhorn\_eps}, \\
     H^{(r)}_{b,s,i,j} &= \frac{R^{(r)}_{b,s,i,j}}{\displaystyle\sum_{p=0}^{N-1}R^{(r)}_{b,s,p,j}+sinkhorn\_eps}, \\
     new\_comb\_res\_mix_{b,s,i,j} &= H^{(L-1)}_{b,s,i,j}.
     \end{aligned}
     $$

     其中$i$为行索引，$j$为列索引；$L=1$时直接输出$H^{(0)}$。`sinkhorn_eps`仅用于上述Sinkhorn计算，与MhcPre算子的Pre分支Sigmoid阶段的`hc_pre_eps`及均方根归一化阶段的`hc_norm_eps`分别独立。

  5. 执行RmsNorm算子功能，对加权合并结果进行均方根归一化，简称rms_norm_1阶段。

     $$
     \begin{aligned}
     h\_in\_rms_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}\operatorname{FP32}(h\_in_{b,s,i})^{2}+rms\_norm\_eps\right), \\
     h\_in\_f32_{b,s,d} &= \operatorname{FP32}(h\_in_{b,s,d})\cdot h\_in\_rms_{b,s}\cdot \operatorname{FP32}(rms\_norm\_1\_weight_d)\cdot rms\_norm\_scale, \\
     y &= \operatorname{Cast}_{dtype(x)}(h\_in\_f32).
     \end{aligned}
     $$

     系数下标`[b,s,n,0]`中的最后一维对应单个混合系数，公式中的nullptr表示未传入对应可选Tensor。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnMegaMhcGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMegaMhc”接口执行计算。

```cpp
aclnnStatus aclnnMegaMhcGetWorkspaceSize(
  const aclTensor  *x,
  const aclTensor  *residual,
  const aclTensor  *postMix,
  const aclTensor  *fn,
  const aclTensor  *mixScales,
  const aclTensor  *mixBases,
  const aclTensor  *rmsNorm1Weight,
  const aclTensor  *combResMixOptional,
  const aclTensor  *shiftedPrevMixOptional,
  const aclTensor  *rmsNorm0WeightOptional,
  const aclTensor  *rmsNormMidWeightOptional,
  double           rmsNormEps,
  double           hcNormEps,
  double           hcPreEps,
  int64_t          opImplMode,
  int64_t          hcMult,
  double           hcPostScale,
  double           sinkhornEps,
  int64_t          numSinkhornIters,
  double           rmsNormScale,
  int64_t          outputMode,
  aclTensor        *newResidualOut,
  aclTensor        *newPostMixOut,
  aclTensor        *newCombResMixOut,
  aclTensor        *newPrevMixOut,
  aclTensor        *yOut,
  aclTensor        *yQuantOut,
  aclTensor        *yScaleOut,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnMegaMhc(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnMegaMhcGetWorkspaceSize

- **参数说明**

  N=`hcMult`，Q=N*(N+2)，K=N*D。表中变量均对应功能说明中的公式。

  <table style="table-layout: fixed; width: 1500px">
  <colgroup>
  <col style="width: 220px">
  <col style="width: 100px">
  <col style="width: 300px">
  <col style="width: 360px">
  <col style="width: 170px">
  <col style="width: 80px">
  <col style="width: 180px">
  <col style="width: 90px">
  </colgroup>
  <thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th></tr></thead>
  <tbody>
  <tr>
    <td>x（const aclTensor*）</td>
    <td>输入</td>
    <td>Attention/MLP输出，对应公式中的x。</td>
    <td>不支持空Tensor。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>[B,S,D]或[T,D]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>residual（const aclTensor*）</td>
    <td>输入</td>
    <td>多head残差状态，对应公式中的residual。</td>
    <td>不支持空Tensor。</td>
    <td>与x相同</td>
    <td>ND</td>
    <td>[B,S,N,D]或[T,N,D]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>postMix（const aclTensor*）</td>
    <td>输入</td>
    <td>MhcPost算子的输出融合阶段的输入系数，对应公式中的post_mix。</td>
    <td>不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[B,S,N,1]或[T,N,1]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>fn（const aclTensor*）</td>
    <td>输入</td>
    <td>MhcPre算子的投影阶段权重，对应公式中的fn，分量顺序为Pre、Post、Res。</td>
    <td>不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[Q,K]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>mixScales（const aclTensor*）</td>
    <td>输入</td>
    <td>MhcPre算子的投影阶段的三路缩放系数，对应公式中的mix_scales，依次对应Pre、Post、Res。</td>
    <td>不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[3]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>mixBases（const aclTensor*）</td>
    <td>输入</td>
    <td>MhcPre算子的系数生成阶段的三路偏置，对应公式中的mix_bases，按[N,N,N²]切分。</td>
    <td>不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[Q]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>rmsNorm1Weight（const aclTensor*）</td>
    <td>输入</td>
    <td>RmsNorm算子的rms_norm_1阶段的逐通道权重，对应公式中的rms_norm_1_weight。</td>
    <td>不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[D]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>combResMixOptional（const aclTensor*）</td>
    <td>可选输入</td>
    <td>MhcPost算子的残差映射阶段的变换矩阵，对应公式中的comb_res_mix。</td>
    <td>本版不支持nullptr。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[B,S,N,N]或[T,N,N]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>shiftedPrevMixOptional（const aclTensor*）</td>
    <td>可选输入</td>
    <td>MhcPre加权合并阶段的外部混合系数，对应公式中的shifted_prev_mix；用于与x_2相乘并沿head维求和，后续rms_norm_1保持执行。</td>
    <td>nullptr时使用本次计算的pre_mix；非空时使用传入值替代pre_mix，形状须与postMix相同，不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[B,S,N,1]或[T,N,1]</td>
    <td>-</td>
  </tr>
  <tr>
    <td>rmsNorm0WeightOptional（const aclTensor*）</td>
    <td>可选输入</td>
    <td>RmsNorm算子的rms_norm_0阶段的逐通道权重，对应公式中的rms_norm_0_weight。</td>
    <td>nullptr时跳过rms_norm_0阶段，直接将x送入MhcPost；非空时不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[D]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>rmsNormMidWeightOptional（const aclTensor*）</td>
    <td>可选输入</td>
    <td>RmsNorm算子的rms_norm_mid阶段的逐通道权重，对应公式中的rms_norm_mid_weight。</td>
    <td>nullptr时跳过RmsNorm算子的rms_norm_mid阶段；非空时不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[K]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>rmsNormEps（double）</td>
    <td>输入</td>
    <td>RmsNorm算子的rms_norm_0、rms_norm_mid和rms_norm_1阶段的稳定项，对应公式中的rms_norm_eps。</td>
    <td>建议值为1e-6；转换为FLOAT32后须有限且大于0。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>hcNormEps（double）</td>
    <td>输入</td>
    <td>用于计算x_2完整N*D行的均方根并缩放投影结果，对应公式中的hc_norm_eps。</td>
    <td>建议值为1e-6；转换为FLOAT32后须有限且大于0。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>hcPreEps（double）</td>
    <td>输入</td>
    <td>MhcPre算子的Pre分支Sigmoid阶段的输出附加项，对应公式中的hc_pre_eps。</td>
    <td>建议值为1e-6；转换为FLOAT32后须有限且大于0。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>opImplMode（int64_t）</td>
    <td>输入</td>
    <td>MhcPre算子的矩阵乘阶段的计算模式，作用于公式中的x_2_rs与proj_w相乘。</td>
    <td>0：FLOAT32计算；1：HF32计算。2（3BF16）为预留取值，本版不支持。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>hcMult（int64_t）</td>
    <td>输入</td>
    <td>对应公式中的N，须与residual的head维一致。</td>
    <td>本版仅支持4。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>hcPostScale（double）</td>
    <td>输入</td>
    <td>MhcPre算子的Post分支Sigmoid阶段的输出倍率，对应公式中的hc_post_scale。</td>
    <td>建议值为2.0；转换为FLOAT32后须有限。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sinkhornEps（double）</td>
    <td>输入</td>
    <td>预留参数。</td>
    <td>本版仅支持1e-6。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>numSinkhornIters（int64_t）</td>
    <td>输入</td>
    <td>预留参数。</td>
    <td>本版仅支持0。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>rmsNormScale（double）</td>
    <td>输入</td>
    <td>预留参数。</td>
    <td>本版仅支持1.0。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>outputMode（int64_t）</td>
    <td>输入</td>
    <td>指定输出模式。0：输出非量化结果；1：输出MX Quant量化结果；2：同时输出量化和非量化结果。</td>
    <td>本版仅支持0；1、2为预留取值，不支持其他取值。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>newResidualOut（aclTensor*）</td>
    <td>输出</td>
    <td>更新后的多head残差，对应公式中的new_residual。</td>
    <td>不支持空Tensor。</td>
    <td>与x相同</td>
    <td>ND</td>
    <td>[B,S,N,D]或[T,N,D]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>newPostMixOut（aclTensor*）</td>
    <td>输出</td>
    <td>MhcPre算子的Post分支系数生成阶段的新系数，对应公式中的new_post_mix。</td>
    <td>不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[B,S,N,1]或[T,N,1]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>newCombResMixOut（aclTensor*）</td>
    <td>输出</td>
    <td>MhcPre算子的Res分支系数生成阶段的新系数，对应公式中的new_comb_res_mix；本版仅输出comb_logits。</td>
    <td>不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[B,S,N,N]或[T,N,N]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>newPrevMixOut（aclTensor*）</td>
    <td>可选输出</td>
    <td>本次MhcPre计算的pre_mix，与外部shifted_prev_mix分开存放于独立UB区域；该输出不回传外部输入。</td>
    <td>shiftedPrevMixOptional非空时必须非空且形状与postMix相同；shiftedPrevMixOptional为nullptr时必须传入nullptr。不支持空Tensor。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[B,S,N,1]或[T,N,1]</td>
    <td>-</td>
  </tr>
  <tr>
    <td>yOut（aclTensor*）</td>
    <td>可选输出</td>
    <td>RmsNorm算子的rms_norm_1阶段计算并应用倍率后的原精度结果，对应公式中的y。</td>
    <td>本版必须输出，不支持空Tensor。</td>
    <td>与x相同</td>
    <td>ND</td>
    <td>[B,S,D]或[T,D]</td>
    <td>√</td>
  </tr>
  <tr>
    <td>yQuantOut（aclTensor*）</td>
    <td>可选输出</td>
    <td>预留参数。</td>
    <td>本版仅支持nullptr。</td>
    <td>FLOAT8_E4M3FN</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>yScaleOut（aclTensor*）</td>
    <td>可选输出</td>
    <td>预留参数。</td>
    <td>本版仅支持nullptr。</td>
    <td>FLOAT8_E8M0</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
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
      <td>executor(aclOpExecutor**)</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px">
  <colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead><tr><th>返回值</th><th>错误码</th><th>描述</th></tr></thead>
  <tbody>
  <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>七个必选输入、combResMixOptional、三个必选输出、workspaceSize或executor存在空指针；yOut为空指针；shiftedPrevMixOptional非空时newPrevMixOut为空指针。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>数据类型、格式、shape或输出布局不符合参数规格；系数最后一维不为1；hcMult不为4；opImplMode不为0或1；预留参数numSinkhornIters不为0或sinkhornEps不为1e-6；参与计算的epsilon转换为FLOAT32后非有限或不大于0；hcPostScale转换为FLOAT32后非有限；预留参数rmsNormScale不为1.0；outputMode不为0（1、2为预留取值，其他值非法）；yQuantOut或yScaleOut非空；shiftedPrevMixOptional为空时newPrevMixOut非空。</td>
  </tr>
  </tbody>
  </table>

## aclnnMegaMhc

- **参数说明**

  <table style="table-layout: fixed; width: 1000px">
  <colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr></thead>
  <tbody>
  <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
  <tr><td>workspaceSize </td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMegaMhcGetWorkspaceSize获取。</td></tr>
  <tr><td>executor</td><td>输入</td><td> op执行器，包含了算子计算流程。 </td></tr>
  <tr><td>stream </td><td>输入</td><td> 指定执行任务的Stream。 </td></tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnMegaMhc默认确定性实现。
- Batch一致性说明：默认非Batch一致性实现，不支持通过aclrtSetSysParamOpt开启Batch一致性。
- 支持推理场景的前向计算，暂不支持反向梯度。
- 输入均位于同一设备，带前导维的Tensor采用相同BS或T布局，不支持广播。
- T、B、S须为正整数，T或B*S不超过UINT32_MAX。
- N=4，Q=24，K=4D。D的取值范围为[16,16384]，且为16的倍数。

## 调用示例

具体编译与执行过程参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mega_mhc.h"

bool Check(int status, const char* step)
{
    if (status == 0) {
        return true;
    }
    std::cerr << step << " failed: " << status << std::endl;
    return false;
}

struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    std::vector<aclTensor*> tensors;
    std::vector<void*> buffers;

    ~Resources()
    {
        // 等待异步任务完成，再释放其引用的资源。
        if (stream != nullptr) {
            Check(aclrtSynchronizeStream(stream), "cleanup aclrtSynchronizeStream");
        }
        for (auto* tensor : tensors) {
            Check(aclDestroyTensor(tensor), "aclDestroyTensor");
        }
        for (auto* buffer : buffers) {
            Check(aclrtFree(buffer), "aclrtFree");
        }
        if (stream != nullptr) {
            Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
        }
        if (deviceSet) {
            Check(aclrtResetDevice(deviceId), "aclrtResetDevice");
        }
        if (initialized) {
            Check(aclFinalize(), "aclFinalize");
        }
    }
};

// 使用0和1初始化输入，FLOAT16的1编码为0x3c00。
aclTensor* CreateTensor(Resources& resources, const std::vector<int64_t>& shape, aclDataType dtype, bool ones = false,
                        bool identity = false)
{
    int64_t count = 1;
    for (auto dim : shape) {
        count *= dim;
    }
    const uint64_t bytes = count * (dtype == ACL_FLOAT16 ? sizeof(uint16_t) : sizeof(float));
    void* deviceData = nullptr;
    if (!Check(aclrtMalloc(&deviceData, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")) {
        return nullptr;
    }
    resources.buffers.push_back(deviceData);
    std::vector<float> fp32(count, ones ? 1.0f : 0.0f);
    std::vector<uint16_t> fp16(count, ones ? 0x3c00 : 0);
    if (identity) {
        const int64_t n = shape.back();
        for (int64_t offset = 0; offset < count; offset += n * n) {
            for (int64_t head = 0; head < n; ++head) {
                fp32[offset + head * n + head] = 1.0f;
            }
        }
    }
    const void* hostData =
        dtype == ACL_FLOAT16 ? static_cast<const void*>(fp16.data()) : static_cast<const void*>(fp32.data());
    if (!Check(aclrtMemcpy(deviceData, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy")) {
        return nullptr;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t axis = static_cast<int64_t>(shape.size()) - 2; axis >= 0; --axis) {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                        shape.data(), shape.size(), deviceData);
    if (tensor != nullptr) {
        resources.tensors.push_back(tensor);
    }
    return tensor;
}

int Run()
{
    // 初始化运行环境、设备及Stream。
    Resources resources;
    if (!Check(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.initialized = true;
    if (!Check(aclrtSetDevice(resources.deviceId), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }

    const int64_t t = 16;
    const int64_t n = 4;
    const int64_t d = 4096;
    const int64_t q = n * n + 2 * n;
    const bool withMid = false;
    const int64_t opImplMode = 0;
    const double rmsNormEps = 1e-6;
    const double hcNormEps = 1e-6;
    const double hcPreEps = 1e-6;
    const int64_t hcMult = n;
    const double hcPostScale = 2.0;
    const double sinkhornEps = 1e-6;
    const int64_t numSinkhornIters = 0;
    const double rmsNormScale = 1.0;
    const int64_t outputMode = 0;

    // 创建输入和输出Tensor。
    auto* x = CreateTensor(resources, {t, d}, ACL_FLOAT16);
    auto* residual = CreateTensor(resources, {t, n, d}, ACL_FLOAT16);
    auto* postMix = CreateTensor(resources, {t, n, 1}, ACL_FLOAT, true);
    auto* fn = CreateTensor(resources, {q, n * d}, ACL_FLOAT);
    auto* mixScales = CreateTensor(resources, {3}, ACL_FLOAT, true);
    auto* mixBases = CreateTensor(resources, {q}, ACL_FLOAT);
    auto* rmsNorm1Weight = CreateTensor(resources, {d}, ACL_FLOAT, true);
    auto* combResMixOptional = CreateTensor(resources, {t, n, n}, ACL_FLOAT, false, true);
    const aclTensor* shiftedPrevMixOptional = nullptr;
    auto* rmsNorm0WeightOptional = CreateTensor(resources, {d}, ACL_FLOAT, true);
    aclTensor* rmsNormMidWeightOptional = withMid ? CreateTensor(resources, {n * d}, ACL_FLOAT, true) : nullptr;
    auto* yOut = CreateTensor(resources, {t, d}, ACL_FLOAT16);
    auto* newPostMixOut = CreateTensor(resources, {t, n, 1}, ACL_FLOAT);
    auto* newCombResMixOut = CreateTensor(resources, {t, n, n}, ACL_FLOAT);
    auto* newResidualOut = CreateTensor(resources, {t, n, d}, ACL_FLOAT16);
    aclTensor* newPrevMixOut = nullptr;
    // 输出原精度结果，预留量化输出必须传入nullptr。
    aclTensor* yQuantOut = nullptr;
    aclTensor* yScaleOut = nullptr;
    if (x == nullptr || residual == nullptr || postMix == nullptr || fn == nullptr || mixScales == nullptr ||
        mixBases == nullptr || rmsNorm1Weight == nullptr || combResMixOptional == nullptr ||
        rmsNorm0WeightOptional == nullptr || (withMid && rmsNormMidWeightOptional == nullptr) || yOut == nullptr ||
        newPostMixOut == nullptr || newCombResMixOut == nullptr || newResidualOut == nullptr) {
        std::cerr << "CreateTensor failed" << std::endl;
        return 1;
    }

    // 第一段接口获取workspace大小及执行器。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (!Check(aclnnMegaMhcGetWorkspaceSize(x, residual, postMix, fn, mixScales, mixBases, rmsNorm1Weight,
                                            combResMixOptional, shiftedPrevMixOptional, rmsNorm0WeightOptional,
                                            rmsNormMidWeightOptional, rmsNormEps, hcNormEps, hcPreEps, opImplMode,
                                            hcMult, hcPostScale, sinkhornEps, numSinkhornIters, rmsNormScale,
                                            outputMode, newResidualOut, newPostMixOut, newCombResMixOut, newPrevMixOut,
                                            yOut, yQuantOut, yScaleOut, &workspaceSize, &executor),
               "aclnnMegaMhcGetWorkspaceSize")) {
        return 1;
    }
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        if (!Check(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace allocation")) {
            return 1;
        }
        resources.buffers.push_back(workspace);
    }
    // 第二段接口执行计算，等待任务完成。
    if (!Check(aclnnMegaMhc(workspace, workspaceSize, executor, resources.stream), "aclnnMegaMhc") ||
        !Check(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
        return 1;
    }
    std::cout << "MegaMhc completed: " << (withMid ? "with_mid" : "no_mid") << ", T=" << t << ", N=" << n << ", D=" << d
              << std::endl;
    return 0;
}

int main()
{
    return Run();
}
```
