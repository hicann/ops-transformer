# mega_mhc

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

- 接口功能：MegaMhc融合多阶段RmsNorm、MhcPost和MhcPre计算，支持可选的输入RmsNorm和中间RmsNorm，更新多路残差状态与Post/Res混合系数，并将各路残差加权合并、归一化为下一子层的输入。

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

  2. 执行MhcPost算子功能，将归一化结果与映射后的残差融合，简称mhc_post阶段。

     $$
     \begin{aligned}
     term1_{b,s,n,d} &= \operatorname{FP32}(post\_mix_{b,s,n,0})\cdot \operatorname{FP32}(x\_1_{b,s,d}), \\
     term2_{b,s,n,d} &=
     \begin{cases}
     \displaystyle\sum_{j=0}^{N-1}\operatorname{FP32}(comb\_res\_mix_{b,s,j,n})\cdot \operatorname{FP32}(residual_{b,s,j,d}), & comb\_res\_mix\neq\mathrm{None},\\
     \operatorname{FP32}(residual_{b,s,n,d}), & comb\_res\_mix=\mathrm{None},
     \end{cases} \\
     x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(x)}\left(term1_{b,s,n,d}+term2_{b,s,n,d}\right).
     \end{aligned}
     $$

     $j$为归约head，$n$为输出head。

  3. 可选执行RmsNorm算子功能，对融合后的完整残差行进行均方根归一化，简称rms_norm_mid阶段。

     当`rms_norm_mid_weight`非空时：

     $$
     \begin{aligned}
     x\_2\_flat &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
     inv\_rms\_mid_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_flat_{b,s,k}^{2}+rms\_norm\_eps\right), \\
     x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(x)}\left(\operatorname{reshape}\left(x\_2\_flat\cdot inv\_rms\_mid\cdot rms\_norm\_mid\_weight,[B,S,N,D]\right)_{b,s,n,d}\right).
     \end{aligned}
     $$

     当`rms_norm_mid_weight`为空时，跳过本阶段，`x_2`保持MhcPost算子的输出融合阶段的输出；随后固定保存：

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

     先执行矩阵乘$x\_2\_rs\cdot proj\_w$，再乘$inv\_rms\_pre$。按照`[N,N,N*N]`切分：

     $$
     (x\_pre,x\_post,x\_comb)=\operatorname{split}(x\_hat,[N,N,N^2]),
     $$

     $$
     (bias\_pre,bias\_post,bias\_res)=\operatorname{split}(mix\_bases,[N,N,N^2]).
     $$

     MhcPre算子的Pre、Post和Res三个分支的系数生成阶段计算如下：

     $$
     \begin{aligned}
     new\_prev\_mix_{b,s,n,0} &= \operatorname{sigmoid}(mix\_scales_0\cdot x\_pre_{b,s,n}+bias\_pre_n)+hc\_pre\_eps, \\
     new\_post\_mix_{b,s,n,0} &= hc\_post\_scale\cdot\operatorname{sigmoid}(mix\_scales_1\cdot x\_post_{b,s,n}+bias\_post_n), \\
     x\_comb &= \operatorname{reshape}(x\_comb,[B,S,N,N]), \\
     bias\_res &= \operatorname{reshape}(bias\_res,[N,N]), \\
     comb\_logits_{b,s,i,j} &= mix\_scales_2\cdot x\_comb_{b,s,i,j}+bias\_res_{i,j}.
     \end{aligned}
     $$

     公式中的new_post_mix是MhcPre算子的Post分支系数生成阶段的新系数，comb_logits是MhcPre算子的Res分支系数生成阶段的新logits，均不覆盖输入post_mix、comb_res_mix。bias_res由长度为N²的一维偏置按行优先顺序还原为[N,N]，再与MhcPre算子的Res分支投影阶段的结果逐元素相加。

     根据MhcPre算子的Pre分支系数生成阶段的系数对完整`x_2`中的各路残差加权合并：

     $$
     \begin{aligned}
     w &= \begin{cases}new\_prev\_mix, & shifted\_prev\_mix=\mathrm{None},\\shifted\_prev\_mix, & shifted\_prev\_mix\neq\mathrm{None},\end{cases} \\
     h\_in_{b,s,d} &= \operatorname{Cast}_{dtype(x)}\left(\sum_{n=0}^{N-1}w_{b,s,n,0}\cdot\operatorname{FP32}(x\_2_{b,s,n,d})\right).
     \end{aligned}
     $$

     MhcPre算子的Res分支Sinkhorn归一化阶段（预留功能）

     令$L=num\_sinkhorn\_iters$。$L=0$时直接返回MhcPre算子的Res分支系数生成阶段的logits，不执行指数变换或归一化：

     $$
     new\_comb\_res\_mix=comb\_logits.
     $$

     $L>0$时，对每个token的$N\times N$矩阵独立执行以下计算。第一次迭代先沿最后一维执行softmax，再执行列归一化：

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

     系数下标`[b,s,n,0]`中的最后一维对应单个混合系数，公式中的None表示未传入对应可选Tensor。

## 函数原型

```python
cann_ops_transformer.mega_mhc(
    x,
    residual,
    post_mix,
    fn,
    mix_scales,
    mix_bases,
    rms_norm_1_weight,
    *,
    comb_res_mix=None,
    shifted_prev_mix=None,
    rms_norm_0_weight=None,
    rms_norm_mid_weight=None,
    rms_norm_eps=1e-6,
    hc_norm_eps=1e-6,
    hc_pre_eps=1e-6,
    inner_precise=0,
    hc_mult=4,
    hc_post_scale=2.0,
    sinkhorn_eps=1e-6,
    num_sinkhorn_iters=0,
    rms_norm_scale=1.0,
    output_mode=0,
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]
```

## 参数说明

$N=hc\_mult$，当前为4；$Q=N(N+2)=24$，$K=N\cdot D$。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x | Tensor | 必选 | Attention/MLP输出，对应公式中的x。 | float16、bfloat16 | `[B,S,D]`或`[T,D]` |
| residual | Tensor | 必选 | MhcPost算子的残差映射阶段的输入，对应公式中的residual。 | float16、bfloat16 | `[B,S,N,D]`或`[T,N,D]` |
| post_mix | Tensor | 必选 | MhcPost算子的输出融合阶段的输入系数，对应公式中的post_mix。 | float32 | `[B,S,N,1]`或`[T,N,1]` |
| fn | Tensor | 必选 | MhcPre算子的投影阶段权重，对应公式中的fn；转置后参与矩阵乘，分量顺序为Pre、Post、Res。 | float32 | `[Q,K]` |
| mix_scales | Tensor | 必选 | MhcPre算子的投影阶段的三路缩放系数，对应公式中的mix_scales_0、mix_scales_1、mix_scales_2。 | float32 | `[3]` |
| mix_bases | Tensor | 必选 | MhcPre算子的Pre、Post和Res分支的系数生成阶段的偏置，对应公式中的mix_bases，按[N,N,N²]切分为bias_pre、bias_post、bias_res。 | float32 | `[Q]` |
| rms_norm_1_weight | Tensor | 必选 | RmsNorm算子的rms_norm_1阶段的逐通道权重，对应公式中的rms_norm_1_weight。 | float32 | `[D]` |
| comb_res_mix | Optional[Tensor] | 可选 | MhcPost算子的残差映射阶段的变换矩阵，对应公式中的comb_res_mix；接口默认值为None，本版必须非None。 | float32 | `[B,S,N,N]`或`[T,N,N]` |
| shifted_prev_mix | Optional[Tensor] | 可选 | 预留参数，默认值为None，本版仅支持None。 | float32 | `[B,S,N,1]`或`[T,N,1]` |
| rms_norm_0_weight | Optional[Tensor] | 可选 | RmsNorm算子的rms_norm_0阶段的逐通道权重，对应公式中的rms_norm_0_weight；接口默认值为None，此时跳过rms_norm_0阶段，直接将x送入MhcPost。 | float32 | `[D]` |
| rms_norm_mid_weight | Optional[Tensor] | 可选 | 默认值为None，RmsNorm算子的rms_norm_mid阶段的逐通道权重，对应公式中的rms_norm_mid_weight；None时跳过本阶段。 | float32 | `[K]` |
| rms_norm_eps | float | 可选 | 默认1e-6，对应公式中的rms_norm_eps，用于RmsNorm算子的rms_norm_0、rms_norm_mid和rms_norm_1阶段。转为float32后须有限且大于0。 | - | - |
| hc_norm_eps | float | 可选 | 默认1e-6，MhcPre算子的均方根归一化阶段的稳定项，对应公式中的hc_norm_eps。转为float32后须有限且大于0。 | - | - |
| hc_pre_eps | float | 可选 | 默认1e-6，对应公式中MhcPre算子的Pre分支Sigmoid阶段的输出附加项hc_pre_eps。转为float32后须有限且大于0。 | - | - |
| inner_precise | int | 可选 | 指定MhcPre算子的矩阵乘阶段的计算模式，作用于公式中的x_2_rs与proj_w相乘。默认值为0。0：使用float32计算；1：使用HF32计算。2（3BF16）为预留取值，本版不支持。 | - | - |
| hc_mult | int | 可选 | 默认值为4，对应公式中的N，须与residual的head维一致；本版仅支持4，其他取值为预留取值。 | - | - |
| hc_post_scale | float | 可选 | MhcPre算子的Post分支Sigmoid阶段的输出倍率，对应公式中的hc_post_scale；默认值为2.0，转换为float32后须有限。 | - | - |
| sinkhorn_eps | float | 可选 | 预留参数，默认值为1e-6，本版仅支持1e-6，不参与计算。 | - | - |
| num_sinkhorn_iters | int | 可选 | 预留参数，默认值为0，本版仅支持0。 | - | - |
| rms_norm_scale | float | 可选 | 预留参数，默认值为1.0，本版仅支持1.0。 | - | - |
| output_mode | int | 可选 | 指定输出模式，默认值为0。0：输出非量化结果；1：输出MX Quant量化结果；2：同时输出量化和非量化结果。不支持其他取值；1、2为预留取值，本版仅支持0。 | - | - |

## 返回值说明

返回`(new_residual, new_post_mix, new_comb_res_mix, new_prev_mix, y, y_quant, y_scale)`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| new_residual | Tensor | 必选 | MhcPost算子的输出融合阶段的残差状态；rms_norm_mid_weight非空时为RmsNorm算子的rms_norm_mid阶段的输出，对应公式中的new_residual，即x_2。 | 与x相同 | `[B,S,N,D]`或`[T,N,D]` |
| new_post_mix | Tensor | 必选 | MhcPre算子的Post分支系数生成阶段的新系数，对应公式中的new_post_mix。 | float32 | `[B,S,N,1]`或`[T,N,1]` |
| new_comb_res_mix | Tensor | 必选 | MhcPre算子的Res分支系数生成阶段的新系数，对应公式中的new_comb_res_mix；本版仅输出comb_logits，Sinkhorn归一化为预留功能。 | float32 | `[B,S,N,N]`或`[T,N,N]` |
| new_prev_mix | Optional[Tensor] | 可选 | 预留参数，本版返回None。 | float32 | `[B,S,N,1]`或`[T,N,1]` |
| y | Optional[Tensor] | 可选 | MhcPre算子的加权合并阶段与RmsNorm算子的rms_norm_1阶段计算并应用倍率后的原精度结果，对应公式中的y；本版必须输出。 | 与x相同 | `[B,S,D]`或`[T,D]` |
| y_quant | Optional[Tensor] | 可选 | 预留参数，本版返回None。 | float8_e4m3fn | - |
| y_scale | Optional[Tensor] | 可选 | 预留参数，本版返回None。 | float8_e8m0fnu | - |

## 约束说明

- y、y_quant至少要输出一个。
- 支持推理场景下的MHC前向计算，暂不支持反向梯度。
- 支持单算子模式调用，暂不支持TorchAir图模式调用。
- 输入均位于同一NPU，格式为ND，前导维采用一致的BS或T布局，不支持广播。
- T、B、S须为正整数，T或B*S不超过UINT32_MAX；本版不支持零元素Tensor。
- N=4，D的取值范围为[16,16384]，且为16的倍数。

## 确定性/Batch一致性

- 确定性说明：默认支持确定性计算。
- Batch一致性说明：默认不支持Batch一致性。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer import mega_mhc

  # 设置设备。
  torch_npu.npu.set_device(0)
  device = "npu:0"
  t, n, d = 1024, 4, 4096
  # 创建输入Tensor。
  x = torch.randn(t, d, device=device, dtype=torch.bfloat16)
  residual = torch.randn(t, n, d, device=device, dtype=x.dtype)
  post_mix = torch.randn(t, n, 1, device=device, dtype=torch.float32)
  comb_res_mix = torch.randn(t, n, n, device=device, dtype=torch.float32)
  fn = torch.randn(n * (n + 2), n * d, device=device, dtype=torch.float32)
  mix_scales = torch.ones(3, device=device, dtype=torch.float32)
  mix_bases = torch.zeros(n * (n + 2), device=device, dtype=torch.float32)
  rms_norm_1_weight = torch.ones(d, device=device, dtype=torch.float32)
  rms_norm_0_weight = torch.ones(d, device=device, dtype=torch.float32)

  # 调用MegaMhc算子。
  new_residual, new_post_mix, new_comb_res_mix, new_prev_mix, y, y_quant, y_scale = mega_mhc(
      x, residual, post_mix, fn, mix_scales, mix_bases, rms_norm_1_weight,
      comb_res_mix=comb_res_mix, shifted_prev_mix=None,
      rms_norm_0_weight=rms_norm_0_weight, rms_norm_mid_weight=None, hc_mult=n,
      rms_norm_eps=1e-6, hc_norm_eps=1e-6, hc_pre_eps=1e-6,
      inner_precise=0, num_sinkhorn_iters=0,
  )
  # 检查输出shape及可选输出的组合约束。
  assert new_prev_mix is None
  assert new_post_mix.shape == (t, n, 1)
  assert new_residual.shape == residual.shape
  assert y is not None
  assert y_quant is None and y_scale is None
  torch_npu.npu.synchronize()
  ```
