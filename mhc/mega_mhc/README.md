# MegaMhc

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：
  MegaMhc融合多阶段RmsNorm、MhcPost和MhcPre计算，支持可选的输入RmsNorm和中间RmsNorm，更新多路残差状态与Post/Res混合系数，并将各路残差加权合并、归一化为下一子层的输入。传入外部Pre混合系数时，还输出本次计算的Pre系数。

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
     w &= \begin{cases}pre\_mix, & shifted\_prev\_mix=\mathrm{None},\\shifted\_prev\_mix, & shifted\_prev\_mix\neq\mathrm{None},\end{cases} \\
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

     系数下标`[b,s,n,0]`中的最后一维对应单个混合系数，公式中的None表示未传入对应可选Tensor。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x | 输入 | Attention/MLP输出，对应公式中的x；shape为[B,S,D]或[T,D]。 | FLOAT16、BFLOAT16 | ND |
| residual | 输入 | 多head残差输入，对应公式中的residual；shape为[B,S,N,D]或[T,N,D]。 | 与x相同 | ND |
| post_mix | 输入 | MhcPost算子的输出融合阶段的输入系数，对应公式中的post_mix；shape为[B,S,N,1]或[T,N,1]。 | FLOAT32 | ND |
| fn | 输入 | MhcPre算子的投影阶段权重，对应公式中的fn，分量顺序为Pre、Post、Res；shape为[Q,K]，Q=N*(N+2)、K=N*D。 | FLOAT32 | ND |
| mix_scales | 输入 | MhcPre算子的系数生成阶段的三路缩放系数，依次对应公式中的mix_scales_0、mix_scales_1、mix_scales_2；shape为[3]。 | FLOAT32 | ND |
| mix_bases | 输入 | MhcPre算子的系数生成阶段的三路偏置，对应公式中的mix_bases，按[N,N,N²]切分为bias_pre、bias_post、bias_res；shape为[Q]。 | FLOAT32 | ND |
| rms_norm_1_weight | 输入 | RmsNorm算子的rms_norm_1阶段的逐通道权重，对应公式中的rms_norm_1_weight，shape为[D]。 | FLOAT32 | ND |
| comb_res_mix | 可选输入 | MhcPost算子的残差映射阶段的变换矩阵，对应公式中的comb_res_mix；默认值为空，本版必须非空。shape为[B,S,N,N]或[T,N,N]。 | FLOAT32 | ND |
| shifted_prev_mix | 可选输入 | MhcPre加权合并阶段的外部混合系数，形状与`post_mix`相同；为空时使用本次计算的`pre_mix`，非空时使用传入值替代`pre_mix`，与`x_2`相乘并沿head维求和。后续`rms_norm_1`保持执行。 | FLOAT32 | ND |
| rms_norm_0_weight | 可选输入 | RmsNorm算子的rms_norm_0阶段的逐通道权重，对应公式中的rms_norm_0_weight，shape为[D]；默认值为空，此时跳过rms_norm_0阶段，直接将x送入MhcPost。 | FLOAT32 | ND |
| rms_norm_mid_weight | 可选输入 | 默认值为空，跳过RmsNorm算子的rms_norm_mid阶段；非空时为RmsNorm算子的rms_norm_mid阶段的逐通道权重，对应公式中的rms_norm_mid_weight，shape为[K]。 | FLOAT32 | ND |
| new_residual | 输出 | 更新后的残差状态，对应公式中的new_residual；未执行rms_norm_mid阶段时为MhcPost算子的输出融合阶段的输出，执行时为RmsNorm算子的rms_norm_mid阶段的输出，shape为[B,S,N,D]或[T,N,D]。 | 与x相同 | ND |
| new_post_mix | 输出 | MhcPre算子的Post分支系数生成阶段的新系数，对应公式中的new_post_mix；shape为[B,S,N,1]或[T,N,1]。 | FLOAT32 | ND |
| new_comb_res_mix | 输出 | MhcPre算子的Res分支系数生成阶段的新系数，对应公式中的new_comb_res_mix；本版仅输出comb_logits，Sinkhorn归一化为预留功能。shape为[B,S,N,N]或[T,N,N]。 | FLOAT32 | ND |
| new_prev_mix | 可选输出 | `shifted_prev_mix`非空时必须输出本次计算的`pre_mix`，形状与`post_mix`相同；为空时不输出。本次系数与外部系数使用独立UB区域，并分别写出和参与加权合并。 | FLOAT32 | ND |
| y | 可选输出 | MhcPre算子的加权合并阶段与RmsNorm算子的rms_norm_1阶段计算后的原精度结果，对应公式中的y；shape为[B,S,D]或[T,D]。接口原型声明为可选，本版必须输出。 | 与x相同 | ND |
| y_quant | 可选输出 | 预留参数，默认不指定，本版不输出。 | FLOAT8_E4M3FN | ND |
| y_scale | 可选输出 | 预留参数，默认不指定，本版不输出。 | FLOAT8_E8M0 | ND |
| rms_norm_eps | 可选属性 | 默认值为1e-6，作用于RmsNorm算子的rms_norm_0、rms_norm_mid和rms_norm_1阶段，对应公式中的rms_norm_eps；转换为FLOAT32后须有限且大于0。 | FLOAT | - |
| hc_norm_eps | 可选属性 | 默认值为1e-6，用于计算`x_2`完整N*D行的均方根并缩放投影结果，对应公式中的hc_norm_eps；转换为FLOAT32后须有限且大于0。 | FLOAT | - |
| hc_pre_eps | 可选属性 | 默认值为1e-6，MhcPre算子的Pre分支Sigmoid阶段的输出附加项，对应公式中的hc_pre_eps；转换为FLOAT32后须有限且大于0。 | FLOAT | - |
| op_impl_mode | 可选属性 | 默认值为0，选择MhcPre算子的矩阵乘阶段的计算模式：0表示FLOAT32计算，1表示HF32计算；2（3BF16）为预留取值，本版不支持。 | INT | - |
| hc_mult | 可选属性 | 默认值为4，对应公式中的N，须与residual的head维一致；本版仅支持4，其他取值为预留取值。 | INT | - |
| hc_post_scale | 可选属性 | MhcPre算子的Post分支Sigmoid阶段的输出倍率，对应公式中的hc_post_scale；默认值为2.0，转换为FLOAT32后须有限。 | FLOAT | - |
| sinkhorn_eps | 可选属性 | 预留参数，默认值为1e-6，本版仅支持1e-6，不参与计算。 | FLOAT | - |
| num_sinkhorn_iters | 可选属性 | 预留参数，默认值为0，本版仅支持0。 | INT | - |
| rms_norm_scale | 可选属性 | 预留参数，默认值为1.0，本版仅支持1.0。 | FLOAT | - |
| output_mode | 可选属性 | 指定输出模式，默认值为0。0：输出非量化结果；1：输出MX Quant量化结果；2：同时输出量化和非量化结果。不支持其他取值；1、2为预留取值，本版仅支持0。 | INT | - |

## 约束说明

- 当前N=4，Q=24，K=4D；D的取值范围为[16,16384]，且为16的倍数。
- 当前只提供前向计算，不提供反向梯度。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | [test_aclnn_mega_mhc](examples/arch35/test_aclnn_mega_mhc.cpp) | 通过[aclnnMegaMhc](docs/aclnnMegaMhc.md)接口调用MegaMhc算子。 |
| PyTorch API | - | 通过[cann_ops_transformer.mega_mhc](docs/torchapi_mega_mhc.md)接口调用MegaMhc算子。 |
