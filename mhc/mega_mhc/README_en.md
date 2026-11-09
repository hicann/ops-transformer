# MegaMhc

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function:
  MegaMhc fuses the computation of multi-stage RmsNorm, MhcPost, and MhcPre, supports an optional input RmsNorm and an optional intermediate RmsNorm, updates the multi-path residual states and Post/Res mixing coefficients, and merges and normalizes the residuals of all paths with weights into the input of the next sublayer. When external Pre mixing coefficients are supplied, it also outputs the Pre coefficients calculated by the current invocation.

- Computation formula:

  The following formulas describe the operator function stage by stage based on the computation functions and execution order of the sub-operators RmsNorm, MhcPost, and MhcPre in the fused operator. $B$ is the batch size, $S$ is the sequence length, $T=B\times S$ is the number of tokens after flattening, $N$ is the number of residual heads, and $D$ is the hidden dimension of each head. The BS layout is calculated using the following formulas. The T layout is equivalent to folding $B$ and $S$ into $T$. $\operatorname{FP32}$ indicates conversion to float32, and $\operatorname{Cast}_{dtype(x)}$ indicates conversion to the data type of input x.

  1. Optionally performs the RmsNorm operator function to perform root mean square normalization on input x, referred to as the rms_norm_0 stage. The following formula is executed only when `rms_norm_0_weight` is not empty. When it is empty, this stage is skipped and $x\_1=x$ is set directly.

     $$
     \begin{aligned}
     x\_fp32_{b,s,d} &= \operatorname{FP32}(x_{b,s,d}), \\
     inv\_rms\_0_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}x\_fp32_{b,s,i}^{2}+rms\_norm\_eps\right), \\
     x\_1_{b,s,d} &= \operatorname{Cast}_{dtype(x)}\left(x\_fp32_{b,s,d}\cdot inv\_rms\_0_{b,s}\cdot rms\_norm\_0\_weight_d\right).
     \end{aligned}
     $$

  2. Performs the MhcPost operator function to fuse `x_1` from the preceding stage with the mapped residual, referred to as the mhc_post stage.

     $$
     \begin{aligned}
     term1_{b,s,n,d} &= \operatorname{FP32}(post\_mix_{b,s,n,0})\cdot \operatorname{FP32}(x\_1_{b,s,d}), \\
     term2_{b,s,n,d} &= \sum_{j=0}^{N-1}\operatorname{FP32}(comb\_res\_mix_{b,s,j,n})\cdot \operatorname{FP32}(residual_{b,s,j,d}), \\
     x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(x)}\left(term1_{b,s,n,d}+term2_{b,s,n,d}\right).
     \end{aligned}
     $$

     $j$ is the reduction head, and $n$ is the output head. This version requires `comb_res_mix` to be non-empty.

  3. Optionally performs the RmsNorm operator function to perform root mean square normalization on the complete fused residual row, referred to as the rms_norm_mid stage.

     When `rms_norm_mid_weight` is not empty:

     $$
     \begin{aligned}
     x\_2\_flat &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
     inv\_rms\_mid_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_flat_{b,s,k}^{2}+rms\_norm\_eps\right), \\
     x\_2_{b,s,n,d} &= \operatorname{Cast}_{dtype(x)}\left(\operatorname{reshape}\left(x\_2\_flat\cdot inv\_rms\_mid\cdot rms\_norm\_mid\_weight,[B,S,N,D]\right)_{b,s,n,d}\right).
     \end{aligned}
     $$

     When `rms_norm_mid_weight` is empty, this stage is skipped, and `x_2` remains the output of the output fusion stage of the MhcPost operator. The following is saved whether or not this stage runs:

     $$
     new\_residual=x\_2.
     $$

  4. Performs the MhcPre operator function to calculate the mixing coefficients and merge the residuals of all paths with weights, referred to as the mhc_pre stage.

     $$
     \begin{aligned}
     x\_2\_rs &= \operatorname{reshape}(\operatorname{FP32}(x\_2),[B,S,N\cdot D]), \\
     inv\_rms\_pre_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{N D}\sum_{k=0}^{ND-1}x\_2\_rs_{b,s,k}^{2}+hc\_norm\_eps\right), \\
     proj\_w &= fn^{T}, \\
     x\_hat &= (x\_2\_rs\cdot proj\_w)\cdot inv\_rms\_pre.
     \end{aligned}
     $$

     The matrix multiplication $x\_2\_rs\cdot proj\_w$ is performed first, and the result is then multiplied by $inv\_rms\_pre$ to generate the current mixing coefficients. This RMS scaling does not modify `x_2`, which is used by the subsequent weighted merge. Split according to `[N,N,N*N]`:

     $$
     (x\_pre,x\_post,x\_comb)=\operatorname{split}(x\_hat,[N,N,N^2]),
     $$

     $$
     (bias\_pre,bias\_post,bias\_res)=\operatorname{split}(mix\_bases,[N,N,N^2]).
     $$

     The coefficient generation stages of the Pre, Post, and Res branches of the MhcPre operator are calculated as follows:

     $$
     \begin{aligned}
     pre\_mix_{b,s,n,0} &= \operatorname{sigmoid}(mix\_scales_0\cdot x\_pre_{b,s,n}+bias\_pre_n)+hc\_pre\_eps, \\
     new\_post\_mix_{b,s,n,0} &= hc\_post\_scale\cdot\operatorname{sigmoid}(mix\_scales_1\cdot x\_post_{b,s,n}+bias\_post_n), \\
     x\_comb &= \operatorname{reshape}(x\_comb,[B,S,N,N]), \\
     bias\_res &= \operatorname{reshape}(bias\_res,[N,N]), \\
     comb\_logits_{b,s,i,j} &= mix\_scales_2\cdot x\_comb_{b,s,i,j}+bias\_res_{i,j}.
     \end{aligned}
     $$

     `pre_mix` contains the mixing coefficients calculated by the current MhcPre invocation. When `shifted_prev_mix` is supplied, `new_prev_mix` returns the current `pre_mix`; otherwise, `new_prev_mix` is not output. This output does not return the external `shifted_prev_mix` values. new_post_mix in the formula is the new coefficient of the coefficient generation stage of the Post branch of the MhcPre operator, and comb_logits is the new logits of the coefficient generation stage of the Res branch of the MhcPre operator. Neither overwrites the input post_mix or comb_res_mix. bias_res is restored from the 1D bias of length N² to [N,N] in row-major order and then added element-wise to the result of the projection stage of the Res branch of the MhcPre operator.

     When `shifted_prev_mix` is empty, the current `pre_mix` is used. Otherwise, the supplied values replace `pre_mix`. The selected coefficients multiply `x_2`, and the results are summed along the head dimension:

     $$
     \begin{aligned}
     w &= \begin{cases}pre\_mix, & shifted\_prev\_mix=\mathrm{None},\\shifted\_prev\_mix, & shifted\_prev\_mix\neq\mathrm{None},\end{cases} \\
     h\_in_{b,s,d} &= \operatorname{Cast}_{dtype(x)}\left(\sum_{n=0}^{N-1}w_{b,s,n,0}\cdot\operatorname{FP32}(x\_2_{b,s,n,d})\right).
     \end{aligned}
     $$

     The supplied `shifted_prev_mix` values do not pass through the Pre branch's Sigmoid or `hc_pre_eps` addition. They replace only the `pre_mix` used by the weighted merge; the Post and Res branches still generate their coefficients. The merged `h_in` still passes through `rms_norm_1` in stage 5. The current `pre_mix` and external `shifted_prev_mix` occupy separate UB regions. External coefficients can be prefetched, and the current coefficient region can be overwritten only after its output transfer completes.

     Sinkhorn normalization stage of the Res branch of the MhcPre operator (reserved function)

     Let $L=num\_sinkhorn\_iters$. This version supports only $L=0$, which directly returns the logits of the coefficient generation stage of the Res branch of the MhcPre operator without exponential transformation or Sinkhorn normalization:

     $$
     new\_comb\_res\_mix=comb\_logits.
     $$

     The following computation for $L>0$ is reserved and is not supported in this version. It operates independently on the $N\times N$ matrix of each token. In the first iteration, softmax is performed along the last dimension first, and then column normalization is performed:

     $$
     \begin{aligned}
     A^{(0)}_{b,s,i,j} &= \operatorname{softmax}_{j}(comb\_logits_{b,s,i,j})+sinkhorn\_eps, \\
     H^{(0)}_{b,s,i,j} &= \frac{A^{(0)}_{b,s,i,j}}{\displaystyle\sum_{p=0}^{N-1}A^{(0)}_{b,s,p,j}+sinkhorn\_eps}.
     \end{aligned}
     $$

     Then, row normalization and column normalization are performed in sequence for $r=1,\ldots,L-1$:

     $$
     \begin{aligned}
     R^{(r)}_{b,s,i,j} &= \frac{H^{(r-1)}_{b,s,i,j}}{\displaystyle\sum_{q=0}^{N-1}H^{(r-1)}_{b,s,i,q}+sinkhorn\_eps}, \\
     H^{(r)}_{b,s,i,j} &= \frac{R^{(r)}_{b,s,i,j}}{\displaystyle\sum_{p=0}^{N-1}R^{(r)}_{b,s,p,j}+sinkhorn\_eps}, \\
     new\_comb\_res\_mix_{b,s,i,j} &= H^{(L-1)}_{b,s,i,j}.
     \end{aligned}
     $$

     Here, $i$ is the row index and $j$ is the column index. When $L=1$, $H^{(0)}$ is output directly. `sinkhorn_eps` is used only for the preceding Sinkhorn computation and is independent of `hc_pre_eps` in the Sigmoid stage of the Pre branch of the MhcPre operator and `hc_norm_eps` in the root mean square normalization stage.

  5. Performs the RmsNorm operator function to perform root mean square normalization on the weighted merging result, referred to as the rms_norm_1 stage.

     $$
     \begin{aligned}
     h\_in\_rms_{b,s} &= \operatorname{rsqrt}\left(\frac{1}{D}\sum_{i=0}^{D-1}\operatorname{FP32}(h\_in_{b,s,i})^{2}+rms\_norm\_eps\right), \\
     h\_in\_f32_{b,s,d} &= \operatorname{FP32}(h\_in_{b,s,d})\cdot h\_in\_rms_{b,s}\cdot \operatorname{FP32}(rms\_norm\_1\_weight_d)\cdot rms\_norm\_scale, \\
     y &= \operatorname{Cast}_{dtype(x)}(h\_in\_f32).
     \end{aligned}
     $$

     The last dimension in the coefficient subscript `[b,s,n,0]` corresponds to a single mixing coefficient. None in the formula indicates that the corresponding optional tensor is not passed in.

## Parameter Description

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| x | Input | Attention/MLP output, corresponding to x in the formula. The shape is [B,S,D] or [T,D]. | FLOAT16, BFLOAT16 | ND |
| residual | Input | Multi-head residual input, corresponding to residual in the formula. The shape is [B,S,N,D] or [T,N,D]. | Same as x | ND |
| post_mix | Input | Input coefficient of the output fusion stage of the MhcPost operator, corresponding to post_mix in the formula. The shape is [B,S,N,1] or [T,N,1]. | FLOAT32 | ND |
| fn | Input | Weight of the projection stage of the MhcPre operator, corresponding to fn in the formula. The component order is Pre, Post, and Res. The shape is [Q,K], where Q=N*(N+2) and K=N*D. | FLOAT32 | ND |
| mix_scales | Input | Three-path scaling coefficients of the coefficient generation stage of the MhcPre operator, corresponding to mix_scales_0, mix_scales_1, and mix_scales_2 in the formula in sequence. The shape is [3]. | FLOAT32 | ND |
| mix_bases | Input | Three-path biases of the coefficient generation stage of the MhcPre operator, corresponding to mix_bases in the formula, split into bias_pre, bias_post, and bias_res according to [N,N,N²]. The shape is [Q]. | FLOAT32 | ND |
| rms_norm_1_weight | Input | Per-channel weight of the rms_norm_1 stage of the RmsNorm operator, corresponding to rms_norm_1_weight in the formula. The shape is [D]. | FLOAT32 | ND |
| comb_res_mix | Optional input | Transformation matrix of the residual mapping stage of the MhcPost operator, corresponding to comb_res_mix in the formula. The default value is empty, which must not be empty in this version. The shape is [B,S,N,N] or [T,N,N]. | FLOAT32 | ND |
| shifted_prev_mix | Optional input | External mixing coefficients for the weighted merge in MhcPre, with the same shape as `post_mix`. When empty, the current `pre_mix` is used. Otherwise, the supplied values replace `pre_mix`, multiply `x_2`, and are summed along the head dimension. The subsequent `rms_norm_1` stage still runs. | FLOAT32 | ND |
| rms_norm_0_weight | Optional input | Per-channel weight of the rms_norm_0 stage of the RmsNorm operator, corresponding to rms_norm_0_weight in the formula. The shape is [D]. The default value is empty. In this case, the rms_norm_0 stage is skipped, and x is sent directly to MhcPost. | FLOAT32 | ND |
| rms_norm_mid_weight | Optional input | The default value is empty, which skips the rms_norm_mid stage of the RmsNorm operator. When the value is not empty, it is the per-channel weight of the rms_norm_mid stage of the RmsNorm operator, corresponding to rms_norm_mid_weight in the formula. The shape is [K]. | FLOAT32 | ND |
| new_residual | Output | Updated residual state, corresponding to new_residual in the formula. When the rms_norm_mid stage is not performed, it is the output of the output fusion stage of the MhcPost operator. When the stage is performed, it is the output of the rms_norm_mid stage of the RmsNorm operator. The shape is [B,S,N,D] or [T,N,D]. | Same as x | ND |
| new_post_mix | Output | New coefficient of the coefficient generation stage of the Post branch of the MhcPre operator, corresponding to new_post_mix in the formula. The shape is [B,S,N,1] or [T,N,1]. | FLOAT32 | ND |
| new_comb_res_mix | Output | New coefficient of the coefficient generation stage of the Res branch of the MhcPre operator, corresponding to new_comb_res_mix in the formula. This version outputs only comb_logits, and Sinkhorn normalization is a reserved function. The shape is [B,S,N,N] or [T,N,N]. | FLOAT32 | ND |
| new_prev_mix | Optional output | Required when `shifted_prev_mix` is supplied; returns the current `pre_mix` with the same shape as `post_mix`. When the input is empty, this output is absent. The current and external coefficients occupy separate UB regions and are used for output and weighted merging, respectively. | FLOAT32 | ND |
| y | Optional output | Original-precision result of the weighted merging stage of MhcPre followed by the rms_norm_1 stage of RmsNorm, corresponding to y in the formula. The shape is [B,S,D] or [T,D]. Although optional in the interface prototype, this output is required in this version. | Same as x | ND |
| y_quant | Optional output | Reserved parameter. It is not specified by default and is not output in this version. | FLOAT8_E4M3FN | ND |
| y_scale | Optional output | Reserved parameter. It is not specified by default and is not output in this version. | FLOAT8_E8M0 | ND |
| rms_norm_eps | Optional attribute | Default value: 1e-6. It applies to the rms_norm_0, rms_norm_mid, and rms_norm_1 stages of the RmsNorm operator, corresponding to rms_norm_eps in the formula. After conversion to FLOAT32, the value must be finite and greater than 0. | FLOAT | - |
| hc_norm_eps | Optional attribute | Default value: 1e-6. Used to calculate the RMS of the complete N*D row of `x_2` and scale the projection result, corresponding to hc_norm_eps in the formula. After conversion to FLOAT32, the value must be finite and greater than 0. | FLOAT | - |
| hc_pre_eps | Optional attribute | Default value: 1e-6. It is the additional output term of the Sigmoid stage of the Pre branch of the MhcPre operator, corresponding to hc_pre_eps in the formula. After conversion to FLOAT32, the value must be finite and greater than 0. | FLOAT | - |
| op_impl_mode | Optional attribute | Default value: 0. It selects the computation mode of the matrix multiplication stage of the MhcPre operator: 0 indicates FLOAT32 computation, and 1 indicates HF32 computation. 2 (3BF16) is a reserved value and is not supported in this version. | INT | - |
| hc_mult | Optional attribute | Default value: 4. It corresponds to N in the formula and must be consistent with the head dimension of residual. Only 4 is supported in this version, and other values are reserved. | INT | - |
| hc_post_scale | Optional attribute | Output scale factor of the Sigmoid stage of the Post branch of the MhcPre operator, corresponding to hc_post_scale in the formula. Default value: 2.0. After conversion to FLOAT32, the value must be finite. | FLOAT | - |
| sinkhorn_eps | Optional attribute | Reserved parameter. Default value: 1e-6. Only 1e-6 is supported in this version, and it does not participate in computation. | FLOAT | - |
| num_sinkhorn_iters | Optional attribute | Reserved parameter. Default value: 0. Only 0 is supported in this version. | INT | - |
| rms_norm_scale | Optional attribute | Reserved parameter. Default value: 1.0. Only 1.0 is supported in this version. | FLOAT | - |
| output_mode | Optional attribute | Specifies the output mode. Default value: 0. 0: outputs the non-quantized result. 1: outputs the MX Quant quantized result. 2: outputs both the quantized and non-quantized results. Other values are not supported. 1 and 2 are reserved values, and only 0 is supported in this version. | INT | - |

## Constraints

- Currently, N=4, Q=24, and K=4D. The value range of D is [16,16384], and D must be a multiple of 16.
- Currently, only forward computation is provided, and backward gradients are not provided.

## Invocation Description

| Invocation Method | Sample Code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_mega_mhc](examples/arch35/test_aclnn_mega_mhc.cpp) | Invokes the MegaMhc operator through the [aclnnMegaMhc](docs/aclnnMegaMhc.md) API. |
| PyTorch API | - | Invokes the MegaMhc operator through the [cann_ops_transformer.mega_mhc](docs/torchapi_mega_mhc.md) API. |
