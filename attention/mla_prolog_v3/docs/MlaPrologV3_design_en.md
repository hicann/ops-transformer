
# MlaPrologV3 Operator Design Description

In recent years, with the rapid development of artificial intelligence technology, large models have made remarkable breakthroughs in natural language processing, computer vision, multimodal interaction, and other fields. However, the complexity and scale of models grow exponentially, which places higher requirements on the underlying computing framework and hardware computing power.
When handling ultra-large-scale parameter models, the traditional matrix operations (such as GEMM) and attention mechanism implementations gradually expose problems such as low computing efficiency, high memory usage, and insufficient parallel scalability, which restrict the real-time performance and resource utilization of model training and inference.

Against this background, the DeepSeek team proposed the Mla operator requirement, which requires collaborative optimization of algorithms and hardware. Through operator fusion, the number of operator calls and overhead time are reduced. While ensuring model accuracy, the inference latency is reduced and hardware resource consumption is decreased. The MlaPrologV3 operator is exactly the fusion part of the Mla operator pre-processing.

## Implementation Principle

Figure 1 Computation flow diagram:

![MlaProlog computation flow diagram](../../../docs/zh/figures/mla_prolog_compute_flow.png)

The implementation follows the computation flow defined by Multi-Head Latent Attention. The overall computation flow is as follows:

1. The input sequence x is multiplied by the downsampling matrix $W^{DQ}$ to perform the rank-reduction transformation, and then normalized to obtain $c^Q$

2. The $W^{UQ}$ and $W^{QR}$ matrices are concatenated to perform rank elevation and mapping computation on $c^Q$. The computation result is then split. Qn normalization and Rope position embedding are performed separately, and Query and Query Rope are finally obtained

3. The $W^{DKV}$ and $W^{KR}$ matrices are concatenated to perform rank reduction and mapping computation on the input sequence x. The computation result is then split. Rope position embedding and normalization of $C^{KV}$ are performed separately, and KR Cache and KV Cache are finally obtained

4. When the output Query is quantized, Rowmax dynamic quantization is performed on the Query, and the quantized Query and the corresponding quantization parameter DequantScaleQNope are finally obtained

5. When queryNormFlag=true, the input $x$ is multiplied by $W^{DQ}$ for downsampling and RmsNorm to obtain the compressed result $c^Q$, which is output as queryNorm

6. When queryNormFlag=true in a quantization scenario (weightQuantMode≠0), dynamic quantization is performed on $c^Q$, and the quantized queryNorm and the corresponding quantization parameter dequantScaleQNorm are finally obtained

For the detailed calculation formulas, see the [Complete Calculation Formulas](#complete-calculation-formulas) section.

## Data Splitting Design

The hardware buffer size is limited, but the amount of computed data is huge, so the data cannot be computed at one time. Tiling is therefore required. Different shapes lead to different splitting axes of the operator, and the splitting axis of the operator affects the functions and performance of the template. Consider the following points:

a. Use all cores to prevent some cores from being idle.

b. The computation assigned to each core is relatively even, to avoid the situation where some cores process too much data while other cores are idle.

c. The amount of data processed by AIC and AIV must match their respective computing power, to prevent AIC or AIV from being idle for a long time.

The MlaPrologV3 operator has multiple Matmul operations:

1. MatmulCq: BS axes are merged as the M axis, HeadSizeCq is the N axis, and HeadSizeX is the K axis. HeadSizeCq is split among cores according to the number of cube cores, and each core is assigned a certain sub-block

2. MatmulCkvKr: BS axes are merged as the M axis, HeadSizeCkvKr is the N axis, and HeadSizeX is the K axis. HeadSizeCkvKr is split among cores into 64 blocks

3. MatmulQcQr: BS axes are merged as the M axis, HeadSizeQcQr is the N axis, and HeadSizeCq is the K axis. HeadSizeQcQr is split among cores according to the number of cube cores, and each core is assigned a certain sub-block

4. MatmulQn: BS axes are merged as the M axis, HeadSizeCkv is the N axis, and dimHeadSizeQc is the K axis

## Pipeline Design

The MlaPrologV3 fused operator contains Vector computation and Cube computation. The computations on the Vector side and the Cube side have dependencies, so CV pipeline design is required.
Otherwise, the C side and the V side are likely to execute in a serial pipeline, which fails to achieve parallel computation and prevents the fused operator from achieving optimal performance:

Figure 2 Pipeline control diagram:

![MlaProlog pipeline control diagram](../../../docs/zh/figures/mla_prolog_pipeline_control.png)

1. During MatmulCq computation, Hcq is split among cores, and a single core does not compute a complete token. Therefore, synchronization control between AIC and AIV (SYNC_MMCQ_NOMRCQ) is required before RmsNormCq computation.

2. During MatmulCkvKr computation, Hcq is also split among cores, and a single core does not compute a complete token. Therefore, synchronization control between AIC and AIV (SYNC_MMCKVKR_NORMCKV) is also required before RmsNormCkv computation.

3. RmsNormCq is split by rows onto different cores. Therefore, synchronization control between AIV and AIC (SYNC_NOMRCQ_MMQCQR) is also required during MatmulQcQr computation.

4. The core splitting strategy differs from that of RopeQr. Therefore, synchronization control between AIC and AIV (SYNC_MMQCQR_ROPEQR) is also required before RopeQr computation.

5. The core splitting strategies of MatmulQcQr and MatmulQn differ, and MatmulQn depends on the output of MatmulQcQr. Therefore, synchronization of all Cube cores (SYNC_ALL_CUBE) is required.

6. Before Vector core computation, synchronization of all Vector cores (SYNC_ALL_VECTOR) is required to ensure pipelined data movement.

## TilingKey Division

TilingKey is of the uint64 type. Each template parameter corresponds to one or more binary bits in TilingKey. The details are as follows:

|Binary Bits|Variable Name|Description|Parameter List|
|-------|------|----|-------|
|0-3|CACHE_MODE|Storage format of KVCache|0-ND (BSND/TND), 1-PA_BSND, 2-PA_NZ, 3-PA_BLK_BSND, 4-PA_BLK_NZ|
|4-5|SCENARIO|Input scenario|0-FP16 (reserved), 1-BF16, 2-quantization scenario|
|6-11|QUANT_MODE|Quantization scenario|0-non-quantization, 1-MMQcQr quantization, 2-MMQcQr quantization+KVcache quantization, 3-MMcqCkvKr quantization+MMQcQr quantization, 4-MMCqCkvkr quantization+MMQcQr quantization+KVcache quantization, 5-MMQcQr quantization+KVcache pertoken-pergroup quantization, 6-MMCqCkvkr quantization+MMQcQr quantization+KVcache pertoken-pergroup quantization, 7-Mxfp8 quantization+MMCqCkvkr quantization+MMQcQr quantization, 8-Mxfp8 quantization+MMCqCkvkr quantization+MMQcQr quantization+KVcache quantization, 9-Mxfp8 quantization+MMCqCkvkr quantization+MMQcQr quantization+KVcache pertoken-pergroup quantization, 10-fp8 quantization+MMcqCkvKr quantization+MMQcQr quantization, 11-fp8 quantization+MMCqCkvkr quantization+MMQcQr quantization+KVcache quantization, 12-hif8 quantization+MMcqCkvKr quantization+MMQcQr quantization, 13-hif8 quantization+MMCqCkvkr quantization+MMQcQr quantization+KVcache quantization, 14-fp8 quantization+MMcqCkvKr quantization+MMQcQr quantization+KVcache pertoken-pergroup quantization, 15-hif8 quantization+MMCqCkvkr quantization+MMQcQr quantization+KVcache pertoken-pergroup quantization|
|12|ENABLE_DEQUANT_OPTIONAL|Enables dequantization. It cannot be used together with ENABLE_GROUP_COMPUTE_OPTIONAL|0-disabled, 1-enabled|
|13|ENABLE_GROUP_COMPUTE_OPTIONAL|Computing power grouping optimization for quantization. It cannot be used together with ENABLE_DEQUANT_OPTIONAL|0-disabled, 1-enabled|
|14-15|EMPTY_TENSOR_MODE|Empty tensor scenario, used when the input tensor dimension is 0|0-no empty tensor, 1-kv_cache/kr_cache is empty, 2-query is empty and the cache is not updated|
|16-17|ACTUAL_SEQ_LEN_MODE|Scenario where actualSeqLen is enabled|0-disabled, 1-actualSeqLen enabled|
|18-19|SPLIT_M_MODE|M splitting mode|0-disabled (split N), 1-enabled (split M)|
|20|ENABLE_ROPE|RoPE switch. Only V4 can be optionally disabled (pass-through output when doRope=false). V1/V2/V3 can only be enabled during compilation|0-disabled (passthrough), 1-enabled|
|21-29|CV_MODE|CV core splitting mode|ASCENDC_TPL_MIX_AIC_1_1(6): 1:1 mode, ASCENDC_TPL_MIX_AIC_1_2(7): 1:2 mode|

## Main Process

```c
// Pseudocode for single-core computation
void Process() {
    loops = compute_step_this_core(); // Compute the loop count of the current core

    for (i = 0; i < loops; i++) {
        updateCurrentStepSize(i, stepSize, allTokenSize);  // Update the M-axis size computed in the current round
        if ASCEND_IS_AIC {
            MatmulCq(tokenXOffset, weightDqOffset, cqResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MMCQ_NORMSCQ_FLG);       // Synchronize Cube and Vector

            MatmulCkvKr(tokenXOffset, weightDkvKrOffset, ckvKrResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MMCKVKR_NORMROPE_FLG);   // Synchronize Cube and Vector

            CrossCoreWaitFlag(SYNC_MMCQ_NORMSCQ_FLG);                     // MatmulQcQr depends on the output of RmsNormCq, so CV inter-core synchronization needs to be inserted

            MatmulQcQr(weightUqQrOffset, qcQrResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MMQCQR_ROPEQR_FLG);      // Synchronize Cube and Vector

            // The core splitting strategies of MatmulQn and MatmulQcQr differ, and MatmulQn depends on the output of MatmulQcQr
            // MatmulQn can start only after MatmulQcQr on all Cube cores completes
            CrossCoreSetFlag<0x0, PIPE_FIX>(SYNC_ALL_CUBE_FLG);
            CrossCoreWaitFlag(SYNC_ALL_CUBE_FLG);

            MatmulQn(qcOffset, weightUkOffset, qnResOffset, mmQnLoopTime);  // The result of MatmulQn is directly output to queryOut, and qnOffset needs to be offset by the Batch axis
        }

        if ASCEND_IS_AIV {
            GetSinCos(tokenIndex);

            CrossCoreWaitFlag(SYNC_MMCQ_NORMSCQ_FLG);  // wait MatmulCq
            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);
            CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);

            RmsNormCq(rmsNormCqOffset);

            // The core splitting strategies of RmsNormCq and MatmulQcQr differ. MatmulQcQr can start only after RmsNormCq on all Vector cores completes
            // The execution of MatmulQcQr can be initiated only after RmsNormCq on all Vector cores completes
            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);
            CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);

            CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_MMCQ_NORMSCQ_FLG);        // Ensure that MatmulQcQr waits for RmsNormCq
            CrossCoreWaitFlag(SYNC_MMCKVKR_NORMROPE_FLG);                   // wait MatmulCkvKr
            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);
            CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);

            RmsNormRopeScatterCkvKr(tokenIndex, rmsNormCkvOffset, ropeKrOffset);
            CrossCoreWaitFlag(SYNC_MMQCQR_ROPEQR_FLG);  // wait MatmulQcQr

            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);
            CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);
            RopeQr(ropeQrOffset, ropeQrResOffset);
        }
    }
}
```

## Complete Calculation Formulas

Combined with the flow diagrams in the preceding sections, the complete calculation formulas of the entire MlaPrologV3 are as follows:

$$
q^N = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ}) \cdot W^{UQ} \cdot W^{UK} \tag{a}
$$

$$
q^R = \mathrm{ROPE}(\alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ}) \cdot W^{QR}) \tag{b}
$$

$$
k^R = \mathrm{ROPE}(x \cdot W^{KR}) \tag{c}
$$

$$
c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV}) \tag{d}
$$

The complete computation flow can be decomposed into the following basic computation units.
Note that $\alpha_q$ and $\alpha_{kv}$ are the scale correction factors of query and key respectively, proposed in the paper [Meituan](https://arxiv.org/abs/2509.01322).
$\alpha_q$ and $\alpha_{kv}$ correspond to qcQrScale and kcScale in the API documentation respectively.

### MatmulCq

The input $x$ is multiplied by the Query downsampling matrix $W^{DQ}$ to perform the downsampling operation, obtaining the compressed Query matrix $c^Q$.
$$
c^Q = x \cdot W^{DQ} \tag{1}
$$

The matrix multiplication modules involved in this section (and subsequent sections) are implemented using the Matmul high-level API in the AscendC Kernel API. For the usage of related APIs, see the development guide [Operator Implementation -> Matrix Programming (High-Level API)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0041.html) on the official website.

### RMSNormCq

RMSNorm (root mean square normalization) is performed row by row on the compressed $Q$ matrix. The RMSNorm operation requires two hyperparameters $\gamma$ and $\epsilon$, which correspond to rmsnormGammaCq and rmsnormEpsilonCq in the API documentation.
$$
c_{\mathrm{norm}}^Q = \mathrm{RmsNorm}(c^Q) \tag{2}
$$
The calculation formula of $\mathrm{RmsNorm}$ is as follows:
$$
\mathrm{RmsNorm}(x) = \gamma \cdot \frac{x_i}{RMS(x)}  \tag{3}
$$
$$
RMS(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon} \tag{4}
$$

### MatmulCkvKr

The input $x$ is multiplied by the Key/Value downsampling matrix (Key and Value share the same downsampling matrix) to obtain the compressed Key/Value matrix $c^{KV}$.
$$
c^{KV} = x \cdot W^{DKV} \tag{5}
$$
The input $x$ is multiplied by the $W^{KR}$ matrix to obtain the $K^R$ matrix used for position embedding computation.
$$
k^R = x \cdot W^{KR} \tag{6}
$$
Based on the properties of matrix multiplication, the $W^{DKV}$ and $W^{KR}$ matrices can be horizontally concatenated into one matrix $[W^{DKV}|W^{KR}]$ for computation. The concatenated matrix corresponds to the weightDkvKr parameter in the API documentation.
$$
c^{KV}k^R = x \cdot [W^{DKV}|W^{KR}] = [x \cdot W^{DKV}|x \cdot W^{KR}] \tag{7}
$$

### RMSNormCkv

RMSNorm (root mean square normalization) is performed row by row on the compressed $KV$ matrix. The RMSNorm operation requires two hyperparameters $\gamma$ and $\epsilon$, which correspond to rmsnormGammaCkv and rmsnormEpsilonCkv in the API documentation.
$$
c_{\mathrm{norm}}^{KV} = \mathrm{RmsNorm}(c^{KV}) \tag{8}
$$
For the RMSNorm calculation, see formulas (3)-(4).

### PostQuant/Dequant

- Basic concepts
  - Quantization: Generally refers to using low-bit-width data (usually INT8) to replace high-bit-width data (usually floating-point numbers) to perform storage, transmission, computation, and other tasks. Sometimes it specifically refers to the computation process of converting floating-point numbers into integers.
  - Dequantization: Restores low-bit-width data to high-bit-width data.
- The basic formula of quantization is:
  $$
  x\_float≈round(x\_float/scale)∗scale
  $$
  PostQuant corresponds to $round(x\_float/scale)$, which scales each value in the data by scale and then performs rounding. Multiplying by the scale parameter corresponds to the dequantization operation.

Currently, two quantization types are included:

  - Dynamic quantization (pertoken): Computes the scale by row (each token). smooth_scale can be used to smooth the input. The corresponding formula is
    $$
    scale=max(abs(x\_float∗smooth\_scale))/Q_{max}
    $$
    where $Q_{max}$ is the maximum value of the quantized output type: INT8 uses 127, FLOAT8_E4M3FN uses 448, and HIFLOAT8 uses 32768. Outputs such as queryOut and queryNorm use this method. Smoothing makes the data smoother so that the data is easier to quantize, as shown in the following figure:

    ![smooth concept](../../../docs/zh/figures/smooth_concept.png)

  - Dynamic quantization (pergroup, mxfp8): Computes the scale with every 32 elements as a group. The scale is of the FLOAT8_E8M0 type. The corresponding formula is
    $$
    scale=2^{\lfloor log_2(max(abs(x\_float))) \rfloor - 8}
    $$
    In the mxfp8 full quantization scenario (weightQuantMode=3), data such as queryOut and queryNorm uses this method.

  - Dynamic quantization (pertoken-pergroup): Computes the scale with every tileSize (currently 128) elements as a group. The formula is the same as that of pertoken, and only the grouping granularity differs. The kvCache pertoken-pergroup quantization scenario (kvCacheQuantMode=3) uses this method.

  - Static quantization (perchannel): Quantizes by column, using one scale for each column of the input tensor. The dequantization parameters of weights such as weightDq, weightUqQr, and weightDkvKr (dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr) and the kvCache perchannel quantization scenario (kvCacheQuantMode=2) all use this method.

  - Static quantization (pertensor): Quantizes the entire tensor with one scale. The kvCache pertensor quantization scenario (kvCacheQuantMode=1) uses this method, corresponding to quantScaleCkv.

### MatmulQcQr

The normalized $C^q$ matrix is multiplied by the Query upsampling matrix $W^{UQ}$ to obtain the $q^C$ matrix.
$$
q^C = c_{norm}^Q \cdot W^{UQ} \tag{9}
$$
The normalized $C^q$ matrix is multiplied by the $W^{QR}$ matrix to obtain the $q^R$ matrix used for position embedding computation.
$$
q^R = c_{norm}^Q \cdot W^{QR} \tag{10}
$$
Similarly, based on the properties of matrix multiplication, the $W^{UQ}$ and $W^{QR}$ matrices can be horizontally concatenated into one matrix $[W^{UQ}|W^{QR}]$ for computation. The concatenated matrix corresponds to the $weightUqQr$ parameter in the API documentation.
$$
q^Cq^R = c_{norm}^Q \cdot [W^{UQ}|W^{QR}] = [c_{norm}^Q \cdot W^{UQ}|c_{norm}^Q \cdot W^{QR}] \tag{11}
$$

### RotaryPosEmb

Rotary Position Embedding (RoPE) is a position embedding algorithm proposed in the paper [Roformer](https://arxiv.org/abs/2104.09864). Its original calculation formula is as follows:
$$
f_{\{q,k\}}(x_m, m) = R_{\Theta,m}^{d}W_{\{q,k\}}x_m \tag{12}
$$

$$
R_{\Theta,m}^{d} =
\left(
\begin{matrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2} \\
\end{matrix}
\right)
\tag{13}
$$
$$
\Theta = \{ \theta_i = 10000^{-2(i-1)/d}, i \in [1,2,...,d/2] \}
$$

where $f_{\{q,k\}}(x_m, m)$ represents the Query/Key vector after the word vector $x_m$ corresponding to the $m$-th token integrates the position information $m$. The calculation formulas of the Query and Key vectors are the same. $d$ is the dimension of the word vector.

After formula (12) is expanded and transformed, it can be expressed in the form of formula (14), where $\otimes$ denotes element-wise cross multiplication.

$$
ROPE(x) = R_{\Theta,m}^{d} x =
\left(
\begin{matrix}
x_0 \\
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_{d-2} \\
x_{d-1}
\end{matrix}
\right)
\otimes
\left(
\begin{matrix}
\cos m\theta_0 \\
\cos m\theta_0 \\
\cos m\theta_1 \\
\cos m\theta_1 \\
\vdots \\
\cos m\theta_{d/2-1} \\
\cos m\theta_{d/2-1}
\end{matrix}
\right)
+
\left(
\begin{matrix}
-x_1 \\
x_0 \\
-x_3 \\
x_2 \\
\vdots \\
-x_{d-1} \\
x_{d-2}
\end{matrix}
\right)
\otimes
\left(
\begin{matrix}
\sin m\theta_0 \\
\sin m\theta_0 \\
\sin m\theta_1 \\
\sin m\theta_1 \\
\vdots \\
\sin m\theta_{d/2-1} \\
\sin m\theta_{d/2-1} \\
\end{matrix}
\right)
\tag{14}
$$

To save the storage space of $cos$/$sin$ and simplify the internal implementation logic, consider that the final $q^R*k^R$ matrix multiplication needs to perform the Reduce operation: adjusting the positions of elements in rows and columns at the same time does not affect the final accumulation result. The computation of the internal $ROPE$ implementation is finally adjusted into the form of formula (15). The input $x$ is split into two parts by odd and even positions, and the inputs of the corresponding $cos$ and $sin$ parts are continuous.

$$
ROPE(x) = R_{\Theta,m}^{d} x =
\left(
\begin{matrix}
x_0 \\
x_2 \\
\vdots \\
x_{d-2} \\
x_{1} \\
x_{3} \\
\vdots \\
x_{d-1}
\end{matrix}
\right)
\otimes
\left(
\begin{matrix}
\cos m\theta_0 \\
\cos m\theta_1 \\
\vdots \\
\cos m\theta_{d/2-1} \\
\cos m\theta_0 \\
\cos m\theta_1 \\
\vdots \\
\cos m\theta_{d/2-1} \\
\end{matrix}
\right)
+
\left(
\begin{matrix}
-x_1 \\
-x_3 \\
\vdots \\
-x_{d-1} \\
x_0 \\
x_2 \\
\vdots \\
x_{d-2}
\end{matrix}
\right)
\otimes
\left(
\begin{matrix}
\sin m\theta_0 \\
\sin m\theta_1 \\
\vdots \\
\sin m\theta_{d/2-1} \\
\sin m\theta_0 \\
\sin m\theta_1 \\
\vdots \\
\sin m\theta_{d/2-1} \\
\end{matrix}
\right)
\tag{15}
$$

In the entire MlaPrologV3 operator, $ROPE$ computation is performed on the $q^R$ matrix and the $k^R$ matrix to obtain the final $queryRope$ and $keyRope$ information. The $keyRope$ information is finally updated to $KRCache$.

$$
q^R = ROPE(q^R)
$$

$$
k^R = ROPE(k^R)
$$

### MatmulQn

The $q^C$ matrix is multiplied by the Key upsampling matrix $W^{UK}$ to obtain the final Query matrix $q^N$.
$$
q^N = q^C \cdot W^{UK} \tag{16}
$$

### KVCache

After the Key/Value results corresponding to the input $x$ are computed, the Key/Value results are updated to the corresponding positions in KVCache. Currently, cacheIndex is introduced to identify the storage positions of the computation results in KVCache. cacheIndex is a 2-dimensional Tensor with the shape [B, S], identifying the target update position of each Token in the Query. Currently, KVCache mainly supports the non-PA (Page Attention) scenario and the PA scenario (ND format storage and NZ format storage).

PA scenario

- KVCache uses ND format storage, and its update process is shown in Figure 3.

  - Assume that the Token/sequence length in the Query is S1. A specific value in the second dimension of cacheIndex indicates the row of KVCache to which the KVCache obtained in this round of computation is to be updated. For example, 11 in the figure indicates that the update is to the position of row 11, which is located in Block2.
  - Figure 3 KVCacheScatter operation (PA scenario - ND format)
  ![KVCacheScatter_PA_ND](../../../docs/zh/figures/KVCacheScatter_PA_ND.jpg)
- KVCache uses NZ format storage, and its update process is shown in Figure 4. For the introduction to the fractal data format, see the guide [Data Layout Format](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0104.html) on the official website.
  - Currently, when KVCache is stored in NZ format, the entire Block is stored in NZ format. Within a small block, the layout is row-major and shaped like the letter Z. Between blocks, the layout is column-major and shaped like the letter N. After the complete H (Hidden States) is divided into small blocks of 32B length, the small blocks that are stored contiguously in ND format need to be stored with a jump of $32B * BlockSize$ in NZ format, that is, the stride needs to be set to $32B * BlockSize$.
  - Figure 4 KVCacheScatter operation (PA scenario - NZ format)
  ![PA_NZ](../../../docs/zh/figures/PA_NZ.jpg)

### KRCache

After the KeyRope result corresponding to the input $x$ is computed, the KeyRope result is updated to the corresponding position in KRCache. Currently, cacheIndex is introduced to identify the storage positions of the computation results in KRCache.

The update logic of KRCache is the same as that of KVCache.
