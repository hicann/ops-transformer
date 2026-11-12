# low_latency_dispatch

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:20:14.985Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**:

  Must be used together with [low_latency_combine](torchapi_low_latency_combine_en.md) to perform token dispatch and combine in MoE parallel deployment.
  - Supports non-quantization, static quantization, pertoken dynamic quantization, pergroup dynamic quantization, mx dynamic quantization, and mx clip dynamic quantization use cases. Quantizes token data first (optional), then performs alltoallv communication in the EP (Expert Parallelism) domain;
  - Supports special expert use cases.

- **Formula**:

  - Quantization use cases:

      If `x_smooth_scale` is not None, perform smoothing/scaling along the expert dimension or by quantization coefficients first; if it is None, use `x` directly:

      $$
      x\_fp32 =
      \begin{cases}
      CastToFp32(x) \times x\_smooth\_scale, & \quad \text{if } x\_smooth\_scale \ne None \\
      CastToFp32(x), & \quad \text{if } x\_smooth\_scale = None
      \end{cases}
      $$

      If quant_mode=0 (non-quantization)

      $$
      quant\_out = x
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      If quant_mode=1 (static quantization)

      $$
      quant\_out = Cast(CastToFp32(x) \times scales, dstType)
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      If quant_mode=2 (dynamic quantization)

      $$
      x\_fp32 = CastToFp32(x) \times scales
      $$

      $$
      dynamic\_scales\_value = dst\_type\_max / Max(Abs(x\_fp32))
      $$

      $$
      quant\_out = Cast(x\_fp32 \times dynamic\_scales\_value, dstType)
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales\_value)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      $$
      dynamic\_scales = alltoall\_dynamic\_scales\_out
      $$

      If quant_mode=3 (pergroup quantization)

      $$
      x\_fp32 = CastToFp32(x) \times scales
      $$

      $$
      dynamic\_scales\_value = dst\_type\_max / Max(Abs(x\_fp32))
      $$

      $$
      quant\_out = Cast(x\_fp32 \times dynamic\_scales\_value, dstType)
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales\_value)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      $$
      dynamic\_scales = alltoall\_dynamic\_scales\_out
      $$

      If quant_mode=4 (mxfp8 quantization)

      $$
      shared\_exp = floor(log_2(max(x))) - emax
      $$

      $$
      dynamic\_scales\_value = 2^{shared\_exp}
      $$

      $$
      quant\_out = Cast(x / dynamic\_scales\_value, dstType)
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales\_value)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      $$
      dynamic\_scales = alltoall\_dynamic\_scales\_out
      $$

      If quant_mode=5 (mxfp8 quantization chip method)

      $$
      max\_abs = max(abs(x))
      $$

      $$
      max\_abs\_clamp = max(max\_abs, 10^{-4})
      $$

      $$
      shared\_exp = ceil(log_2(max\_abs\_clamp / fp8\_max))
      $$

      $$
      dynamic\_scales\_value = 2^{shared\_exp}
      $$

      $$
      quant\_out = CastToFp8(x / dynamic\_scales\_value)
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales\_value)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      $$
      dynamic\_scales = alltoall\_dynamic\_scales\_out
      $$

  - Special expert cases:

      Zero expert case, i.e., `zero_Expert_Num` is not 0:

      $$
      Moe(ori\_x) = 0
      $$

      Copy expert case, i.e., `copy_Expert_Num` is not 0:

      $$
      Moe(ori\_x) = ori\_x
      $$

      Constant expert scenario, where `const_expert_num` is not 0:

      $$
      Moe(ori\_x) = const\_expert\_alpha\_1 * ori\_x + const\_expert\_alpha\_2 * const\_expert\_v
      $$

      For parameters ori\_x, const\_expert\_alpha\_1, const\_expert\_alpha\_2, and const\_expert\_v, see the [low_latency_combine](torchapi_low_latency_combine_en.md) documentation.

## Function Prototype

```python
MoeDistributeBuffer.low_latency_dispatch(x, topk_idx, num_experts, *, quant_mode=0, comm_alg="", x_smooth_scale=None, x_active_mask=None, topk_weights=None, zero_expert_num=0, copy_expert_num=0, const_expert_num=0, elastic_info=None, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, expert_token_nums_type=1, num_max_dispatch_tokens_per_rank=0, y_dtype=None, x_dtype=None, x_smooth_scales_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Mandatory

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Token data used for computation, to be sent to other devices based on `topk_idx`. Must be a 2D tensor representing BS tokens. Data format: ND. Non-contiguous tensors are supported.</td>
        <td>bfloat16, float16, uint8, float8_e5m2, float8_e4m3fn</td>
        <td>(BS, H)</td>
    </tr>
    <tr>
        <td>topk_idx</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>TopK expert indices for each token, determining which experts each token is sent to. Must be a 2D tensor. Data format: ND. Non-contiguous tensors are supported. Corresponds to the `topk_idx` input of [low_latency_combine](torchapi_low_latency_combine_en.md). The values in the tensor range from [0, num_experts), and the K values in the same row cannot be duplicated.</td>
        <td>int32</td>
        <td>(BS, K)</td>
    </tr>
    <tr>
        <td>num_experts</td>
        <td>int</td>
        <td>Required</td>
        <td>Number of MoE experts. Value range: [1, 1024]. The following condition must be met: num_experts%(ep_world_size - shared_expert_rank_num)=0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quant_mode</td>
        <td>int</td>
        <td>Optional</td>
        <td>Quantization mode. 0: non-quantization (default), 1: static quantization, 2: pertoken dynamic quantization, 3: pergroup dynamic quantization, 4: mx dynamic quantization, 5: mx clip dynamic quantization. When `quant_mode` is 2, 3, 4, or 5, `dynamic_scales` cannot be None. When `quant_mode` is 0 and the input `x` is of a low-precision data type such as hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, or float4_e1m2, `dynamic_scales` also cannot be None. In other use cases, `dynamic_scales` is None.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_alg</td>
        <td>str</td>
        <td>Optional</td>
        <td>Communication affinity memory layout algorithm. The current version supports three input values: "", "fullmesh_v1", and "fullmesh_v2". The default value "" enables the fullmesh_v1 template; "fullmesh_v1" enables the fullmesh_v1 template; "fullmesh_v2" enables the fullmesh_v2 template.</td>
        <td>str</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_smooth_scale</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>            <ul>
                <li>Quantization smoothing parameter or quantization coefficient.</li>
                <li>If `quant_mode` is 0, do not pass this parameter if `x` is bfloat16 or float16; a valid tensor is required if `x` is a low-precision data type such as hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, or float4_e1m2.</li>
                <li>Required if `quant_mode` is 1.</li>
                <li>Optional if `quant_mode` is 2 or 3.</li>
                <li>Do not pass this parameter if `quant_mode` is 4 or 5.</li>
                <li>If used for expert-dimension smoothing, this parameter must be a 2D tensor. If shared experts exist, the shape is (shared_expert_num+num_experts, H); otherwise, the shape is (num_experts, H). The data format is ND. Non-contiguous tensors are not supported.</li>
                <li>For shape constraints in other quantization coefficient use cases, see "quant_mode-related constraints".</li>
            </ul>
        </td>
        <td>float</td>
        <td>            <ul>
                <li>(shared_expert_num+num_experts, H)</li>
                <li>(num_experts, H)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>x_active_mask</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Indicates whether tokens participate in communication. Must be a 1D or 2D tensor. If the input is 1D, the shape is (BS, ); if the input is 2D, the shape is (BS, K). The data format must be ND. Non-contiguous tensors are supported. If the input is 1D, true indicates that the corresponding token participates in communication, and all true values must be placed before false values. For example, {true, false, true} is invalid input. If the input is 2D, true indicates that the `topk_idx` corresponding to the current token participates in communication. If all K `bool` values of the current token are false, the current token does not participate in communication. By default, all tokens participate in communication. If the BS value differs across devices, all tokens must be valid.</td>
        <td>bool</td>
        <td>            <ul>
                <li>(BS, )</li>
                <li>(BS, K)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>topk_weights</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Denotes the topK expert weights of each token.</td>
        <td>float32</td>
        <td>(BS, K)</td>
    </tr>
    <tr>
        <td>zero_expert_num</td>
        <td>int</td>
        <td>Optional</td>
        <td>Number of zero experts. Value range: [0, MAX_int32), where MAX_int32 = 2^31 - 1. Valid zero expert IDs: [num_experts, num_experts+zero_expert_num).</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>copy_expert_num</td>
        <td>int</td>
        <td>Optional</td>
        <td>Number of copy experts. Value range: [0, MAX_int32), where MAX_int32 = 2^31 - 1. Valid copy expert IDs: [num_experts+zero_expert_num, num_experts+zero_expert_num+copy_expert_num).</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>const_expert_num</td>
        <td>int</td>
        <td>Optional</td>
        <td>Number of constant experts. Value range: [0, MAX_int32), where MAX_int32 = 2^31 - 1. Valid constant expert IDs range from [moe_expert_num+zero_expert_num+copy_expert_num, moe_expert_num+zero_expert_num+copy_expert_num+const_expert_num).</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>elastic_info</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Reserved parameter. Not supported in the current version. Pass the default value None.</td>
        <td>Not supported currently</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expert_shard_type</td>
        <td>int</td>
        <td>Optional</td>
        <td>Shared expert device arrangement type. Currently only 0 is supported, indicating that shared expert devices are placed before MoE expert devices.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>shared_expert_num</td>
        <td>int</td>
        <td>Optional</td>
        <td>Number of shared experts. Each shared expert can be replicated and deployed on multiple devices. Value range: [0, 4]. 0 indicates no shared expert. The default value is 1.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>shared_expert_rank_num</td>
        <td>int</td>
        <td>Optional</td>
        <td>Number of devices for shared experts. Value range: [0, ep_world_size). 0 indicates no shared expert. A non-zero value must satisfy shared_expert_rank_num%shared_expert_num=0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expert_token_nums_type</td>
        <td>int</td>
        <td>Optional</td>
        <td>Value type of the output `expert_token_nums`. Value range: [0, 1]. 0 indicates the prefix sum of the number of tokens received by each expert. 1 indicates the number of tokens received by each expert (default).</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_max_dispatch_tokens_per_rank</td>
        <td>int</td>
        <td>Optional</td>
        <td>Number of tokens on each device. Specifies the maximum BS size when BS differs across ranks. The default value is 0 when BS is the same on each rank.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>y_dtype</td>
        <td>int</td>
        <td>Optional</td>
        <td>            <ul>
                <li>Logical data type of the `expand_x` output. The default value is None, which is deduced from `quant_mode` and the input type.</li>
                <li>In Ascend 950PR&950DT products use cases, this parameter specifies a low-precision output type.</li>
                <li>`quant_mode=1` supports int8 and hifloat8.</li>
                <li>`quant_mode=2` supports int8, float8_e4m3fn, and float8_e5m2.</li>
                <li>`quant_mode=3` supports float8_e4m3fn and float8_e5m2.</li>
                <li>`quant_mode=4` or `quant_mode=5` supports float8_e4m3fn, float8_e5m2, float4_e2m1, and float4_e1m2.</li>
                <li>Use the integer enumeration corresponding to the ACL data type in `cann_ops_transformer.common`.</li>
            </ul>
        </td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_dtype</td>
        <td>int</td>
        <td>Optional</td>
        <td>Represents the logical data type of the input `x`. The default value is None, which is derived from the PyTorch storage type of `x`. In low-precision scenarios, it can be set to ACL data types such as hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, and float4_e1m2. Among them, hifloat8, float4_e2m1, and float4_e1m2 are stored as uint8 on the PyTorch side.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_smooth_scales_dtype</td>
        <td>int</td>
        <td>Optional</td>
        <td>Represents the logical data type of `x_smooth_scale`. The default value is None, which is derived from the PyTorch storage type of `x_smooth_scale`. In low-precision scenarios, ACL data types such as float8_e8m0 can be specified. For example, when `x` is float8_e5m2 or float8_e4m3fn, `x_smooth_scale` supports float or float8_e8m0; when `x` is float4_e2m1 or float4_e1m2, `x_smooth_scale` requires float8_e8m0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Return Values

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>expand_x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Represents the token data received by the local device. It must be a 2D tensor, where A indicates the maximum number of tokens that may be received in the EP communicator. When no quantization is performed and `y_dtype` is not specified, the data type is the same as that of `x`; when quantization is performed or `y_dtype` is specified, the data type is determined by `quant_mode` and `y_dtype`, and low-precision outputs such as int8, hifloat8, float8_e4m3fn, float8_e5m2, float4_e2m1, and float4_e1m2 are supported. The data format is ND. Non-contiguous tensors are supported.</td>
        <td>bfloat16, float16, int8, uint8, float8_e5m2, float8_e4m3fn</td>
        <td>(A, H); (A, H/2) for float4_e2m1/float4_e1m2</td>
    </tr>
    <tr>
        <td>dynamic_scales</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>            <ul>
                <li>Represents the computed dynamic quantization parameters.</li>
                <li>Output is generated when `quant_mode` is 2, 3, 4, or 5.</li>
                <li>Output is also generated when `quant_mode` is 0 and input `x` is a low-precision type such as hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, or float4_e1m2.</li>
                <li>float8_e8m0 is represented as uint8 on the PyTorch side.</li>
                <li>The supported data format is ND. Non-contiguous tensors are supported.</li>
            </ul>
        </td>
        <td>float, uint8</td>
        <td>            <ul>
                <li>pertoken dynamic quantization: (A,)</li>
                <li>pergroup dynamic quantization: (A, Ceil(H, 128))</li>
                <li>mx dynamic quantization/mx clip dynamic quantization: (A, Align2(Ceil(H, 32)))</li>
                <li>low-precision non-quantized input: (A, dim1), where dim1 is determined by the second dimension of `x_smooth_scale`</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>assist_info_for_combine</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Number of tokens sent to the same expert. It must be a 1D tensor with ND data format, and supports non-contiguous tensors. Corresponds to the `assist_info_for_combine` input of [low_latency_combine](torchapi_low_latency_combine_en.md).</td>
        <td>int32</td>
        <td>(A * 128, )</td>
    </tr>
    <tr>
        <td>expert_token_nums</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Number of tokens actually received by each expert on the current device. It must be a 1D tensor, supports the ND data format, and supports non-contiguous tensors.</td>
        <td>int64</td>
        <td>(local_expert_num,)</td>
    </tr>
    <tr>
        <td>ep_recv_counts</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Indicates the number of tokens received by each device in the EP communicator (the token counts are expressed as a prefix sum). Must be a 1D tensor, supports the ND data format, and supports non-contiguous tensors. Corresponds to the `ep_send_counts` input of [low_latency_combine](torchapi_low_latency_combine_en.md).</td>
        <td>int32</td>
        <td>(ep_world_size*local_expert_num, )</td>
    </tr>
    <tr>
        <td>expand_scales</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Output after alltoallv is performed on `topk_weights` together with `x`.</td>
        <td>float32</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Constraints

- This API supports use in inference scenarios.
- This API supports the single-operator mode and torchair graph mode. `low_latency_dispatch` and `low_latency_combine` must be used together.
- For each input tensor, the constraints on the list[Tensor] length, transposition, and non-contiguous tensor support are as follows:

    <table style="undefined;table-layout: fixed; width:800px"><colgroup>
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 150px">
    <col style="width: 250px">
    </colgroup>
    <thead>
    <tr>
        <th>Parameter</th>
        <th>list[Tensor] length</th>
        <th>Whether transposed</th>
        <th>Whether non-contiguous tensors are supported</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>x</td>
            <td>-</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>topk_idx</td>
            <td>-</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>x_smooth_scale</td>
            <td>-</td>
            <td>No</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>x_active_mask</td>
            <td>-</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
    </tbody>
    </table>

- The variables used in Shape are as follows:
  - A: maximum number of tokens received by the current device. The value range is as follows:
    - For shared experts, A=BS\*shared\_expert\_num/shared\_expert\_rank\_num must be satisfied.
    - For MoE experts, when `num_max_dispatch_tokens_per_rank` is 0, A \>= BS \* ep\_world\_size \* min\(local\_expert\_num, K\) must be satisfied; when `num_max_dispatch_tokens_per_rank` is not 0, A \>= num_max\_dispatch\_tokens\_per\_rank \* ep\_world\_size \* min\(local\_expert\_num, K\) must be satisfied.

  - H: hidden size. The value is \[1024, 8192\].

  - BS: batch sequence size, indicating the number of tokens finally output by the local device. Value range: 0<BS≤512.

  - K: number of experts selected by topK. Value range: 0<K≤16. In addition, 0 < K ≤ num\_experts + zero_expert_num + copy_expert_num + const_expert_num must be satisfied.

  - local\_expert\_num: number of experts on the local device.
    - For shared expert devices, local\_expert\_num is 1.
    - For MoE expert devices, local\_expert\_num=num\_experts/\(ep\_world\_size-shared\_expert\_rank\_num), and 0 < local_expert_num * ep_world_size ≤ 2048 must be satisfied.

- In different product models, with different communication algorithms, or in different versions, the element values in the Tensor outputs `assist_info_for_combine`, `ep_recv_counts`, and `expand_scales` of `low_latency_dispatch` may differ. Pass these tensors directly to the corresponding parameters of `low_latency_combine`. Other model logic must not depend on these values.
- During the API call, the values of `num_experts`, `expert_shard_type`, `shared_expert_num`, `shared_expert_rank_num`, and `num_max_dispatch_tokens_per_rank` must be consistent across all devices. `expert_shard_type` and `num_max_dispatch_tokens_per_rank` must also be consistent across different layers in the network and consistent with the corresponding parameters of [low_latency_combine](torchapi_low_latency_combine_en.md).
- In this scenario, a single card contains two DIE (short for "die" or "bare die"), so "this card" in the parameter description refers to a single DIE.
- num_experts + zero_expert_num + copy_expert_num + const_expert_num < MAX_int32.
- Related constraints:
  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>:
    - `quant_mode=0` indicates the non-quantization scenario. `x_smooth_scale` is not passed, and the supported data types of `expand_x` are float16 and bfloat16.
    - `quant_mode=2` indicates the pertoken dynamic quantization scenario. The supported data type of `expand_x` is int8. `x_smooth_scale` is optional. If a valid Tensor is passed, its shape is (num_experts, H), and the shape of the output `dynamic_scales` is (A,).
  <!-- end id7 -->
  <!-- npu="A3" id8 -->
  - <term>Atlas A3 products</term>:
    - The topK expert weight feature is not supported. `topk_weights` must be set to None, and the output `expand_scales` is invalid.
    - `quant_mode=0` indicates the non-quantization scenario. `x_smooth_scale` is not passed, and the supported data types of `expand_x` are float16 and bfloat16.
    - `quant_mode=2` indicates the pertoken dynamic quantization scenario. The supported data type of `expand_x` is int8. `x_smooth_scale` is optional. If shared experts exist and a valid Tensor is passed, its shape is (shared_expert_num + num_experts, H). If no shared expert exists, its shape is (num_experts, H). The shape of the output `dynamic_scales` is (A,).
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    - `topk_weights` can be a valid Tensor or None. If a valid Tensor is passed, the topK expert weight feature is enabled. The valid Tensor is a 2D Tensor with shape (BS, K), and the output `expand_scales` is valid. If None is passed, the topK expert weight feature is not enabled, and the output `expand_scales` is invalid. Empty Tensors are not supported.
    - `quant_mode=0` indicates the non-quantization scenario. If `x` is float16 or bfloat16, `expand_x` can be the same as `x` or be specified as hifloat8 using `y_dtype`, and `x_smooth_scale` must be None. If `x` is hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, or float4_e1m2, `x_smooth_scale` must be a valid Tensor, and `expand_x` has the same logical type as `x`. If `x` is hifloat8, the logical type of `x_smooth_scale` is float. If `x` is float8_e5m2 or float8_e4m3fn, the logical type of `x_smooth_scale` is float or float8_e8m0. If `x` is float4_e2m1 or float4_e1m2, the logical type of `x_smooth_scale` is float8_e8m0, and H must be an even number. In this case, the shape of `x_smooth_scale` is (BS, dim1), where dim1 <= H.
    - `quant_mode=1` indicates the static quantization scenario. The supported data types of `expand_x` are int8 and hifloat8, and `x_smooth_scale` must be a valid Tensor. If `expand_x` is int8, `x_smooth_scale` can represent the quantization factor with shape (1,); the smooth weight shared by all experts with shape (H,); or the quantization factor fused with the smooth weight of each expert, with shape (shared_expert_num + num_experts, H) when shared experts exist and (num_experts, H) when no shared expert exists. If `expand_x` is hifloat8, the shape of `x_smooth_scale` must be (1,).
    - `quant_mode=2` indicates the pertoken dynamic quantization use case. `expand_x` supports int8, float8_e4m3fn, and float8_e5m2. `x_smooth_scale` is optional. If a valid Tensor is passed in, its expert-dimension shape rule is the same as that of `quant_mode=1`, and the output `dynamic_scales` shape is (A,).
    - `quant_mode=3` indicates the pergroup dynamic quantization use case. `expand_x` supports float8_e4m3fn and float8_e5m2. `x_smooth_scale` is optional. If a valid Tensor is passed in, its expert-dimension shape rule is the same as that of `quant_mode=1`, and the output `dynamic_scales` shape is (A, Ceil(H, 128)).
    - `quant_mode=4` indicates the mx dynamic quantization use case, and `quant_mode=5` indicates the mx clip dynamic quantization use case. `expand_x` supports float8_e4m3fn, float8_e5m2, float4_e2m1, and float4_e1m2. `x_smooth_scale` must be None, and the output `dynamic_scales` shape is (A, Align2(Ceil(H, 32))). If `expand_x` is float4_e2m1 or float4_e1m2, H must be an even number.
  <!-- end id9 -->
  - In this document, `Ceil(H, N)` indicates `(H + N - 1) // N`, and `Align2(x)` indicates rounding up to an integer multiple of 2.
- HCCL communicator buffer size:
    Before calling this API, check whether the configured HCCL communicator buffer size is appropriate. The unit is MB.
  - When the HCCL\_BUFFSIZE environment variable is used, the buffer size is managed at the communicator granularity and defaults to 200 MB. Each communicator exclusively occupies a memory region of size "2*HCCL\_BUFFSIZE".
  - Within the EP communicator, when comm\_alg is set to "fullmesh_v1" or "": the size setting must meet the requirement \>= 2 \* \(local\_expert\_num \* max\_bs \* ep\_world\_size \* Align512\(Align32\(2 \* H\) + <b>44</b>\) + \(K + shared\_expert\_num\) \* max\_bs \* Align512\(2 \* H\)\).
  - Within the EP communicator, when comm\_alg is set to "fullmesh_v2": the size setting must meet the requirement \>= 2 \* \(local\_expert\_num \* max\_bs \* ep\_world\_size \* 480Align512\(Align32\(2 \* H\) + <b>44</b>\) + \(K + shared\_expert\_num\) \* max\_bs \* Align512\(2 \* H\)\).
  - where 480Align512(x) = ((x+480-1)/480)\*512,Align512(x) = ((x+512-1)/512)\*512,Align32(x) = ((x+32-1)/32)\*32.
  <!-- npu="950" id18 -->
  - For <term>Ascend 950PR&950DT products</term>, when quant\_mode=0 and x\_smooth\_scale is not empty, the dispatch token length is max(Align32(2 \* H) + 44, Align32(Align32(2 \* H) + 4 \* H) + 16), followed by alignment according to the preceding rules.
  - For <term>Ascend 950PR&950DT products</term>, the combine term in the preceding formulas is calculated using `480Align512(2 * H)`.
  <!-- end id18 -->
  - The size of the communicator to create can be calculated by calling the MoeDistributeBuffer.get_low_latency_ccl_buffer_size API.
  <!-- npu="950" id19 -->
  - For <term>Ascend 950PR&950DT products</term>, you can specify the size of the internally allocated communicator buffer using `ccl_buffer_size` when initializing `MoeDistributeBuffer`. This parameter is in MB and can use the value returned by `MoeDistributeBuffer.get_low_latency_ccl_buffer_size`.
  - For <term>Ascend 950PR&950DT products</term>, when both the HCCL\_BUFFSIZE environment variable and a positive `ccl_buffer_size` are configured, the buffer allocated through `ccl_buffer_size` takes precedence. When this parameter is 0, the default communicator buffer is used.
  <!-- end id19 -->

- The "/" in the formulas in this document indicates integer division.

- Communication domain usage constraints:

  - The `low_latency_dispatch` and `low_latency_combine` operators in a model support only the same EP communicator, and no other operators are allowed in this communicator.

  - Nodes in a communicator must belong to one SuperPoD. Cross-SuperPoD communication is not supported.

- Version compatibility constraints:

     In static graph mode, starting from TorchNPU 8.0.0, the TorchNPU framework performs strict validation on the output of the last node in a static graph by comparing the Meta inference result with the inferShape inference result. If the graph contains only one Dispatch operator and the CANN version is earlier than the TorchNPU version, a shape mismatch error is reported. In this case, upgrade the CANN version. For details, see "Related Product Version Compatibility" in *[TorchNPU Release Notes](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md)*.

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Call in single-operator mode:

  ```python
  import os
  import torch
  import random
  import torch_npu
  import numpy as np
  from torch.multiprocessing import Process
  import torch.distributed as dist
  from torch.distributed import ReduceOp
  from cann_ops_transformer.ops import MoeDistributeBuffer

  # Control mode
  quant_mode = 2  # 2 indicates dynamic quantization
  is_dispatch_scales = True  # For dynamic quantization, you can choose whether to pass scales
  input_dtype = torch.bfloat16  # Output dtype
  server_num = 1
  server_index = 0
  port = 50001
  master_ip = '127.0.0.1'
  dev_num = 16
  world_size = server_num * dev_num
  rank_per_dev = int(world_size / server_num)  # Number of dies per host
  shared_expert_rank_num = 0  # Number of shared experts
  num_experts = 32  # Number of MoE experts
  bs = 8  # Number of tokens
  h = 7168  # Length of each token
  k = 8
  random_seed = 0
  ep_world_size = world_size
  moe_rank_num = ep_world_size - shared_expert_rank_num
  local_moe_expert_num = num_experts // moe_rank_num
  globalBS = bs * ep_world_size
  is_shared = (shared_expert_rank_num > 0)
  is_quant = (quant_mode > 0)
  zero_expert_num = 1
  copy_expert_num = 1
  const_expert_num = 0


  def gen_const_expert_alpha_1():
      const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
      return const_expert_alpha_1


  def gen_const_expert_alpha_2():
      const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
      return const_expert_alpha_2


  def gen_const_expert_v():
      const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
      return const_expert_v


  def get_new_group(rank):
      ep_ranks = list(range(ep_world_size))
      ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
      if rank in ep_ranks:
          print(f"rank:{rank} ep_ranks:{ep_ranks}")
      return ep_group


  def get_hcomm_info(rank, comm_group):
      if torch.__version__ > '2.0.1':
          hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
      else:
          hcomm_info = comm_group.get_hccl_comm_name(rank)
      return hcomm_info


  def get_dispatch_kwargs_warmup(
      x_warm_up, topk_idx_warm_up,
  ):
      x_warm_up = x_warm_up.to(input_dtype).npu()
      topk_idx_warm_up = topk_idx_warm_up.to(torch.int32).npu()
      return {
          'x': x_warm_up,
          'topk_idx': topk_idx_warm_up,
          'x_active_mask': None,
          'shared_expert_num': 0,
          'shared_expert_rank_num': shared_expert_rank_num,
          'num_experts': num_experts,
          'quant_mode': 2,
          'num_max_dispatch_tokens_per_rank': 16,
      }


  def run_npu_process(rank):
      torch_npu.npu.set_device(rank)
      rank = rank + 16 * server_index
      # Initialize the distributed process group.
      dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
      ep_group = get_new_group(rank)
      ep_hcomm_info = get_hcomm_info(rank, ep_group)

      # Create the input tensor.
      x = torch.randn(bs, h, dtype=input_dtype).npu()
      topk_idx = torch.tensor([[5, 7, 17, 4, 2, 6, 11, 16],
                              [10, 12, 13, 15, 19, 4, 18, 1],
                              [19, 33, 1, 17, 9, 5, 0, 32],
                              [19, 11, 17, 0, 10, 5, 7, 9],
                              [10, 16, 11, 17, 33, 8, 9, 3],
                              [12, 19, 5, 7, 1, 3, 18, 16],
                              [11, 9, 13, 16, 12, 33, 17, 14],
                              [16, 4, 9, 5, 0, 10, 11, 17]], dtype=torch.int32).npu()
      topk_weights = torch.randn(bs, k, dtype=torch.float32).npu()

      scales_shape = (1 + num_experts, h) if shared_expert_rank_num else (num_experts, h)
      if is_dispatch_scales:
          scales = torch.randn(scales_shape, dtype=torch.float32).npu()
      else:
          scales = None

      const_expert_alpha_1 = None
      const_expert_alpha_2 = None
      const_expert_v = None

      # Construct the communicator buffer.
      distribute_buffer = MoeDistributeBuffer(ep_group)

      # Call dispatch to perform token quantization and EP-domain alltoallv distribution.
      expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, _ = distribute_buffer.low_latency_dispatch(
          x=x,
          topk_idx=topk_idx,
          shared_expert_num=0,
          shared_expert_rank_num=shared_expert_rank_num,
          num_experts=num_experts,
          quant_mode=quant_mode,
          num_max_dispatch_tokens_per_rank=globalBS,
          zero_expert_num=zero_expert_num,
          copy_expert_num=copy_expert_num,
          const_expert_num=const_expert_num)

      if is_quant:
          expand_x = expand_x.to(input_dtype)

      # Pass the dispatch outputs to combine to complete token retrieval.
      x = distribute_buffer.low_latency_combine(x=expand_x,
                                              topk_idx=topk_idx,
                                              topk_weights=topk_weights,
                                              assist_info_for_combine=assist_info_for_combine,
                                              ep_send_counts=ep_recv_counts,
                                              shared_expert_num=0,
                                              shared_expert_rank_num=shared_expert_rank_num,
                                              num_experts=num_experts,
                                              num_max_dispatch_tokens_per_rank=globalBS,
                                              ori_x=x,
                                              const_expert_alpha_1=const_expert_alpha_1,
                                              const_expert_alpha_2=const_expert_alpha_2,
                                              const_expert_v=const_expert_v,
                                              zero_expert_num=zero_expert_num,
                                              copy_expert_num=copy_expert_num,
                                              const_expert_num=const_expert_num)
      print(f'rank {rank} npu finished! \n')


  if __name__ == "__main__":
      print(f"bs={bs}")
      print(f"num_max_dispatch_tokens_per_rank={globalBS}")
      print(f"shared_expert_rank_num={shared_expert_rank_num}")
      print(f"num_experts={num_experts}")
      print(f"k={k}")
      print(f"quant_mode={quant_mode}", flush=True)
      print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
      print(f"ep_world_size={ep_world_size}", flush=True)

      if shared_expert_rank_num > ep_world_size:
          print("shared_expert_rank_num cannot be greater than ep_world_size")
          exit(0)
      if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
          print("ep_world_size must be an integer multiple of shared_expert_rank_num")
          exit(0)
      if num_experts % moe_rank_num != 0:
          print("num_experts must be an integer multiple of moe_rank_num")
          exit(0)
      p_list = []
      for rank in range(rank_per_dev):
          p = Process(target=run_npu_process, args=(rank,))
          p_list.append(p)

      for p in p_list:
          p.start()
      for p in p_list:
          p.join()

      print("run npu success.")
  ```

- Graph mode call:

    ```python
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time
    from cann_ops_transformer.ops import MoeDistributeBuffer

    # Control mode
    quant_mode = 2  # 2 indicates dynamic quantization
    is_dispatch_scales = True  # Scales are optional for dynamic quantization
    input_dtype = torch.bfloat16  # Output dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # Number of dies per host
    shared_expert_rank_num = 0  # Number of shared experts
    num_experts = 32  # Number of MoE experts
    bs = 8  # Number of tokens
    h = 7168  # Length of each token
    k = 8
    random_seed = 0
    ep_world_size = world_size
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = num_experts // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)

    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 0

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self, group_ep):
            super().__init__()
            self.group = group_ep
            self.distribute_buffer = MoeDistributeBuffer(group_ep)

        def forward(self, x, topk_idx, group_ep,
                    ep_world_size, ep_rank_id, expert_shard_type, shared_expert_rank_num, num_experts,
                    scales, quant_mode, num_max_dispatch_tokens_per_rank, topk_weights, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
            output_dispatch_npu = self.distribute_buffer.low_latency_dispatch(
                x=x,
                topk_idx=topk_idx,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            output_combine_npu = self.distribute_buffer.low_latency_combine(
                x=expand_x_npu,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                assist_info_for_combine=assist_info_for_combine_npu,
                ep_send_counts=ep_recv_counts_npu,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                ori_x=x,
                const_expert_alpha_1=const_expert_alpha_1,
                const_expert_alpha_2=const_expert_alpha_2,
                const_expert_v=const_expert_v,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            x = output_combine_npu
            x_combine_res = output_combine_npu
            return [x_combine_res, output_combine_npu]

    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2

    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v

    def get_new_group(rank):
        ep_ranks = list(range(ep_world_size))
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            print(f"rank:{rank} ep_ranks:{ep_ranks}")
        return ep_group

    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    def get_dispatch_kwargs_warmup(
        x_warm_up, topk_idx_warm_up,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        topk_idx_warm_up = topk_idx_warm_up.to(torch.int32).npu()
        return {
            'x': x_warm_up,
            'topk_idx': topk_idx_warm_up,
            'x_active_mask': None,
            'expert_shard_type': 0,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'num_experts': num_experts,
            'scales': None,
            'quant_mode': 2,
            'num_max_dispatch_tokens_per_rank': 1,
        }

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://{master_ip}:{port}'
        )
        ep_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        # Create the input tensors
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        topk_idx = torch.tensor([
            [0, 8, 4, 1, 6, 12, 14, 17],
            [14, 10, 7, 3, 0, 12, 11, 17],
            [12, 0, 5, 11, 19, 4, 6, 18],
            [17, 3, 4, 10, 18, 0, 1, 2],
            [13, 16, 9, 10, 15, 6, 7, 14],
            [17, 15, 14, 8, 16, 18, 3, 12],
            [4, 12, 2, 17, 15, 3, 9, 10],
            [16, 7, 12, 9, 18, 3, 19, 17]
        ], dtype=torch.int32).npu()
        topk_weights = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + num_experts, h) if shared_expert_rank_num else (num_experts, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        elastic_info = None
        const_expert_alpha_1 = None
        const_expert_alpha_2 = None
        const_expert_v = None
        model = MOE_DISTRIBUTE_GRAPH_Model(ep_group)
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(
            x, topk_idx, ep_hcomm_info, ep_world_size,
            rank, 0, shared_expert_rank_num, num_experts, scales,
            quant_mode, globalBS, topk_weights, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
            zero_expert_num, copy_expert_num, const_expert_num
        )
        torch.npu.synchronize()
        print(f'rank {rank} npu finished! \n')

        time.sleep(10)


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"num_max_dispatch_tokens_per_rank={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"num_experts={num_experts}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num cannot be greater than ep_world_size")
            exit(0)

        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size must be an integer multiple of shared_expert_rank_num")
            exit(0)

        if num_experts % moe_rank_num != 0:
            print("num_experts must be an integer multiple of moe_rank_num")
            exit(0)

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```
