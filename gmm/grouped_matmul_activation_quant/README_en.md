# GroupedMatmulActivationQuant

English | [简体中文](./README.md)

## Product Support

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: fuses GroupedMatmul, activation, and quant computation. The current version supports the MXFP8 and MXFP4 input scenarios on the WeightNz path. The activation function supports only gelu_tanh, and the quantized output supports FP8 and FP4.

- Calculation formula:
  - <term>Ascend 950PR&950DT products</term>:

    <details>
    <summary>MX quantization scenario:</summary>

    - Definitions:
      - $E$ denotes the number of experts, $M$ denotes the total number of tokens, $K$ denotes the input feature dimension, and $N$ denotes the output feature dimension.
      - $blocksize$ denotes the group size that shares the exponent in MX quantization. Currently, only 64 is supported.
      - **·** denotes matrix multiplication, and **⊙** denotes element-wise multiplication.

    - Determine the token range corresponding to each group according to group_list.

    - Perform GroupedMatmul and dequantization for each group. By default, the intermediate result is of the FLOAT32 type:

      $$
      C_i = (X_i \cdot weight_i) \odot x\_scale_{i\ Broadcast} \odot weight\_scale_{i\ Broadcast}
      $$

    - Perform the gelu_tanh activation:

      $$
      S_i = GeluTanh(C_i)
      $$

      The current underlying kernel implementation uses the gelu_sigmoid function to approximate gelu_tanh:

      $$
      \operatorname{GeluTanh}(x) =
      \frac{x}{1 + \exp\left(-1.595769121 \times \left(x + 0.044715 \times x^3\right)\right)}
      $$

    - Perform MX dynamic quantization on the activation result in groups of $blocksize=32$ along the N axis, and output the quantized result $y$ and the quantization factor $y\_scale$.

    </details>

## Parameter Description

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format | Shape |
| ------ | -------------- | ---- | -------- | -------- | ----------- |
| x | Input | Left matrix, corresponding to $X$ in the formula. | FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2 | ND | 2D, in the format `(M, K)` |
| group_list | Input | Grouping information, corresponding to `group_list` in the formula. | INT64 | ND | 1D, in the format `(E,)` |
| weight | Dynamic input | Dynamic tensorList of the right matrix. Only a length of 1 is supported. For MXFP8, the type is FLOAT8_E4M3FN; for MXFP4, any FLOAT4 type is allowed, and x and weight must both be of FLOAT4 types. | FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2 | FRACTAL_NZ | viewShape is 3D, and storageShape is 5D |
| weight_scale | Dynamic input | MX quantization factor of `weight`, a dynamic tensorList. Only a length of 1 is supported. | FLOAT8_E8M0 | ND | 4D |
| bias | Dynamic input | bias dynamic tensorList. This parameter must be empty in the current MX scenario. | FLOAT | ND | - |
| x_scale | Optional input | MX quantization factor of `x`. This parameter must be passed in the current MX scenario. | FLOAT8_E8M0 | ND | 3D, in the format `(M, ceil(K / 64), 2)` |
| y | Output | Output matrix after activation and quantization. FP4 output supports only the MXFP4 input scenario. | FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2 | ND | 2D, in the format `(M, N)` |
| y_scale | Output | MX quantization factor of the output `y`. | FLOAT8_E8M0 | ND | 3D, in the format `(M, ceil(N / 64), 2)` |
| activation_type | Required attribute | Activation function type. Currently, only `"gelu_tanh"` is supported. | STRING | - | - |
| transpose_weight | Optional attribute | Specifies whether `weight` is transposed. The default value is false. | BOOL | - | - |
| group_list_type | Optional attribute | Specifies the grouping mode of the `group_list` input. The values 0 and 1 are supported, and the default value is 0. | INT64 | - | - |
| tuning_config | Optional attribute | Reserved tuning parameter. The default value is `[0]`. This parameter is not used yet. | LIST_INT | - | - |
| quant_mode | Optional attribute | Quantization mode. Currently, only `"mx"` is supported. | STRING | - | - |
| y_dtype | Optional attribute | Data type of the output `y`. The default value is DT_UNDEFINED, in which case the output type follows `x`. The supported values are 35 (FLOAT8_E5M2), 36 (FLOAT8_E4M3FN), 40 (FLOAT4_E2M1), and 41 (FLOAT4_E1M2). | INT64 | - | - |
| round_mode | Optional attribute | Quantization rounding mode. The default value is `"rint"`. Currently, only `"rint"` is supported. | STRING | - | - |
| scale_alg | Optional attribute | MX quantization scale algorithm. The default value is 0. The values 0, 1, and 2 are supported. Value 0 supports all output types; value 1 supports only the FP8 output; value 2 supports only the FLOAT4_E2M1 output. | INT64 | - | - |
| dst_type_max | Optional attribute | Specifies the value of maxType, corresponding to Amax(DType) in the formula. The default value is 0.0. When scale_alg is 2, 0.0 or [6.0, 12.0] is supported; for other scale_alg values, only 0.0 is supported. | FLOAT | - | - |

## Constraint Description

- `x` supports only non-transposed input; `weight` supports non-transposed and transposed input, and the transpose attribute of `weight_scale` must be consistent with that of `weight`.
- `N` must be an integer multiple of 64, and the minimum valid value of N is 64 in non-empty scenarios; the value range of the first dimension of `group_list` is [1, 1024].
- When `group_list_type=0`, `group_list` must be a non-negative, monotonically non-decreasing sequence, and the last value must not be greater than M; when `group_list_type=1`, all elements must be non-negative, and the sum must not be greater than M. The caller must ensure the element values of the Tensor at runtime, and the first-stage interface does not read device-side data.
- In the MXFP4 scenario, `K` must be an even number and cannot be 2, and the trailing dimension of the NZ storageShape of weight is 64; in the MXFP8 scenario, the trailing dimension is 32.
- In non-empty scenarios, K must be greater than 0.
- `scale_alg=1` supports only the FLOAT8 output; `scale_alg=2` supports only the FLOAT4_E2M1 output; the FLOAT4_E1M2 output supports only `scale_alg=0`.
- When `bias` is empty, nullptr, an empty tensorList, or an empty tensorList of length 1 whose element shape is (0) is supported.

## Interface Invocation

| Invocation Method | Sample | Description |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_grouped_matmul_activation_quant_weight_nz](examples/arch35/test_aclnn_grouped_matmul_activation_quant_weight_nz.cpp) | Calls the GroupedMatmulActivationQuant operator in the mode of [aclnnGroupedMatmulActivationQuantWeightNz](docs/aclnnGroupedMatmulActivationQuantWeightNz_en.md). |
| PyTorch API | - | Calls the grouped_matmul_activation_quant operator through the [grouped_matmul_activation_quant](docs/torchapi_grouped_matmul_activation_quant_en.md) API. |
