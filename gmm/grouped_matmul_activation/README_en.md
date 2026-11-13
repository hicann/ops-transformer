# GroupedMatmulActivation

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: fuses matrix multiplication grouped along the M axis (Grouped Matrix Multiplication, GMM) and Swish-Gated Linear Unit (SwiGLU) activation for the expert forward computation of a Mixture of Experts (MoE) network. The input and weight use MXFP4 E2M1, the weight is in FRACTAL_NZ format, the output is BFLOAT16, and output quantization is not performed.
- Computation formula: apply the corresponding E8M0 scale factors to the input and weight of each expert, and then perform matrix multiplication.

  $$
  C_i = \operatorname{DequantMX}(X_i,S_i^X) \times \operatorname{DequantMX}(W_i,S_i^W)
  $$

  Each E8M0 scale factor applies to 32 consecutive elements along the K axis. When `swiglu_mode=2`, the matmul result is split into front and back halves along the N axis. The front half is the activation branch, and the back half is the linear branch:

  $$
  u_i = \min(C_i[..., :N/2], clamp\_limit)
  $$

  $$
  v_i = \operatorname{clip}(C_i[..., N/2:],-clamp\_limit,clamp\_limit)+glu\_bias
  $$

  $$
  y_i = \operatorname{Cast}_{BF16}\left(\frac{u_i}{1+\exp(-glu\_alpha\times u_i)}\times v_i\right)
  $$

  `M` is the total number of input tokens, `K` is the input feature dimension, `N` is the matmul output dimension, `E` is the number of experts, and `i` is the expert index.

## Parameter Description

The following table uses the parameter names and order defined by the operator. For the parameter mappings of ACLNN and PyTorch, see the corresponding API documents.
The transposed shapes of weight and weight_scale in the table are the shapes passed to the operator IR after channel normalization, not the view shapes passed in directly by ACLNN callers. For the difference between the two, see [ACLNN Constraints](docs/aclnnGroupedMatmulActivationWeightNz.md#约束说明).

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :---: | :--- | :--- | :---: |
| x | Input | Activation matrix with logical shape `(M,K)`. | FLOAT4_E2M1 | ND |
| x_scale | Input | Scale factor of the activation, with shape `(M,K/64,2)`. | FLOAT8_E8M0 | ND |
| group_list | Input | Expert grouping information, with shape `(E)`. | INT64 | ND |
| weight | Input | Dynamic TensorList with a length of 1, containing the weights of all experts. The logical shape is `(E,K,N)` when not transposed and `(E,N,K)` when transposed. | FLOAT4_E2M1 | FRACTAL_NZ |
| weight_scale | Input | Dynamic TensorList with a length of 1. The shape is `(E,K/64,N,2)` when not transposed and `(E,N,K/64,2)` when transposed. | FLOAT8_E8M0 | ND |
| bias | Input | Dynamic TensorList. Reserved matmul bias, which must currently be empty. | BFLOAT16 | ND |
| y | Output | SwiGLU result, with shape `(M,N/2)`. | BFLOAT16 | ND |
| dequant_mode | Optional attribute | Dequantization mode. The value 2 indicates MX dequantization. Default value: 2. | INT | - |
| dequant_dtype | Optional attribute | Data type of the intermediate matmul result. The value 0 indicates FLOAT32. Default value: 0. | INT | - |
| act_type | Optional attribute | Activation type. The value is `"swiglu"`. Default value: `"swiglu"`. | STRING | - |
| group_list_type | Optional attribute | The value is 0 or 1. The value 0 indicates cumulative end positions, and 1 indicates the number of tokens in each group. Default value: 0. | INT | - |
| tuning_config | Optional attribute | Reserved tuning parameter, which must currently be an empty list. Default value: `[]`. | LIST_INT | - |
| swiglu_mode | Optional attribute | Default value: 2, indicating SwiGLU that splits the input into front and back halves, applies clipping before activation, and applies alpha/bias. The value 0 is reserved for internal compatibility calls of the earlier WeightNzV3. Public APIs support only 2. | INT | - |
| clamp_limit | Optional attribute | Clipping upper limit, which must be a finite positive number. Default value: 7.0. | FLOAT | - |
| glu_alpha | Optional attribute | Scaling coefficient of the sigmoid input, which must be a finite number. Default value: 1.702. | FLOAT | - |
| glu_bias | Optional attribute | Bias of the linear branch, which must be a finite number. Default value: 1.0. | FLOAT | - |
| transpose_weight | Optional attribute | Default value: false. false indicates the `(E,K,N)` view of the weight, and true indicates the `(E,N,K)` view. weight_scale must use the corresponding layout. ACLNN infers this attribute from the view. | BOOL | - |

## Constraints

- Grouping is performed only along the M axis. M must be greater than 0. K and N must be positive numbers that satisfy 128 alignment. The number of experts E ranges from `[1,1024]`.
- `weight` and `weight_scale` pass the data of all E experts through one tensor respectively. The data cannot be split into multiple TensorList elements by expert.
- When `group_list_type=0`, the grouping list must be non-negative and monotonically non-decreasing, and the last value must not be greater than M. When `group_list_type=1`, the number of tokens in each group must be non-negative, and the total must not exceed M.
- The PyTorch API uses `activation_type` and does not expose `dequant_mode`, `dequant_dtype`, or `transpose_weight`. For constraints on the CPU grouping list and input storage, see [PyTorch API document](docs/torchapi_grouped_matmul_activation.md).

## Invocation

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| aclnn API | [Runnable Example](examples/arch35/test_aclnn_grouped_matmul_activation_weight_nz.cpp) | Invoked through the two-stage APIs of [aclnnGroupedMatmulActivationWeightNz](docs/aclnnGroupedMatmulActivationWeightNz.md). See [Build and Run](examples/README_en.md). |
| PyTorch API | [Invocation Example](docs/torchapi_grouped_matmul_activation.md#调用示例) | Invoked through the [grouped_matmul_activation](docs/torchapi_grouped_matmul_activation.md) API. |

For the method of building the operator package, see [Build from Source](../../docs/zh/install/quick_install.md).
