# GroupedMatmulActivation Example

This example calls `aclnnGroupedMatmulActivationWeightNz` with MXFP4 E2M1 A4W4 inputs and WeightNZ weights. It applies SwiGLU mode 2 and outputs BF16 directly, without output quantization or yScale.

## Build and Run

Install an operator package containing GroupedMatmulActivation, then run from the repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --run_example grouped_matmul_activation eager cust --example_name=grouped_matmul_activation_weight_nz --vendor_name=custom --soc=ascend950
```

Set the environment script path and `vendor_name` according to your installation. See the [repository example instructions](../../../docs/zh/invocation/quick_op_invocation.md) for the command format.

## Example Configuration

- One expert, M=32, K=128, N=128, with non-transposed weights. The BF16 output shape is `(32,64)`.
- x and weight contain ones: each FP4 byte contains two elements, encoded as `0x22`. E8M0 scales are one, encoded as 127.
- The logical weight shape is `(1,128,128)` and the NZ storage shape is `(1,2,8,16,64)`. Shapes and strides count logical elements; FP4 allocation uses half a byte per element. Constant inputs need no additional NZ byte rearrangement.
- groupListType=1; groupList is `[32]` on the device. bias and tuningConfig are empty.
- dequantMode=2, dequantDtype=0, actType="swiglu", swigluMode=2.
- clampLimit=7, gluAlpha=1.702, gluBias=1.

The example executes both ACLNN stages, synchronizes, and copies all BF16 output elements to the host. With all-one inputs, both matmul branches produce 128 and are clamped to 7. The expected result is `7 / (1 + exp(-1.702 * 7)) * (7 + 1)`, approximately 56. Each element is checked for a finite value and BF16 error tolerance; a failure returns a nonzero status.

## Torch and ACLNN Parameters

This example calls ACLNN directly with `dequantMode=2` and `dequantDtype=0` from its header. The Torch API does not expose `dequant_mode`, `dequant_dtype`, or `transpose_weight`: its wrapper fixes the dequantization mode and intermediate dtype, and infers transposition from the weight and scale views. Do not pass these three attributes to Torch.

Torch retains `activation_type`, `group_list_type`, `tuning_config`, `swiglu_mode`, `clamp_limit`, `glu_alpha`, and `glu_bias`. Currently, only `activation_type="swiglu"` and `swiglu_mode=2` are supported.
