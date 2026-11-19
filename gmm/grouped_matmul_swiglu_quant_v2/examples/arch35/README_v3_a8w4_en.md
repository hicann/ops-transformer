# GMMSwigluQuantWeightNzV3 MX A8W4 Example

English | [简体中文](./README_v3_a8w4.md)

This example uses the two-stage `aclnnGroupedMatmulSwigluQuantWeightNzV3` API to run A8W4, SwiGLU mode 2, optional smoothScale, and MX output quantization.

After installing a compatible operator package, run the following commands from the repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --run_example grouped_matmul_swiglu_quant_v2 eager cust --example_name=grouped_matmul_swiglu_quant_weight_nz_v3_a8w4 --vendor_name=custom --soc=ascend950
```

Adjust the environment script path and vendor_name to match the installation. One execution checks scaleAlg=0 and scaleAlg=1, each with smoothScale omitted and supplied.

- M=33, K=N=256, E=2; groupListType=1, with group sizes `[1,32]`.
- x uses E4M3FN and weight uses E2M1 FRACTAL_NZ, both filled with ones; xScale and weightScale use E8M0 and are filled with ones.
- The weight API view is `(E,K,N)` with strides `(K*N,1,K)`; its physical storage is `(E,K/64,N/16,16,64)`.
- The weightScale API view is `(E,K/64,N,2)` with strides `((K/64)*N*2,2,(K/64)*2,1)`; its physical storage is `(E,N,K/64,2)`.
- smoothScale uses FP32, ND, and shape `(E,N/2)`. The two experts use factors of 1 and 2, respectively; pass nullptr to omit smoothScale.
- swigluMode=2, clampLimit=7, gluAlpha=1.702, and gluBias=1. Output y uses E4M3FN with shape `(M,N/2)`; yScale uses E8M0 with shape `(M,ceil((N/2)/64),2)`.

The program copies y and yScale back to the host, decodes them, and checks the reconstructed values element by element. A failed check returns a nonzero exit code. For all-one inputs, matrix multiplication followed by clamping yields an expected value of approximately 56 without smoothing; with smoothing, the expected value for the second expert is approximately 112.
