# GMMSwigluQuantWeightNzV3 MX A8W4 示例

示例通过 `aclnnGroupedMatmulSwigluQuantWeightNzV3` 两段式接口执行 A8W4、SwiGLU mode 2、可选 smoothScale 和 MX 输出量化。

安装匹配的算子包后，在仓库根目录运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --run_example grouped_matmul_swiglu_quant_v2 eager cust --example_name=grouped_matmul_swiglu_quant_weight_nz_v3_a8w4 --vendor_name=custom --soc=ascend950
```

环境脚本路径与 vendor_name 按实际安装配置填写。运行一次依次验证 `scaleAlg=0/1`，每种算法分别传入空 smoothScale 和非空 smoothScale。

- M=33、K=N=256、E=2；groupListType=1，分组为 `[1,32]`。
- x 为 E4M3FN、weight 为 E2M1 FRACTAL_NZ，数值均为 1；xScale/weightScale 为 E8M0，数值均为 1。
- weight 的接口 view 为 `(E,K,N)`，stride 为 `(K*N,1,K)`，物理存储为 `(E,K/64,N/16,16,64)`。
- weightScale 的接口 view 为 `(E,K/64,N,2)`，stride 为 `((K/64)*N*2,2,(K/64)*2,1)`；物理存储为 `(E,N,K/64,2)`。
- smoothScale 为 FP32、ND、`(E,N/2)`，两个专家分别使用 1 和 2；无 smoothScale 时传 nullptr。
- swigluMode=2、clampLimit=7、gluAlpha=1.702、gluBias=1，输出 y 为 E4M3FN、`(M,N/2)`，yScale 为 E8M0、`(M,ceil((N/2)/64),2)`。

程序回传 y 与 yScale，解码并逐元素校验重建后的数值，失败返回非零。全 1 输入经过矩阵乘和 clamp 后，无 smoothing 时预期值约为 56；非空 smoothing 时第二个专家的预期值约为 112。
