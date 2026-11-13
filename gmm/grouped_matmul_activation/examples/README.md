# GroupedMatmulActivation 示例

本示例调用 `aclnnGroupedMatmulActivationWeightNz`，输入为 MXFP4 E2M1 A4W4、WeightNZ，执行 SwiGLU mode 2 后直接输出 BF16，不执行输出量化，不产生 yScale。

## 编译运行

安装包含 GroupedMatmulActivation 的算子包，在仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --run_example grouped_matmul_activation eager cust --example_name=grouped_matmul_activation_weight_nz --vendor_name=custom --soc=ascend950
```

环境脚本路径和 `vendor_name` 按实际安装配置填写。命令格式参考[仓库标准示例说明](../../../docs/zh/invocation/quick_op_invocation.md)。

## 示例配置

- 单专家，M=32、K=128、N=128，非转置权重；BF16 输出形状为 `(32,64)`。
- x 和 weight 全为 1：FP4 每字节两个元素，编码为 `0x22`；E8M0 scale 全为 1，编码为 127。
- weight 逻辑形状为 `(1,128,128)`，NZ 存储形状为 `(1,2,8,16,64)`；shape 和 stride 按逻辑元素数填写，FP4 内存按半字节计算。常量输入无需额外重排 NZ 字节。
- groupListType=1，groupList 为 `[32]`，放在 Device；bias、tuningConfig 为空。
- dequantMode=2、dequantDtype=0、actType="swiglu"、swigluMode=2。
- clampLimit=7、gluAlpha=1.702、gluBias=1。

示例执行两段式 ACLNN 调用、同步并回传全部 BF16 输出。全 1 输入的两路矩阵乘结果均为 128，经过 clamp 后均为 7，预期结果为 `7 / (1 + exp(-1.702 * 7)) * (7 + 1)`，约为 56。逐元素检查有限值和 BF16 误差，失败时返回非零状态。

## Torch 与 ACLNN 参数区别

本示例直接调用 ACLNN，使用头文件中的 `dequantMode=2`、`dequantDtype=0`。Torch 接口不暴露 `dequant_mode`、`dequant_dtype`、`transpose_weight`：反量化模式和中间结果类型由封装固定传入，转置状态按权重及 scale 的视图识别。Torch 调用时不要传入这三个属性。

Torch 仍保留 `activation_type`、`group_list_type`、`tuning_config`、`swiglu_mode`、`clamp_limit`、`glu_alpha`、`glu_bias`，其中当前仅支持 `activation_type="swiglu"` 和 `swiglu_mode=2`。
