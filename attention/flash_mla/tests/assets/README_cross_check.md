# FlashMla：CPU 模拟 CUDA 的三方比较

三份结果使用同一组最终输入：

| 角色 | 来源 | 比较器位置 |
| --- | --- | --- |
| NPU | 被测 FlashMla 算子 | `output` |
| golden | 原有 Ascend 混合精度 CPU golden | `third_party` |
| bench | 新增 DeepSeek FlashMLA dense CUDA 数值路径的 CPU 模拟 | `golden`（共同基准） |

对应 `CrossCheckComparison(npu, bench, ..., third_party=golden)`，与
SparseFlashAttention 的自定义比较器使用相同的角色映射。衡量 NPU 相对
bench 的误差与原 golden 相对 bench 的误差之比。

**这里的 bench 模拟 CUDA 中间精度，不是全 FP32 高精度真值。**
不需要安装 FlashMLA、CUDA，也不需要 GPU 服务。

## 模拟来源与范围

固定参照 DeepSeek FlashMLA 的 dense decode 版本
[`c2067be3eaa0f2e98e10854c30898139d5d01d36`](https://github.com/deepseek-ai/FlashMLA/tree/c2067be3eaa0f2e98e10854c30898139d5d01d36)：

- `flash_mla/flash_mla_interface.py`：同名 `flash_mla` 接口。
- `csrc/kernels/config.h`：64-token KV 块，QK 维度 576，V 维度 512。
- `csrc/kernels/splitkv_mla.cu`：`wg0_bunch_0` / `wg1_bunch_0` 的
  online softmax、中间 P 类型转换，以及输出归一化/LSE。

[`impl/cuda_bench.py`](impl/cuda_bench.py) 独立实现数值计算：

1. FP16/BF16 Q、K 转 FP32 后做 QK，模拟低精度输入和 FP32 累加。
2. 分数乘 `softmax_scale * log2(e)`，按 64-token 块维护最大值。
3. 使用 `exp2` 求未归一化权重，并按最大值变化重缩放已有 sum 和输出累加器。
4. sum 使用未舍入的 FP32 权重；PV 之前将权重舍入到 Q 的 FP16/BF16。
5. PV 和输出累加保留 FP32，最后除以 sum，输出转回 Q dtype；LSE 为 FP32。

布局转换、分页恢复、变长长度与右对齐 causal mask 复用原 golden 的适配层，
注意力数值计算不调用原 `attention_single`。支持原适配层的 BSND/BNSD/TND、
PA_BBND/PA_BNBD/PA_NZ，以及 TND→NTD 等输出转换。不同 NPU 页长恢复后，
CPU bench 仍以 CUDA 的 64-token 块计算。

这是**单 KV split 的数值模型**：不复现 GPU SM 调度决定的 split 分区及 combine，
不模拟 WGMMA 指令内部舍入、双 warpgroup 归约次序、融合 FMA 和设备 exp2 近似。
因此不承诺与实卡 CUDA 逐位相同，也不能用它宣称通过了真实 CUDA 三方验证。
布局、页长和 head 数由适配层转换，不代表 CUDA 原始接口支持这些所有组合。
空 KV、完全被 causal 屏蔽和 `seqused_q` 填充行采用算子契约：output=0、LSE=+inf；
CUDA 极端非有限值的硬件特例不在模拟范围内。非 causal 的非零任意 mask 会明确报错，
因为被参照的 CUDA 接口只有 causal 开关。

## TTK 接入

- `customize_inputs` 完成长度、页表和 mask 填充后，在 `--compare cross_check`
  模式下计算并缓存当前用例的 golden/bench，仅保留一个用例的输出。
- golden 回调读取缓存，避免 TTK 的 Promote 改变 FP16/BF16 中间舍入语义。
- `spec.compare` 显式接收 `compare_context`，按 `testcase_name` 找对应 bench。
- 未指定 cross_check 时保留原双方比较，不额外计算 bench。
- cross_check 缺少缓存、用例名不匹配或 TTK 比较器不可用会明确失败，不回退为双方比较。
- LSE 开启时也参与三方比较；关闭时双方参考均为 `None`，按 `SUPPRESSED` 处理。
- 不支持跳过输入生成、只加载旧 golden 文件的手工回放：文件里没有 CPU bench 缓存。

L0 阈值沿用 SparseFlashAttention：MARE/MERE/RMSE 比值为 10/2/2，
小值域按输出 dtype 取一阶 ULP 尺度：FP16 为 `2^-10`，BF16 为 `2^-8`
（TTK 框架 BF16 默认值；分母低于可表示精度会把末位翻转放大），
小值绝对误差阈值为 `2^-16`，小值错误点数比不超过 2。
FP32 的 LSE 使用相同参数。比值分母保护和 NaN/Inf 判定使用 TTK 内置实现。

## 运行

在已配置 CANN、自定义算子包和 Python 扩展的环境中，从 ops-test-kit 目录执行：

```bash
OPS_TRANSFORMER=/path/to/ops-transformer
python3 -m ttk e2e \
  -i "$OPS_TRANSFORMER/attention/flash_mla/tests/assets/testcase/flash_mla_e2e_smoke.csv" \
  --plugin "$OPS_TRANSFORMER/attention/flash_mla/tests/assets/spec.py" \
  --compare cross_check --warmup false --run 1 --pc 1 --seed 42 \
  -o /tmp/flash_mla_cpu_cuda_cross_check.csv
```

需要支持动态 metadata `(-1,)` 和 `compare_context` 的 TTK 版本。
metadata 仍由原 `npu_preprocess` 生成，不参与 CPU bench 计算。

部分 TTK 版本在执行自定义 compare 之前，会因 `--compare cross_check` 尝试通用
XPU provider 查询，并输出无远端 provider 的警告。本插件使用本地缓存完成比较，
不依赖该查询结果；以自定义比较结果和最终 CSV 的 `precision_status` 为准。
不要把通用 XPU 查询告警当成本插件已运行 CUDA 或发生了双方比较回退。
