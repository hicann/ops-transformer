# FlashAttn：全 FP32 bench 的三方比较

三份结果使用同一组最终输入：

| 角色 | 来源 | 比较器位置 |
| --- | --- | --- |
| NPU | 被测 FlashAttn 算子 | `output` |
| golden | 现有 golden（`impl/golden.py`），已按 FlashMLA 的方式对齐算子中间精度 | `third_party` |
| bench | 复用现有 golden 适配层的全 FP32 bench（`impl/fp32_bench.py`） | `golden`（共同基准） |

对应 `CrossCheckComparison(npu, bench, ..., third_party=golden)`，与
SparseFlashAttention、FlashMLA 使用相同的角色映射。衡量 NPU 相对 bench 的
误差与 golden 相对 bench 的误差之比。

golden 侧按 FlashMLA golden（`attention/flash_mla_with_kvcache`）的方式
对齐算子中间精度：未归一化 exp 权重量化到输入精度再进 PV（算子
vector->Cube 交接的 P cast）、sum 用未舍入权重、PV FP32 累加、最后除以
FP32 sum；其余数值链与原来一致，双方比较使用同一份 golden。

**这里的 bench 不做任何中间 cast：不把 P 舍入到低精度、不引入任何低精度
中间量，全程 FP32 计算。它不是 CUDA 中间精度的模拟**（与 FlashMLA 的
CUDA bench 不同），不需要安装任何第三方库。

## 数值路径与范围

- `impl/fp32_bench.py` 复用现有 golden 的
  布局/长度/页表/mask 适配层（布局 BSND/BNSD/TND、PA_BBND/PA_BNBD/PA_NZ、
  分页恢复、变长长度、mask_mode 0/3/4）。
- bench 在 `compute_mode="full"` 下运行，任何阶段都不做中间 cast；
  golden 与 bench 的唯一区别是未归一化 exp 权重量化到输入精度后参与 PV。
- 支持 attention sinks（float32、1-D `(Q_N,)`）：golden 与 bench 均在 softmax 分母中叠加 `exp(sink)` 项。

## TTK 接入

- `customize_inputs` 完成长度、页表和 mask 填充后，在 `--compare cross_check`
  模式下计算并缓存当前用例的 golden/bench，仅保留一个用例的输出。
- golden 回调读取缓存，避免 TTK 的 Promote 改变 FP16/BF16 输入精度。
- `spec.compare` 显式接收 `compare_context`，按 `testcase_name` 找对应 bench。
- 未指定 cross_check 时保留原双方比较，不额外计算 bench。
- cross_check 缺少缓存、用例名不匹配或 TTK 比较器不可用会明确失败，不回退为
  双方比较；不支持跳过输入生成、只加载旧 golden 文件的手工回放。

L0 阈值沿用 SparseFlashAttention：MARE/MERE/RMSE 比值为 10/2/2，
小值域按输出 dtype 取一阶 ULP 尺度：FP16 为 `2^-10`，BF16 为 `2^-8`，
小值绝对误差阈值为 `2^-16`，小值错误点数比不超过 2。FP32 的 LSE 使用
相同参数。

## 运行

在已配置 CANN、自定义算子包和 Python 扩展的环境中，从 ops-test-kit 目录执行：

```bash
OPS_TRANSFORMER=/path/to/ops-transformer
python3 -m ttk e2e \
  -i "$OPS_TRANSFORMER/attention/flash_attn/tests/assets/testcase/flash_attn_e2e_smoke.csv" \
  --plugin "$OPS_TRANSFORMER/attention/flash_attn/tests/assets/spec.py" \
  --compare cross_check --warmup false --run 1 --pc 1 --seed 42 \
  -o /tmp/flash_attn_fp32_cross_check.csv
```

需要支持 `compare_context` 和自定义三方比较的 TTK 版本。部分 TTK 版本在
执行自定义 compare 之前会因 `--compare cross_check` 尝试通用 XPU provider
查询并输出无远端 provider 的警告；本插件使用本地缓存完成比较，不依赖该
查询结果，以自定义比较结果和最终 CSV 的 `precision_status` 为准。
