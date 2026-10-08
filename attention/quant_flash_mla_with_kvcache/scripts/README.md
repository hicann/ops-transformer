# QMLA 编译、验证与采集

入口：`attention/quant_flash_mla_with_kvcache/build_and_run.sh`。
固定使用 conda `qmla`、Ascend950、最多 10 个可用 CPU 核和 10 个编译任务。
主算子与 `quant_flash_mla_with_kvcache_metadata` 一起构建，安装到
`build_out/qmla_install`，Python wheel 安装到 `qmla` 环境。

当前容器 cgroup 只读。资源启动器每 100 ms 检查进程树 RSS，达到 18 GiB
立即终止，预留 2 GiB 余量。它不是 20 GiB 硬限制，不能排除瞬时超限；
接受这一限制方式后，显式设置 `QMLA_ALLOW_RSS_WATCHDOG=1`：

```bash
export QMLA_ALLOW_RSS_WATCHDOG=1
bash attention/quant_flash_mla_with_kvcache/build_and_run.sh all
bash attention/quant_flash_mla_with_kvcache/build_and_run.sh profile
```

`build` 只构建/安装；`run` 只验证。`run`/`all` 后的参数透传给 golden 脚本。
例如 `run --golden-device=cpu` 使用 CPU reference；默认是 NPU reference。
默认每次直接生成输入，全部结果只保留在内存，不保存或加载 tensor。
只有显式传入 `--cache-dir=<目录>` 才启用原来的缓存工作流。
原有 `--mode=cpu` 名称和 `*_cpu_output.pt` 缓存名保留兼容，它们表示 reference
阶段/结果，实际计算设备由 `--golden-device` 决定。默认在 NPU 上分块生成随机 tensor、量化、转换布局、计算 reference 和比较误差；
CPU 只处理少量 block table/metadata 调度信息、统计标量和缓存 I/O。
`--gen-device=cpu` 可恢复原 NumPy 生成方式；NPU 种子固定，但与 NumPy 随机序列不同。
NPU 比较按 4M 元素分块，保持原 QMLA 阈值，并拒绝 NaN/不匹配的 Inf。

默认场景保持原脚本配置：B=1、96 个 Q heads、1 个 KV head、Q 长度 8、
KV 长度 102400、PA_NZ、FP8 E4M3、BF16 输出、不返回 LSE。

采集命令依据 `ops-transformer-testkit/core/profiler.py` 和 `core/msopprof.py`：

```bash
msprof --output=<dir> --aic-mode=task-based <qmla-python> <golden.py> --mode=npu --flush-l2 --warmup=5 --runs=20
```

先运行 `all` 验证精度，再运行 `profile`。每次采集都按固定种子重新生成默认
case 的输入，采集模式不执行 golden/比较。性能统计只保留
主算子的 20 次测量，排除前 5 次预热；每轮用 256 MiB FP32 buffer 做 sum 后同步，
与 testkit 的默认 FLUSH_MODE=read 一致。耗时汇总包含均值、中位数和按 testkit
冷启动口径去掉一个最大值、一个最小值后的均值；MFU 使用后者。metadata 在准备阶段调用一次。
默认 case 的矩阵计算量为 171127603200 FLOPs，FP8 峰值按 756.94 TFLOPS。
只采集普通性能，不运行深度指标和流水图采集。输出位于
`build_out/qmla_profile/<timestamp>`：原始采集文件、完整日志、命令 JSON 和
`performance_summary.json`。

已在 qmla 环境、Ascend950PR 上完成默认 case 的编译和 NPU 验证：
393216 个输出元素全部通过精度检查；无缓存运行 CPU 进程树峰值约 1.94 GiB。

优化前默认 case 刷 L2 后结果：预热 5 次、测量 20 次；按框架冷启动口径去掉
一个最大值和最小值后平均 338.193722 μs，MFU 66.848716%。
全 20 次均值 344.359150 μs，中位数 322.793500 μs，存在耗时抖动。

优化后的保留版本为 268.571111 μs、MFU 84.178138%，85% 目标尚未达到。
修改说明、退化实验和复现命令见 [优化记录](../optimization_notes.md)。
