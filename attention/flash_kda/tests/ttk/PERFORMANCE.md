# FlashKDA A5/A6 统一性能测量

两侧直接读取唯一的 `cases/ttk_cases.csv` 中 25 条 profiling 用例，保持相同输入、seed、融合/缓存开关及训推语义。不再复制或生成 A6 CSV。metadata 使用架构原生容量，公共传输准备在测量外；性能用例不做精度断言。

## 两项指标

| 指标 | 纳入范围 | 排除范围 |
| --- | --- | --- |
| 主 kernel | profiler 中当前模板完整计算 kernel 的 `Duration(us)` | metadata、Cast/Transpose/Zero 等辅助核、预热、首次编译 |
| 端到端 | 每次重新生成 metadata、前向内部转换/分配、公开三项入口执行（包括公开 chunk_states 的 FP32→BF16 转换）和 NPU 同步的墙钟时间 | 输入构造/H2D、预热、首次编译、profiler 开销 |

两者单独采样。端到端不通过“主 kernel 加一次 metadata 时间”估算，也不包装或替换算子校验函数。每侧都在同一设备/stream 上串行执行，默认预热 10 次、采样 50 次、独立测量 3 轮，取三轮均值的中位数。两侧设备应独占，记录相应 SDK/wheel 指纹；不能将旧 0.8.2 的 A6 测量直接当作当前 v3 基线。

2026-11-17 接口适配后的数据须与相同接口、输入、seed、输出开关和计时边界的 A5 结果比较。旧接口 BNSD out 先转为相同逻辑布局再核对数值；训练新增状态返回的端到端开销单独标注。保留 A6 修改前后数据；缺少可比 A5 数据时逐例标为待验收，不能判为达标。`benchmark.py` 在 environment.json 中记录固定参考提交作为 `interface_contract`；比较工具拒绝缺失或不匹配的接口标识，旧报告需在匹配版本重新采集，不能仅手改标识。

## 推荐：同一脚本完成两种计时

先完成对应架构的真实编译和安装冒烟。以下命令分别在 A5、A6 匹配环境的仓库根目录执行；`benchmark.py` 默认 `KDA_DUT=installed`，三个接口来自专用 wheel 并要求 Native require；源码测量显式设置 `KDA_DUT=source`，独立归档。`SDK_TAG` 填实际 wheel/source 指纹，输出目录必须位于仓库外且不存在。

```bash
# A5 环境
export KDA_DUT=installed
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python attention/flash_kda/tests/ttk/benchmark.py \
  --arch arch35 --sdk-tag "$SDK_TAG" --templates auto staged persistent \
  --output /tmp/kda-a5-perf
python attention/flash_kda/tests/ttk/kda_runtime.py --check-native /tmp/kda-a5-perf/evidence

# A6 环境：DSL/OpKit 0.8.3，匹配 v3 包
export CANNBOTDSL_NPU_ARCH=dav-9201
export CANNBOTDSL_CACHE_DIR="$(mktemp -d /tmp/kda-a6-cache.XXXXXX)"
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python attention/flash_kda/tests/ttk/benchmark.py \
  --arch arch92 --sdk-tag "$SDK_TAG" --templates auto staged persistent \
  --output /tmp/kda-a6-perf
python attention/flash_kda/tests/ttk/kda_runtime.py --check-native /tmp/kda-a6-perf/evidence
```

只测自动选路时使用 `--templates auto`；先定位关键 shape 时加 `--cases P22-001 P24-001`。双模板顺序按 case/轮次交替，强制模板仅为测试入口，不增加算子参数。每个 profiler 目录必须恰好包含 50 条选定模板的主 kernel 记录；缺失、混入其他模板或统计到预热都会报错，不回退使用 Host 时间。完整调度、有效核数、DV、输入哈希、源码身份、主 kernel 名称和原始采样保存在输出目录。

将两侧结果复制到同一机器后比较：

```bash
python attention/flash_kda/tests/ttk/compare_performance.py \
  /tmp/kda-a5-perf /tmp/kda-a6-perf > /tmp/kda-performance-comparison.json
```

比较器拒绝输入哈希、属性、case/template、轮次或采样协议不一致的数据。逐例两项 A6/A5 比值均不超过 1 才通过；未达标退出码为 1。不同架构的 metadata、有效核数和实际自动模板允许不同，不要求两侧 metadata 哈希相同。

## TTK 入口

两架构分别使用自己的 adapter、输入构造和 ABI 校验，共用 `assets/performance.py` 计时工具。先设置架构、安装入口、PYTHONPATH 和独立证据目录，再选择同一 CSV 的 profiling 编号：

```bash
KDA_PERF_IDS="$(python "$KDA_TTK/kda_cases.py" --suite profiling --ids)"
KDA_PERF_TIMING=kernel KDA_PERF_TEMPLATE=auto python -m ttk e2e \
  -i "$KDA_TTK/cases/ttk_cases.csv" -t "$KDA_PERF_IDS" \
  --plugin "$KDA_PLUGIN" --run 50 --warmup true -o "$KDA_OUT/kernel.csv"

KDA_PERF_TIMING=e2e KDA_PERF_TEMPLATE=auto python -m ttk e2e \
  -i "$KDA_TTK/cases/ttk_cases.csv" -t "$KDA_PERF_IDS" \
  --plugin "$KDA_PLUGIN" --run 50 --warmup true \
  --task-prof=false -o "$KDA_OUT/e2e.csv"
python "$KDA_TTK/kda_runtime.py" --check-native "$KDA_EVIDENCE_DIR"
```

可将 `KDA_PERF_TEMPLATE` 改为 staged 或 persistent；强制模板调用所选来源内的私有 metadata 构建器，不新增公开算子参数。auto 使用公开 metadata 接口。TTK 端到端从 evidence 的 `e2e_us` 读取完整墙钟采样；主 kernel 从 profiler 提取。独立三轮和最终两指标对比优先使用上面的 benchmark；运行结束后检查输出 evidence 下 Native 命中。无精度断言的性能数据不代表训练缓存精度验收。

## 本轮 A6 搬运优化与复测

共享 Stage1 将 Q/K 搬运拆为 issue/finish：当前任务的 Q/K 使用者和公开缓存写回完成后，复用原 UB 槽预取下一任务，尝试与当前 Cube 三角逆计算重叠。staged 和 persistent 共用此路径，未增加 Q/K 双缓冲或改变归一化、门控及逆矩阵运算。

`0832a5119` 的 persistent 跨组 UB 状态驻留已撤回。用户在 A6 对同一用例做 bisect：`06397b81a`、`347985d5a` 通过，`0832a5119` 在两模板 final_state 交叉检查中失败，报告最大偏差约 0.0236。当前计算源码恢复为 `347985d5a`，保留本节 Q/K 预取及统一性能测量。persistent 继续使用 DV128、四 chunk 分组和 FP32 GM 组间交接；编译缓存键及路由特征也一并恢复。

原组间 `h` 与其 TensorSpec 均为 FP32，GM 写回直接复制 `ub_state_fp32`；两种路径在发布 L1 时使用同一 BF16 转换。因此 bisect 能确认回归提交，尚不能证明驻留路径“更精确”或 staged 多了低精度舍入。UB 跨 rewind 生命周期及实际编译产物仍需设备证据，本地控制流模拟不能验证它们。

更新后使用新进程及独立缓存执行公共 TTK 正确性与 13 条红线，再运行 25 条性能用例；保留原精度要求，单条用例通过不等于这些套件已全部通过。不要把 `0832a5119` 的数据作为当前实现的有效性能基线。
