# CCU AllToAll + MXFP8 MatMul 通算融合示例（ccu_alltoall）

CCU 直调（`<<<>>>` kernel 直调）AllToAll 集合通信 + Blaze MXFP8 MatMul **双流通算融合**示例：
一次 `AllToAllMatmulCustom` 调用同时下发两个 kernel——streamCcu 上的 `__ccu_host__` 通信
kernel 与 streamAicore 上的 AIC 融合计算 kernel，通信与计算在 tile 级 overlap。

- 数据语义（K-并行 GEMM）：`Y[M/R,N] = Σ_r recvData[r块,:] @ x2[r*K段,:]`（L0C 跨源累加，
  self 项直读 x1）；`x1=[M,K]` FP8 E4M3FN（token 轴均分 R 块做 A2A），
  `workspace=[recvData|recvScale]`，`x2=[R*K,N]` 完整权重不通信，输出 BF16
- 端到端 ~230us（4 rank，M=8192 / tileM=512 / K=2048 / N=3584；口径见「性能测试方法」）
- 仅支持 Ascend 950（dav-3510）+ CANN ≥ 9.2.0；rank 数 = 本机 NPU 数（进程内每卡一线程）

## 目录结构

```
ccu_alltoall/
├── src/                                 # 单翻译单元（main.cpp 唯一编译源，-xasc）
│   ├── main.cpp                         # 示例入口：读数据 → FillTiling → 调用 → 校验 → 释放
│   ├── all_to_all_matmul_ccu.asc        # 锚点/高级接口（§1 host 编排 / §2 AIC kernel 入口 + Custom）
│   ├── ccu_all_to_all.asc               # CCU 通信服务（契约+引擎+资源 / tiling 契约 / host 装配 / __ccu_host__ kernel）
│   ├── all_to_all_matmul_tiling_data.h  # tiling 聚合结构体 + FillAllToAllMatmulTilingData（SWAT）
│   └── all_to_all_matmul_aic.h          # AIC 计算逻辑（同步原语 / WaitPolicy + matmul 编排）
├── scripts/
│   ├── gen_data.py                      # 数据生成（位型闭式，与原 C++ 生成器逐位一致）
│   ├── run.sh                           # 数据(按需) + 构建 + 运行 / --perf msprof 采集
│   └── parse_prof.py                    # msprof op_summary 解析
├── CMakeLists.txt                       # 自包含构建入口（独立 cmake，见下）
├── README.md
└── .gitignore                           # build/ data/ output/ prof/
```

include 链（单 TU，顺序即依赖顺序）：

```
main.cpp(-xasc) → all_to_all_matmul_ccu.asc（锚点，唯一引入）
                → { ccu_all_to_all.asc → all_to_all_matmul_tiling_data.h → all_to_all_matmul_aic.h } → 锚点体
```

后缀规则：`.asc` = 定义 kernel 入口【单次引入】，`.h` = 纯逻辑。

matmul kernel / tiling / utils 继续**原位引用** apace 仓（不拷贝），构建时经
`-I <repo>/mc2/common/op_kernel` 以 `apace/...` 前缀解析：

- `apace/kernel/matmul/quant_batch_matmul/all_to_all_qbmm_mx_kernel.h`
- `apace/utils/op_state_dump.h`
- `apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h`（hcomm/urma 共享件）
- `apace/tiling/quant_matmul_tiling_swat.h`

## 机制摘要（引自源文件头）

- **双流架构**（`AllToAllMatmulCustom` 一次调用下发两个 kernel）：
  - streamCcu：`CcuAllToAllMesh1DKernel`（`__ccu_host__`）——wake 后单类发布一次，循环
    dataCommTurn 轮（轮 0 = data 块 0 + scale 整块搭载），每轮末 PostSync 落地确认 +
    EventRecord 通知 AIC
  - streamAicore：`AllToAllQuantMatmulKernelCcu`——block 0 清事件残值 → commit（wake），
    全核融合 matmul 逐 tile 轮询 round-done，tile 级通算 overlap
- **事件契约**（AIC↔CCU 同步）：`ev[0] = wake`（AIC commit → CCU process，循环前唯一一次）；
  `ev[t+1] = data 轮 t done`（CCU notify → AIC wait，轮 0 含其搭载的 scale 整块）
- **参数直达 stage**：host 每次调用把通信参数块刷入 `kernelArg->stage`（随 CCU kernel
  launch 按值到达，不 chase 二级指针），kernel 启动即自读并按轮闭式派生
  （`LoadRoundArgs`）；AIC 不经手通信参数
- **单 TU 与 kernel 名冻结**：`ccu_all_to_all.asc` 的 `__ccu_host__` kernel 每个二进制只能
  引入一次（锚点 `all_to_all_matmul_ccu.asc` 是唯一引入处）；kernel 名
  `CcuAllToAllMesh1DKernel` 参与 binaryCacheTag 哈希（tag = 名哈希 + rankId，单进程多
  rank 线程共存的正确性依赖），永不改名
- **命名规则**（词根=功能域，防演化漂移）：`Ccu` = CCU 传输原语（只碰寄存器/实例/通道
  语义，与具体集合通信算法无关）；`AllToAll` = 服务函数（语义绑定 AllToAll 算法）

## 环境与构建

- Ascend 950（dav-3510），CANN ≥ 9.2.0（依赖 asc-comm 正式版 host launch 头
  `ccu_host_launch.h`）
- 依赖 ops-tensor（blaze / tensor_api，pin 43d554d）：优先复用
  `mc2/common/op_kernel/apace/.wtdeps/ops-tensor`（存在即用），否则自动 FetchContent 拉取到
  `build/third_party/`（需网络）
- ⚠️ overlay 坑：若本机 `~/Ascend` 下多个 cann-* 目录互相 overlay（`asc/include` 混入旧版
  tensor_api），构建/运行前显式 `export ASCEND_HOME_PATH=<cann-9.2.0 路径>`，勿依赖 latest 软链
- 构建（examples 树无统一构建入口，本示例独立 cmake）：

```bash
export ASCEND_HOME_PATH=~/Ascend/cann-9.2.0/x86_64-linux        # 或 apace/.wtdeps/cann
export PATH=~/Ascend/cann-9.2.0/x86_64-linux/ccec_compiler/bin:$PATH
cmake -S . -B build
cmake --build build --target ccu_alltoall --parallel 8
```

## 数据生成与运行

数据由 `scripts/gen_data.py` 生成（纯 stdlib；位型闭式与原 apace ST C++ 生成器逐位一致），
运行时 main 读取 `data/`（`--data <dir>` 可选）：

```bash
python3 scripts/gen_data.py --M 8192 --K 2048 --N 3584 --ranks 4 --out data
export HCCL_OP_EXPANSION_MODE=CCU_SCHED        # CCU 调度模式（必须）
./build/ccu_alltoall precision 8192 512 2048 3584    # M tileM K N
./build/ccu_alltoall perf 8192 512 2048 3584 10      # + loop
./build/ccu_alltoall --data /path/to/data precision 512 128 256 256
```

或直接用编排脚本（数据按需生成 + 构建 + 运行）：

```bash
bash scripts/run.sh                                       # 数据(按需)+构建+precision 默认 512 128 256 256
bash scripts/run.sh precision 8192 512 2048 3584          # 任意 case（meta 不匹配自动重生成）
bash scripts/run.sh perf 8192 512 2048 3584 10            # perf 直跑
bash scripts/run.sh --perf perf 8192 512 2048 3584 10     # perf + msprof 采集（prof/）
bash scripts/run.sh --gen precision 8192 512 2048 3584    # 仅生成数据
bash scripts/run.sh --skip-build ...                      # 跳过编译
```

rank 数 = 本机 NPU 数（`npu-smi info -l` 探测，`CCU_A2A_RANKS` 环境变量覆盖）；约束：
`M` 须被 rank 数整除，`blockRows(=M/R)` 须被 `tileM` 整除（dataCommTurn ≤ 16），`K` 须为
64 的倍数（MXFP8）。

### 数据文件

| 文件 | 形状/大小 | 说明 |
|---|---|---|
| `meta.txt` | 一行 `M K N R` | shape 与 rank 数一致性契约（不匹配报错并提示再生成命令） |
| `x1_rank{r}.bin` | [M,K] FP8 E4M3FN | rank r 的输入（行常量编码，行级溯源） |
| `x1_scale_rank{r}.bin` | M×(K/64×2) B | E8M0 全 1.0 |
| `x2.bin` | [R×K,N] FP8 E4M3FN | DN 列主序权重（各 rank 共享，不通信） |
| `x2_scale.bin` | R×(K/64×2)×N B | E8M0 全 1.0 |
| `y_expect_rank{r}.bin` | [M/R,N] BF16 | 期望输出（precision 比对；perf 跳过读取） |

## 验证语义

- 期望值为闭式位型：`Y[m] = K × Σ_{r=0}^{R-1} 2^((2i+r+m)%4 + r%2)`（各加数均为 2 的幂，
  float32 精确表示，取高 16 位即 BF16 位型）——**位级断言零舍入噪声**
- precision 模式：输出与 `y_expect_rank{r}.bin` 逐元素 BF16 位型比对，全部一致才 PASS
  （`=== ALL RANKS PASS ===`）；无论成败输出落盘 `output/<rank>/npu_out.bin`
- 行级溯源数据可分类定位错误：源 rank 混淆 / 目标路由混淆 / 行轮错位 / 字节损坏

## 性能测试方法

### 测试命令

```bash
# 环境准备（每次登录后）
export HCCL_OP_EXPANSION_MODE=CCU_SCHED

# 精度验证
./build/ccu_alltoall precision 8192 512 2048 3584

# 性能测试（参数：M tileM K N loop）
./build/ccu_alltoall perf 8192 512 2048 3584 20

# msprof 采集（板端标准口径）
msprof --application="./build/ccu_alltoall perf 8192 512 2048 3584 20" \
       --output=./prof --ai-core=on
```

### 看结果

```bash
# 方法一：直接看程序输出（event 计时，端到端口径）
# 关注 "steady avg=XXXus/iter" 行

# 方法二：msprof 解析（kernel duration 口径，标准指标）
# 打开 prof/PROF_*/mindstudio_profiler_output/op_summary_*.csv
# 过滤 Op Name 含 "AllToAllQuantMatmulKernelCcu" 的行
# 取 Task Duration(us) 列，每 rank 剔除首帧后取中位
```

**指标选哪个**：

- **rank 均值**：4 个 rank 各自取中位再平均——用于与历史基线对比（240us 口径）
- **makespan**（每帧取 4 rank 最慢者）：帧的真实耗时——用于报吞吐
- 大体积段（≥8MB）两口径基本一致；小体积段（≤4MB）因偏斜主导，只看 makespan

### 为什么插入重型算子

**问题**：4 张卡的 host 线程各自提交 kernel（Refresh → CCU launch → AIC launch），提交时间
有先后（~50-300us 偏差）。先提交的 rank 其 AIC kernel 先启动，但要等最慢 rank 的数据——
等待时间被计入它的 duration，导致 **rank 间耗时悬殊 50+us**，数据不可信。

**解决**：每次迭代结束后，在 streamAicore 上插入 4×128MB memset（~430us 设备占用）。设备
忙期间，所有 host 线程完成下一帧提交。当 memset 结束时，4 rank 的 kernel 同时启动——
**rank 极差从 50+us 收敛到 5~21us**。

**原理**：

```
无重型算子：  rank0 提交(快) → kernel 启动 → 等数据(~50us) → 计算
              rank3 提交(慢) → kernel 启动 → 计算（数据已到）

有重型算子：  rank0 提交 → memset(~430us) ← 期间 rank3 也完成提交
              rank3 提交 → memset(~430us)
              memset 结束 → 4 rank kernel 同时启动 → 同时计算
```

**注意**：msprof 解析时按 kernel name 过滤（只看 AllToAllQuantMatmulKernelCcu），memset
自动排除在统计外。重型算子只影响小体积段（≤4M 收益 −17~−31%），大体积段（≥8M）通信/计算
主导，无影响。

### 常用 case

| 用途 | 命令参数（M tileM K N） |
|---|---|
| 目标 case（16MB） | `8192 512 2048 3584` |
| 小 case 冒烟 | `512 128 256 256` |
| 数据量扫描 | `8192 512 {128\|256\|512\|1024\|2048\|4096\|8192} 3584` |

## 与 apace 仓库的关系

- 本示例是 CCU 变体的**唯一副本**（自 apace 三层结构迁移：`core/ccu_comm` 四件 +
  `kernel/fusions/all_to_all_quant_matmul` CCU 三件 + `tests/st/all_to_all_quant_matmul/ccu`，
  合并为 src/ 五件；机制零改动，仅 include/注释路径随搬移调整）
- matmul kernel / tiling / utils 原位引用 apace（见「目录结构」节清单），构建依赖仓库布局
- hcomm / urma 通信变体的 ST 仍在 `mc2/common/op_kernel/apace/tests/st/all_to_all_quant_matmul/`
