# AllGatherMatmulV3（ubmem融合方案）

本文档介绍 `experimental/mc2/all_gather_matmul_v3` 中基于 HCCL **ubmem 窗口（MTE window）** 的
AllGather + MX 量化矩阵乘融合方案，包括接口说明、支持范围、详细设计、编译部署与测试方法。

> 同目录族的 mc2 URMA 方案（`mc2/all_gather_matmul_v3`）通过 URMA all-gather 收集左矩阵后再做
> MX 量化矩阵乘；本方案改用ubmem 窗口完成 gather，AIV 通过读模式，直接把远端数据按照K方向切分的方式分批搬进本卡窗口、AIC 从窗口读取参与 MMAD，从而把通信与计算融合在同一个 kernel。

---

## 1. 方案概览

- **融合语义**：各卡持有自己的左矩阵 `x1[M, K]` 与 MX scale，AllGather 后得到
  `A_all[rankSize * M, K]`，再与本卡右矩阵 `x2` 做 MX 量化矩阵乘，输出 `y[rankSize * M, N]`。

  $$
  y = \sum_{g=0}^{\lceil K/32 \rceil}\left( AllGather(x1) @ x2 \right) \cdot
      \left( AllGather(x1\_scale) \cdot x2\_scale \right)
  $$

  scale 按 K 轴每 32 个元素一组（group size = 32），每组 2 个 e8m0 scale 值。

- **算子语义如图**：
![alt text](all_gather_process.png)
  1. Host 侧创建 ubmem 通信上下文；
  2. AIV：把本卡 `x1`/`x1_scale` 拷入本卡窗口对应槽位，并把其余 rank 的 `x1`/`x1_scale`
     拉入本卡窗口；按 K 方向分批次写入，每轮写完后通知AIC启动matmul计算；
  3. AIC：从窗口读取 `A_all` 与 scale（首轮allgather结束即可开始 MMAD），与本地 `x2` 完成矩阵乘，
     最后K方向全部算完后结果经 fixpipe 直接写 `y`。

- **主要收益**：
  - AIC 按 K 方向多轮次计算（而非等全部 AllGather 完成），首轮切K数据就绪即开始计算，掩盖通信时延，适用于decode小M场景；

---

## 2. 支持范围

### 2.1 产品

| 产品 | 支持 |
| --- | --- |
| Ascend 950PR / 950DT 系列 | 支持 |
| Atlas A3 / A2 / 推理系列 / 训练系列 | 不支持 |

### 2.2 dtype 组合

| x1 | x2 | y | 说明 |
| --- | --- | --- | --- |
| e4m3 | e4m3 | bf16 / fp16 | 同 dtype |
| e5m2 | e5m2 | bf16 / fp16 | 同 dtype |
| e4m3 | e5m2 | bf16 / fp16 | fp8 异构 |
| e5m2 | e4m3 | bf16 / fp16 | fp8 异构 |
| fp4(e2m1, uint8 打包) | fp4 | bf16 / fp16 | 仅 ND，x1/x2 必须同为 fp4 |

- fp8 场景 x1/x2 可任意混用；
- fp4 以 uint8 打包（2 元素/字节），实际传入末维为 `K/2`，需传 `x1_dtype=x2_dtype=296`。

### 2.3 右矩阵布局

| 布局 | 存储形态 | kernel LayoutB |
| --- | --- | --- |
| ND + 转置 B | `[N, K]`（对外可传 `.t()` 视图） | `dn_ext_layout_ptn` |
| ND + 非转置 B | 连续 `[K, N]` | `nd_ext_layout_ptn` |
| NZ + 转置 B | `(ceil(K/32), ceil(N/16), 16, 32)` | `zn_layout_ptn` |
| NZ + 非转置 B | `(ceil(N/32), ceil(K/16), 16, 32)` | `nz_layout_ptn` |

- scale 布局随方向切换：转置 B 为 `[N, ceil(K/64), 2]`，非转置 B 为 `[ceil(K/64), N, 2]`；
- fp4 仅支持 ND。

### 2.4 其他限制

- `rank_size ∈ {2, 4, 8}`；
- `K ∈ [256, 65535)`；`fp4` 场景 K 须为偶数；
- `bias` 暂不支持（必须传 `None`）；
- `gather_out` / `amax_out` 为预留输出，当前返回 `(0,)` 空占位 tensor。

---

## 3. 接口说明

### 3.1 函数原型

```python
cann_ops_transformer.all_gather_quant_matmul(
    x1,
    x2,
    group,
    *,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode=None,
) -> (Tensor, Tensor, Tensor)
```

ubmem 路径必须使用 `comm_mode="ubmem"`（默认值即 `"ubmem"`）。MX 量化路径必须传
`x1_scale`/`x2_scale`；传 `None` 会在 meta / torch extension / aclnn 任一层直接报错。

### 3.2 参数说明

| 参数 | 类型 | 必选 | 说明 | dtype | 格式 / shape |
| --- | --- | --- | --- | --- | --- |
| `x1` | Tensor | 是 | 各卡本地左矩阵。fp4 场景 uint8 打包，末维为 `K/2`，配合 `x1_dtype=296` | fp8_e4m3fn / fp8_e5m2 / uint8(fp4) | ND，`(M, K)`（fp4 为 `(M, K/2)`） |
| `x2` | Tensor | 是 | 右矩阵，支持转置/非转置两种方向 | fp8_e4m3fn / fp8_e5m2 / uint8(fp4) | ND：`[N,K]` 的 `.t()` 视图或连续 `[K,N]`；NZ：2D 逻辑 shape 或 4D storage |
| `group` | ProcessGroup | 是 | HCCL 通信域 | - | - |
| `bias` | Tensor | 否 | **ubmem 路径不支持，必须为 None** | - | - |
| `x1_scale` | Tensor | 是 | x1 的 MX scale | e8m0（uint8 存储时配 `x1_scale_dtype=293`） | `(M, ceil(K/64), 2)` |
| `x2_scale` | Tensor | 是 | x2 的 MX scale，布局随右矩阵方向 | e8m0（同上） | 转置：`(N, ceil(K/64), 2)`；非转置：`(ceil(K/64), N, 2)` |
| `group_sizes` | List[int] | 否 | MX 仅支持 `[1, 1, 32]`，None 时自动推导 | int | - |
| `x1_dtype` | int | 否 | None=按 tensor dtype；fp4 传 `296` | int | - |
| `x2_dtype` | int | 否 | None=按 tensor dtype；fp4 传 `296` | int | - |
| `x1_scale_dtype` | int | 否 | None=按 tensor dtype；uint8 存储 e8m0 传 `293` | int | - |
| `x2_scale_dtype` | int | 否 | 同上 | int | - |
| `y_dtype` | int | 否 | 输出 dtype：`15`(BF16，默认) / `5`(FP16)； | int | - |
| `comm_mode` | str | 否 | ubmem 路径必须为 `"ubmem"` | str | - |

### 3.3 返回值

| 参数 | 说明 | dtype | shape |
| --- | --- | --- | --- |
| `y` | 融合计算结果 | 由 `y_dtype` 指定（BF16/FP16） | `(M * rankSize, N)` |
| `gather_out` | 预留，未实现 | 同 `x1` | `(0,)` |
| `amax_out` | 预留，未实现 | float32 | `(0,)` |

### 3.4 调用示例（4 卡，ND 转置 B）

以下示例假设已完成第 5 节的算子包与 torch extension 安装，并已 `source` CANN 的 `set_env.sh`。

```python
import math

# 默认使用 CANN 的 HCCL 内置 buffer；若运行时提示窗口超限，再按提示设置
# os.environ["HCCL_BUFFSIZE"] = "2000"  # 单位 MB
import torch
import torch_npu
import torch.distributed as dist
import cann_ops_transformer


def run(rank, world_size, m, k, n):
    torch_npu.npu.set_device(rank % 4)
    dist.init_process_group("hccl", rank=rank, world_size=world_size,
                            init_method="tcp://127.0.0.1:29500")
    group = dist.new_group(ranks=list(range(world_size)))

    x1 = torch.randn(m, k).to(torch.float8_e4m3fn).npu()
    x2 = torch.randn(n, k).to(torch.float8_e4m3fn).npu()          # 物理 [N, K]
    x1_scale = torch.randint(0, 256, (m, math.ceil(k / 64), 2), dtype=torch.uint8).npu()
    x2_scale = torch.randint(0, 256, (n, math.ceil(k / 64), 2), dtype=torch.uint8).npu()

    y, _, _ = cann_ops_transformer.all_gather_quant_matmul(
        x1,
        x2.t(),                        # ND 转置 B：传 [N,K] 的 .t() 视图
        group,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=[1, 1, 32],
        x1_scale_dtype=293,
        x2_scale_dtype=293,
        y_dtype=15,                    # BF16
        comm_mode="ubmem",
    )
    torch.npu.synchronize()
    print(f"[rank {rank}] y={tuple(y.shape)} {y.dtype}")
    dist.barrier()
    dist.destroy_process_group()
```

其他布局的构造方式：

```python
# ND 非转置 B：物理 [K, N]，直接传连续 tensor；x2_scale 为 [ceil(K/64), N, 2]
x2_kn = torch.randn(k, n).to(torch.float8_e4m3fn).npu()

# NZ 转置 B：对物理 [N, K] 做 format_cast(29)；x2_scale 为 [N, ceil(K/64), 2]
x2_nz = torch_npu.npu_format_cast(x2, 29)

# NZ 非转置 B：对物理 [K, N] 做 format_cast(29)；x2_scale 为 [ceil(K/64), N, 2]
x2_kn_nz = torch_npu.npu_format_cast(x2_kn, 29)
```

fp4 场景：`x1/x2` 用 uint8 打包，调用时传
`x1_dtype=296, x2_dtype=296`，且仅支持 ND。

---

## 4. 详细方案设计

整体流程（状态同步 → AIV gather → AIC matmul）如下图：

![alt text](design.png)

### 4.1 状态区逻辑

每个 rank 的通信窗口 = HCCL 内置 buffer，前 4KB 为状态区，其后依次是 data 区
（`rankSize` 个 x1 槽位）与 scale 区（`rankSize` 个 scale 槽位）。状态区按
`srcRank × coreIdx` 划槽（每槽 32B，实际写 1 个 float）。

- **本地就绪**：`CopyLocalData` 完成后，每个非 leftover 核向本组负责的 peer 写 1 个
  `flag=1.0`，槽位为 `(myRank, myCoreIdx)`；
- **配对等待**：每个核只轮询本卡窗口里 `(remoteRank, myCoreIdx)` 的槽位，读到
  `1.0 ± 0.5` 即认为配对的远端分片已就绪，随后清零；
- **1:1 握手**：不做全局 Sum、不要求所有 rank 到齐；leftover 核不参与；
- 作用：先确认"本地数据已写好、远端对应分片已就绪"，再做远端搬运，避免读写竞争。

### 4.2 allgather 阶段

**本地填槽**（`CopyLocalData`）：

- 除最后 1 个 V 核外，其余核按 M 行均分，把本 rank 的 x1 从 GM 搬到本卡窗口的 self 槽位；
- 最后 1 个 V 核单独搬运本 rank 的 scale；
- 完成后 `SyncAll`，保证本地槽位对所有核可见。

**远端分组**（只对 `rankSize-1` 个远端 rank 分组）：

```
groupSize = V核数 / (rankSize - 1)
groupIdx  = aivId / groupSize
peer      = (curRankId + 1 + groupIdx) % rankSize   # 组间轮转
leftover  = groupIdx >= rankSize - 1                # 多余核，不参与远端搬运
kDim      = min(ceil(K/512), groupSize, 2)          # K 方向最多 2 个核
mDim      = min(groupSize / kDim, M)                # M 方向均分
```

- 组内 `kDim` 个核按奇偶 512 元素切片覆盖 K，`mDim` 个核按行切分 M；
- 每组最后 1 个核兼搬该远端 rank 的 scale；
- 示例：V核 = 24、rankSize = 4 → 8 核/组，kDim = 2、mDim = 4；
- leftover 核不搬远端数据（本 rank 数据由本地拷贝完成），但仍参与轮间 `SyncAll` 与 flag 计数。

**逐轮搬运**：

- 粒度：512 个 K 元素为一切片，一轮 = 1024 个 K（kDim=2 时两个切片交替）；
- 单轮：远端 GM → UB（ping/pong）→ 本卡窗口 GM；
- 每轮结束：`SyncAll` 轮间屏障 → `CrossCoreSetFlag` 通知 AIC 本轮数据可读。

### 4.3 matmul 阶段

- **MX 量化矩阵乘**：A/B 以 fp8/fp4 原值参与计算；e8m0 block scale（每 32 个 K 一组）
  由 mad_mx 指令在乘加过程中直接应用（等价于 A、B 各乘自己的 scale 后立即相乘累加）；
- **按轮等待**：AIC 读第 `iter` 段 `kL1` 前，累计等待 `ceil((iter+1)*kL1/1024)` 次 flag，
  保证该段所有列已 gather 完成；
- **装载顺序**：B/bias 不依赖 gather，先发 GM→L1；再 `CrossCoreWaitFlag`；随后装载
  A、scaleA（scaleB 随 B 一起）；L1→L0 → MMAD；
- **输出**：整个 K 方向在 L0C 累加完成后，fixpipe 一次写 `y`（bf16/fp16）。

### 4.4 优化手段

- **K 轮流水**：AIC 不等完整 AllGather，首轮就绪即开算，通信与计算按轮重叠；
- **B/bias 预取**：把不依赖 gather 的装载提前到等待 flag 之前，隐藏部分等待时间；
- **小行数远端预读**：
  - 一轮的搬运量（payload）= 本核行数 × 每切片字节数；该数据能一次装进一个 UB
    buffer 时称为"单 chunk"（`行数 ≤ xBufSize_ / 每切片字节数`），否则要按 buffer
    容量分多次 GM→UB 拷贝，称为"多 chunk"；
  - 仅当"单 chunk"且单核行数 ≤ 14 时才启用预读：下一轮的远端读（MTE2）提前到
    本轮写（MTE3）和轮间屏障之前发出，与之重叠；轮间 `SyncAll` 只等 MTE3
    （`{PIPE_MTE3, PIPE_MTE3}`）；
  - "多 chunk"或大 payload 不启用预读，避免抢当前轮带宽；

---

## 5. 编译、部署与运行

### 5.1 环境要求

- 硬件：Ascend 950PR / 950DT，至少 `world_size` 张卡（示例 4 卡）；
- 软件：CANN toolkit、Python 3.10、torch / torch_npu、numpy、ml_dtypes；
- 编译依赖：gcc/g++（torch extension 首次调用时需要 JIT 编译）、Python `build` 包
  （`pip install build`，用于打 whl；缺失时 `build.sh` 会跳过 whl 构建且不报错）；
- 环境变量：

```bash
source $CANN_HOME/set_env.sh                 # 例如 /home/xxx/Ascend/cann-9.3.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3     # 数量须与 --world-size 一致
```

- 默认不要设置 `HCCL_BUFFSIZE`：ubmem 窗口直接使用 HCCL 内置 buffer；若运行时报窗口超限，
  再按提示适当增大 `HCCL_BUFFSIZE`（单位 MB）。

### 5.2 编译并安装自定义算子包

```bash
cd ops-transformer
source $CANN_HOME/set_env.sh
bash build.sh --pkg --experimental --soc=ascend950 --ops=all_gather_matmul_v3 -j16
./build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet
```

- `--pkg` 产出 `build_out/cann-ops-transformer-custom_linux-x86_64.run`，安装到
  `$ASCEND_OPP_PATH/vendors/custom_transformer`；
- 安装后确认：`ls $ASCEND_OPP_PATH/vendors/custom_transformer`；
- 安装脚本会提示把 `op_api/lib` 加入动态库路径；直接写 Python 调用时必须设置（`run_test.sh`
  已内置）：
  `export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/custom_transformer/op_api/lib/:$LD_LIBRARY_PATH`；
- 只改 `torch_extension/` 或 `test/` 时不需要重编算子包。

### 5.3 安装 torch extension

```bash
# 构建 whl（带 --ops 的 --pkg 流程不会自动构建 whl，需单独执行本命令）
bash build.sh --torch_extension --experimental
pip install --force-reinstall --no-deps build_out/cann_ops_transformer-*.whl
```

- 这里**不要带 `--ops`**：带 `--ops` 会生成 `cann_ops_transformer_<vendor>` 包，导入名不再是
  `cann_ops_transformer`，与测试脚本/示例不一致；不带 `--ops` + `--experimental` 生成全量实验
  算子包，导入名为 `cann_ops_transformer`；
- whl 是纯 Python 包（op 定义 + `csrc` 源码），**不预编译 .so**：首次调用算子时，
  `OpBuilder` 通过 `torch.utils.cpp_extension.load` 按源码 JIT 编译，缓存到
  `~/.cache/torch_extensions`；
- JIT 依赖 gcc/g++、torch/torch_npu 头文件以及 `ASCEND_HOME_PATH` 指向的 CANN include；
- 首次调用会多花几十秒到几分钟编译，之后复用缓存；修改 `torch_extension/csrc` 后会自动重编
  （按源码指纹判断）。安装后可在 `bash test/run_test.sh --list-cases` 的
  `[ENV] cann_ops_transformer=` 行确认实际使用的包路径（run_test.sh 会把用户 site-packages
  放到 PYTHONPATH 最前）。

### 5.4 运行

```bash
cd experimental/mc2/all_gather_matmul_v3

# 精度：单 case（4 卡）
bash test/run_test.sh --mode precision --cases m48 --world-size 4 \
  --x1-dtype e4m3 --x2-dtype e4m3 --y-dtype bf16 --x2-format ND --x2-orient trans

# 默认 18 个 shape / 12 个非对齐 M shape
bash test/run_test.sh --mode precision --cases all --world-size 4 \
  --x1-dtype e4m3 --x2-dtype e4m3 --y-dtype bf16 --x2-format ND --x2-orient trans
bash test/run_test.sh --mode precision --cases odd --world-size 4 \
  --x1-dtype e4m3 --x2-dtype e4m3 --y-dtype bf16 --x2-format ND --x2-orient trans

# 自定义 shape（per-rank M=144, K=7168, N=4400）
bash test/run_test.sh --mode precision --m 144 --k 7168 --n 4400 --world-size 4 \
  --x1-dtype e4m3 --x2-dtype e4m3 --y-dtype bf16 --x2-format ND --x2-orient trans
```

- 完整参数、性能模式与 CSV 说明见第 6 节；
- 也可以不使用测试脚本，直接 `import cann_ops_transformer` 调用
  `all_gather_quant_matmul`（见 3.4 示例）。

---

## 6. 测试与验证

测试脚本位于 `test/`，内置 18 个默认 shape（`m4` ~ `m96`，K=3072，N=2560）：

```bash
# 精度（CPU golden 反量化 matmul 比对）
bash test/run_test.sh --mode precision --cases all --world-size 4 \
  --x1-dtype e4m3 --x2-dtype e5m2 --y-dtype fp16 --x2-format NZ --x2-orient trans

# 性能 + 掩盖率（仅 e4m3 x e4m3 -> bf16）
bash test/run_test.sh --mode perf --cases m48,m96 --world-size 4
```

- 精度模式支持全部 10 种 dtype 组合与 4 种布局，参考值为 CPU golden（fp32 反量化 matmul）；
- 性能模式按 profiler 解析各 rank 的 `op_summary`，跨 rank 取最大、后 5/10 次迭代平均，
  由 rank0 汇总写 CSV；
- 非对齐 M 组（`m5` ~ `m93`）用 `--cases odd` 运行；`--cases all,odd` 可一次跑完 30 个 shape；

---

## 7. 已知限制与注意事项

1. **bias 不支持**：ubmem 路径必须传 `None`；
2. **gather_out / amax_out 未实现**：返回 `(0,)` 占位；
3. **fp4 仅 ND**：fp4 + NZ 直接拦截；且 x1/x2 必须同为 fp4；
4. **NZ 仅 fp8**；NZ 4D 分形存储按 N 的 16/32 对齐分块（转置 16、非转置 32），y 的 N 仍为逻辑 N（由 `x2_scale` 决定）；
5. **K == N 歧义**：2D 输入无法仅从形状区分方向，按 `x2_scale` 布局消歧
   （`[N,kd,2]` 视为转置、`[kd,N,2]` 视为非转置）；scale 也方形时直接报错；
6. **窗口容量**：窗口必须落在 HCCL 内置 buffer 内，超出时报错并提示增大 `HCCL_BUFFSIZE`；
7. **确定性**：默认支持确定性计算。

---
