# flash_attn_minimal 算子详解文档

## 一、算子设计

### 1.1 功能与架构

本算子基于华为昇腾 NPU 的 AscendC 框架，采用 **Host（CPU）拉起 + Device（NPU）跑 Kernel** 的架构，实现标准自注意力计算 $Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V$。接口参数详见 [README.md](./README.md)。

算子约束：Q/K/V 数据类型为 `bfloat16`，`D=128`，`S1=S2` 且被 128 整除，`N2` 整除 `N1`（支持 GQA），计算稠密（全量）注意力。输入为 `BSND` 布局，B/N1/N2/S1/S2 全部作为运行时参数传入 Kernel，无独立的 Host Tiling 阶段。

调用链路：**Python** `torch.ops.ascend_ops.flash_attn_minimal(q,k,v,softmaxScale)` → **C++ 入口** `FlashAttnNpu()`（`torch_interface.cpp`）→ **Kernel 调度** `FaKernel`（`fa_kernel_interface.h`）→ 按 core 类型分发 `CubeFunc`（`fa_block_cube.h`）/ `VectorFunc`（`fa_block_vec.h`）。

### 1.2 基本块与多核切分策略

Kernel 以 **M=128、N=128、D=128** 为基本 tile（模板参数 `M_BASE / N_BASE / D_SIZE`，`fa_kernel_interface.h`）。每个 AIC Cube core 配 2 个 AIV Vector core（`KERNEL_TYPE_MIX_AIC_1_2`），每个 AIV 处理半个 M（`M_BASE/2 = 64` 行），因此 BMM 结果 UB 中以 `64×128` 为半块单位。

Cube 和 Vector 均按 **BN1 = B×N1**（batch × query-head 总数）对 core 均分：

- AIC core（`CubeFunc`，`fa_block_cube.h`）：`bn1PerCore = ceil(BN1/blockDim)`，处理 `bn1 ∈ [aicIdx×bn1PerCore, +bn1PerCore)`。
- AIV core（`VectorFunc`，`fa_block_vec.h`）：`aicIdx = aivIdx / 2`，`subBlockIdx = GetSubBlockIdx()`（0/1），两个 AIV 分别负责该 tile 上半 / 下半 64 行，处理范围与配对的 AIC 一致。
- blockDim 由 `platform_ascendc::CalcTschBlockDim(64, ...)` 计算（`torch_interface.cpp`），入口使用 `fixedAivNum=64`。

GQA 映射：Cube 和 Vector 在每个 BN1 迭代内通过 `n2Head = n1Head * N2 / N1` 计算对应的 KV head（`fa_block_cube.h` / `fa_block_vec.h` 隐含在 `bn1` 拆解中），无需 G 轴合轴。

两层主循环（`fa_block_cube.h` / `fa_block_vec.h`）：

```
for bn1 in range(start, end):          ← 当前 core 负责的 head 区间
    bIdx   = bn1 / N1
    n1Head = bn1 % N1
    n2Head = n1Head * N2 / N1          ← GQA：query head → KV head 映射
    for s1Block in range(numS1Blocks):  ← num S1 blocks = ceil(S1/M_BASE)
        for s2Block in range(numS2Blocks):  ← num S2 blocks = ceil(S2/N_BASE)
            C1(s1Block,s2Block) → V1 → C2 → V2   ← 顺序执行，无软件流水
```

对于典型 shape `B=1, N1=32, N2=1, S1=8192, S2=8192`：

| 常量 | 值 | 含义 |
|------|-----|------|
| `BN1` | `1×32 = 32` | 总 head 组数（即总 query head 数） |
| `numS1Blocks` | `8192/128 = 64` | 每个 head 的 S1 块数 |
| `numS2Blocks` | `8192/128 = 64` | 每行 KV(flash) 块数 |

blockDim（AIC 视角）通常为 32 → `bn1PerCore = 32/32 = 1`，即 **每个 AIC core 处理 1 个 query head**，每个 head 内 64×64 = 4096 个 tile。

每行 attend 全部 S2，且一行不跨 core，各 core 产出即为最终结果，无需跨核归约。

### 1.3 内存划分

NPU 上存储层次从大到小为 GM → L1 → L0A/L0B/L0C → UB。所有 buffer 均以**显式线性地址**分配（`LocalTensor<T>(TPosition, addr, elems)`），Cube 与 Vector 对**共享区（UB 的 BMM 结果、L1 的 P 矩阵）使用相同地址**以完成跨核数据交换。

#### GM（Global Memory）

Q/K/V 及输出统一为 `BSND` 格式（`FaGmTensor<T, GmFormat::BSND>`，`offset_calculator.h`）。workspace 为空 tensor（`at::empty({0})`），Kernel 不使用。

#### L1（AIC 侧暂存，`fa_block_cube.h`）

| Buffer | 大小 | 说明 |
|--------|------|------|
| `l1P`（P matrix，`M×N` bf16） | 128×128×2 = 32KB | AIV 写（`CopySoftmaxToL1`）、AIC 读（BMM2） |
| `l1Q`（Q tile，`M×D` bf16） | 128×128×2 = 32KB | 每个 S1 块首轮 S2 加载后复用 |
| `l1K`（K tile，`N×D` bf16） | 128×128×2 = 32KB | 每个 S2 块重载 |
| `l1V`（V tile，`N×D` bf16） | 128×128×2 = 32KB | 每个 S2 块重载 |
| **L1 总计** | **128KB** | 从 0 地址线性排布 |

#### L0（AIC Cube 专用，`fa_block_cube.h`）

| 空间 | 大小 | 说明 |
|------|------|------|
| **L0A** | `128×128` bf16 × 2 | double-buffer，存 Q（BMM1）或 P（BMM2） |
| **L0B** | `128×128` bf16 × 2 | double-buffer，存 K（BMM1）或 V（BMM2） |
| **L0C** | `128×128` fp32 × 1 | 单 buffer，存 Matmul 结果 |

L0A/L0B 使用 ping-pong 双缓冲（`l0aIdx`/`l0bIdx` 在 `MatmulFull` 内 `1-idx` 轮换），L0C 为单 buffer（每次 BMM 前 `get_buf PIPE_M` → `Mmad` → `Fixpipe` 后 `rls_buf`）。

#### UB（AIV Vector 专用 + AIC Fixpipe 写入，`fa_block_vec.h`）

AIC 通过 Fixpipe（L0C→UB）把矩阵乘结果写入 UB 供 AIV 消费。`ubBmm1`/`ubBmm2`/`l1P` 在 Cube 和 Vector 中地址一致（跨核共享），其余为 AIV 私有。

| Buffer | 大小 | 说明 |
|--------|------|------|
| `ubBmm1` | 64×128 fp32 = 32KB | BMM1 结果，Cube→UB，Vec 消费 |
| `ubBmm2` | 64×128 fp32 = 32KB | BMM2 结果，Cube→UB，Vec 消费 |
| `softmaxSum` | 64 fp32 = 256B | Softmax 分母 sum（按行索引） |
| `softmaxMax` | 64 fp32 = 256B | Softmax 分子 max（按行索引） |
| `softmaxExp` | 64 fp32 = 256B | Flash Update 指数比例因子（按 tile 索引） |
| `commonBuf` | 512B | 通用临时空间（sum/max 更新用） |
| `stage1Buf` | (64+2)×128 bf16 ≈ 16.5KB | Softmax 结果 P（cast bf16），搬往 L1 |
| `stage2Buf` | 64×128 fp32 = 32KB | Flash Update 累加器 |
| `castBuf` | 与 `stage2Buf` **同址复用**（64×128 bf16） | 输出 fp32→bf16 就地 downcast |

> `stage2Buf` 与 `castBuf` 共享同一 UB 尾部地址：`Cast` 读 fp32（4B/elem）写 bf16（2B/elem），写指针慢于读指针，可安全就地转换以节省 UB。

### 1.4 跨核同步

AIC 与 AIV 是不同物理 core，跑同一份二进制但走不同分支（`#ifdef __DAV_C310_CUBE__`）。一个 tile 内顺序执行为 **C1（BMM1）→ V1（Softmax）→ C2（BMM2）→ V2（FlashUpdate/输出）**，跨核用 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 做生产者-消费者同步。

Cube 和 Vector 在内层 S2 循环中**顺序推进**：Cube 做完 BMM1 后设置 `FLAG_BMM1_READY`，Vector 等待该 flag → 做 Softmax + 拷 L1P → 设置 `FLAG_BMM1_READY`（释放）+ `FLAG_L1P_READY`（通知 Cube）；Cube 等待 `FLAG_L1P_READY` + `FLAG_BMM2_READY` → 做 BMM2 → 设置 `FLAG_BMM2_READY`；Vector 等待 `FLAG_BMM2_READY` → 做 Rescale → 释放 `FLAG_BMM2_READY`。此顺序执行未使用 PRELOAD 软件流水。

跨核 flag 定义（`fa_block_cube.h` / `fa_block_vec.h`）：

| Flag | ID | 含义 | 设置方 | 等待方 |
|------|----|------|--------|--------|
| `FLAG_BMM1_READY` | 0 | BMM1 UB 已就绪 | Cube | Vector |
| `FLAG_L1P_READY` | 1 | L1P 已就绪 | Vector | Cube |
| `FLAG_BMM2_READY` | 2 | BMM2 UB 已就绪 | Cube | Vector |

每个 flag 另有 `+16` 镜像对应第 2 个 AIV 子核（`fa_block_cube.h` 等）。Vector 侧初始化预置 `FLAG_BMM1_READY` 和 `FLAG_BMM2_READY`（释放状态，`fa_block_vec.h`），Cube 主循环结束后 barrier 等待这两个 flag 及其 `+16` 镜像，确保退出前 AIV 已消费完共享 buffer（`fa_block_cube.h`）。

AIV 侧本地 `get_buf/rls_buf` 事件号：`stage1Buf` / `ComputeSoftmax` 用 `E_VEC1_ID = 0`；`stage2Buf` / `Cast` / 输出用 `E_VEC2_ID = 1`，两者不冲突。

---

## 二、代码结构说明

### 2.1 整体架构概览

```
torch_interface.cpp              ← PyTorch 注册 + 入口（输入校验，直接拉起 Kernel）
  └─ op_kernel/fa_kernel_interface.h   ← Device Kernel 入口（按 core 类型 #ifdef 分发 CubeFunc/VectorFunc）
       ├─ kernel/fa_block_cube.h       ← AIC Cube：Q/K/V 搬运 + BMM1/BMM2 + Fixpipe
       └─ kernel/fa_block_vec.h        ← AIV Vector：Softmax + Flash Update + 输出
```

附属功能模块：

| 目录/文件 | 功能 |
|------|------|
| `op_kernel/fa_kernel_public.h` | 公共头/命名空间引入 + 工具函数（Max/Min/Clip） |
| `op_kernel/memcopy/offset_calculator.h` | GM 多维索引（`FaGmTensor<BSND>`）+ 坐标结构体 + 地址计算 |
| `op_kernel/memcopy/memory_copy.h` | `CopyMatrixGmToL1`（ND→NZ） + `CopyAttnOutBN1UbToGm`（UB→GM） |
| `op_kernel/matmul/matmul.h` | `MatmulFull`（L1→L0A/L0B `LoadData` + `Mmad`） + `MMParam` |
| `op_kernel/vector/vf/vf_softmax.h` | VF 指令加速的 Flash Softmax：`ProcessVec1Vf`（update / no_update，`EQ_128` 路径）+ `UpdateExpSumAndExpMax` |
| `op_kernel/vector/vf/vf_softmax_impl/` | `vf_softmax_aligned128_{no_update,update}.h` 具体实现 + `vf_softmax_const.h` 常量 |
| `op_kernel/vector/vf/vf_rescale.h` | `FlashUpdate` / `FlashUpdateLast` / `LastDiv`（Flash Update Rescale） |

### 2.2 Torch 接口：`torch_interface.cpp`

唯一入口 `FlashAttnNpu()`（`torch_interface.cpp`）：
1. `CheckInput` 校验 bf16、4 维、`D=128`，且 `S1=S2`、`S1%128=0`、`N1>=N2`、`N1%N2=0`、batch 维度一致。
2. 取平台信息与当前 NPU stream，分配同形状输出与空 workspace。
3. 固定 `fixedAicNum=32 / fixedAivNum=64`，`blockDim = CalcTschBlockDim(64, GetCoreNumAic(), GetCoreNumAiv())`。
4. 以 `FaKernel<<<blockDim, nullptr, aclstream>>>(gq, gk, gv, go, gws, blockDim, scale, B, N1, N2, S1, S2)` 拉起 Kernel，随后 `aclrtSynchronizeStream` 同步。

算子通过 `TORCH_LIBRARY(ascend_ops, ...)` 注册 schema `flash_attn_minimal(Tensor q, Tensor k, Tensor v, float softmaxScale=0) -> Tensor`，通过 `TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, ...)` 将实现绑定到 NPU 后端。`PyInit__C` 提供 Python 模块入口，算子注册完全自包含于本文件，无需外部 `npu_ops_def.cpp`。

### 2.3 Device Kernel：`op_kernel/`

#### 2.3.1 入口与分发（`fa_kernel_interface.h`）

`FaKernel` 接收运行时形状参数（`B, N1, N2, S1, S2`），设置 `KERNEL_TYPE_MIX_AIC_1_2` 与 `InitSocState()` 后，按编译宏分发：`__DAV_C310_CUBE__` → `CubeFunc<M,N,D>`，否则 → `VectorFunc<M,N,D>`。模板默认 `M=N=D=128`。

#### 2.3.2 Cube 侧（`fa_block_cube.h`）

`CubeFunc()` 建立 GM 张量（`BSND` 格式）、分配 L1/L0/UB，然后按 `bn1 → s1Block → s2Block` 三层循环顺序执行：

核心函数：

- **`ComputeMm1`**（BMM1：Q@Kᵀ，`fa_block_cube.h`）：
  1. 等待 `FLAG_BMM1_READY`（ubBmm1 空闲）。
  2. `isFirstS2Loop` 时 `CopyMatrixGmToL1`（Q GM→L1，ND→NZ），该 S1 块后续 S2 轮复用。
  3. 每块 `CopyMatrixGmToL1`（K GM→L1，ND→NZ）。
  4. `MatmulFull`（L1→L0A/L0B ping-pong → `Mmad` → L0C），`isRightTranspose=true` 实现 Kᵀ。
  5. `FixpipeMm1`（L0C→UB，`dualDstCtl=1` 拆成两份 `64×128` fp32 给 2 个 AIV）→ `CrossCoreSetFlag(FLAG_BMM1_READY)`。

- **`ComputeMm2`**（BMM2：P@V，`fa_block_cube.h`）：
  1. 等待 `FLAG_L1P_READY` + `FLAG_BMM2_READY`。
  2. `CopyMatrixGmToL1`（V GM→L1，ND→NZ）。
  3. `MatmulFull`（L1P@V，`realM=mActual`）→ `FixpipeMm2`（L0C→UB dual-dest）→ `CrossCoreSetFlag(FLAG_BMM2_READY)`。

#### 2.3.3 Vector 侧（`fa_block_vec.h`）

`VectorFunc()` 分配共享 UB（`ubBmm*`、`l1P`）与私有 UB，初始化 2 个跨核 flag 为释放状态，然后按 `bn1 → s1Block → s2Block` 三层循环顺序执行：

- **`ComputeSoftmax`**（`fa_block_vec.h`）：等待 `FLAG_BMM1_READY` → 首块走 `ProcessVec1Vf<...false>`（no_update），非首块走 `<...true>`（update）：逐行 `QKᵀ×scale → row_max → exp(x-max) → sum → cast bf16`；非首块再 `UpdateExpSumAndExpMax` 合并历史 max/sum。
- **`CopySoftmaxToL1`**（`fa_block_vec.h`）：把 `stage1Buf`（P，bf16）`DataCopy` 到 `l1P`（按 `subBlockIdx` 定位上/下半），`CrossCoreSetFlag` 释放 `FLAG_BMM1_READY` + 设置 `FLAG_L1P_READY`。
- **`ComputeRescale`**（`fa_block_vec.h`）：等待 `FLAG_BMM2_READY` → `isFirstS2Loop` 时 `DataCopy` 初始化 `stage2Buf`；中间块 `FlashUpdate`（`stage2 = stage2·expMax + cur`）；末块 `FlashUpdateLast`（累加后除以 sum）→ `CrossCoreSetFlag` 释放 `FLAG_BMM2_READY`。
- **`CopyAttnOutToGm`**（`fa_block_vec.h`）：仅 `isLastS2Loop` 触发：`isFirstS2Loop` 时先 `LastDiv`（除以 sum），然后 `Cast` fp32→bf16（就地写入 `castBuf`），`CopyAttnOutBN1UbToGm` 写回 GM（`BSND` 格式写回 `s1BlockBase + subBlockIdx*halfM` 位置）。

---

## 三、典型 Case 代入走读

以 `B=1, N1=32, N2=1, S1=8192, S2=8192, D=128`（GQA，G=32）为例，运行于 32 AIC + 64 AIV。

### 3.1 多核分配

- `BN1 = B×N1 = 1×32 = 32`。
- blockDim（AIC 视角）= 32 → `bn1PerCore = ceil(32/32) = 1`。**每个 AIC core 处理 1 个 query head**，配 2 个 AIV core 处理该 head 的上/下半 64 行。
- 每个 core 内：`numS1Blocks=64` × `numS2Blocks=64` = 4096 个 tile，顺序执行。

### 3.2 单 tile 计算流程

一个 tile 处理 1 个 query head（`n1Head`）的 S1 块（`s1Block × 128` 行），attend KV head（`n2Head = n1Head × 1 / 32 = 0`）。

**Step A — BMM1（AIC Cube）**

1. **Copy Q**（仅 `s2Block==0`）：`CopyMatrixGmToL1` 从 `Q[b=0, n1=n1Head, s1=s1BlockBase, d=0]` 读 128 行 → L1（NZ，128×128 bf16 = 32KB）。同一 S1 块的后续 S2 轮复用此 L1Q。
2. **Copy K**：`K[0, 0, s2Block×128, 0]` → L1（NZ，32KB）。
3. **Matmul**：`Q(128×128) × Kᵀ(128×128)` → L0C。
4. **Fixpipe** L0C→UB：dual-dest 拆成 2×`64×128` fp32，通知两个 AIV。

**Step B — Vec1 / Softmax（AIV Vector，每个子核 64 行）**

1. 等待 `FLAG_BMM1_READY` → `ProcessVec1Vf`（首块 no_update / 非首块 update，`EQ_128`）：`QKᵀ×scale → max → exp → sum → cast bf16` 写入 `stage1Buf`。
2. 非首块 `UpdateExpSumAndExpMax` 更新 `softmaxSum`/`softmaxMax`/`softmaxExp`。
3. `DataCopy` P(bf16) → `l1P`，`CrossCoreSetFlag` 释放 BMM1 + 通知 Cube（L1P ready）。

**Step C — BMM2（AIC Cube）**

1. 等待 `FLAG_L1P_READY` + `FLAG_BMM2_READY` → **Copy V**：`V[0, 0, s2Block×128, 0]` → L1（NZ，32KB）。
2. `MatmulFull` `P(128×128) × V(128×128)`（`realM=mActual`）→ Fixpipe L0C→UB dual-dest → 通知 AIV。

**Step D — Vec2 / Flash Update（AIV Vector）**

- 行首块：`DataCopy` 初始化 `stage2Buf`；
- 中间块：`FlashUpdate`（`stage2 = stage2·expMax + cur`）；
- 行末块：`FlashUpdateLast`（累加 + 除以 sum）→ `LastDiv`（全零行补除）→ `Cast` bf16（就地）→ `CopyAttnOutBN1UbToGm` 写回输出对应位置。

如此，每行 64 个 S2 块通过 flash online-softmax 累加为最终输出；32 个 query head 分布在 32 个 AIC core 上并行处理，覆盖全部输出。

---

## 四、性能测试

以下场景在 950DT 平台上完成测试，由于当前实现没有做Preload计算，各阶段串行执行，MFU较低。

| B | Q_N | KV_N | Q_S | KV_S | D | Duration | MFU |
|---|-----|------|-----|------|---|----------|-----|
| 1 | 32 | 1 | 8192 | 8192 | 128 | 7853.29 | 32.40% |
