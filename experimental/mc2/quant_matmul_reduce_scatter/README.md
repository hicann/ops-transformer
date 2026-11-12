# QuantMatmulReduceScatter

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>       |    ×     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：完成 MX 量化 `Matmul` 计算和 `ReduceScatter` 通信的融合，先计算后通信。融合算子内部实现计算和通信（基于ubmem 窗口 -- MTE window）流水并行。

- 计算公式：

    $$
    output_j=ReduceScatter\left(\sum_{i}\ \mathrm{dequant}(x1_i)\cdot\mathrm{dequant}(x2_i)\right)[j\cdot M/R : (j+1)\cdot M/R]
    $$

    - x1 指 Matmul 计算的左矩阵（MX 量化存储，fp8 或 fp4）。
    - x2 指 Matmul 计算的右矩阵（MX 量化存储，fp8 或 fp4）。
    - x1_scale / x2_scale 指对应的 e8m0 逐块缩放因子（每 32 个元素一组）。
    - M 为 x1 的行数（总行数），R 为 rank_size，N 为 x2 的列数。
    - 每个 rank 输出 `ReduceScatter` 后自身的行分片 `[M/R, N]`。

## 方案概览

- **融合语义**：各卡持有自己的左矩阵 `x1[M, K]`、右矩阵 `x2` 及对应 MX scale，先在本地完成 MX 量化矩阵乘，再对结果做 ReduceScatter；各 rank 输出自己负责的 `[M/R, N]` 行分片。**先计算后通信（compute-first）**。

  $$
  y_j = ReduceScatter\left(\sum_{i}\ \mathrm{dequant}(x1_i)\cdot\mathrm{dequant}(x2_i)\right)[j\cdot M/R : (j+1)\cdot M/R]
  $$

  scale 按 K 轴每 32 个元素一组（group size = 32），每组 2 个 e8m0 scale 值。

- **执行形态**：

  ```
  rank i（每卡）：
    x1[M,K] / x2[N,K]|[K,N]（本地）──▶ AIC: MX Matmul（L0C 结果 FP32）
                                          │ fixpipe（DualDst SplitM，M 对半）
                                          ▼
                            AIV 两个子核 UB（FP32 槽）──cast──▶ y dtype(bf16/fp16)
                                          │ AllToAll（UB → 各 peer 的 ubmem 窗口）
                                          ▼
                    SyncAll + status flag barrier ──▶ ReduceAdd（读各 peer 窗口本 rank 的 M/R 行）
                                          │ FP32 累加
                                          ▼
                                       y[M/R, N]
  ```

  1. Host 侧创建 ubmem 通信上下文（各 rank 内置 HCCL buffer 地址表）；
  2. AIC：`BlockMmad` 完成 MX 量化矩阵乘，L0C 结果经 fixpipe **直写两个 AIV 子核的 UB**（不落本卡 GM）；
  3. AIV：把 UB 中的 FP32 结果 cast 到 y dtype 后 **AllToAll** 发送到各 peer 的 ubmem 窗口；
  4. AIV：`SyncAll` + status flag 屏障后，各 rank 从 peer 窗口读取本 rank 的 `[M/R, N]` 行分片并 **ReduceAdd**（FP32 累加）写 y。

- **主要收益**：
  - 通信（AllToAll）被矩阵乘结果产出掩盖，无独立 ReduceScatter 算子；
  - 计算与通信粒度按 block 级细粒度切分，在 M 较小的场景下提升通算掩盖率；
  - L0C→UB 直通 + DualDst 按 M 对半分摊到 2 个 AIV 子核，Matmul 输出结果无需落盘GM后再搬入核内，提升通信带宽利用率；
  - ReduceAdd 阶段以 FP32 累加，避免低精度输出 dtype 累加误差。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnQuantMatmulReduceScatterGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulReduceScatter”接口执行计算。

```cpp
aclnnStatus aclnnQuantMatmulReduceScatterGetWorkspaceSize(
    const aclTensor* context,
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* x1Scale,
    const aclTensor* x2Scale,
    const char*      group,
    int64_t          rankSize,
    const aclTensor* y,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnQuantMatmulReduceScatter(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

### torch extension API

```python
def quant_matmul_reduce_scatter(
    x1: torch.Tensor,          # 左矩阵 [M, K]；fp4 打包时 [M, K/2]（uint8）
    x2: torch.Tensor,          # 右矩阵：转置 B 传 [N,K] 的 .t() 视图 或 ZN 分形；非转置 B 传 [K,N] 连续 或 NZ 分形
    x1_scale: torch.Tensor,    # [M, ceil(K/64), 2]（e8m0）
    x2_scale: torch.Tensor,    # 转置 [N, ceil(K/64), 2] / 非转置 [ceil(K/64), N, 2]
    group,                     # 通信域名称（HcclGetCommName 返回的 commName）
    *,
    x1_dtype: Optional[int] = None,        # e4m3=292 / e5m2=291 / fp4=296
    x2_dtype: Optional[int] = None,
    x1_scale_dtype: Optional[int] = None,  # 293（uint8 打包承载时）
    x2_scale_dtype: Optional[int] = None,
    y_dtype: Optional[int] = None,         # bf16=283 / fp16=257
    comm_mode: Optional[str] = "ubmem",
) -> torch.Tensor:             # y: [M/rank_size, N]
```

## aclnnQuantMatmulReduceScatterGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 320px">
    <col style="width: 320px">
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 120px">
    <col style="width: 100px">
    </colgroup>
    <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>context</td>
        <td>输入</td>
        <td>UBMEM 通信上下文（HCCL window 资源经 host 侧 HcclAllocComResourceByTiling 创建并包装）。</td>
        <td>内部由 torch_extension / 调用方自动创建并复用，通常无需显式构造。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x1</td>
        <td>输入</td>
        <td>Matmul 计算的左矩阵输入，即计算公式中的 x1。</td>
        <td>
            <ul>
                <li>shape 为（m，k）。</li>
                <li>支持原生 fp8 存储；fp4 以 uint8 打包（2 元素/字节，末维为 k/2），需配合 x1_dtype 枚举。</li>
            </ul>
        </td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>输入</td>
        <td>Matmul 计算的右矩阵输入，即计算公式中的 x2。</td>
        <td>
            <ul>
                <li>转置方向由 x2 布局自动推断（aclnn 内推导）：DN/ZN 的 B 物理存储 [N,K]（.t() 视图或 NZ [N,K]）；ND/NZ 的 B 物理存储 [K,N]（连续或 NZ 分形）。</li>
                <li>ND 格式支持原生 fp8 或 fp4 uint8 打包（fp4 打包沿末维，DN 打包 k、ND 打包 n）。</li>
                <li>FRACTAL_NZ 格式仅支持 fp8，分形 storage 为 4 维；fp8 的 NZ 构造需经 uint8 中转（npu_format_cast 不支持 fp8 dtype）。</li>
                <li>fp4 仅支持 ND 格式。</li>
            </ul>
        </td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND、FRACTAL_NZ</td>
        <td>2（或 4，NZ storage）</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>x1Scale</td>
        <td>输入</td>
        <td>x1 的 MX e8m0 缩放因子。</td>
        <td>
            <ul>
                <li>shape 为（m，ceil(k/64)，2），每 32 个元素一组一个 scale。</li>
            </ul>
        </td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2Scale</td>
        <td>输入</td>
        <td>x2 的 MX e8m0 缩放因子。</td>
        <td>
            <ul>
                <li>转置（DN/ZN）shape 为（n，ceil(k/64)，2）。</li>
                <li>非转置（ND/NZ）shape 为（ceil(k/64)，n，2）。</li>
            </ul>
        </td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3</td>
        <td>×</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>通信域名称。</td>
        <td>通过 Hccl 提供的接口 “extern HcclResult HcclGetCommName(HcclComm comm, char* commName);” 获取，其中 commName 即为 group。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>rankSize</td>
        <td>输入</td>
        <td>通信域内的卡数（ReduceScatter 的 rank_size）。</td>
        <td>取值 2、4、8、16，且 M 必须为 rankSize 的整数倍。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>y</td>
        <td>输出</td>
        <td>ReduceScatter 通信与 Matmul 计算的结果，即计算公式中的 output。</td>
        <td>
            <ul>
                <li>shape 为（m/rank_size，n）。</li>
            </ul>
        </td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在 Device 侧申请的 workspace 大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回 op 执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **返回值**

    返回 aclnnStatus 状态码，具体参见 [aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
    </colgroup>
    <thead>
    <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的 context、x1、x2、x1_scale、x2_scale 或 y 是空指针。</td>
    </tr>
    <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>x1、x2、x1_scale、x2_scale 或 y 的数据类型 / 数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
        <td>group 或 rank_size 不符合约束要求。</td>
    </tr>
    </tbody></table>

## aclnnQuantMatmulReduceScatter

- **参数说明**
    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 173px">
    <col style="width: 133px">
    <col style="width: 860px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在 Device 侧申请的 workspace 大小，由第一段接口 <code>aclnnQuantMatmulReduceScatterGetWorkspaceSize</code> 获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op 执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的 stream。</td>
    </tr>
    </tbody></table>

- **返回值：**

    返回 aclnnStatus 状态码，具体参见 [aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 产品与通信引擎约束：
  - 仅 <term>Ascend 950PR&950DT系列产品</term>，apace / ubmem / MTE-window 路径。
- 参数说明中 shape 涉及的变量说明：
  - m 为卡数 rank_size 的整数倍，rank_size 支持 2、4、8。
  - fp4 场景下 k（逻辑 K）必须为偶数（2 元素/字节打包）。
  - k 支持范围 [64, 20480]，k 对齐64。
  - n 支持范围 (0, 20480]。
- 性能优势区间：**total m ≤ 512**。
- dtype 配对约束：
  - x1/x2 支持 fp8（e4m3 / e5m2 可任意异构配对）与 fp4（仅 fp4+fp4 配对）。
  - x1_scale / x2_scale 固定为 e8m0。
  - y 支持 bf16 / fp16。
- format / 布局约束：
  - x1 恒为 ND。
  - x2 支持 ND / FRACTAL_NZ；fp4 仅支持 ND。
  - x2 的格式与 transpose_x2 正交：DN（ND+转置）/ ND（ND+非转置）/ ZN（NZ+转置）/ NZ（NZ+非转置）四种组合均合法。
  - fp8 的 FRACTAL_NZ 数据需经 uint8 承载字节中转构造（torch_npu 的 npu_format_cast 不支持 fp8 dtype）。
- 精度：融合算子默认确定性实现，L0C 结果经 DualDst 直写 AIV 的 UB（FP32）后再 cast 到 y dtype，ReduceScatter 阶段以 FP32 累加。

## 详细设计

### 4.1 总体数据流

1. **Host（torch_extension / aclnn）**
   - 根据 `group` 取得 HCCL comm，创建/复用 ubmem 通信上下文（`CommUbmemContext`）；
   - 按 (dtype, format, transpose) 推断 N（含 fp4 打包折算）并分配 `y[M/R, N]`；
   - 调 aclnn → tiling → kernel launch。
2. **AIC（计算阶段）**
   - `BlockMmad` 完成 MX 量化矩阵乘，L0C 结果为 FP32；
   - fixpipe（`L0C2UB_MODE_DUAL_DST_SPLIT_M`）把 L0C 按 M 对半**直写两个 AIV 子核的 UB FP32 槽**；
   - 每 tile 完成后 `Sync::NotifyVector(0)` 通知对应 AIV；下一 tile fixpipe 前 `Sync::WaitForVector(1)` 等 AIV ack。
3. **AIV（all-to-all + reduce 阶段）**
   - `Sync::WaitForCube(0)` 等本 tile 数据就绪 → cast FP32→y dtype → `Sync::NotifyCube(1)` ack → AllToAll 到各 peer 窗口；
   - 全部 tile 发送完成后 `SyncAll<true>` + status flag 屏障 → ReduceAdd：从各 peer 窗口读本 rank 行分片、FP32 累加、cast 回 y。

### 4.2 ubmem 通信窗口

- **上下文**：`CommUbmemContext` 由 HCCL 建链生成，包含 `rankId`、`rankSize`、`commBufferAddrs[rankSize]`；torch_extension 按 group 名缓存复用，避免重复建链。
- **每个 rank 内置 buffer 的布局**（host/kernel 同源常量）：
  - `[0, APACE_MTE_STATE_WIN_SIZE)`（1MB）：元数据 / status 区；
  - 其后为数据区：各 peer 的 AllToAll 落点按 `dstRank * APACE_EP_RANK_OFFSET_STEP`（1024B）步进错开。
- **数据/元数据分离**：kernel 侧数据区地址 = `commBufferAddrs[dstRank] + APACE_MTE_STATE_WIN_SIZE + dstRank * APACE_EP_RANK_OFFSET_STEP`，不覆盖 status / 元数据区（对齐 R14 硬红线）。

### 4.3 分核策略

- 核配比 **AIC:AIV = 1:2**（`KERNEL_TYPE_MIX_AIC_1_2`）；
- **DualDst SplitM**：一个 AIC 的 L0C 结果按 M 对半交给两个 AIV 子核并行 cast/发送；
- AIV 侧 ReduceAdd 按 `tpRankM = M/rankSize` 行在全部 AIV 上 `SplitToCore` 均分，**多核并行归约**（避免单核归约写竞争）。

### 4.4 通算流水与同步

- **INTRA 子核域内 AIC↔AIV 同步**（flags：fwd=0、ack=1、vec1 域 = id+16）：
  - AIC：`Sync::NotifyVector(0)`；下一 tile 前 `Sync::WaitForVector(1)`；
  - AIV：`Sync::WaitForCube(0)` → `Sync::NotifyCube(1)`；
- **发送缓冲 ping-pong**：`MTE3_V / V_MTE3` HardEvent 保证上一 tile 的 MTE3 发送完成后再复用 UB 缓冲；
- **跨 rank 屏障**：全部 AllToAll 后 `SyncAll<true>` + `WriteStatusToWin / ReadStatus` 状态 flag 屏障：
  - **状态区**：每个 rank 的通信窗口首部为状态区（`APACE_MTE_STATE_WIN_SIZE`，1MB，存放完成标志），数据区从其后的 `rankId * APACE_EP_RANK_OFFSET_STEP` 开始，二者不重叠；
  - `WriteStatusToWin`：本 rank 把行分片写入各目标窗口后，向各目标窗口状态区写入完成标志 `1.0`；
  - `ReadStatus`：归约前从自己窗口状态区轮询各发送方的完成标志，全部就绪后才开始 `ReduceAdd`；读完将状态槽复位为 `0.0`，供下一轮/后续迭代复用。
- **HardEvent 配对**：`MTE2_V / V_MTE2 / V_MTE3 / S_MTE3 / S_MTE2 / MTE2_S` 均同迭代 Set/Wait 配对。

### 4.5 调用链路

```
quant_matmul_reduce_scatter (Python API)
  ├─ dtype / y_dtype 枚举、uint8 bitcast、fp4 打包 N 推导、bias/group_sizes/comm_mode 校验
  └─ torch.ops → aclnnQuantMatmulReduceScatterGetWorkspaceSize
       └─ CheckDtypeAndFormat / CheckAttrs / CheckShape
            └─ QuantMatmulReduceScatter tiling
                └─ kernel launch
```

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [aclnn_quant_matmul_reduce_scatter.cpp](./op_api/aclnn_quant_matmul_reduce_scatter.cpp) | 两段式 aclnn 接口实现，含 dtype/format/attr 校验与 x2 归一化。 |
| torch_extension | [quant_matmul_reduce_scatter.py](./torch_extension/quant_matmul_reduce_scatter.py) | Python 直调接口（含 x1_dtype/x2_dtype/y_dtype 等枚举），内部经 torch_extension csrc 透传 aclnn。 |
| 测试脚本 | [quant_matmul_reduce_scatter_test.py](./test/quant_matmul_reduce_scatter_test.py) | 覆盖 dtype（e4m3/e5m2/fp4）× y（bf16/fp16）× x2 format（ND/NZ）× 转置（trans/non-trans）的精度与性能测试。 |

## 测试与验证

### 测试脚本

| 文件 | 功能 |
| :--- | :--- |
| [quant_matmul_reduce_scatter_test.py](./test/quant_matmul_reduce_scatter_test.py) | 精度（precision）/ 性能（perf）测试脚本：多卡（torch.distributed）执行，CPU 侧生成量化数据与 golden 参考，输出精度比对或性能统计；内置 `DEFAULT_CASES` 用例表。 |
| [run_test.sh](./test/run_test.sh) | 一键入口：未指定 dtype/format/orient 时批跑 36 组组合（dtype × y × format × orient）；也透传任意参数给测试脚本。 |

### 使用示例

```bash
# 四卡精度，批跑 36 组 dtype×format×orient 组合
bash run_test.sh --world-size 4

# 指定内置用例（含踩坑点回归用例）
bash run_test.sh --mode precision --cases test_case_1,test_case_2 --world-size 4

# 自定义形状（--m 为每卡输出行数，输入 M = m * world-size）
bash run_test.sh --mode precision --m 64 --k 2048 --n 3584 --world-size 4

# 性能模式 + 小算子对照
bash run_test.sh --mode perf --world-size 8 --with-baseline

# 查看内置用例
python quant_matmul_reduce_scatter_test.py --list-cases
```
