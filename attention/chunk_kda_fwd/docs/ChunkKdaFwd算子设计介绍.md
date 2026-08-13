# ChunkKdaFwd算子设计介绍

## 1 设计目标

1. 顶层接口对齐非 CP 的 FLA `chunk_kda_fwd`。
2. 不新增公开算子原型；A5 特化复用既有 `ChunkKdaFwd` L0 调用接口和
   `chunk_kda_fwd` device kernel launch 入口。
3. A2/A3/A5 使用同一数学定义；A5 保留 regbase 双发射特化。
4. 输入 layout 与输出 layout 解耦。
5. FwdH 由 KDA 在 arch22/arch35 下分别维护，支持 key-wise gate 和 `state_v_first`。

## 2 计算过程

### 2.1 接口分层

```text
raw g -> ChunkKdaFwd[
    gate cumsum -> Prepare/Post-WU -> FwdH -> Finalize
] -> attn_out
```

接口调用链按层级划分为：

```text
cann_ops_transformer.ops.chunk_kda_fwd
  -> PyTorch C++ extension wrapper
  -> aclnnChunkKdaFwd
  -> ChunkKdaFwd L0 注册与 launcher
  -> chunk_kda_fwd device kernel launch 入口
  -> Gate / Prepare / Post-WU / FwdH / Finalize 内部阶段
```

`aclnnChunkKdaFwd` 负责公开 layout 的连续化和必要视图转换，并调用一次已注册的
`ChunkKdaFwd` L0 接口。L0 launcher 继续使用 `chunk_kda_fwd` device kernel launch 入口。
Gate、Prepare、Post-WU、FwdH 和 Finalize 只是该 kernel 内部计算阶段，不注册
`ChunkKdaFwdPrepare`、`ChunkKdaFwdPostWu` 或 `ChunkKdaFwdFinalize` L0 算子。模板选择所需信息
均由既有输入、shape 和属性推导，不增加公开属性或接口字段。

### 2.2 计算阶段

#### 2.2.1 Gate Cumsum

将 raw/已激活 gate 转为 FP32 chunk-local log2 累计值：

```text
gamma = g                                                        use_gate_in_kernel=false
gamma = -exp(A_log) * softplus(g + dt_bias)                      use_gate_in_kernel=true, safe_gate=false
gamma = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))        use_gate_in_kernel=true, safe_gate=true
gk = chunk_local_cumsum(gamma) / ln(2)
```

#### 2.2.2 Prepare

只读取 `q/k/v/gk/beta` 及变长元数据，产生：

```text
Aqk, Akk, qg, w_seed, u_seed
```

```text
qg[i] = q[i] * exp2(gk[i])
w_seed[i] = beta[i] * k[i] * exp2(gk[i])
u_seed[i] = beta[i] * v[i]
```

score 点积以 BF16/FP16 为输入、FP32 累积。`Akk=(I+L)^{-1}` 的 `L`、单位阵、
对角块求逆、分块合并、Cube 输入/累积和 workspace 均为 FP32，并显式关闭 HF32；只在完整
FP32 求解结束后将公开 `Akk` 写回为 q dtype。

#### 2.2.3 Post-WU

只读取 `k/gk/w_seed/Akk/u_seed`，产生：

```text
w, u, kg
w = Akk @ w_seed
u = Akk @ u_seed
kg[i] = k[i] * exp2(gk[last] - gk[i])
```

`Akk` 的 head 循环按 `H_v` 执行，GQA 映射只在读取 q/k head 时换算，避免按 `H_k` 重复或漏算。
Post-WU 是否作为单独的内部阶段执行由 tiling 决定。满足 A5 dense-aligned 融合条件时，Prepare
直接调用 Post-WU 计算组件；其他场景在 Prepare 结束并完成同步后执行内部 Post-WU 阶段。两种
方式都复用同一 `ChunkKdaFwd` L0 调用接口和 device kernel launch 入口。

#### 2.2.4 FwdH

读取 `kg/w/u/gk` 和可选 `initial_state`，计算 chunk 间状态更新：

```text
v_new = u - w @ h_prev
h_next = exp2(gk_last) * h_prev + kg^T @ v_new
h[chunk] = h_prev
final_state = h_after_last_chunk
```

arch22 和 arch35 均使用 `ChunkKdaFwd` 目录内的 KDA 私有 FwdH 实现，不依赖其他算子的
kernel 源码。跨 chunk 的 `h_prev/h_next` 固定保存在 FP32 workspace 或 FP32 `final_state` 中；
公开 `h` 仅作为反向保存快照按 q dtype 写回，不能反向参与下一 chunk 的状态更新。

#### 2.2.5 Finalize

只读取 `qg/Aqk/v_new/h`，以 FP32 累积计算：

```text
attn_out = scale * (qg @ h) + Aqk @ v_new
```

`scale` 在第一项完成 FP32 点积后施加，不预先量化进 BF16 `qg`。

kernel 内直接按 BSND/TND 写出 `attn_out`。供反向使用的中间量保持 BNSD/NTD。

## 3 数值设计

- `q`、`k`、`v`和公式中以输入dtype保存的中间量作为矩阵乘输入，矩阵乘使用FP32累加。
- `Akk=(I+L)^{-1}`的$L$、单位阵、对角块求逆、分块合并、Cube输入和累加保持FP32，不启用HF32。
- chunk间状态$h_c$和`final_state`的计算保持FP32，不使用公开`h`的低精度快照继续下一个chunk的状态传播。
- `attn_out`的两项矩阵乘和加法使用FP32累加，在最终写回时转为输出dtype。

## 4 输出布局与重计算

### 4.1 状态布局

内部状态统一使用 `[...,K,V]`。`state_v_first=true` 时，L2 在进入 FwdH 前转置 initial state。
内部 `hCompute` 始终保持 head-major 供 Finalize 消费；公开 `hOut` 在 L2 导出边界转为
sequence-major，并按 `state_v_first` 决定末两维顺序。`final_state` 按序列排列，与 FLA 顶层
输出一致。

### 4.2 重计算策略

L2 不理解 autograd 重计算策略。`final_state/gk/w/u/qg/kg/v_new/h` 是相互独立的
`OPTIONAL_OUTPUT`；非空指针表示导出，空指针表示不公开该结果。arch35 快路径为隐藏输出传递
固定 ABI 占位，并由 tiling 在 kernel workspace 中承接实际中间结果；通用路径使用同一规则。
公开输出存在时直接作为内部目标使用。

PyTorch API包装层对齐fla-org `chunk_kda_fwd`提交
`0f0f0c97af39343855b43bbbaddcedfda5cb9d77`：

- `attn_out` 始终返回。
- `final_state` 只在 `output_final_state=true` 时返回。
- `gk` 在 `use_gate_in_kernel=false` 或 `disable_recompute=true` 时返回。
- `Aqk` 始终返回。
- `Akk` 始终返回。
- `w` 在 `disable_recompute=true` 时返回。
- `u` 在 `disable_recompute=true` 时返回。
- `qg` 在 `disable_recompute=true` 时返回。
- `kg` 在 `disable_recompute=true` 时返回。
- `v_new` 在 `disable_recompute=true` 时返回。
- `h` 在 `disable_recompute=true` 或 `return_intermediate_states=true` 时返回。
- `initial_state` 由Python层原对象透传，不是aclnn输出。

内部 `hCompute` 与公开 `hOut` 是两个生命周期：`hCompute` 是 FwdH 到 Finalize 的必需
head-major 阶段结果，`hOut` 为空时由 kernel workspace 承接；`hOut` 非空时，L2 提供 head-major
临时输出并在导出边界转为 sequence-major。该规则对齐非 CP 的低层 12 返回值接口；
第 12 项 `initial_state` 由 Python 层原对象透传。

## 5 Tiling设计

本文中的“模板”指由 tiling 或 kernel 条件选择、具有独立数据切分或流水组织的计算路径。
C++ 中仅用于 dtype、常量传播或代码复用的普通 `template` helper 不单独列为场景模板。

### 5.1 接口与实现目录

`ChunkKdaFwd` 保留一套 L0 注册、launcher 和 `op_kernel/chunk_kda_fwd.cpp` device kernel launch
入口。A2/A3 通用实现位于 `op_kernel/*.h`，A5 特化位于 `op_kernel/arch35/*.h`，A5 host 派生
选项位于 `op_host/arch35/chunk_kda_fwd_tiling_impl.h`。Prepare、Post-WU 和 Finalize 的 `.h`
文件是同一 kernel 内部计算组件，不是独立 L0 算子接口。

### 5.2 编译期shape模板

| tiling key | 编译期常量 | A2/A3 实现 | A5 实现 | 覆盖场景 |
| --- | --- | --- | --- | --- |
| `key=1` 通用 shape 模板 | `COMPILE_BT/K/V=0/0/0` | 根目录通用实现 | 根目录通用实现 | 不满足 `chunk=64,K=128,V=128` 的 dense、tail、varlen、GQA 等场景 |
| `key=2` chunk64/K128/V128 模板 | `COMPILE_BT/K/V=64/128/128` | 根目录通用实现，使用编译期常量 | `arch35` 特化实现 | `chunk=64,K=V=128` 的 dense、tail、varlen 和不同 head 数 |

`SetTilingKey` 只检查 chunk、K、V，不检查 SoC、layout、是否 varlen 或是否存在尾块。因此 tiling
key 表示 shape 模板族，不表示平台、接口或算子版本。平台差异由同一 key 编译时的架构分支选择。

### 5.3 A5 key2执行组合模板

`ConfigureChunkKdaFwdArch35` 从既有输入和输出实例化状态派生四个内部选项。选项不属于公开
属性，不改变 L0、aclnn L2 或 Python 接口。

| 执行组合 | 选择条件 | 内部阶段顺序 |
| --- | --- | --- |
| standalone Gate | `computeGateInPrepare=false` | Gate 完整执行并同步，再进入 Prepare |
| Prepare 内联 Gate | BF16 q/k/v、raw g 为 FP32、存在 `A_log`、`use_gate_in_kernel=true`、`safe_gate=true` | Prepare AIV 在 gate-product 流水中计算 chunk-local gk |
| A5 dense FwdH | BF16 且非 varlen、`T % 64 == 0` | 使用 `ChunkKdaFwdFwdH` arch35 dense 模板 |
| Prepare/Post-WU 阶段内流水融合 | A5 key2、BF16、`safe_gate=true`、dense-aligned、`H_v % 2 == 0`，且未选择下一行模板 | Prepare AIC 按 head pair 生产后，直接调用 Post-WU 组件分批消费 |
| Post-WU/FwdH 阶段内流水融合 | 上一行条件成立，同时 Gate 已在 Prepare 内计算，且不导出 `qg/v_new/h` | dense FwdH 直接完成 W/U 和状态传播所需计算 |
| 独立内部 Post-WU 阶段 | 上述两种 Post-WU 融合均未选择 | Prepare 完成同步后调用 `RunChunkKdaPostWu` |
| 通用 FwdH backend | FP16、varlen、存在非对齐尾块或不满足 dense FwdH 条件 | 选择 KDA 私有 `KdaFwdHTileShapes128/256` 模板 |

因此“Prepare/Post-WU 融合”只描述满足表中条件的执行组合，不表示所有 key2 场景都取消了
内部 Post-WU 阶段。特别是 varlen 和非对齐尾块使用独立内部 Post-WU 阶段。

### 5.4 Gate与Prepare内部模板

| 模板 | 选择条件 | 主要实现 |
| --- | --- | --- |
| Gate 通用 chunk-cumsum | standalone Gate | 复用 `KdaGateCumsum::DispatchKdaGateCumsum`，支持 raw/已激活 gate 和 safe true/false |
| Gate/Prepare arch35 regbase | A5 key2、BF16、raw FP32、`A_log`、safe gate | 16-row gate tile、两套 UB staging，MTE2 与 VEC 交叠；gk、qg、w seed 和 kg 因子成批生成 |
| Prepare 通用 score/solve | key1，以及 A2/A3 key2 | 通用 Aqk/Akk MMAD、causal mask、三角 solve 和 seed 写回 |
| Prepare arch35 key2 通用边界 | A5 key2 的 odd head、safe false、tail 或 varlen 等未命中成批融合的场景 | 保留同一数学定义，使用有界 score queue、AIC/AIV 握手和 padded tail |
| Prepare arch35 safe-gate head-pair | A5 key2、safe gate、`H_v % 2 == 0` | 两个 head lane，共享两级 score queue；32-row score block；Akk solve 与后续 score 搬运交叠 |
| Prepare arch35 full-chunk fused-score | 上一行且 `curT=64`，并命中 fused Post-WU | QK/Akk 两次 MMAD 使用手工 L1/L0 流水，Aqk/Akk mask 和 beta seed 在写回前完成 |

### 5.5 Post-WU内部模板

| 模板 | 选择条件 | 主要实现 |
| --- | --- | --- |
| Post-WU 通用模板 | key1，以及 A2/A3 key2 | 通用 Cube/Vector 计算 `w=Akk@w_seed`、`u=Akk@u_seed`，并生成 kg |
| Post-WU arch35 full-chunk head-pair | A5 key2、`curT=64`、融合组合 | 每批最多 4 个 chunk task、每 task 两个 head lane；Prepare 生产后直接消费 |
| Post-WU arch35 full-chunk standalone | A5 key2、`curT=64`、未融合 | W/U 两套 MTE2 staging 与 MMAD/Fixpipe 流水；Kg 使用 16-row tile 和 32-row 双缓冲 |
| Post-WU arch35 tail 16-63 | A5 key2、`16<=curT<64` | padded Cube 与按有效行写回，保留通用尾块同步协议 |
| Post-WU arch35 BF16 sub-16 | A5 key2、BF16、`1<=curT<16` | `ComputeTailWuRegbaseArch35` 先完整暂存 W/U seed 和 Akk；单 AIV FP32 regbase 累加并写回，避免 W seed 与 W 输出原地复用时的跨 AIV 读写竞争 |
| Post-WU arch35 sub-16 fallback | A5 key2、非 BF16、`1<=curT<16` | 使用通用逐行 Vector 路径；key1 和 A2/A3 由 Post-WU 通用模板覆盖 |

Post-WU sub-16 模板的最坏 staging 为
`2*15*128*sizeof(BF16)+15*64*sizeof(BF16)=9,600 B`，不新增 workspace 或公开输出。

### 5.6 FwdH内部模板

| 模板 | 选择条件 | 主要实现 |
| --- | --- | --- |
| `KdaFwdHTileShapes128` | 通用 backend 且 `V<=128` | KDA 私有 arch22/arch35 FwdH；arch22 对完整/不小于 16 行的 chunk 使用 BF16 输入、FP32 累加的 Cube MMAD 计算 `kg^T@v_new`，仅 sub-16 尾块使用有界 Vector 兜底；跨 chunk 状态保持 FP32，支持 dense/varlen/tail |
| `KdaFwdHTileShapes256` | 通用 backend 且 `128<V<=256` | V=256 分块，其他语义与 128 模板一致 |
| `ChunkKdaFwdFwdH` arch35 dense | A5 key2、BF16、非 varlen、`T%64=0` | 16-token sub-chunk、4-slot L1 staging，AIC MMAD/Fixpipe 与 AIV gate/state 更新协作 |
| arch35 dense fused Post-WU/FwdH | `fusePostWuIntoFwdH=true` | FwdH 从 Prepare seed 直接形成 W/U 与 v_new，不导出未请求的 qg/v_new/h |

arch35 dense FwdH 的 state、W 和 v_new L1 slot 使用 ready/free credit 保护复用。AIV 在最后一个
task 完成后显式消费 AIC 返还的 terminal free credit，再退出 kernel，避免未消费 credit 进入后续执行。

### 5.7 Finalize内部模板

| 模板 | 选择条件 | 主要实现 |
| --- | --- | --- |
| Finalize 通用模板 | key1，以及 A2/A3 key2 | 通用 Cube 计算两项矩阵乘，AIV 完成 FP32 相加、类型转换与 sequence-major 写回 |
| Finalize arch35 dense-aligned pipeline | A5 key2、非 varlen、`T%64=0` | `ProcessOutAicPipelinedArch35` 使用双 L1 slot，使下一输出 tile 的 MTE2 与当前 MTE1/MMAD/Fixpipe 交叠 |
| Finalize arch35 full-chunk staged | A5 key2、`curT=64` 且未命中 dense-aligned pipeline | 分阶段计算 `scale*(qg@h)` 与 `Aqk@v_new`，保持 FP32 中间结果 |
| Finalize arch35 tail 16-63 | A5 key2、`16<=curT<64` | padded Cube 计算两项结果，AIV 按有效行相加并写回 |
| Finalize arch35 BF16 sub-16 | A5 key2、BF16、`1<=curT<16` | `ComputeTailOutputRegbaseRows` 一次暂存 h、v_new、qg 和 Aqk，单次 MTE2-to-V 交接后以 FP32 regbase 直接累加两项，绕开尾块标量 EventID 链 |
| Finalize arch35 sub-16 fallback | A5 key2、非 BF16、`1<=curT<16` | 通用 FP32 Vector 逐行归约；key1 和 A2/A3 由 Finalize 通用模板覆盖 |

Finalize sub-16 模板的最坏 staging 为
`(128*128+2*15*128+15*16)*sizeof(BF16)=40,928 B`，小于现有 `gateWritebackBuf_`
在 key2 BF16 下的 `40,960 B`，因此阈值固定为 16，不依赖运行时试探。

所有模板保持相同公式、公开输出顺序、可选输出语义和 layout 契约。模板选择不得改变
`ChunkKdaFwd` L0、`aclnnChunkKdaFwd` 或 `cann_ops_transformer.ops.chunk_kda_fwd` 的接口。

## 6 流水设计

- Prepare 的右矩阵在 L1 驻留，避免 K/K^T 重复搬运和重复转置。
- AIC 使用 L1/L0 双缓冲组织 MTE2、MTE1、Cube、Fixpipe。
- AIV 使用输入 staging ping-pong，使下一 tile MTE2 与当前 tile VEC 重叠。
- A5 VEC 路径使用 regbase 双发射；数值主计算仍保持 FP32。
- inter-sub-chunk 合并使用独立 workspace 区域，避免阻塞主 tile 流水。

性能结论只使用 `msopprof`。目标回归 case 由本算子测试目录维护。
A5 的 H96/T16K、K=V=128、chunk=64、BF16 关键场景以 10 ms 为目标值；门禁读取
`ChunkKdaFwd` 主 kernel 和 BSND 输入的 `beta` 布局转换的全部有效样本，保守累加两个组件的
各自最大值；合计超过 11 ms 或任一组件未采集到有效样本都失败。
同一关键场景使用非零输入连续执行 50 次，要求 `attn_out/final_state/Aqk/Akk` 二进制一致。

## 7 验证矩阵

- 平台：A2/A3/A5。
- dtype：FP16/BF16。
- layout：BSND/BNSD/TND/NTD。
- gate：raw/已激活、safe true/false。
- Shape：K=128，V=128/256，chunk=64/128，dense/varlen/tail/GQA。
- 属性：final state、重计算策略、`state_v_first`。
