# MinimaxSparseAttentionSplitKv 算子设计文档

## 1. 算子目标

`minimax_sparse_attention_split_kv` 面向 Prefill 阶段的 Sparse Attention，采用 **KV-Gather-Q** 方案：以 KV block 为主轴分任务，同一个 KV block 只加载一次，然后遍历所有选择该 block 的 Q token，计算该 block 对这些 Q token 的 partial attention。

目标实现如下数据流：

1. Phase 1 完成 `QK + softmax + PV` 的 partial 计算。
2. Phase 1 以 `1 logical KV block × 1 KV head` 为 task 粒度。
3. Q token 通过 host 提供的反向索引 gather，Q 在 GM 中可能非连续，因此 Q 需要逐行搬运到连续的片上临时区后参与矩阵乘。
4. Phase 1 将每个 `(qToken, kvHead, slotK)` 的 `O_partial`（未归一化 P×V）、`rowMax`、`rowSum` 写入 workspace；**不在 Phase 1 做 `O / rowSum`**。
5. Phase 2 在 Vector 侧读取每个 Q 的 topK partial，复用 IFA FlashDecode 路径（`CopyLseIn` → `ComputeScaleValue` → `CopyAccumOutIn`（`O/rowSum`）→ `ReduceFinalRes`）完成 rescale 与 combine，写最终 `attention_out`。
6. Phase 1 与 Phase 2 在同一个 kernel 内执行，中间通过 `SyncAll` 做全局同步。

稀疏选择关系由 host 预构建的反向索引张量表达；block 有效长度由 kernel 根据 `actual_seq_lengths_kv` 与 `blockSize` 在片上计算。

## 2. 输入、输出与辅助张量

### 2.1 算子输入

KV 支持四种存储（由 `block_table` 与 attr **`inputLayout`** 共同决定）：

1. **Paged KV Cache**（`block_table` 非空）：`key`/`value` 为 4 维 `[num_physical_blocks, blockSize, kvHeads, D]`。`query` 必须是 TND，`inputLayout="TND"`。
2. **TND 连续内存**（`block_table` 为空，`inputLayout="TND"`）：`query`/`key`/`value` 为 `[T, N, D]`。token 按 `actual_seq_lengths*` 在 T 轴 packed。
3. **BNSD 连续内存**（`block_table` 为空，`inputLayout="BNSD"`）：训练图常用。`query` `[B, Nq, S, D]`，`key`/`value` `[B, Nkv, S_kv, D]`。同一 head 的 token 连续（stride = D）；CSR 的 `qToken` 为 padded 展平 id `b * S + t`。
4. **BSND 连续内存**（`block_table` 为空，`inputLayout="BSND"`）：`query` `[B, S, Nq, D]`，`key`/`value` `[B, S_kv, Nkv, D]`。同一 token 的 heads 连续（stride = D），Q gather 与 TND 相同；K/V 用 padded `b * S + tok` 寻址。

BNSD 与 BSND 都是 4 维，**不能只靠 shape 区分**（`N==S` 时 shape 完全一样），必须传 `inputLayout`。

| Tensor | Shape | Type | 说明 |
| --- | --- | --- | --- |
| `query` | TND：`[T, Nq, D]`；BNSD：`[B, Nq, S, D]`；BSND：`[B, S, Nq, D]` | bf16/fp16 | 4D 的 `qToken` 是 `b * S + t`。 |
| `key` | PA：`[num_physical_blocks, blockSize, kvHeads, D]`；TND：`[T_kv, kvHeads, D]`；BNSD：`[B, kvHeads, S_kv, D]`；BSND：`[B, S_kv, kvHeads, D]` | bf16/fp16 | 有 `block_table` 为 PA；否则与 `inputLayout` 一致。 |
| `value` | 与 `key` 相同 | bf16/fp16 | 与 `key` 相同。 |
| `block_table` | `[batch, maxBlocksPerBatch]` | int32 | **可选**。传入则为 PA（仅 TND query）；不传则为连续内存。 |
| `k2q_row_ptr` | `[kvHeads, totalPackedRows + 1]` | int32 | Dense packed-row CSR 行指针；`totalPackedRows = sum(ceil(kv_seqlen/blockSize))`。 |
| `k2q_q_indices` | `[kvHeads, totalQTokens * topK]` | int32 | CSR 列数据：global `qToken` id，按 head 分段存储。 |
| `k2q_slot_indices` | `[kvHeads, totalQTokens * topK]` | int32 | 每条边在 workspace 中的 `slotK`，范围 `[0, topK)`。 |
| `actual_seq_lengths` | `[batch]` | int32 | **必选**。每个 batch 的 Q sequence length。允许 **0**：padding / dummy 请求。TND 模式下 `batch` 取该 tensor（或 `actual_seq_lengths_kv`）的长度。 |
| `actual_seq_lengths_kv` | `[batch]` | int32 | **必选**。每个 batch 的 KV sequence length。允许 **0**，与 `actual_seq_lengths` 一起标识 dummy 请求。 |


### 2.2 反向索引输入约束

`k2q_row_ptr`、`k2q_q_indices`、`k2q_slot_indices` 由 host 在调用算子前构建（CSR 格式，对齐 CUDA `build_k2q_csr_reference`），并作为 GM 输入传入 kernel，不放入 workspace。

语义约定：

1. Host 侧 forward 选块为 `select_idx[kvHeads, totalQTokens, topK]`（可用 global logical id 表达，构建 CSR 时转为 batch-local q2k）与 `select_num_idx[kvHeads, totalQTokens]`。
2. **`totalPackedRows`** = `sum_b ceil(kv_seqlen[b]/blockSize)`，与 MSA/CUDA `build_k2q_csr` 一致；未选中的块对应 CSR 空行（`row_ptr[r+1]==row_ptr[r]`）。`kv_seqlen[b]=0` 的 padding 请求贡献 0 行，不进入 Phase1 packed row。
3. Phase1 task：`taskIdx = packedRow * kvHeads + kvHeadIdx`，其中 `packedRow` 由 `(batch, local_kv_block)` 按 MSA 规则打包。
4. stride 分核遍历 `taskIdx`；`packedRow = taskIdx / kvHeads` 变化时在循环内增量步进 MSA coord（见 §5.1 / §6.1），再按 PA/`block_table` 或 TND token 偏移取 K/V block。单测 `decode_packed_row` 用搜索式解码校验同一语义。
5. `csrStart = k2q_row_ptr[kvHeadIdx, packedRow]`，`csrEnd = k2q_row_ptr[kvHeadIdx, packedRow + 1]`，`numQTokens = csrEnd - csrStart`。
6. `qToken = k2q_q_indices[kvHeadIdx, csrStart + qi]`（global flattened q id），`slotK = k2q_slot_indices[...]`。
7. `k2qNnzUpperBound = k2q_q_indices.shape[1]`，通常为 `totalQTokens * topK`。
8. GQA 下不同 `kvHeadIdx` 可有不同 topK 选块。


### 2.3 输出

| Tensor | Shape | Type | 说明 |
| --- | --- | --- | --- |
| `attention_out` | 与 `query` 相同 | bf16/fp16 | TND `[T, Nq, D]` / BNSD `[B, Nq, S, D]` / BSND `[B, S, Nq, D]`。 |
| `softmax_lse` | flag=true：TND `[T, Nq, 1]` / BNSD `[B, Nq, S, 1]` / BSND `[B, S, Nq, 1]`；flag=false：`[0]` | fp32 | 与 FIA 一致：`softmaxLseFlag` 控制是否写出。公式 `lse = log(sum_k rowSum[k] * exp(rowMax[k] - max_k)) + max_k`，由 Phase2 `ComputeScaleValue_VF` 在 combine 时得到。 |

`softmaxLse` 是 IR 必选输出（与 FIA 相同）。`softmaxLseFlag=false` 时 infershape 为 `[0]`，aclnn 构造占位 tensor，kernel **不写** 该 GM。

### 2.4 属性

Attr 顺序必须与 `def.cpp` / infershape / tiling / L0 `OP_ATTR` 一致。

| Attr | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `numKeyValueHeads` | int | `1` | KV head 数。 |
| `scaleValue` | float | `0.0` | QK scale，通常 `1/sqrt(D)`。 |
| `blockSize` | int | `128` | KV block token 数。 |
| `topK` | int | `8` | 每个 Q token 选中的 KV block 数。 |
| `innerPrecise` | int | `4` | `0` / `1` / `4`，见 §3.1.1。 |
| `softmaxLseFlag` | bool | `false` | 是否写出 `softmax_lse`。 |
| `inputLayout` | string | `"TND"` | `"TND"` / `"BNSD"` / `"BSND"`（大写，与 FA/FIA 一致）。`nullptr` 或空串按 `"TND"`。 |

Rank 必须匹配：TND ↔ 3 维，BNSD/BSND ↔ 4 维。Paged KV（`block_table` 非空）只允许 `"TND"`。

## 3. Tiling 数据定义

### 3.1 基础字段

| 字段 | 含义 |
| --- | --- |
| `batch` | batch 数。PA 来自 `block_table.shape[0]`；TND 非 kvcache 来自 `actual_seq_lengths_kv` 的长度；BNSD/BSND 来自 `query.shape[0]`。 |
| `numHeads` | Q head 数。TND/BNSD：`query` 的 N 维；BSND：`query.shape[2]`。 |
| `kvHeads` | KV head 数，优先来自 attr，否则 PA 来自 `key.shape[2]`、TND 来自 `key.shape[1]`、BNSD 来自 `key.shape[1]`、BSND 来自 `key.shape[2]`。 |
| `groupSize` | `numHeads / kvHeads`，GQA 中一个 KV head 对应的 Q head 数。 |
| `embeddingSize` | head size D。当前主目标为 D=128。 |
| `blockSize` | KV block token 数，通常为 128。 |
| `topK` | 每个 Q token 选择的 KV block 数（attr）。 |
| `totalQTokens` | Q token 总数。TND：`query.shape[0]`；BNSD/BSND：`B * S`（含 padding）。 |
| `totalPackedRows` / `numKvBlocks` | `k2q_row_ptr.shape[1] - 1` = `sum_b ceil(kv_seqlen[b]/blockSize)`。tiling 字段名为 `numKvBlocks`。 |
| `maxBlocksPerBatch` | PA：`block_table.shape[1]`，用于 `block_table` 行 stride。TND：0，不参与寻址。 |
| `isPageAttention` | `1`：paged KV cache；`0`：连续内存。 |
| `layoutType` | `0`：TND；`1`：BNSD；`2`：BSND。来自 attr `inputLayout`。 |
| `qSeqLen` / `kvSeqLen` | BNSD/BSND 的 padded S（来自 shape）；TND 为 0。 |
| `softmaxLseFlag` | `1`：Phase2 写出 `softmax_lse`；`0`：跳过（输出 shape `[0]`）。 |
| `k2qNnzUpperBound` | `k2q_q_indices.shape[1]`，CSR 数据区每 head 的上界，通常 `totalQTokens * topK`。 |
| `totalTaskNumP1` | Phase 1 task 数，`totalPackedRows * kvHeads`。 |
| `totalTaskNumP2` | Phase 2 task 数，`totalQTokens * kvHeads`。 |
| `accumOutSize` | workspace 中 O_partial float 元素总数，`totalQTokens * kvHeads * topK * groupSize * D`。 |
| `lseStatSize` | workspace 中 softmaxMax 或 softmaxSum 各自 float 元素总数，`totalQTokens * kvHeads * topK * groupSize`。 |
| `workSpaceSize` | `libapiWorkspaceSize + userWorkspace` 字节数。 |
| `innerPrecise` | 内部计算精度。`0`：fp32 softmax + fp32 `O_partial`；`1`：bf16 softmax + bf16 `O_partial`；`4`（默认）：bf16 softmax + fp32 `O_partial`。其它值 tiling / aclnn 均拒绝。 |
| `tilingKey` | `20001`（BF16 `innerPrecise=4`）、`20002`（`=1`）、`20003`（`=0`）、`20004`（FP8 E4M3FN Q/K/V + BF16 out）。 |

### 3.1.1 innerPrecise 与 tilingKey

| `innerPrecise` | Softmax S（QK fixpipe） | `O_partial` | tilingKey |
| --- | --- | --- | --- |
| **0** ALL_HIGH | fp32（NoQuant） | fp32 | `20003` `INNER_HIGH` |
| **1** ALL_LOW | bf16 | bf16 | `20002` `INNER_LOW` |
| **4** MIXED（默认） | bf16 | fp32 | `20001`（BF16 QKV） / `20004`（FP8 QKV，attentionOut 仍为 BF16） |

`innerPrecise=0` 走独立 fp32 softmax epilogue（`block_epilogue_online_softmax_arch35_reg_high_prec.hpp`）：按行 `scale → causal -inf mask → max → exp(S-max) → sum`，P cast 为 bf16 后再 ND→zN 写入 L1，供 PV 使用。跨 block 的 rescale 仍在 Phase2，与 `=4` 相同（fp32 `O_partial`）。

**高精度 tiling 约束（host `CheckTilingConstraints`）**：fp32 S 把 UB 占满，不能沿用 bf16 的「只换 tilingKey」假设。Atlas A5 UB=256KB，kernel 固定：

```text
S   2 stage * 16384 * 4B = 128KB   # offset 0；bf16 路径只有 64KB
P   2 stage * 16384 * 2B =  64KB   # offset 128KB（bf16 路径在 64KB）
tmp                    32KB        # offset 192KB，row-max/sum 破坏性拷贝 + ND P
stats  gmUb/glUb                   # offset 224KB = 7*32KB，与 SM_UB_GM_OFFSET 对齐
```

因此 `innerPrecise=0` 必须同时满足：

1. `D == 128`，`blockSize ∈ (0, 128]`（与 `L0_TILE_M/N`、tilingKey `D128` 一致）。
2. 单 AIV 的 S tile `Align16(ceil(gCount/2)*groupSize) * Align16(blockSize) ≤ 8192`（tmp=32KB）。`groupSize=16, blockSize=128` → `64*128=8192` 刚好满；`groupSize=128` 单核吃整块 M=128 会撑爆 tmp，tiling 直接失败。
3. 统计 UB：`ceil(gCount/2) * Align8(groupSize) ≤ 64`（`SM_ROW_MAX_ELEM_NUM`）。
4. 平台 `ubSize ≥ stats 末端`。Workspace 仍按 fp32 `O_partial` 计（与 `innerPrecise=4` 相同），不必为 0 再加倍 GM。

`innerPrecise=1/4` 的 S 为 bf16，不受 tmp=8192 这条限制，但仍受 D/blockSize/groupSize 与 stats 64 条约束。

### 3.2 blockDim 策略

同一个 kernel 内包含 Cube Phase 1 和 Vector Phase 2，因此 `blockDim` 不能只按照 Phase 1 计算。推荐：

```text
totalTaskP1 = totalPackedRows * kvHeads
totalTaskP2 = totalQTokens * kvHeads
blockDim = min(max(totalTaskP1, totalTaskP2), max(aicNum, aivNum))
blockDim = max(blockDim, 1)
```

Phase 1 使用：

```text
for taskIdx = blockIdx; taskIdx < totalTaskNumP1; taskIdx += blockDim
```

Phase 2 使用：

```text
for taskIdx = blockIdx; taskIdx < totalTaskNumP2; taskIdx += blockDim
```

如果目标平台 AIC/AIV core 数不同，需要保证无效 core 不越界；每个 phase 都必须用自己的 `totalTaskNumPx` 做循环边界。

## 4. Workspace 布局

workspace 只保存 Phase 1 到 Phase 2 的 partial 数据，不保存反向索引表。布局与 IFA FlashDecode 一致：**O / max / sum 三块独立 buffer**；max/sum 在 GM 中为 **紧凑连续** `float`（每 slot 长度 `groupSize`），Phase 2 读入 UB 时再展开为 IFA VF 所需的 32B 行对齐布局。

### 4.1 总大小

```text
slotOElems    = groupSize * D
slotStatElems = groupSize                         # max/sum 每 slot 紧凑 float 个数
taskSlots     = totalQTokens * kvHeads * topK

accumOutSize  = taskSlots * slotOElems            # O_partial，fp32
lseStatSize   = taskSlots * slotStatElems         # softmaxMax 或 softmaxSum 各一块

userWorkspace = (accumOutSize + lseStatSize * 2) * sizeof(float)
workspaceSize = libapiWorkspaceSize + userWorkspace
```

tiling 字段：`accumOutSize`、`lseStatSize`（float 元素个数，非字节）。

kernel 入口使用 `GetUserWorkspace(workspace)` 时，tiling 侧申请的 workspace 必须包含 `libapiWorkspaceSize`。

GM 指针划分（顺序固定）：

```text
accumOutGm   = userWorkspace + 0
softmaxMaxGm = userWorkspace + accumOutSize * sizeof(float)
softmaxSumGm = userWorkspace + (accumOutSize + lseStatSize) * sizeof(float)
```

Phase 1 启动前 **建议** 将 `softmaxMax` / `softmaxSum` 预置为无效：`rowMax[gh] = -inf`，`rowSum[gh] = 0`（未写入 slot 可被 Phase 2 识别为无效）。

**当前 Arch35 kernel 实现**：kernel 入口由 VEC 侧显式初始化 workspace 的 `softmaxMax` / `softmaxSum` 连续区域（`InitWorkspaceStats`，`minimax_sparse_attention_split_kv_kernel_arch35.h`）。在 `InitSyncFlags()` 之后、`SyncAll<false>()` 之前调用：UB 上各 `Duplicate` 一次（`ubMax<-(-inf)`、`ubSum<-0`），单次循环按 `WS_INIT_CHUNK`（4096 fp32）分块同时 `DataCopyPad` 写 `softmaxMax` 与 `softmaxSum`（尾块按 `lseStatSize_ % WS_INIT_CHUNK` 长度处理）；各 VEC sub-block 按 chunk 下标 stride 分核。末尾 `PipeBarrier<PIPE_ALL>`，由随后的 `SyncAll<false>()` 兼作 init 完成屏障。CUBE 核空转。`accumOut` 不预置（Phase2 `ReduceFinalRes` 的 `lseMaxBuf_ <= NEG_INF_LSE` 跳过非法 split）。Golden 侧亦显式将 `ws_max` 初始化为 `-inf`。

### 4.2 逻辑 shape 与线性偏移

对每个 `(qToken, kvHeadIdx)` task，topK 个 slot 在各自 buffer 内连续存放：

```text
taskIdx = qToken * kvHeads + kvHeadIdx

# accumOut: [topK, groupSize, D]  fp32，未归一化 O_partial = P × V
oBase(taskIdx, slotK) = taskIdx * topK * slotOElems + slotK * slotOElems
O_partial[gh, d]      -> accumOutGm[oBase + gh * D + d]

# softmaxMax / softmaxSum: [topK, groupSize]  fp32，紧凑存储
statBase(taskIdx, slotK) = taskIdx * topK * slotStatElems + slotK * slotStatElems
rowMax[gh]               -> softmaxMaxGm[statBase + gh]
rowSum[gh]               -> softmaxSumGm[statBase + gh]
```

注意：

1. Phase 1 写入的 `O_partial = P×V` **未** 按 `rowSum` 归一化（与 IFA decode 不同：IFA 在 split 内 `RowDivs` 后写 `accumOut`）。Prefill 将 `/rowSum` **推迟到 Phase 2 的 `CopyAccumOutIn`**，再与 IFA 相同的 `ComputeScaleValue` scale 做 `ReduceFinalRes`。
2. `rowMax` / `rowSum` 各为每个 Q head 一份，长度为 `groupSize`；无效 slot 语义为 `rowSum = 0`（Golden 另将 `rowMax = -inf`）。
3. O 与 max/sum **不再** 打包在同一 `partialSize` 槽内；Phase 2 分别按上式寻址三块 buffer。
4. Phase 2 `CopyLseIn`：从 GM 按 task 批量读 compact max/sum，`Broadcast` 展开为 IFA VF 所需的 `[topK, dealRow, 8]`；`ComputeScaleValue_VF` 会 **覆写** UB 中 `lseSum` 为 scale，故 `CopyAccumOutIn` 必须从 **`gmSoftmaxSum` 再读** 原始 `rowSum`。

## 5. Batch offset、packed row 与有效长度

### 5.1 Batch offset

kernel 根据 `actual_seq_lengths` 和 `actual_seq_lengths_kv` 在片上计算 batch offset：

```text
batchQOffset[0] = 0
batchKvBlockOffset[0] = 0
for b in [0, batch):
  batchQOffset[b + 1] = batchQOffset[b] + q_seqlen[b]
  batchKvBlockOffset[b + 1] = batchKvBlockOffset[b] + ceil(kv_seqlen[b] / blockSize)
```

**Phase1** 不直接使用跨 batch 展平的 `kvBlockIdx`，而是通过 `(packedRow → batchIdx, localBlockIdx)` MSA coord 步进定位 logical KV block（§6.1）：

```text
# PA
physicalBlock = block_table[batchIdx * maxBlocksPerBatch + localBlockIdx]
kvBlockBase   = physicalBlock * blockSize * kvHeads * D + kvHeadIdx * D

# TND 连续内存
kvTokenStart  = sum(kv_seqlen[0..batchIdx)) + localBlockIdx * blockSize
kvBlockBase   = kvTokenStart * kvHeads * D + kvHeadIdx * D

# BNSD 连续内存
kvBlockBase   = (batchIdx * kvHeads + kvHeadIdx) * S_kv * D + localBlockIdx * blockSize * D

# BSND 连续内存
kvBlockBase   = (batchIdx * S_kv + localBlockIdx * blockSize) * kvHeads * D + kvHeadIdx * D
```

Q / `attention_out` / LSE 的 GM 偏移（`qToken` 为 CSR 给出的展平 id）：

```text
# TND / BSND：同一 token 的 heads 连续
qOffset = qToken * Nq * D + h * D
# BNSD：同一 head 的 tokens 连续
qOffset = (b * Nq + h) * S * D + t * D     # qToken = b * S + t

# LSE
TND  [T, N, 1]     offset = qToken * Nq + h
BNSD [B, N, S, 1]  offset = (b * Nq + h) * S + t
BSND [B, S, N, 1]  offset = qToken * Nq + h
```

**Q token** 反查所属 batch（causal 用 Q 侧 sequence length）。TND 的 `qToken` 是 packed flatten；BNSD/BSND 是 padded flatten `b * S + t`：

```text
# TND
qBatchIdx = first b where qToken < batchQOffset[b + 1]
localQIdx = qToken - batchQOffset[qBatchIdx]

# BNSD / BSND
localQIdx = qToken % S
```

`batchKvBlockOffset` 仍可用于 host 侧将 global logical block id 转为 batch-local id（构建 q2k/k2q CSR 时）。

> **打包顺序澄清**：`batchKvBlockOffset` 是 host 侧 **row-major 累加**（按 batch 顺序累 `ceil(kv_seqlen/blockSize)`），仅用于 global logical block id ↔ batch-local id 互转，**不等于** Phase1 的 `packedRow`。Phase1 `packedRow` 采用 **column-major MSA 打包**：外层遍历 `localBlockIdx`，内层遍历 `batchIdx`，即顺序为 `(batch=0,blk=0), (batch=1,blk=0), ..., (batch=B-1,blk=0), (batch=0,blk=1), ...`。该顺序由 `InitPackedRowCoord` / `AdvancePackedRowCoord` 在 kernel 内增量步进实现，并与 golden `_build_packed_row_map` / `_init/_advance_packed_row_coord` 对齐。`totalPackedRows = sum_b ceil(kv_seqlen[b]/blockSize)` 与打包顺序无关，两种打包下总数一致。
>
> **`q_len=kv_len=0` 的 padding 请求**：`ceil(0/blockSize)=0`，该 batch **不占用 packed row**。`InitPackedRowCoord` 从 `(0,0)` 起跳过 `KvRowsPerBatch==0` 的 batch，因此 packedRow 0 是第一个非空请求，而不是无条件的 batch 0。中间的 dummy 由 `AdvancePackedRowCoord` 同样跳过。全 batch 都是 dummy 时 `totalPackedRows=0`，Phase1 不进任务循环。

### 5.2 Block 有效长度

每个 logical KV block 的有效 token 数在 kernel 内根据 `actual_seq_lengths_kv` 计算：

```text
kvSeqlenBatch = kv_seqlen[batchIdx]
tailRemain    = kvSeqlenBatch - localBlockIdx * blockSize
validSize     = min(blockSize, tailRemain)   # tailRemain > 0 时有效
```

- 非尾块：`validSize = blockSize`
- 尾块：`validSize = kvSeqlenBatch % blockSize`；若整除则 `validSize = blockSize`
- `tailRemain <= 0` 的 block 不应出现在有效 `numKvBlocks` 范围内
- `kv_len=0`：`numBlocksB=0`，`validSize=0`，Phase1 `continue`（不读 K/V、不写 workspace）

## 6. Phase 1：KV-centric partial compute

### 6.1 Task 映射

```text
taskIdx    -> (packedRow, kvHeadIdx)
packedRow  = taskIdx / kvHeads
kvHeadIdx  = taskIdx % kvHeads
(batchIdx, localBlockIdx) from inline MSA coord step when packedRow advances
```

Phase1 分核：

```text
for taskIdx = blockIdx; taskIdx < totalTaskNumP1; taskIdx += blockDim
```

每个 task 处理：

```text
1 logical KV block × 1 KV head × all selected Q tokens of this block
```

### 6.2 片上数据流

目标实现应按 tile 化矩阵乘实现，而不是逐元素 GM 标量循环：

1. PA：从 `block_table` 读取 `physicalBlockId`。TND：用 `cumKvStart + localBlockIdx * blockSize` 得到 token 起点。
2. 根据 §5.2 计算 `validSize`。
3. 将该 logical block 当前 `kvHeadIdx` 的 `K[validSize, D]`、`V[validSize, D]` 搬到 L1/片上缓冲；每个 `(packedRow, kvHeadIdx)` task 加载一次，矩阵乘仅用前 `validSize` 行。
4. 从 CSR 得到 `numQTokens`；`packedRow` 变化时 inline 步进得 `(batchIdx, localBlockIdx)`，再按 §5.1 取 K/V 起点。
5. 按 `qi in [0, numQTokens)` 遍历 Q 列表。
6. 对每个 `qi`：
   - `qToken = k2q_q_indices[kvHeadIdx, csrStart + qi]`。
   - 对每个 Q token，搬运其 `groupSize` 个 Q head：`query[qToken, kvHeadIdx * groupSize + gh, :]`。
   - Q 在 GM 中可能非连续，必须逐行搬运；搬入后在片上整理成连续矩阵 `Q_tile[M, D]`，其中 `M = qTile * groupSize`。
7. Cube 执行 `S = Q_tile[groupSize, D] × K[D, validSize]`（QK 按 block 全长 `validSize` 计分）。
8. Vector softmax / PV 的有效 KV 长度为 `causalValidLen`（见 §6.4）：对 `S` 的前 `causalValidLen` 列做 online softmax（**当前实现以有效列长截断**，非显式写 `-inf` mask；Golden 对越界列显式置 `-inf`）。
9. Vector 对每 row 计算 `rowMax`、`rowSum`（跨 block 的 LSE 在 Phase2 `ComputeScaleValue_VF` 中按 `softmaxLseFlag` 合并写出）。
10. Vector 得到 `P = exp(S - rowMax)`（未除 `rowSum`），cast bf16 供 PV。
11. Cube 执行 `O_partial = P[groupSize, causalValidLen] × V[causalValidLen, D]`，fixpipe 写 GM（**不做** `/rowSum`）。
12. Vector 将 `rowMax`、`rowSum` 写入 workspace；`O_partial` 由 Cube PV 写入。

### 6.3 GQA 映射

一个 KV head 对应 `groupSize` 个 Q head：

```text
qHeadIdx = kvHeadIdx * groupSize + gh
for gh in [0, groupSize):
  query row = query[qToken, qHeadIdx, :]
```

同一个 `(packedRow, kvHeadIdx)` task 内，K/V 只加载一次，`groupSize` 个 Q head 共享该 K/V block。

### 6.4 Causal mask

每个 Q token 的 causal 边界不同，不能只按 block 统一 mask。

对每个 `qToken`（causal 边界由 **Q 所在 batch** 的 sequence length 决定；KV block 位置由 **packedRow 解码得到的 `localBlockIdx`** 决定）：

```text
qBatchIdx      = FindBatchForQToken(qToken)
localQIdx      = qToken - batchQOffset[qBatchIdx]
qPosition      = kv_seqlen[qBatchIdx] - q_seqlen[qBatchIdx] + localQIdx
kvStartPos     = localBlockIdx * blockSize    # localBlockIdx 来自 packedRow 的 MSA coord
causalValidLen = 0 if q_seqlen==0 or kv_seqlen==0 or (BNSD/BSND and localQIdx >= q_seqlen)
                 else 0 if qPosition < kvStartPos
                 else min(validSize, qPosition - kvStartPos + 1)
```

mask 规则（Golden 参考；kernel 等价地仅对前 `causalValidLen` 列做 softmax/PV）：

```text
for col in [0, validSize):
  if col >= validSize or col >= causalValidLen:
    score[row, col] = -inf   # Golden 显式 mask；kernel 以 causalValidLen 截断
```

当 `causalValidLen == 0` 时，kernel/golden **跳过写入**该 partial，依赖 workspace 预置 `rowMax == -inf`、`rowSum == 0`，Phase 2 自动跳过：

```text
# 不写入 workspace；slot 保持 rowMax == -inf, rowSum == 0
```

### 6.5 workspace 写出

Phase 1 每个 `(qToken, kvHeadIdx, slotK)` 写入三块 buffer：

```text
csrStart = k2q_row_ptr[kvHeadIdx, packedRow]
slotK    = k2q_slot_indices[kvHeadIdx, csrStart + qi]
taskIdx  = qToken * kvHeads + kvHeadIdx

oBase   = taskIdx * topK * groupSize * D + slotK * groupSize * D
statBase = taskIdx * topK * groupSize + slotK * groupSize
```

写出顺序：

```text
O_partial[gh, d] -> accumOutGm[oBase + gh * D + d]          # Cube PV fixpipe 写 GM
rowMax[gh]       -> softmaxMaxGm[statBase + gh]             # Vector 连续 DataCopyPad
rowSum[gh]       -> softmaxSumGm[statBase + gh]
```

O 写 workspace 时按 Q 行逐行写，因为 Q token 是 gather 得到的，不能假设连续。max/sum 按 subcore 负责的 `gh` 行连续写入对应 slot。

## 7. Phase 2：Vector-only CombineScale（IFA FlashDecode 路径）

Phase 2 实现与 `incre_flash_attention` 的 `FlashDecodeCompute` / `CombineSplitKVRes` 对齐，文件：`block_epilogue.hpp`。

```text
CopyLseIn(max, sum)
  -> ComputeScaleValue_VF   # lseSum UB 覆写为 scale；softmaxLseFlag=true 时同时写 LSE 到独立 UB 并 DataCopyPad 到 softmaxLse GM [T,N,1]
  -> CopyAccumOutIn         # 读 O_partial，从 gmSoftmaxSum 读 rowSum，O_norm = O / rowSum
  -> ReduceFinalRes_VF      # out += scale[k] * O_norm[k]
  -> CopyFinalResOut
```

各步骤说明：

- **`CopyLseIn`**：从 GM 读当前 task 的 compact `softmaxMax` / `softmaxSum`（逻辑 shape `[topK, groupSize]`）。当前实现：对 `startRow=0` 且 `dealRowCount×topK` 连续段做 `DataCopyPad`，再 `Broadcast` 到 UB 布局 `[topK, dealRow, 8]`（32B 行对齐）。**当前实现仅支持 `loopCount == 1`（即 `groupSize <= gSplitSize`）**：`CopyLseIn` 中 `taskBase` 未加 `startRow` 偏移、`blockLen = dealRowCount * topK * sizeof(float)` 假设整段连续，当 `groupSize > gSplitSize` 分多片 combine 时既漏掉 `startRow` 之后的行、又跨 split 拼接错误数据。大 `groupSize` 分片 `startRow` gather 路径待实现（与 §9 item 3 一致），在完成前应避免 `groupSize > gSplitSize` 的 shape。
- **`ComputeScaleValue_VF`**：输入 UB 中 `(rowMax, rowSum)`，输出 **scale 权重** 写回 `lseSum` UB（与 IFA 相同公式）。`softmaxLseFlag=true` 时额外计算 `lse = log(Σ_k rowSum[k] * exp(rowMax[k]-max)) + max`，UB 布局 `[groupSize, 8]`，再 `DataCopyPad` 到 `softmax_lse`（TND/BSND 同一 token 的 heads 连续；BNSD 同一 token 的 heads 间隔 S）。
- **`CopyAccumOutIn`**：按 split 索引 `j` 从 `accumOutGm` 读未归一化 `O_partial`；**必须从 `gmSoftmaxSum` 读取原始 `rowSum`**（因 `lseSum` UB 已被 scale 覆盖），对每 row 做 `O_norm = O_partial / rowSum`（当前 kernel 用 `Divs`；Golden 对 `rowSum<=0` 置 0）。
- **`ReduceFinalRes_VF`**：对 topK 个 split 累加 `scale[j] * O_norm[j]` 到 `dst`。

### 7.1 Task 映射

```text
taskIdx   -> (qToken, kvHeadIdx)
qToken    = taskIdx / kvHeads
kvHeadIdx = taskIdx % kvHeads
```

每个 task 输出该 `qToken` 在一个 KV head group 下的 `groupSize` 个 Q head：

```text
attention_out[qToken, kvHeadIdx * groupSize + gh, :]
```

Phase 2 分核（仅 `__DAV_VEC__` 执行）：

```text
coreIdx  = GetBlockIdx()   # VEC 线性 AIV id，范围 [0, GetBlockNum()*GetSubBlockNum())
coreNum  = GetBlockNum() * GetSubBlockNum()   # GetBlockNum() 为 AIC block 数（AIV 数的一半）
for taskIdx = coreIdx; taskIdx < totalTaskNumP2; taskIdx += coreNum:
    qToken = taskIdx / kvHeads
    if IsPaddingQToken(qToken):   # BNSD/BSND: q_len=kv_len=0 or t>=q_len
        continue                  # 不写 attention_out / softmax_lse
    FlashDecodeCompute(taskIdx, totalTaskNumP2, ...)
```

> 历史实现曾单次调用 `FlashDecodeCompute(GetBlockIdx(), ...)` 且无循环，
> 导致 `totalTaskNumP2 > AIV 数` 时尾部 `(qToken, kvHead)` 不被 combine（输出恒 0）。
> 另曾误用 `stride = GetBlockNum()`（仅为 block 数），实际应乘 `GetSubBlockNum()`。

### 7.2 Combine 公式（kernel 与 golden 一致）

对每个 `(qToken, kvHeadIdx, gh)`，在 topK 个 partial 上：

```text
# Step A: ComputeScaleValue_VF（IFA split-KV；softmaxLseFlag 控制是否写 LSE）
max_global = max_k(rowMax[k])

scale[k] = rowSum[k] * exp(rowMax[k] - max_global)
scale[k] /= sum_j scale[j]

# Step B: CopyAccumOutIn（Prefill 相对 IFA Phase1 的 RowDivs）
O_norm[k] = O_partial[k] / rowSum[k]     # rowSum 从 gmSoftmaxSum 读；无效时 Golden 置 0

# Step C: ReduceFinalRes_VF
out[d] = sum_k scale[k] * O_norm[k][d]
```

等价于 IFA 在 split 内先 `O/rowSum` 再按相同 scale combine。Phase 1 GM 中 **`O_partial` 保持未归一化**；归一化发生在 Phase 2 `CopyAccumOutIn`。

### 7.3 无效 slot 处理

无效 slot 语义：`rowSum[gh] <= 0`（Golden 同时设 `rowMax = -inf`）。

```text
# Phase 1 未写入 -> workspace 保持初始值（kernel: rowMax=-inf, rowSum=0，由 InitWorkspaceStats 预置）
# Phase 2:
CopyLseIn             -> 读 compact max/sum 到 UB，Broadcast 成 [topK, dealRow, 8]（无效 slot: rowMax=-FLT_MAX, rowSum=0）
ComputeScaleValue_VF  -> 读 lseMax（只读不改）算 scale，写回 lseSum UB；无效 slot 的 scale 为 0
ReduceFinalRes_VF     -> 按 split j 从 UB 读 lseMaxBuf_[j*dealRow*8]（slot 首行 rowMax）：
                         <= NEG_INF_LSE 则该 slot 非法，跳过 CopyAccumOutIn 与 ReduceFinalRes_VF；
                         scale[j]==0 不贡献
```

> **非法 slot 跳过（UB 判 -inf）**：`ReduceFinalRes`（`block_epilogue.hpp`）在每个 split `j` 进入 `CopyAccumOutIn` 前，从 UB 的 `lseMaxBuf_[j * dealRowCount * FP32_ONE_BLOCK_SIZE]`（broadcast 布局 `[topK, dealRowCount, 8]`，split `j` 首行 rowMax）读标量。`CopyLseIn` 已把 GM compact max 读入 UB 并 Broadcast，`ComputeScaleValue_8_VF` 只读 `lseMax` 不覆写，故该值在 `ComputeScaleValue` 之后仍为原始 rowMax。`InitWorkspaceStats` 把未写入 slot 的 `rowMax` 预置为 `-FLT_MAX`（`NEG_INF_LSE = -3.4028235e38f`），而有效 slot 的 rowMax 是有限 score 派生值，故 `firstRowMax <= NEG_INF_LSE` 等价于 slot 非法。非法 split 直接 `continue`，不执行 `CopyAccumOutIn`（避免 `Divs(O, 0)` 产生 `inf/nan`）也不执行 `ReduceFinalRes_VF`（避免 stale UB 累加），依赖 `ComputeScaleValue` 已得 `scale[j]=0` 保证 0 贡献。`CopyAccumOutIn` 内部不做逐行除零 pad。该判断从 UB 读取，无 GM 往返。

规则：

1. Phase 1 仅对 CSR 中且 `causalValidLen > 0` 的 `(qToken, slotK)` 写入有效 partial。
2. 未被 Phase 1 写入的 slot 不参与有效 combine（依赖 `rowSum=0` → scale=0）。
3. `causalValidLen == 0` 时跳过 Phase 1 写入，Phase 2 自动跳过。

### 7.4 输出写回

```text
outAddr = qToken * numHeads * D + (kvHeadIdx * groupSize + gh) * D + d
```

Phase 2 必须全在 Vector 侧完成，不再触发 Cube 计算。

## 8. CPU Golden 参考实现

`tests/test_minimax_sparse_attention_split_kv_golden.py` 与 kernel **任务划分、CSR 消费、MSA coord 步进、causal 规则、Phase2 combine** 对齐：

| 项 | Kernel | CPU Golden |
| --- | --- | --- |
| Phase1 分核 | `taskIdx += blockDim` stride | `taskIdx` 顺序 0..totalTaskNumP1-1（等价 coreNum=1） |
| coord | `Init/AdvancePackedRowCoord` inline | `_init/_advance_packed_row_coord` inline |
| Phase1 partial | `accumOut` + `softmaxMax` + `softmaxSum` 三块 GM | `ws_o / ws_max / ws_sum`，shape `[T,kvH,topK,...]` |
| Phase1 O | 未归一化 `P×V` fp32 | 同左 |
| Phase2 scale | `ComputeScaleValue_VF`（`softmaxLseFlag` 来自 tiling） | `_compute_scale_weights` |
| Phase2 O/rowSum | `CopyAccumOutIn` 从 `gmSoftmaxSum` 读 rowSum 再 `Divs` | `o_norm = o_partial / rowSum`（`rowSum<=0` 置 0） |
| Phase2 combine | `ReduceFinalRes_VF` | `sum(scale * o_norm)` |
| QK/SM 中间精度 | 由 `innerPrecise` 决定：`0` 为 QK fp32 S + Vector fp32 softmax + P cast bf16；`4` 为 bf16 S softmax + fp32 `O_partial`；`1` 为 bf16 S + bf16 `O_partial` | fp32 softmax；P cast bf16 供 PV（对齐 `innerPrecise=0`） |
| ws_max 初值 | 依赖 runtime zero-fill | 显式 `-inf` |

Golden 不模拟 Cube tile 搬运；数值路径与 §7.2 一致。

另保留 `cpu_golden_prefill_qcentric_bf16` 作 Q-centric 交叉校验。

## 9. Kernel 实现边界

目标 kernel 不应停留在以下标量参考形态：

1. 不应在 GM 上对每个元素重复读取 K/V 做点积。
2. 不应对每个 `(qToken, qHead, d, n)` 用标量 C++ 循环模拟 QK 和 PV。
3. 不应在 Phase 1 对同一 `(packedRow, kvHeadIdx)` task 的 K/V 反复从 GM 读取。
4. Phase 2 必须能跳过无效 partial（`rowMax == -inf` 或 `rowSum <= 0`，或 causal 全 mask）。

当前 `minimax_sparse_attention_split_kv_kernel_arch35.h` 已具备：

- Phase1/2 按 `totalTaskNumP1/P2` stride 分核；Phase1 使用 `coreIdx = GetBlockIdx()`（VEC 侧 `/ GetSubBlockNum()`）；Phase2 在 VEC 侧 `coreIdx = GetBlockIdx()`（线性 AIV id）、`coreNum = GetBlockNum() * GetSubBlockNum()` 循环覆盖全部 P2 任务。
- Phase1 `BlockMmadQK` / `BlockMmadPV` + `LoadKResident` / `LoadVResident`；`PRE_LAUNCH=2` 流水。
- MSA packed row coord 增量步进；CSR 驱动 Q gather + `causalValidLen` + workspace 写出。
- Phase2 `block_epilogue.hpp`：IFA FlashDecode combine + **`CopyAccumOutIn` 从 GM 读 rowSum 做 O 归一化**；`ReduceFinalRes` 按 slot 首值 `rowSum[0]<=0` 跳过非法 split 的 `CopyAccumOutIn` / `ReduceFinalRes_VF`。
- Kernel 入口 `InitWorkspaceStats`：VEC 侧显式预置 `softmaxMax=-inf` / `softmaxSum=0`。

仍待完善：

1. Q 仍逐 token gather，尚未 `qTile > 1` 批量拼 Q。
2. ~~Kernel workspace 显式预置 `rowMax=-inf`；非法 slot 跳过~~ —— 已实现（`InitWorkspaceStats` + `ReduceFinalRes` 首值判断跳过 `CopyAccumOutIn`）。
3. `CopyLseIn` 在 `groupSize > gSplitSize` 分片时按 `startRow` 正确 gather max/sum。
4. ~~Phase2 `GetBlockIdx` 与 Phase1 VEC 核映射关系统一验证（AIC/AIV 混合 blockDim）~~ —— 已实现：`Phase2CombineScale` 改为 `subLinearIdx = GetBlockIdx()*GetSubBlockNum()+GetSubBlockIdx()`、`stride = GetBlockNum()*GetSubBlockNum()` 的循环，覆盖全部 `totalTaskNumP2` 任务，且区分两个 VEC 子核。

## 10. 后续 kernel 修改建议

建议按以下顺序继续演进：

1. **正确性加固**：~~workspace 预置；`CopyAccumOutIn` 除零保护~~ —— 已完成；剩余 `CopyLseIn` 大 `groupSize` 分片 gather。
2. Phase 1 扩展 `qTile > 1`：批量 gather Q，拼 `M = qTile * groupSize` 做 Cube QK/PV。
3. 性能：
- PRE_LAUNCH 流水（已启用，可继续调优）
- double buffer / L1/L0 preload
- score/softmax UB 复用
- K/V L1 load优化，一次性可load多块(L1总大小512K，一块K/V是128\*128\*2=32K)
- 空 CSR 行 task 负载均衡
- Regbase 优化
- combine 阶段跳过无效 slot（当前主要依赖 scale=0）
