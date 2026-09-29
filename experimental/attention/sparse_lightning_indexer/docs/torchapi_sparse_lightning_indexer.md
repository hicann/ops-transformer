# torch API: sparse_lightning_indexer

## 功能说明

`sparse_lightning_indexer`（candidate consumer）在 LightningIndexerV2 加权 ReLU 打分框架上消费
source 算子（`lightning_indexer_candidate`，即 aclnnLightningIndexerV2 candidate 模式）
输出的候选块索引，做候选外 leak 降级 TopK。仅支持 ascend910b/ascend910_93
（Atlas A2/A3 系列，arch22），无 ascend950 路径。


## 接口定义

```python
cann_ops_transformer.sparse_lightning_indexer(
    Tensor q, Tensor k, Tensor w, int topk, Tensor candidate_topk_indices, *,
    Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None,
    Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? block_table=None,
    Tensor? output_idx_offset=None, Tensor? metadata=None, int max_seqlen_q=-1,
    str layout_q="BSND", str layout_k="BSND",
    int mask_mode=0, int cmp_ratio=1, Tensor? candidate_block_length=None,
    int candidate_block_size=8) -> (Tensor, Tensor)
```

返回 `(sparse_indices, sparse_values)`：

- `sparse_indices`：INT32，BSND `[B,S1,N2,topk]` / TND `[T,N2,topk]`（leak 降级 topk）；
- `sparse_values`：恒空 tensor `(0,)`（return_value 固定 0，接口不暴露该参数）。

## 参数约束

| 参数 | 约束 |
|---|---|
| q / k / w | 同 `lightning_indexer`（BF16/FP16 同 dtype、FP32；D=128、N2=1；BSND/TND/PA_BBND；q/k 各维与 numel 必须 >0） |
| candidate_topk_indices | **必传** Tensor 且 INT32、numel>0；BSND `[B,S1,N2,candBlocks]` / TND `[query.dim0,N2,candBlocks]`；candBlocks = 末维 ∈ (0,2048] 且 64 的倍数（由输入 shape 推导，无属性）；值域 [0,numBlocks) 或 -1、槽位无序、允许重复/越界/-1 脏数据为跨算子契约（host/device 不校验，越界候选自然按非候选降级） |
| candidate_block_size | [2,64] 内的 2 的幂，默认 8；**与 source 侧必须一致** |
| candidate_block_length | 预留，仅 None 或 numel==0 的空 tensor（非空报错） |
| topk | (0, 2048] |
| return_value | 不暴露，固定 0（leak 降级语义下 Values 输出无意义） |
| metadata | 可选、**纯透传**（与 `lightning_indexer` 一致）：None 直接下发 aclnn，arch22 kernel 不消费 metadata；如需分核信息由调用方显式调用 `lightning_indexer_metadata` 生成后传入（需 CANN ≥ 9.2.0，更低版本 libopapi 缺少 `aclnnLightningIndexerV2Metadata` 符号） |
| seqused_q / output_idx_offset | 接口对齐保留，arch22 kernel 不消费 |
| 其余 | 同 `lightning_indexer`（layout 组合/PA block_table 等，详见 aclnn 文档 [aclnnSparseLightningIndexer.md](aclnnSparseLightningIndexer.md)） |

## 封装层断言（跨算子契约执行点）

Python 层（`_sparse_li_check_args`）：

- `candidate_topk_indices` 必为 Tensor 且 dtype 为 int32，末维 ∈ (0,2048] 且 64 的倍数；
- `topk` ∈ (0, 2048]；
- `candidate_block_size` 为 [2,64] 内的 2 的幂；
- `candidate_block_length` 为 None 或 numel==0（与 host 校验一致）。

C++ 层（csrc TORCH_CHECK）：`q`/`k`/`candidate_topk_indices` numel>0 且 shape 各维 >0，
`topk`>0，`candidate_topk_indices` 为 int32 且末维同规，`candidate_block_size`、
`candidate_block_length` 同上；最终由 host tiling 校验兜底。

## 图模式说明

本算子无 GE converter 注册（NPU-only 自定义算子）；`torch.compile` 场景经 PrivateUse1/fallback
路径以 eager kernel 执行（已 `torch.compiler.allow_in_graph`）。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_transformer_custom as cot

q = torch.randn(2, 8, 8, 128, dtype=torch.float16).npu()
k = torch.randn(2, 2048, 1, 128, dtype=torch.float16).npu()
w = torch.rand(2, 8, 8, dtype=torch.float32).npu()

# source（LIV2 candidate 模式，仅Atlas A2/A3支持开启）：生成候选块索引（同一 forward 内与
# consumer 配对；返回4元组，第3个输出 candidate_topk_indices 即 consumer 的候选输入）
sparse_idx, _, cand, _ = cot.lightning_indexer_candidate(
    q, k, w, 128, candidate_topk_blocks=2048, candidate_block_size=8,
    mask_mode=3, cmp_ratio=1)

# consumer：消费候选，输出 leak 降级 topk（metadata 缺省 None 纯透传，arch22 kernel 不消费）
out_idx, out_val = cot.sparse_lightning_indexer(
    q, k, w, 128, cand,
    candidate_block_size=8, mask_mode=3, cmp_ratio=1)
```

## 输出语义

- 候选外可达位置：分数降级为 NEG_HUGE（-1e30f，位型 0xF149F2CA）参与排序（不取消入选资格）；
- 不可达位置：输出 -1；
- regime 2（候选内有效数 < topk < 可达数）：候选外可达位置以降级分数填充入选；
- 全候选（覆盖全部块号）输入下与 `lightning_indexer` 结果逐行一致。

> **注意：** 输出槽位顺序（含候选外填充次序）不作对外承诺，板上仅保证有效槽集合一致与 -1 槽数正确。
