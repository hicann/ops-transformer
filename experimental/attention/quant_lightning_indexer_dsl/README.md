# Quant Lightning Indexer DSL

QLI 对 MXFP4 Q/K 计算带权归约分数、mask 和 TopK，可同时生成 candidate block 索引。本文以 Torch 接口 的实际签名为准。

## 必选数据输入

T1 为所有 batch 查询 token 总数，P 为物理页数，PA 为页大小，N1 支持 32 或 64。

| 参数 | dtype | 物理 shape / 说明 |
|---|---|---|
| q | uint8 | `[T1,N1,64]`，打包 MXFP4 |
| k | uint8 | `[P,PA,1,64]`，打包 MXFP4 |
| w | float32 | `[T1,N1]` |
| descale_q | uint8 | `[T1,N1,2,2]`，E8M0 位表示 |
| descale_k | uint8 | `[P,PA,1,2,2]`，E8M0 位表示；必选 |

K 和 K scale 分开传入；QLI 的 K 由 Cube 读取，VEC1 使用 BF16 计算。不要使用旧名 q_descale/k_descale。

## 完整接口签名

```python
def quant_lightning_indexer(
    q,
    k,
    w,
    descale_q,
    descale_k,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q='TND',
    layout_k='TND',
    return_value=False,
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    ...

def quant_lightning_indexer_metadata(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q='TND',
    layout_k='TND',
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    ...
```

`candidate_topk_blocks=-1, candidate_block_size=-1` 关闭 candidate；开启时分别传正整数容量和 8，例如 2048、8。metadata 与计算调用使用相同配置。

## 当前支持范围

当前实现只支持 `layout_q="TND"`、`layout_k="PA_BBND"`，计算接口与 metadata 均使用参数名 `layout_k`。虽然 schema 中 K 布局默认值为 `"TND"`，当前调用必须显式指定 `"PA_BBND"`；接受 `cu_seqlens_k` 不代表已经实现 K=TND。

逻辑 head_dim=128，KV head 数为 1；MXFP4 每两个元素打包为一个 uint8，所以 Q/K 的数据维度为 64 字节。PA 页大小 `block_size` 从 K 的 shape 推导，支持 16 的倍数且不超过 1024，和 candidate 的 8-token block 是两个不同概念。没有固定 128K 的 token 上限。

所有实际计算输入 Tensor 必须位于同一 NPU。序列信息及索引使用 int32；除 PA 存储允许第 0 轴 padding 外，应使用连续 Tensor。

## 公共可选参数

| 参数 | 默认值 | 含义及当前约束 |
|---|---|---|
| cu_seqlens_q | None | int32 `[B+1]`，TND 查询的累计长度；B>1 时必需 |
| cu_seqlens_k | None | 接口预留；当前 PA_BBND 路径不参与寻址或长度计算 |
| seqused_q | None | int32 `[B]`，各 batch 有效查询长度 |
| seqused_k | None | int32 `[B]`，各 batch 有效 KV 长度；省略时按当前 PA 容量处理 |
| cmp_residual_k | None | int32 `[B]`，压缩 causal 边界的残余长度；mask_mode=3 且 cmp_ratio!=1 时必需，其他组合传 None |
| block_table | None | int32 `[B, max_pages]`，逻辑页到物理页的映射；B>1 时必需，省略只支持单 batch 恒等映射 |
| output_idx_offset | None | int32 `[T1,1]`，仅对有效输出 token 索引加偏移，-1 保持不变 |
| metadata | None | int32 `[1024]`，必须由调用方先调用对应 metadata 接口生成；省略或传 None 直接报错 |
| max_seqlen_q | -1 | 查询长度上界；-1 使用推导路径，建议显式传入 |
| mask_mode | 0 | 支持 0、3；3 为 causal 场景 |
| cmp_ratio | 1 | 压缩比例，支持 `[1,128]` |
| layout_q / layout_k | "TND" / "TND" | 当前必须传 "TND" / "PA_BBND" |
| return_value | False | 是否输出 TopK 分数；性能测试使用 False |

`topk` 为必选整数，范围 `[1,8192]`；`quant_mode` 为必选整数，当前为 1（MXFP4）。可选参数表示 schema 允许省略，不表示所有参数组合都可省略。

## Metadata 参数与返回值

metadata 不接收 Q/K 数据、block_table 或 PA block_size。`num_heads_q`、`num_heads_k`、`head_dim`、`topk` 是必选的关键字参数。序列参数沿用上表语义，另外有以下属性：

| 参数 | 默认值 | 说明 |
|---|---|---|
| batch_size | None | 可由 cu_seqlens_q 或 seqused_q 的 shape 推导，否则默认单 batch；显式值必须与长度 Tensor 一致 |
| max_seqlen_q | -1 | 显式查询长度上界；省略时从查询序列信息推导，可能读取设备数据到 host |
| max_seqlen_k | -1 | 原始 KV 长度上界，非 candidate token 数；省略时从 seqused_k 推导，可能读取设备数据到 host |
| layout_k | "TND" | 当前必须显式传 "PA_BBND"，与计算接口一致 |

metadata 返回 NPU 上固定 shape `[1024]` 的 int32 Tensor，由 AICPU 生成调度信息。计算 kernel 消费其中的分核边界，支持 LD 和非 LD，不需要调用者自行划分核。复用要求序列长度、压缩/掩码参数和调度配置不变；QSLI 还要求 candidate_block_length 不变。metadata 与计算算子的对应参数必须一致。计算接口保留 metadata=None 的签名默认值以兼容参数顺序，但DSL 运行时入口拒绝 None，不再内部生成 metadata。

Torch 接口包含 NPU（PrivateUse1）实现、metadata fallback 注册和 register_fake。fake 仅描述输出 shape/dtype；fallback 不表示支持 CPU 执行 NPU 算子。显式生成 metadata 可避免把生成开销混入计算算子的计时。


## 返回值

始终返回四个 Tensor：

| 返回值 | dtype | shape |
|---|---|---|
| indices | int32 | `[T1,1,topk]` |
| values | bfloat16 | return_value=True 时为 `[T1,1,topk]`，否则为 `[0]` |
| candidate_indices | int32 | 开启时为 `[T1,1,candidate_topk_blocks]`，否则为 `[0]` |
| candidate_length | int32 | 开启时为 `[T1,1]`，否则为 `[0]` |

无效 token 索引用 -1 填充；有效 token 不足 topk 时不产生越界索引。candidate 输出为逻辑 8-token block 索引，有效数量由 candidate_length 给出；不足容量的后缀为 -1。TopK 不承诺同分索引稳定排序。

## 调用示例

以下示例假设已按上述 shape 准备同一 NPU 上的输入，B、S1、S2 为对应长度上界，压缩 residual 由调用方按输入语义提供。开启 candidate，同时关闭分数返回：

```python
from cann_ops_transformer.ops.ds41 import (
    quant_lightning_indexer,
    quant_lightning_indexer_metadata,
)

metadata = quant_lightning_indexer_metadata(
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    cmp_residual_k=cmp_residual_k,
    batch_size=B, max_seqlen_q=S1, max_seqlen_k=S2,
    num_heads_q=q.shape[1], num_heads_k=1, head_dim=128, topk=512,
    mask_mode=3, cmp_ratio=2, layout_q="TND", layout_k="PA_BBND",
    candidate_topk_blocks=2048, candidate_block_size=8,
)
indices, values, candidate_indices, candidate_length = quant_lightning_indexer(
    q, k, w, descale_q, descale_k, 512, 1,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q, seqused_k=seqused_k,
    cmp_residual_k=cmp_residual_k, block_table=block_table,
    output_idx_offset=output_idx_offset, metadata=metadata,
    max_seqlen_q=S1, mask_mode=3, cmp_ratio=2,
    layout_q="TND", layout_k="PA_BBND", return_value=False,
    candidate_topk_blocks=2048, candidate_block_size=8,
)
```

关闭 candidate 时，两处调用均省略 candidate 参数或都传 `-1, -1`，仍接收四个返回值。不能把 QLI 的独立 K/scale 直接当作 QSLI 的融合 K。

## 编译与验证

Torch 接口分别调用 `ops.quant_lightning_indexer_dsl` / `ops.quant_sparse_lightning_indexer_dsl` 及对应 metadata 模块。使用 native 包时须保持源码、Torch 接口和导出产物版本一致。静态配置变化是否命中取决于包内导出的组合与缓存，不能仅凭第二次调用耗时推断；metadata 调用与 DSL 计算 kernel 编译是不同环节。

pytest 公共调用会通过真实 Torch 接口先生成 metadata，再传给计算接口。回归时记录实际加载的实现及其校验值。批跑用例由调用者提供，算子目录不附带用例表。
