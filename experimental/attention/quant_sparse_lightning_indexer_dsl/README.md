# Quant Sparse Lightning Indexer DSL

QSLI 在 candidate 指定的逻辑块上执行 gather、MXFP4 QK、BF16 权重归约、mask、TopK 和可选 LD 归并。本文以 Torch 接口 的实际签名为准。

## 必选数据输入与融合 K

T1 为所有 batch 查询 token 总数，P 为物理页数，PA 为页大小；N1=32。

| 参数 | dtype | 物理 shape / 说明 |
|---|---|---|
| q | uint8 | `[T1,32,64]`，打包 MXFP4 |
| k | uint8 | `[P,PA/8,544]`，融合 K 和 E8M0 scale |
| w | float32 | `[T1,32]` |
| descale_q | uint8 或 float8_e8m0fnu | `[T1,32,2,2]`，E8M0 |
| candidate_block_indices | int32 | `[T1,1,2048]`，逻辑 8-token block 索引 |
| candidate_block_length | int32 | `[T1,1]`，每行有效 candidate 前缀长度，范围 `[0,2048]` |

每个 candidate block 连续存储 `K[8,64]` 的 512B，随后是 `scale[8,4]` 的 32B，共 544B。`candidate_block_size` 为必选整数，当前为 8。`descale_k=None` 是可选预留参数，当前 PA_BBND 融合 K 路径必须为 None。

测试数据可在调用和计时之外打包，key_bytes 和 key_scale_bytes 均为 uint8 位表示：

```python
packed_k = torch.cat((
    key_bytes.reshape(pages, block_size // 8, 512),
    key_scale_bytes.reshape(pages, block_size // 8, 32),
), dim=-1)
```

接口不会额外调用打包或解包小算子。V0 搬入融合数据，分别以 ND 写入 K 和 scale staging，再由 Cube 读取并转换格式。PA 物理地址为 `physical_page * k.stride(0) + block_within_page * 544`。

## 完整接口签名

```python
def quant_sparse_lightning_indexer(
    q,
    k,
    w,
    descale_q,
    candidate_block_indices,
    candidate_block_length,
    topk,
    quant_mode,
    candidate_block_size,
    *,
    descale_k=None,
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
):
    ...

def quant_sparse_lightning_indexer_metadata(
    candidate_block_length,
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
    quant_mode,
    candidate_block_size,
    mask_mode=0,
    cmp_ratio=1,
    layout_q='TND',
    layout_k='TND',
):
    ...
```

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


QSLI metadata 额外必选 `candidate_block_length` Tensor，以及关键字属性 `quant_mode=1`、`candidate_block_size=8`；这些属性没有默认值。metadata 依据实际 candidate 长度生成调度边界，调用方不必把 candidate 数量换算为 max_seqlen_k。

## 返回值与有效索引

返回 `(indices, values)`：indices 为 int32 `[T1,1,topk]`；return_value=True 时 values 为 BF16 `[T1,1,topk]`，否则为空 Tensor `[0]`。输出是原始逻辑 token 索引，不是 candidate 数组内位置。

candidate_block_length 之外的后缀不参与计算；有效 token 不足 topk 时索引填 -1，全无效行输出全 -1。output_idx_offset 只作用于有效索引。TopK 不承诺同分索引稳定排序。

## 调用示例

以下示例假设已按上表准备同一 NPU 上的输入，B、S1、S2 为对应长度上界，cmp_residual_k 为当前压缩 causal 输入对应的残余长度：

```python
from cann_ops_transformer.ops.ds41 import (
    quant_sparse_lightning_indexer,
    quant_sparse_lightning_indexer_metadata,
)

metadata = quant_sparse_lightning_indexer_metadata(
    candidate_block_length,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q, seqused_k=seqused_k,
    cmp_residual_k=cmp_residual_k,
    batch_size=B, max_seqlen_q=S1, max_seqlen_k=S2,
    num_heads_q=32, num_heads_k=1, head_dim=128,
    topk=512, quant_mode=1, candidate_block_size=8,
    mask_mode=3, cmp_ratio=2, layout_q="TND", layout_k="PA_BBND",
)
indices, values = quant_sparse_lightning_indexer(
    q, packed_k, w, descale_q,
    candidate_block_indices, candidate_block_length, 512, 1, 8,
    descale_k=None, cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q, seqused_k=seqused_k,
    cmp_residual_k=cmp_residual_k, block_table=block_table,
    output_idx_offset=output_idx_offset, metadata=metadata,
    max_seqlen_q=S1, mask_mode=3, cmp_ratio=2,
    layout_q="TND", layout_k="PA_BBND", return_value=False,
)
```

## Batch 一致性

当 `torch_npu.npu._get_deterministic_level()` 返回 3 时，V0 按 candidate 原始顺序逐块调用 mem_copy，每块搬入 544B。普通路径在 DMA 地址和间距合法时使用 ascvec.copy_gm2ub 成对搬运，否则回退单行搬运，不交换候选顺序。该开关作为运行时参数传入，不新增公开 Torch 参数。

运行环境必须支持 deterministic level=3；测试不把其他 level 当作 level=3。更换 kernel ABI 后须重新导出匹配的 native 包，不能混用旧二进制。

## 编译与验证

Torch 接口分别调用 `ops.quant_lightning_indexer_dsl` / `ops.quant_sparse_lightning_indexer_dsl` 及对应 metadata 模块。使用 native 包时须保持源码、Torch 接口和导出产物版本一致。静态配置变化是否命中取决于包内导出的组合与缓存，不能仅凭第二次调用耗时推断；metadata 调用与 DSL 计算 kernel 编译是不同环节。

pytest 公共调用会通过真实 Torch 接口先生成 metadata，再传给计算接口。回归时记录实际加载的实现及其校验值。批跑用例由调用者提供，算子目录不附带用例表。
