# MixedQuantSparseFlashMla

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

MixedQuantSparseFlashMla（MQSMLA）用于混合量化稀疏 MLA 注意力计算。Q 为 BF16；窗口 KV 使用 per-token-group FP8 E4M3，压缩 KV 使用 per-token-group FP4 E2M1，窗口侧 group size 为 32、压缩侧为 16、scale 为 BF16。算子按稀疏逻辑索引和 block table 从分页 KV 中收集数据，反量化后计算注意力，Key 和 Value 共享同一份 latent 数据。

- `ORI_SPARSE`：仅使用 ori 侧。
- `ORI_CMP_SPARSE`：同时使用 ori 与 cmp 侧，两侧 KV 一起参与 softmax。

设 query 行 t、头 h 对应的有效 KV 集合为 I(t)，反量化后为 k_j=v_j，缩放系数为 s，sink 为 a_h（所有 batch/query 共用），数学语义为：

$$
z_{t,h,j}=s\langle q_{t,h},k_j\rangle,\quad
Z_{t,h}=\exp(a_h)+\sum_{j\in I(t)}\exp(z_{t,h,j})
$$

$$
out_{t,h}=\frac{\sum_{j\in I(t)}\exp(z_{t,h,j})v_j}{Z_{t,h}},\qquad
lse_{t,h}=\log Z_{t,h}.
$$

sink 只进入分母，没有对应的 Value 项。显式传入零 sink 时分母仍包含 exp(0)=1。实际实现使用分块在线 softmax、BF16 概率中间结果和 FP32 累加。

## 编译复用

DSL 入口使用 `Dim + TensorSpec` 首次编译，并在当前进程内用带锁缓存复用编译产物。相同静态配置重复调用直接命中缓存；query 行数 T1、KV 物理页数和 workspace 字节长度为动态维度，改变这些维度也可复用。

batch、K、block size、KV 页距、block table 容量、softmax scale、设备/核数、可选 Tensor 组合以及 sinks/LSE 形态仍属于静态配置，改变后首次调用会编译新的产物。缓存不跨进程保证复用；编译失败不会入缓存。所有接口张量仍原样传入，不做 torch 布局转换。

## 参数说明

### Attention 接口

仅 q 为位置参数；其余参数均为 keyword-only。签名中可选 Tensor 默认 None，当前支持模式的必填要求见“约束说明”。

```python
mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
) -> tuple[Tensor, Tensor]
```


### 维度说明

| 命名 | 含义 |
| :--- | :--- |
| `B` | batch大小 |
| `T1` | 所有batch中query有效token数之和 |
| `N1` | Q头数 |
| `N2` | KV头数，当前固定为1 |
| `D` | Q和反量化后KV的逻辑head dim，当前固定为512 |
| `G` | 每个KV头对应的Q头数，`G=N1/N2` |
| `K1` | `ori_sparse_indices`最后一维的容量 |
| `K2` | `cmp_sparse_indices`最后一维的容量 |
| `blocknum1` | `ori_kv`物理block数量 |
| `blocknum2` | `cmp_kv`物理block数量 |
| `blocksize1` | `ori_kv`每个物理block包含的token数 |
| `blocksize2` | `cmp_kv`每个物理block包含的token数 |
| `max_num_blocks_per_seq1` | `ori_block_table`第二维容量，即单batch最大逻辑block数 |
| `max_num_blocks_per_seq2` | `cmp_block_table`第二维容量，即单batch最大逻辑block数 |

### 输入和属性

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `q` | Tensor | 必选 | Query输入。 | `bfloat16` | ND | `(T1, N1, 512)` |
| `ori_kv` | Tensor | 可选 | 窗口KV Cache，Key和Value共享同一份MLA latent数据。按token进行group size为32的FP8 E4M3量化，以`uint8`字节视图传入；每个token占544字节。 | `uint8` | ND | `(blocknum1, blocksize1, N2, 544)` |
| `cmp_kv` | Tensor | 可选 | 压缩KV Cache。`ORI_CMP_SPARSE`模式必选，`ORI_SPARSE`模式不传。按token进行group size为16的FP4 E2M1量化，以两个FP4元素打包为一个字节的`uint8`视图传入；每个token占320字节。 | `uint8` | ND | `(blocknum2, blocksize2, N2, 320)` |
| `ori_sparse_indices` | Tensor | 可选 | 窗口KV的逻辑token索引。传入length时每行前`ori_topk_length`个元素有效、其余填`-1`；不传length时全部K1列有效。 | `int32` | ND | `(T1, N2, K1)` |
| `cmp_sparse_indices` | Tensor | 可选 | 压缩KV的逻辑token索引。`ORI_CMP_SPARSE`模式必选，`ORI_SPARSE`模式不传。传入length时每行前`cmp_topk_length`个元素有效、其余填`-1`；不传length时全部K2列有效。 | `int32` | ND | `(T1, N2, K2)` |
| `ori_block_table` | Tensor | 可选 | `ori_kv`的逻辑block到物理block映射表。 | `int32` | ND | `(B, max_num_blocks_per_seq1)` |
| `cmp_block_table` | Tensor | 可选 | `cmp_kv`的逻辑block到物理block映射表。`ORI_CMP_SPARSE`模式必选，`ORI_SPARSE`模式不传。 | `int32` | ND | `(B, max_num_blocks_per_seq2)` |
| `cu_seqlens_q` | Tensor | 可选 | 各batch的query累积序列长度。首元素为0，末元素为`T1`。算子不生成缺省值，缺传时拦截报错。 | `int32` | ND | `(B+1,)` |
| `seqused_q` | Tensor | 可选 | 各batch中实际使用的query token数，仅作交叉校验：若传入，其值必须等于对应batch的query跨度（当前不支持query行padding）。不能替代`cu_seqlens_q`（不能由它推导累积序列长度）。 | `int32` | ND | `(B,)` |
| `seqused_ori_kv` | Tensor | 可选 | 预留参数。上游语义为各batch中实际使用的窗口KV token数；当前实现仅校验其为`int32`且形状为`(B,)`，不解析、不消费其内容，传入与否不影响计算结果。 | `int32` | ND | `(B,)` |
| `seqused_cmp_kv` | Tensor | 可选 | 预留参数。上游语义为各batch中实际使用的压缩KV token数；当前实现仅校验其为`int32`且形状为`(B,)`，不解析、不消费其内容，传入与否不影响计算结果。 | `int32` | ND | `(B,)` |
| `ori_topk_length` | Tensor | 可选 | 默认None，使用全部K1列。 每个query实际使用的窗口稀疏索引数，是`ori_sparse_indices`有效前缀的准确长度，取值范围为`[0, K1]`。长度为0时跳过该侧KV；两侧均为0时输出为0，开启LSE时其值为sinks；可不传（None）。 | `int32` | ND | `(T1, N2)` |
| `cmp_topk_length` | Tensor | 可选 | 默认None，使用全部K2列。 每个query实际使用的压缩稀疏索引数，是`cmp_sparse_indices`有效前缀的准确长度，取值范围为`[0, K2]`。仅在`ORI_CMP_SPARSE`模式可传。 当前不支持空query；可不传（None）。 | `int32` | ND | `(T1, N2)` |
| `sinks` | Tensor | 可选 | 各Q头的可学习attention sink，只进softmax分母。算子不生成缺省值，缺传时拦截报错；不需要sink时显式传入全0张量。 | `float32` | ND | `(N1,)`，所有 batch/query 共用 |
| `metadata` | Tensor | 可选 | 分核任务表，驱动多核的query行划分。前blocks项为各核分到的首个query行序号（m-tile起点），后blocks项为各核分到的query行数；所有核的区间合起来必须恰好无缝无叠地覆盖`[0, T1)`。算子不生成缺省值，缺传时拦截报错；须先调用`mixed_quant_sparse_flash_mla_metadata(ori_topk_length, cmp_topk_length, cu_seqlens_q=..., num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1, has_cmp_kv=...)`生成（两阶段调用）。 | `int32` | ND | `(2 * blocks,)` |
| `quant_mode` | int | 必选 | 量化模式，必须显式传入`1`，无默认值。Q为BF16；`ori_kv`为per-token-group FP8 E4M3、group size为32、scale为BF16；`cmp_kv`为per-token-group FP4 E2M1、group size为16、scale为BF16。 | `int32` | - | - |
| `softmax_scale` | float | 可选 | Softmax前QK乘积的缩放系数。默认值为`None`，表示使用`1/sqrt(512)`。 | `float32` | - | - |
| `layout_q` | string | 可选 | Q布局，当前仅支持`"TND"`，默认值为`"TND"`。 | string | - | - |
| `layout_kv` | string | 可选 | KV布局，当前仅支持`"PA_BBND"`，默认值为`"PA_BBND"`。 | string | - | - |
| `return_softmax_lse` | bool | 可选 | 是否计算Softmax的log-sum-exp结果，默认值为`False`；始终返回`softmax_lse` Tensor，关闭时为空Tensor。 | bool | - | - |


### 输出

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 / 形状 |
| :--- | :--- | :--- | :--- | :--- |
| attn_out | 输出 | 注意力结果，与 q 同设备 | BFLOAT16 | ND / `(T1,64,512)` |
| softmax_lse | 输出 | 开启 LSE 时有效；关闭时仍返回空 Tensor | FLOAT32 | ND / `(1,T1,64)` 或 `(0,)` |

### Metadata 接口

两阶段调用时先生成 metadata，再将其原样传给 attention。两个 topk_length 为必选位置 Tensor，其余参数 keyword-only，顺序如下：

```python
mixed_quant_sparse_flash_mla_metadata(
    ori_topk_length,
    cmp_topk_length,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=True,
) -> Tensor
```

| 参数 | 类型 / 形状 | 默认值 | 当前用途 |
| :--- | :--- | :--- | :--- |
| `ori_topk_length` | int32 Tensor `(T1,N2)` | 必传 | 校验形状；长度值暂不影响分核 |
| `cmp_topk_length` | int32 Tensor `(T1,N2)` | 必传 | 校验形状；长度值暂不影响分核 |
| `cu_seqlens_q` | int32 Tensor `(B+1,)` | None | 优先以最后一项确定T1 |
| `seqused_q` | int32 Tensor `(B,)` | None | 无cu_seqlens_q时以总和确定T1 |
| `seqused_ori_kv` | int32 Tensor `(B,)` | None | 预留 |
| `seqused_cmp_kv` | int32 Tensor `(B,)` | None | 预留 |
| `batch_size` | int | None | 无query长度Tensor时与max_seqlen_q相乘 |
| `max_seqlen_q` | int | None | 同上，仅适用于各batch均为该长度 |
| `max_seqlen_ori_kv` | int | None | 预留 |
| `max_seqlen_cmp_kv` | int | None | 预留 |
| `num_heads_q` | int | 必传 | 当前校验为64 |
| `num_heads_kv` | int | 必传 | 当前校验为1 |
| `head_dim` | int | 必传 | 当前校验为512 |
| `quant_mode` | int | 必传 | 当前校验为1 |
| `layout_q` | str | TND | 当前仅支持TND |
| `layout_kv` | str | PA_BBND | 当前仅支持PA_BBND |
| `has_ori_kv` | bool | True | 当前必须为True |
| `has_cmp_kv` | bool | True | 控制是否计入cmp侧固定负载 |
| 返回 `metadata` | int32 Tensor `(x,)` | — | x=2*blocks；前半段起点、后半段数量 |

上述取值检查与分核逻辑属于DSL单文件实现；torch_extension直接转发给 net wheel 中的 `ops.mixed_quant_sparse_flash_mla` 实现，得到真实分核结果。DSL在两个topk_length所在NPU生成metadata；两者必须同形状、同设备。query长度信息若传入，派生T1必须与topk_length首维一致；未传时直接使用其首维。其他预留字段当前不解析、不校验内容，不参与分核。

`blocks`为设备Cube核数，通过`torch.npu.get_device_properties(...).cube_core_num`获取；metadata生成使用topk_length所在NPU，tiling使用`q.device`，调用方须使它们位于同一NPU。workspace容量和kernel发射核数均使用tiling查询结果。

query token总数依次由`cu_seqlens_q[-1]`、`seqused_q.sum()`或`batch_size * max_seqlen_q`派生；均未提供时使用`ori_topk_length.shape[0]`。`has_cmp_kv`默认True；`ORI_SPARSE`必须显式传False。该函数按均摊负载模型（每个query的负载视为恒定：`ori_kv` 128个token，`ORI_CMP_SPARSE`模式下再加`cmp_kv` 512个token）用贪心均衡算法切分，返回int32 `(2 * blocks,)`分核任务表（等权重下即近似均分）；返回值**直接落在NPU设备上**，可原样传给算子入口。


### KV 量化与分页布局

每个 token 数据的物理和逻辑顺序均为 `nope[448] + rope[64]`，其后紧接 scale，ori 每连续 32 个元素共享一个 BF16 scale，共 16 个 scale；cmp 每连续 16 个元素共享一个 BF16 scale，共 32 个 scale。输入均为 uint8 字节视图，token 内没有额外 padding。

| 输入 | 数据字节范围 | scale 字节范围 | 每 token 字节数 |
| :--- | :--- | :--- | :--- |
| ori_kv | nope `[0,448)`，rope `[448,512)`；FP8 E4M3 | `[512,544)`，16 个 BF16 scale | 544 |
| cmp_kv | nope `[0,224)`，rope `[224,256)`；FP4 E2M1 | `[256,320)`，32 个 BF16 scale | 320 |

FP4 偶数位置元素存于低 4 bit，奇数位置元素存于高 4 bit。第 i 个 scale 覆盖物理顺序中的元素 `[g*i,g*(i+1))`，ori 的 g=32，cmp 的 g=16。

ori/cmp 使用各自的 block size、block table 和逻辑索引空间。batch b 内逻辑 token p 的地址为：

```text
physical_page = block_table[b, p // block_size]
byte_offset = physical_page * kv.stride(0) + (p % block_size) * row_bytes
```

当前 N2=1。KV 仅第 0 轴允许非连续，stride 必须为 `(page_stride,N2*row_bytes,row_bytes,1)`，且 `page_stride >= block_size*N2*row_bytes`。ori/cmp 页距可不同，不要求是 row_bytes 的整数倍；页间 padding 不参与计算。其他输入必须连续且与 q 位于同一 NPU，运行入口不整理布局或搬运输入。

## 约束说明

- 当前仅支持`layout_q="TND"`和`layout_kv="PA_BBND"`。
- 当前仅支持`quant_mode=1`，不再支持旧版608字节/token的KV编码。
- `D`固定为512，`N2`固定为1；`N1`固定为64（不再支持N1=128的SPLIT_G形态及其他N1取值）。
- `ORI_SPARSE`和`ORI_CMP_SPARSE`均不接收mask、window或压缩率参数。`seqused_ori_kv`和`seqused_cmp_kv`为预留参数，当前实现不解析其内容；KV的有效计算长度只由对应的稀疏索引和`topk_length`指定。
- 对每个query，`ori_sparse_indices`的前`ori_topk_length`项必须是合法的窗口KV逻辑token索引，其余项必须填`-1`。
- 对每个query，`cmp_sparse_indices`的前`cmp_topk_length`项必须是合法的压缩KV逻辑token索引，其余项必须填`-1`。
- 两侧`topk_length`独立可选；不传时对应全部K列参与计算（不得包含`-1`）。传入时必须为准确前缀长度且不得超过K。长度为0时跳过该侧KV；两侧均为0（无cmp时ori为0）的query不发射KV tile，输出写0，开启LSE时写对应head的sinks。
- `cu_seqlens_q`必须显式传入：首元素必须为0，整体非递减，末元素必须为`T1`；算子不生成缺省值，也不能由`seqused_q`推导。
- `ori_block_table`和`cmp_block_table`中的有效物理block编号不得越界；每个有效稀疏索引映射到的逻辑block必须位于对应block table的有效范围内。
- `sinks`必须显式传入（不需要sink时传全0张量）；`metadata`必须显式传入（用`mixed_quant_sparse_flash_mla_metadata`生成，两阶段调用）。`metadata`为分核任务表：`int32`且形状为`(2 * blocks,)`；前blocks项是各核的query行起点、后blocks项是各核的query行数量，各核区间必须恰好覆盖`[0, T1)`（无缝、无重叠）。
- q、ori_kv、cmp_kv、稀疏索引及block table等输入不支持空Tensor（各维度必须大于0）。
- 各参数shape中使用相同符号的维度必须保持一致。


- `sinks` 仅支持 `(64,)`，所有 batch/query 共用。
- metadata 接口的两个 topk_length 始终必传，即使 has_cmp_kv=False 也需传同形状全零 cmp 长度。attention 省略长度时，调用方须为 metadata 单独准备全 K 长度；不使用的 cmp 侧准备全零长度。
- torch 层只分配输出并调用 DSL，不做输入拦截；输入校验由 DSL host 完成。

## 调用说明

依赖已配置的 CANN、PyTorch、torch_npu 和 CANNBotDSL 环境。通过仓库 torch_extension 构建流程安装 cann_ops_transformer 后使用：

| 调用方式 | 入口 | 说明 |
| :--- | :--- | :--- |
| PyTorch API | `cann_ops_transformer.ops.ds41` | 分配输出，调用 DSL，返回 `(attn_out,softmax_lse)` |
| torch.ops | `torch.ops.cann_ops_transformer.ds41` | 导入扩展完成注册后调用同名接口 |
| DSL 直调 | net wheel 的 `ops.mixed_quant_sparse_flash_mla` | 同序参数末尾增加必传 keyword `out`、`lse`；写入缓冲区，返回 None |

DSL 文件更名后，需同时重新构建并安装 net native wheel 和本仓 PyTorch wheel。
版本号不变时使用 `python -m pip install --no-deps --force-reinstall` 安装两个新包，
确保 metadata 和 attention 均通过 `ops.mixed_quant_sparse_flash_mla` 导入。

`ds41` 为与 compressor_v2 共用的实际模块名。MQSMLA 的 torch 调用为 `torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla(...)`，metadata 同层；底层 schema 使用 `ds41.mixed_quant_sparse_flash_mla` 与 `ds41.mixed_quant_sparse_flash_mla_metadata`，与主线注册名互不冲突。下面的函数接收已按上述格式打包并置于 NPU 的输入，可用于 ori-only 或 ori+cmp；两侧 attention 长度均省略，使用全部 K 列：

```python
import torch
import torch_npu
from cann_ops_transformer.ops.ds41 import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)


def attention(q, ori_kv, ori_indices, ori_table, cu_seqlens_q, sinks,
              cmp_kv=None, cmp_indices=None, cmp_table=None):
    t1 = q.shape[0]
    has_cmp = cmp_kv is not None
    ori_lengths = torch.full((t1, 1), ori_indices.shape[-1],
                             dtype=torch.int32, device=q.device)
    cmp_lengths = torch.full((t1, 1), cmp_indices.shape[-1] if has_cmp else 0,
                             dtype=torch.int32, device=q.device)
    metadata = mixed_quant_sparse_flash_mla_metadata(
        ori_lengths, cmp_lengths,
        cu_seqlens_q=cu_seqlens_q,
        num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1,
        has_ori_kv=True, has_cmp_kv=has_cmp,
    )
    return mixed_quant_sparse_flash_mla(
        q,
        ori_kv=ori_kv, cmp_kv=cmp_kv,
        ori_sparse_indices=ori_indices, cmp_sparse_indices=cmp_indices,
        ori_block_table=ori_table, cmp_block_table=cmp_table,
        cu_seqlens_q=cu_seqlens_q,
        sinks=sinks, metadata=metadata,
        quant_mode=1, return_softmax_lse=True,
    )
```

DSL 直调时使用相同输入，额外传入连续 BF16 `out=torch.empty_like(q)` 与 FP32 `lse=torch.empty((1,T1,64),dtype=torch.float32,device=q.device)`；关闭 LSE 时 lse 形状为 `(0,)`。调用后直接使用缓冲区，不能解包 DSL 返回值。

Meta/Fake 仅推导输出，不执行 DSL；当前未提供 GE 图转换实现。

## 测试说明

测试资产全部位于 [tests](tests/)，不依赖旧项目的 mqsmla 目录或文档。从 ops-transformer 根目录执行：

```bash
bash build.sh --torch_extension
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
python3 -m pytest -q -s experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/test_mqsmla.py -ra
```

[test_case_paramset.py](tests/test_case_paramset.py) 中的 `TEST_PARAMS` 定义命名参数组，`ENABLED_PARAMS` 选择运行组，各参数取值做笛卡尔积展开。所有用例统一通过已安装 wheel 的 `torch.ops` 调用 metadata 和 attention，不再按文件路径加载 DSL 源码；修改算子后须重新执行上述构建与安装命令。默认包含 decode 用例及 `torch_extension` 接口回归组，覆盖两模式与 LSE 开关。CPU golden 与输入生成使用测试侧独立常量，不导入 kernel。

CPU golden 独立执行反量化、稀疏 gather 与注意力计算，使用测试侧契约常量。`kv_axis0_noncontiguous` 控制 KV 页间非连续输入，统一作用于所有活动KV侧，框架固定页间padding为64字节，storage起点为0；活动KV侧默认显式传入topk_length，可用omit_ori_topk_length / omit_cmp_topk_length省略对应attention参数，手动指定长度优先，未指定的侧才按`fullK` / `random`生成全K长度/随机有效前缀。新增用例和形状组合可修改参数集或 Excel 表格。


### Excel 批量用例

Excel 模式复用同一个 `tests/test_mqsmla.py` 入口，每行运行一个独立用例并显示 `Testcase_Name`，继续进行 CPU golden 与已安装 wheel 的 `torch.ops` 精度比较。指定 Excel 后只运行该表格，不追加 Python 参数集；未指定时仍运行 `ENABLED_PARAMS`。

安装读取依赖 `python3 -m pip install openpyxl`，从仓库根目录执行：

```bash
python3 -m pytest -q -s experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/test_mqsmla.py \
  --excel experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/excel/example.xlsx --sheet decode
# 与主线批量用例读取方式一致的环境变量；命令行参数优先。
MQSMLA_EXCEL=/absolute/path/my_cases.xlsx MQSMLA_SHEET=decode \
  python3 -m pytest -q -s experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/test_mqsmla.py
# 先检查收集结果；也可用 -k excel_ori_cmp 选择单个用例。
python3 -m pytest --collect-only -q experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/test_mqsmla.py \
  --excel experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/excel/example.xlsx --sheet prefill
```

- 支持 `.xlsx`，默认 sheet 为 `decode`，第一行为列名；必需列和值为 `Testcase_Name`（唯一非空文本）和 `template_run_mode`（`ORI_SPARSE` / `ORI_CMP_SPARSE`）。空行忽略，空单元格、文本 `None` / `null` 使用测试默认值。
- 可添加 `备注` 列描述泛化目标；读取器忽略备注内容，不传入测试参数。精简白盒表位于 `tests/excel/mqsmla_whitebox_cases.xlsx`，使用默认 `decode` sheet，当前启用的典型参数集场景位于表格前部，其后为B、S2、block_size等形状参数代表值。
- 形状列：`B`、`S1`、`S2`、`S2C`、`K1`、`K2`、`block_size1`、`block_size2`。`S1/S2/S2C` 可填整数或 `[1, 3]` 这样的 B 元素列表，列表表示每个 batch 的长度，不做笛卡尔积。
- `cu_seqlens_q` / `seqused_q`：可填一维整数列表，分别为 B+1 个累积偏移（从0开始）和 B 个实际query长度。例如 `B=3`、`cu_seqlens_q=[0,1,4,6]`、`seqused_q=[1,3,2]` 生成 `q.shape=(6,64,512)`。可只传其中一项；只传 `seqused_q` 时测试框架生成 `cu_seqlens_q`。两者都传必须满足相邻差分等于 `seqused_q`，每个batch长度须为正，不支持padding。任一字段传入时忽略 `S1`；均未传时才使用 `S1`（标量表示每个batch等长，也兼容B元素列表）。参数集中写成 `"cu_seqlens_q": [[0,1,4,6]]` / `"seqused_q": [[1,3,2]]`，也接受整数Tensor候选值；保存/回放保留这些输入。
- `ori_topk_length` / `cmp_topk_length`：手动指定的有效前缀长度，优先于对应fullK/random模式。可填长度为T的列表`[17,32,64]` / `[[17],[32],[64]]`（按TND顺序，每个query一个长度，T1由query长度字段决定）。每项必须在`[0,K1]` / `[0,K2]`内；K1/K2仍是indices容量，不会随有效长度改变。活动侧长度同时为0的空query不支持。两侧独立处理，只有空单元格/None/null才使用该侧fullK/random生成默认长度。
- 参数集仍按候选列表展开：`"ori_topk_length": [[17,32,64]]`表示一组T1=3的逐query长度；`[None]`表示按模式生成。当前decode参数按B分别配置T个长度；大prefill用例仍可省略长度字段并使用fullK。白盒表仅保留S1=1的decode用例。
- 输入控制列：`ori_kv_topk_mode` / `cmp_kv_topk_mode`（`fullK` / `random`）、`seed`、`dist`（仅支持 `norm`，默认值也是 `norm`）、`softmax_scale`（留空使用默认缩放）。
- 布尔列：`return_softmax_lse`、`kv_axis0_noncontiguous`，接受 Excel 布尔值、大小写 TRUE/FALSE 或 1/0。第0轴非连续仅通过 `kv_axis0_noncontiguous` 开关控制，不配置两侧独立开关、padding大小或storage offset。
- 兼容列：主线 `K` 映射为压缩侧 `K2`；`cmp_ratio` 可按 `S2 // cmp_ratio` 推导 `S2C`，显式给出冲突值时报错。可填固定契约列 `layout_q=TND`、`layout_kv=PA_BBND`、`N1=64`、`N2=1`、`D=512`、`rope_head_dim=64`、`quant_mode=1`。
- 读取方式参考主线，字段语义以此 DSL 支持子集为准，不直接执行主线所有模板。非空未知列、未支持的模式/布局、非法值、重复名称和公式单元格会在收集时明确报错，包含文件、sheet 和行号；不会静默跳过整表或忽略非空参数。请粘贴公式计算后的数值。

所有结果仍由 pytest 汇总，可用 `--junitxml=/absolute/path/results.xml` 保存报告。

### test_run.sh 与输入/golden保存回放

`tests/test_run.sh` 参考主线封装以下命令，复用 `tests/test_mqsmla.py`。请先加载上文CANN/Python环境；执行NPU测试前构建安装wheel。`PYTHON`可指定Python可执行文件。

| 命令 | 行为 |
| --- | --- |
| `single` | 按当前paramset生成输入和golden并运行；可用`--excel`切换用例来源 |
| `batch_save` | 默认从36条decode白盒Excel生成CPU输入和golden，每例保存一个.pt；无需NPU或wheel |
| `batch_exec` | 读取目录中的.pt运行NPU并比较，不读取Excel，不重新生成输入/golden |
| `batch` | 每例生成并保存，再读取该文件运行NPU；数据默认保留 |

```bash
# 在仓库根目录执行；可将目录改为需要长期保存数据的位置。
bash experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/test_run.sh batch_save \
  --pt-dir /tmp/mqsmla_data
bash experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/test_run.sh batch_exec \
  --pt-dir /tmp/mqsmla_data --device-id 0
# 按Excel生成、保存并运行
bash experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/test_run.sh batch \
  --excel experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/excel/mqsmla_whitebox_cases.xlsx \
  --sheet decode --pt-dir /tmp/mqsmla_data
# 参数集运行；-- 后可传pytest筛选、收集或报告参数
bash experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/test_run.sh single -- -k ori_cmp_sparse_8k_decode
bash experimental/attention/mixed_quant_sparse_flash_mla_dsl/tests/test_run.sh batch_exec \
  --pt-dir /tmp/mqsmla_data -- --junitxml=/tmp/mqsmla_replay.xml
```

支持`--excel`、`--sheet`、`--pt-dir`、`--device-id`，以及`--`后的pytest参数。默认数据目录为`tests/mqsmla_testcase`，默认sheet为decode；传入的相对路径以调用者当前目录为准，脚本本身可从任意目录调用。`MQSMLA_EXCEL`、`MQSMLA_SHEET`、`MQSMLA_PT_DIR`也可设置默认值。

每个.pt包含格式版本、完整测试参数、CPU Q/sinks/cu_seqlens/稀疏索引/有效长度、已打包的uint8 KV物理页与block table、golden输出和LSE。回放使用保存的参数和KV数据，按统一非连续开关重建固定padding布局；metadata由当前NPU核数重新生成。加载使用`weights_only=True`并检查格式及golden形状；文件是本DSL测试格式，不兼容主线的.pt文件。

文件名前缀保留用例顺序，后缀含参数摘要。相同用例重新保存会原子替换对应文件；保存失败不留下半个.pt。回放按文件名顺序运行目录内所有.pt，可用`-k`筛选；不同用例集使用独立目录，避免把历史文件混入本轮回放。大prefill输入与golden需要较多磁盘空间，按用例逐个生成保存。

直接pytest也支持`--data-mode run/save/replay/save-run`与`--pt-dir`；默认run保持原测试行为。save模式仅执行CPU数据准备，replay模式完全以文件为用例来源。

保存/回放验证：两个小用例（ORI_SPARSE连续KV、ORI_CMP_SPARSE非连续KV且开启LSE）保存2/2通过，读取保存文件后的真实NPU精度比较2/2通过。另验证了逐Tensor读写一致、回放不调用生成/golden、原子写失败保留旧文件、损坏/不兼容文件报错、跨目录和空格路径，以及默认36条decode用例收集。未执行完整36条NPU批跑。


## net Native wheel 调用

DSL 实现统一维护在 `cannbot-arena-ds41/net/ops/`，三个算子通过
`net/native_package/run-build.sh --group ds41` 编译到同一个
`cannbot-arena-net-ops` wheel。先安装该 wheel，再构建并安装本仓的
`cann_ops_transformer` wheel。torch_extension 仅保留 torch 注册和接口适配，
运行时从已安装的 `ops` 包导入 host 入口，不再打包本目录的 DSL 副本。
pytest 仍通过已安装的 torch wheel 调用 `torch.ops`。

Native 包预编译范围由导出配置决定；默认覆盖已有单算子 smoke 配置，
其他静态配置需通过 `CANNBOTDSL_DS41_PROFILES` 指定 JSON 后重新打包。
详见 net 工程 `native_package/README.md`。

## 当前动态 native 调用方式

与 net/ops/flash_kda.py 相同：使用 Dim + TensorSpec 编译动态契约，@register导出，
运行时复用同一编译入口和进程内缓存。首次调用生成IR以匹配wheel内的二进制，
命中后不执行后端编译；设置CANNBOTDSL_NATIVE_BINARY_MODE=require禁止未命中回退。
CANNBOTDSL_NATIVE_BINARY_REPORT报告应包含installed_hit且没有backend_pipeline。

T、B、S2/S2C、物理页数、两侧页大小和页距、scale及核数动态；页大小支持1..1024，
含两侧不同、非2幂与页间padding。N1=64、N2=1固定，head_dim当前512，K1/K2静态。
分页采用运行时标量查表，本次未做性能验收。

默认wheel包含有cmp（K1/K2=128/512）及无cmp（K1=128）两种模式，均关闭LSE、
一维sinks `[N1]`、活动侧length可独立传入或省略。无cmp attention的cmp输入全为None；metadata传零cmp
length并设置has_cmp_kv=False。其他结构组合需要追加导出profile。
既有pytest共7个用例覆盖两种模式、动态尺寸、页距和T超过核数的场景。

### DS41 require 模式

三个算子的 callable 缓存入口固定设置 `CANNBOTDSL_NATIVE_BINARY_MODE=require`，
无需调用方 export。这是 DSL 的进程级策略；首次 `.compile()` 仍生成 IR 匹配
native key，未命中报 `Native binary required but not found`，不回退后端编译。
打包时 native collector 仍正常编译、收集二进制。

MQSMLA 默认包含有 cmp（K1/K2=128/512）和无 cmp（K1=128）两种模式，
均为一维 sinks `[N1]`、关闭 LSE、支持活动侧 topk_length 传入或省略。
T、B、S2/S2C、物理页数、页大小（1..1024）、页距和 scale 动态；
N1=64、N2=1、head_dim=512 固定。改变 K1/K2、开启 LSE，需要通过 `CANNBOTDSL_DS41_PROFILES` 额外导出对应配置。
数值 topk_length 可变，但 Tensor 有无属于静态结构。

当前 sinks 契约已对齐主线：仅接受连续 FP32 `[N1]`（N1=64），
所有 batch/query 共用同一组 head sinks。kernel 不按 batch 偏移读取；
默认有/无 cmp 的 native 二进制均使用一维 sinks。二维 `[B,N1]` 输入会校验报错。
测试输入和独立 CPU golden 使用同一契约，旧二维 sinks 的 `.pt` 数据需重新 batch_save。

当前默认 MQSMLA wheel 预编译 6 种 topk_length 结构：有 cmp 时 ori/cmp
长度分别可传 Tensor 或 None（4 种），无 cmp 时 ori 长度可传 Tensor 或 None（2 种）。
省略 attention 的某侧长度表示该侧全部 K 个索引有效，不允许 -1 填充；
metadata 接口仍要求长度 Tensor，调用方可单独传全 K 长度，无 cmp 侧传零长度。
测试参数 omit_ori_topk_length / omit_cmp_topk_length 支持参数集及 Excel；
测试先按完整长度计算 metadata，再省略对应 attention keyword。默认仍关闭 LSE。

保存用例格式为 `mqsmla-dsl-input-golden-v2`，布局标记 `nope448-rope64-scale`。旧 rope/nope 格式不能直接回放，需用 `batch_save` 重新生成输入与独立 CPU golden。白盒工作簿为 `tests/excel/mqsmla_whitebox_cases.xlsx`，保存目录为 `tests/mqsmla_whitebox_cases/`。
