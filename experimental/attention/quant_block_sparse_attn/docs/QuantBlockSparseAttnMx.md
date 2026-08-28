# QuantBlockSparseAttnMx

维度编号说明：本文中"第 N 维"按从 1 开始计数；代码表达式如 `shape[2]`、`dim[0]` 保留从 0 开始的写法。



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

QuantBlockSparseAttnMx 是 QuantBlockSparseAttn 的 MXFP8 全量化路径（`quant_mode=2`），用于 FP8 量化场景下的分块稀疏注意力计算。算子根据 `sparse_indices` 和 `sparse_seq_len` 指定的稀疏块索引，只对每个 Query block 选中的 KV block 执行注意力计算，并支持 PagedAttention 形式的 KV Cache 存储。

与 `quant_mode=1`（FP8）路径相比，MXFP8 全量化路径 Query/Key 采用 **per-token-group** 的量化模式，Value 采用 **per-channel-group** 的量化模式：Q/K 按 D 轴每 32 个元素共享一个 `FLOAT8_E8M0` scale，V 按 S 轴每 32 个元素共享一个 `FLOAT8_E8M0` scale。P scale 支持 `FLOAT8_E8M0` 和 `FLOAT32` 两种数据类型。

计算语义如下：

$$
P = \text{softmax}(((Q\times q\_descale) \times (K^T\times k\_descale ^T)) \times softmax\_scale, mask)
$$

$$
O = (quant(P \times p\_scale)\times (V \times v\_descale)) / p\_scale
$$

其中 `K`、`V` 由 `block_table` 和 `sparse_indices` 从 PageAttention KV Cache 中按块寻址获得。`q_descale`、`k_descale`、`v_descale` 均为 `FLOAT8_E8M0` 格式，每 32 个数据元素对应一个 scale 值（Q/K 沿 D 轴分组，V 沿 S 轴分组）。`p_scale` 为 per-tensor，支持 `FLOAT8_E8M0` 或 `FLOAT32` 数据类型。`mask_mode=0` 表示不加 mask，`mask_mode=3` 表示 causal mask。

## 接口说明

该算子通过 PyTorch 扩展注册为 `torch.ops.custom.npu_quant_block_sparse_attn`，与 `quant_mode=1` 共用同一接口签名，通过显式传入 `quant_mode=2` 选择 MXFP8 全量化路径。PyTorch 入口会创建输出 Tensor，并调用底层算子接口。

### PyTorch 接口原型

```python
torch.ops.custom.npu_quant_block_sparse_attn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    p_scale: Optional[Tensor],
    sparse_indices: Tensor,
    sparse_seq_len: Tensor,
    atten_mask: Optional[Tensor],
    softmax_scale: float,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    seqused_q: Optional[Tensor] = None,
    seqused_kv: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    metadata: Optional[Tensor] = None,
    layout_kv: str = "PA_BNBD",
    layout_q: str = "TND",
    layout_sparse_indices: str = "B_N_Qb_Kb",
    layout_out: str = "TND",
    quant_mode: int = 1,
    mask_mode: int = 3,
    return_softmax_lse: bool = False,
) -> Tuple[Tensor, Tensor]
```

## 参数说明

维度符号说明：

- `B`：Batch size。
- `S1`：单个 batch 的 Query 最大长度，`S1 <= max_Qb * sparse_q_block_size`。
- `S2`：单个 batch 的 KV 最大长度，`S2 <= max_Kb * sparse_kv_block_size`。
- `T1`：所有 batch 的 Query 有效 token 数之和。
- `N1`：Query head 数。
- `N2`：KV head 数。
- `G`：GQA 分组数，`G = N1 / N2`。
- `D`：Q/K head dim，当前实现固定为 128。
- `D_v`：V head dim，当前实现固定为 128。
- `max_Qb`：Query block 最大数量，对应 `sparse_indices` 第 3 维和 `sparse_seq_len` 第 3 维。
- `max_Kb`：每个 Query block 最多保存的稀疏 KV block 索引数量，对应 `sparse_indices` 第 4 维。
- `max_block_num_per_batch`：每个 batch 在 `block_table` 中可索引的最大逻辑 KV block 数，对应 `block_table` 第 2 维。
- `block_num`：PageAttention KV Cache 物理 block 总数。
- `pa_block_size`：PA KV Cache 的物理 block 大小，对应 `key`/`value` 第 3 维；必须为 `sparse_kv_block_size` 的正整数倍且不超过 1024。
- `quant_group_size`：MXFP8 量化分组大小，固定为 32；Q/K scale 按 D 轴每 32 元素一个 scale，V scale 按 S 轴每 32 元素一个 scale。


- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 132px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    <col style="width: 119px">
    <col style="width: 272px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>Query 输入。</td>
      <td>不支持空 Tensor。PyTorch 接入层会将该输入转为连续 Tensor。MXFP8 路径仅支持 TND 布局。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>(T1,N1,D)</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>PageAttention KV Cache 中的 Key。</td>
      <td>layout_kv 仅支持 PA_BNBD。按 4D PA BNBD 视图传入。第 3 维为 pa_block_size，可不等于 sparse_kv_block_size。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>(block_num,N2,pa_block_size,D)</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>PageAttention KV Cache 中的 Value。</td>
      <td>layout_kv 仅支持 PA_BNBD。按 4D PA BNBD 视图传入。第 3 维为 pa_block_size，与 key 第 3 维一致。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>(block_num,N2,pa_block_size,D_v)</td>
    </tr>
    <tr>
      <td>q_descale</td>
      <td>输入</td>
      <td>Query 反量化缩放因子，per-token-group。</td>
      <td>每 32 个 D 元素对应一个 e8m0 scale；末维 2 表示相邻两个 32-group 的 scale 打包。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>(T1,N1,D/64,2)</td>
    </tr>
    <tr>
      <td>k_descale</td>
      <td>输入</td>
      <td>Key 反量化缩放因子，per-token-group。</td>
      <td>按 5D PA 视图传入。每 32 个 D 元素对应一个 e8m0 scale。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>(block_num,N2,pa_block_size,D/64,2)</td>
    </tr>
    <tr>
      <td>v_descale</td>
      <td>输入</td>
      <td>Value 反量化缩放因子，per-channel-group。</td>
      <td>按 5D PA 视图传入。每 32 个 S 元素对应一个 e8m0 scale，沿 S 轴分组。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>(block_num,N2,pa_block_size/64,D_v,2)</td>
    </tr>
    <tr>
      <td>p_scale</td>
      <td>输入</td>
      <td>softmax 概率 FP8 量化缩放因子，per-tensor。</td>
      <td>允许传空（None 或 shape size == 0），传空时 kernel 侧使用默认值 1.0 进行量化计算；非空传入时 shape 必须为 (1)，数据类型支持 FLOAT8_E8M0 或 FLOAT32。</td>
      <td>FLOAT8_E8M0 或 FLOAT32</td>
      <td>ND</td>
      <td>(1) 或空</td>
    </tr>
    <tr>
      <td>sparse_indices</td>
      <td>输入</td>
      <td>稀疏 KV block 索引。</td>
      <td>sparse_indices 仅支持 B_N_Qb_Kb，有效元素表示逻辑 KV block id；逻辑块索引不能重复，并且有效的在前，-1在后，表示不使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,N1,max_Qb,max_Kb)</td>
    </tr>
    <tr>
      <td>sparse_seq_len</td>
      <td>输入</td>
      <td>每个 Query block 对应的有效 KV block 数。</td>
      <td>每个值应在 [0, max_Kb] 范围内；值为 0 时对应稀疏任务输出置零，LSE 置为 -FLT_MAX。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,N1,max_Qb)</td>
    </tr>
    <tr>
      <td>atten_mask</td>
      <td>输入</td>
      <td>Attention mask。</td>
      <td>mask_mode=0 时不使用；mask_mode=3 时为 causal mask。</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>mask_mode=0 时可不传或传空指针；mask_mode=3:(2048,2048)</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>输入</td>
      <td>Query 累积序列长度。</td>
      <td>TND + PA_BNBD 场景必传，实际 Q 长度由相邻前缀差计算。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1)</td>
    </tr>
    <tr>
      <td>cu_seqlens_kv</td>
      <td>输入</td>
      <td>预留参数。</td>
      <td>预留参数，必须传空（None）；传入非空 Tensor 时 host 侧将拦截并报错。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>输入</td>
      <td>预留参数。</td>
      <td>预留参数，必须传空（None）；传入非空 Tensor 时 host 侧将拦截并报错。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seqused_kv</td>
      <td>输入</td>
      <td>每个 batch 的 KV 实际使用长度。</td>
      <td>TND + PA_BNBD 场景必传。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B)</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>输入</td>
      <td>PageAttention block 映射表。</td>
      <td>第 1 维必须等于 B。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,max_block_num_per_batch)</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输入</td>
      <td>负载均衡元数据。</td>
      <td>MXFP8 路径下为可选输入，不允许传空（None）；如传入则必须为由 npu_quant_block_sparse_attn_metadata 生成的有效 1D INT32 Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(metadata_size)</td>
    </tr>
    <tr>
      <td>softmax_scale</td>
      <td>输入属性</td>
      <td>QK 结果缩放因子。</td>
      <td>取值范围必须为 (0, 1]，传入 &lt;=0 或 &gt;1 的值将被 host 侧拦截。常用值为 1 / sqrt(D)。</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_q_block_size</td>
      <td>输入属性</td>
      <td>Query 方向稀疏 block 大小。</td>
      <td>支持 64 或 128，且必须与 sparse_kv_block_size 相等。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_kv_block_size</td>
      <td>输入属性</td>
      <td>KV 方向稀疏 block 大小。</td>
      <td>支持 64 或 128，且必须与 sparse_q_block_size 相等。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>输入属性</td>
      <td>KV 数据布局。</td>
      <td>仅支持 "PA_BNBD"。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>输入属性</td>
      <td>Query 数据布局。</td>
      <td>MXFP8 路径仅支持 "TND"。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_sparse_indices</td>
      <td>输入属性</td>
      <td>稀疏索引布局。</td>
      <td>仅支持 "B_N_Qb_Kb"。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_out</td>
      <td>输入属性</td>
      <td>输出布局。</td>
      <td>仅支持 "TND"。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>输入属性</td>
      <td>量化模式。</td>
      <td>支持1 和 2，1 表示 FP8量化，2表示 MXFP8 量化；MXFP8要求显式传入2，否则会默认走入FP8量化。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>输入属性</td>
      <td>mask 模式。</td>
      <td>支持 0 和 3，0 表示无 mask，3 表示 causal。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_softmax_lse</td>
      <td>输入属性</td>
      <td>是否返回 softmax lse。</td>
      <td>True 时返回有效 softmax lse；False 时不返回有效 softmax lse。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_out</td>
      <td>输出</td>
      <td>Attention 输出。</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T1,N1,D_v)</td>
    </tr>
    <tr>
      <td>softmax_lse</td>
      <td>输出</td>
      <td>softmax log-sum-exp。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>return_softmax_lse=True:(T1,N1); return_softmax_lse=False:()</td>
    </tr>
  </tbody>
  </table>

## 返回值

PyTorch 接口返回 `(attention_out, softmax_lse)`：

- `attention_out`：BF16 Tensor，最后一维为 `D_v`。MXFP8 路径固定按 TND 语义返回 `(T1,N1,D_v)`。
- `softmax_lse`：FLOAT32 Tensor。`return_softmax_lse=True` 时返回 `(T1,N1)`（注意：与 `quant_mode=1` 的 `(N1,T1)` 布局不同）；`return_softmax_lse=False` 时返回无有效 LSE 的占位 Tensor。


## 约束说明

### 约束类型说明

QuantBlockSparseAttnMx 算子约束分为 4 个档位，按约束复杂程度递增分为单参数约束、存在性约束、一致性约束和特性交叉约束，各档位约束内容如下：

- 单参数约束：对于单个接口参数的约束，包含 Tensor 和 Attribute。
  - 对于 Tensor，单参数约束包含 shape 维度、每一维度取值、dtype、format、是否为空 Tensor、是否连续或 stride 形态等校验。
  - 对于 Attribute，单参数约束包含属性取值范围和默认值语义。
- 存在性约束：约束特定场景下，特性参数组内必须传入某参数，或不支持传入某参数。
- 一致性约束：特性参数组内，各个参数间的 shape、dtype、layout、head 数、序列长度、block 数等一致性约束。
- 特性交叉约束：涉及多个参数组，不同参数组间的交叉约束。

### 特性参数组

<table><thead>
  <tr>
    <th>特性参数组</th>
    <th>参数字段名称</th>
    <th>字段分组</th>
    <th>字段类型</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="10">公共参数组</td>
    <td>query</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>key</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>value</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>softmax_scale</td>
    <td>ATTR</td>
    <td>float</td>
  </tr>
  <tr>
    <td>sparse_q_block_size</td>
    <td>ATTR</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>sparse_kv_block_size</td>
    <td>ATTR</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>layout_q</td>
    <td>ATTR</td>
    <td>string</td>
  </tr>
  <tr>
    <td>layout_out</td>
    <td>ATTR</td>
    <td>string</td>
  </tr>
  <tr>
    <td>quant_mode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>attention_out</td>
    <td>OUTPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="4">量化参数组</td>
    <td>q_descale</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>k_descale</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>v_descale</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>p_scale</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="3">稀疏索引参数组</td>
    <td>sparse_indices</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>sparse_seq_len</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>layout_sparse_indices</td>
    <td>ATTR</td>
    <td>string</td>
  </tr>
  <tr>
    <td rowspan="2">Paged Attention参数组</td>
    <td>block_table</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>layout_kv</td>
    <td>ATTR</td>
    <td>string</td>
  </tr>
  <tr>
    <td rowspan="4">ActualSeqLen参数组</td>
    <td>cu_seqlens_q</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>cu_seqlens_kv</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>seqused_q</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>seqused_kv</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="2">Attention Mask参数组</td>
    <td>atten_mask</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>mask_mode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>Metadata参数组</td>
    <td>metadata</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="2">SoftmaxLSE参数组</td>
    <td>return_softmax_lse</td>
    <td>ATTR(OPTIONAL)</td>
    <td>bool</td>
  </tr>
  <tr>
    <td>softmax_lse</td>
    <td>OUTPUT</td>
    <td>Tensor</td>
  </tr>
</tbody></table>

### 基准信息说明

资料约束中，常见字段释义如下：

| 命名 | 含义 |
| :---: | :--- |
| MXFP8全量化 | `query`、`key`、`value` 为 `FLOAT8_E4M3FN`，`q_descale`、`k_descale`、`v_descale` 为 `FLOAT8_E8M0`，`p_scale` 为 `FLOAT8_E8M0` 或 `FLOAT32` |
| PA_BNBD | Paged Attention KV Cache 排布，逻辑形态为 `[block_num, N2, pa_block_size, D或D_v]`。 |
| 4D PA | 当前支持的 KV Cache 存储形态；接口传入 4D `key`、4D `value` 和 5D `k_descale`、5D `v_descale` 视图。 |
| BatchSize | Batch 数，对应 `sparse_indices`、`sparse_seq_len`、`block_table` 的第 1 维。 |
| QueryTokenNum | 所有 batch 的 Query 有效 token 数之和，对应 `query` 的 T 轴。 |
| QueryMaxSeqLen | 单个 batch 的 Query 实际最大序列长度，由 `cu_seqlens_q` 的相邻前缀差语义决定。 |
| KeyValueMaxSeqLen | 单个 batch 的 KV 实际最大序列长度，由 `seqused_kv` 的数值语义决定。 |
| N1 | Query head 数，对应 `query` 的 N 轴和 `sparse_indices` 的第 2 维。 |
| N2 | KV head 数，对应 `key`、`value` 的 N 轴。 |
| G | GQA 分组数，`G = N1 / N2`。 |
| D | Query/Key head dim，当前固定为 128。 |
| D_v | Value head dim，当前固定为 128。 |
| max_Qb | Query block 最大数量，对应 `sparse_indices` 第 3 维和 `sparse_seq_len` 第 3 维。 |
| max_Kb | 每个 Query block 最多保存的稀疏 KV block 索引数量，对应 `sparse_indices` 第 4 维。 |
| block_num | PA KV Cache 物理 block 总数，对应 4D `key` 第 1 维。 |
| pa_block_size | PA KV Cache 物理 block 大小，对应 `key`/`value` 第 3 维；必须为 `sparse_kv_block_size` 的正整数倍且不超过 1024。 |
| max_block_num_per_batch | 每个 batch 在 `block_table` 中可索引的最大逻辑 block 数，对应 `block_table` 第 2 维。 |
| key.stride(0) | 相邻物理 PA block 的外步长，表示 Key、Value、`k_descale`、`v_descale` 各视图在 0 轴上的非连续存储跨度。 |
| quant_group_size | MXFP8 量化分组大小，固定为 32；每 32 个数据元素对应一个 e8m0 scale 值。 |

### 参数组约束

#### 公共参数组

- 单参数约束

  - `query`、`key`、`value` 数据类型仅支持 `FLOAT8_E4M3FN`，数据格式仅支持 ND。
  - `query` 仅支持 3D Tensor，MXFP8 路径下 `layout_q` 必须为 `"TND"`，shape 为 `(QueryTokenNum, N1, D)`。
  - `query` 不支持空 Tensor，且 `D` 当前固定为 128。
  - `key` 仅支持 4D PA 形态，shape 为 `(block_num, N2, pa_block_size, D)`，其中 `D` 固定为 128，`pa_block_size` 必须为 `sparse_kv_block_size` 的正整数倍且不超过 1024。
  - `value` 仅支持与 `key` 对应的 4D PA 形态，shape 为 `(block_num, N2, pa_block_size, D_v)`，其中 `D_v` 固定为 128，`pa_block_size` 必须与 `key` 第 3 维一致。
  - `attention_out` 数据类型为 `BFLOAT16`，数据格式为 ND。输出 shape 固定为 `(QueryTokenNum, N1, D_v)`。
  - `sparse_q_block_size` 和 `sparse_kv_block_size` 各支持 64 或 128，且二者必须相等。
  - `layout_q` 仅支持 `TND`。
  - `layout_out` 仅支持 `TND`。
  - `quant_mode` 支持1 和 2，1表示`FP8`量化，2表示 `MXFP8`量化。
  - `softmax_scale` 为 float 属性，取值范围必须为 `(0, 1]`，常用值为 `1 / sqrt(D)`。

- 存在性约束

  - `query`、`key`、`value`、`q_descale`、`k_descale`、`v_descale`、`sparse_indices`、`sparse_seq_len` 必须传入；`p_scale` 允许传空，传空时使用默认值 1.0 进行量化计算；`atten_mask` 在 `mask_mode=3` 时必须传入，在 `mask_mode=0` 时可不传或传空指针。
  - `metadata` 在 MXFP8 路径下为必选输入，不允许传空（None）。
  - 当前算子仅支持 4D PA_BNBD KV Cache 输入和 BF16 attention_out 输出 以及FP32 softmax_lse。

- 一致性约束

  - `N1` 必须是 `N2` 的整数倍。
  - `sparse_indices` 必须按 `(BatchSize, N1, max_Qb, max_Kb)` 传入，`sparse_seq_len` 必须按 `(BatchSize, N1, max_Qb)` 传入；当前主算子默认这两个 Tensor 的 shape 正确，不在 host 中对 max_Qb 和 max_Kb 进行拦截校验。
  - `block_table` 必须按 `(BatchSize, max_block_num_per_batch)` 传入。主算子使用 `sparse_indices.shape[2]` 作为 `max_Qb`，使用 `block_table.shape[1]` 作为 `max_block_num_per_batch`，二者均必须大于 0。
  - `BatchSize`、`N1`、`N2`、`G` 均必须大于 0。`BatchSize`<= `65536`，`N1` <= `128`，`N2` <= `8`，`G` <= `16`。
  - `S1`、`S2` 均小于 `20M` ，不拦截。
  - `D` 和 `D_v` 必须均为 128。

- 特性交叉约束

  - `query`、`key`、`value` 的 head dim 必须与量化参数、稀疏 block 参数和输出 head dim 保持一致。
  - `layout_q` 固定为 TND，`attention_out`、`softmax_lse` 按固定输出语义返回。

#### 量化参数组

- 单参数约束

  - `q_descale`、`k_descale`、`v_descale` 数据类型仅支持 `FLOAT8_E8M0`，数据格式仅支持 ND。
  - `p_scale` 数据类型支持 `FLOAT8_E8M0` 或 `FLOAT32`，数据格式仅支持 ND。
  - `q_descale` 表示 Query per-token-group 反量化缩放，shape 为 4D `(QueryTokenNum, N1, D/64, 2)`，其中末维 2 表示相邻两个 32-group scale 打包，`D/64` 为 D 轴 scale group 对数。
  - `k_descale` 表示 Key per-token-group 反量化缩放，需与 PA KV Cache 的物理 block、KV head、block 内 token 和 D-group 对应，shape 为 5D `(block_num, N2, pa_block_size, D/64, 2)`。
  - `v_descale` 表示 Value per-channel-group 反量化缩放，沿 S 轴每 32 个 token 分一组，shape 为 5D `(block_num, N2, pa_block_size/64, D_v, 2)`。
  - `p_scale` 表示 softmax 概率 per-tensor 静态量化缩放，允许传空，传空时使用默认值 1.0 进行量化计算；非空传入时 shape 必须为 `(1)`，数据类型为 `FLOAT8_E8M0` 或 `FLOAT32`。

- 存在性约束

  - `quant_mode=2` MXFP8 场景下 `q_descale`、`k_descale`、`v_descale` 必须传入；`p_scale` 允许传空，传空时使用默认值 1.0 进行量化计算。

- 一致性约束

  - `q_descale` 的 token/head/D-group 维度必须与 `query` 的 `QueryTokenNum`、`N1`、`D` 对齐。
  - `k_descale` 的 PA block、KV head、block 内 token、D-group 维度必须与 `key`、`block_table`、`pa_block_size`、`D` 对齐。
  - `v_descale` 的 PA block、KV head、S-group、D_v 维度必须与 `value`、`block_table`、`pa_block_size`、`D_v` 对齐。
  - `p_scale` 非空时数值应大于 0，且是有限正数；Tiling 阶段无法读取 Tensor 数值，该数值合法性由调用者保证。


- 特性交叉约束

  - `quant_mode=2` 时，`q_descale`、`k_descale`、`v_descale` 数据类型为 `FLOAT8_E8M0`；`p_scale` 数据类型为 `FLOAT8_E8M0` 或 `FLOAT32`。

#### 稀疏索引参数组

- 单参数约束

  - `sparse_indices`、`sparse_seq_len` 数据类型仅支持 `INT32`，数据格式仅支持 ND。
  - `layout_sparse_indices` 当前仅支持 `B_N_Qb_Kb`。
  - `sparse_indices` 必须为 4D Tensor，shape 为 `(BatchSize, N1, max_Qb, max_Kb)`，其中 `max_Kb` 必须大于 0。
  - `sparse_seq_len` 必须为 3D Tensor，shape 为 `(BatchSize, N1, max_Qb)`。

- 存在性约束

  - `sparse_indices` 和 `sparse_seq_len` 均为必传输入。

- 一致性约束

  - `sparse_indices` 和 `sparse_seq_len` 的 shape 必须与 `BatchSize`、`N1`、`max_Qb` 语义保持一致；当前主算子 host 默认这两个 Tensor 的 shape 正确，不在 host 中对 max_Qb 和 max_Kb 的正确性进行校验，输入正确性由用户外部保证。
  - `sparse_seq_len[B,N1,max_Qb]` 表示对应 Query block 的有效 KV block 数；值为 0 时对应稀疏任务输出置零，LSE 置为 `-FLT_MAX`。
  - `sparse_indices` 的有效元素表示逻辑 KV block id；逻辑块索引不能重复，并且有效的在前，-1在后，表示不使用。

- 特性交叉约束

  - `sparse_indices` 中的逻辑 KV block id 需要能通过 `block_table` 映射到合法 PA 物理 block，映射逻辑由用户外部保证。
  - Tiling 阶段无法读取 Tensor 数值，`sparse_indices`、`sparse_seq_len` 的数值合法性由调用者保证。

#### Paged Attention参数组

- 单参数约束

  - `layout_kv` 当前仅支持 `PA_BNBD`。
  - `block_table` 数据类型仅支持 `INT32`，数据格式仅支持 ND。当前 PA 执行路径依赖 `block_table` 做逻辑 block 到物理 block 的映射，必须传入有效 `block_table`。
  - `block_table` 的 shape 必须为 `(BatchSize, max_block_num_per_batch)`，第 1 维必须等于 `BatchSize`。
  - PA 物理 block 外步长由 host 侧从 `key.stride(0)` 推导，不再作为输入属性传入。

- 存在性约束

  - 当前算子面向 Paged Attention KV Cache 场景，不支持普通连续 KV 输入。
  - 需要使用逻辑 KV block 到物理 block 映射时，必须传入 `block_table`。

- 一致性约束

  - 当前仅支持 4D PA 输入。
  - 4D PA 输入下，接口分别传入 `key`、`value`、`k_descale`、`v_descale` 对应的 4D/4D/5D/5D 视图。
  - `key` stride 必须满足 `[key.stride(0), pa_block_size * D, D, 1]`。
  - `value` stride 必须满足 `[value.stride(0), pa_block_size * D_v, D_v, 1]`。
  - `value.stride(0)` 必须等于 `key.stride(0)`，表示 Value 视图与 Key 视图使用相同的物理 PA block 外步长。
  - `k_descale` shape 必须为 `[block_num, N2, pa_block_size, D/64, 2]`。
  - `v_descale` shape 必须为 `[block_num, N2, pa_block_size/64, D_v, 2]`。
  - `pa_block_size` 必须为 `sparse_kv_block_size` 的正整数倍且不超过 1024。
  - `block_table` 的有效值必须在 `[0, block_num - 1]` 范围内，该逻辑由用户外部保证。

- 特性交叉约束

  - `block_table` 的 Batch 维度必须与 `sparse_indices`、`sparse_seq_len`、ActualSeqLen 参数组中的 BatchSize 一致；
  - `sparse_indices` 的有效逻辑 block id 必须小于 `max_block_num_per_batch`，否则 `block_table` 映射越界。
  - `block_table` 数值合法性由调用者保证。

#### ActualSeqLen参数组

- 单参数约束

  - `cu_seqlens_q`、`seqused_kv` 数据类型仅支持 `INT32`，数据格式仅支持 ND。
  - `cu_seqlens_q` shape 为 `(BatchSize + 1)`，`seqused_kv` shape 为 `(BatchSize)`。

- 存在性约束

  - TND + PA_BNBD 场景必须传入 `cu_seqlens_q` 和 `seqused_kv`。
  - `cu_seqlens_kv` 和 `seqused_q` 为预留参数，必须传空；传入非空 Tensor 时 host 侧将拦截并报错。

- 一致性约束

  - `cu_seqlens_q` 应从 0 开始单调非降，末尾值等于 QueryTokenNum。
  - `seqused_kv` 每个元素应在 `[0, KeyValueMaxSeqLen]` 范围内。
  - `seqused_*`、`cu_seqlens_*` 的 Batch 语义必须与 `sparse_indices`、`sparse_seq_len`、`block_table` 保持一致。

- 特性交叉约束

  - Tiling 阶段无法读取 Tensor 数值，`cu_seqlens_q` 和 `seqused_kv` 的数值合法性由调用者保证。

#### Attention Mask参数组

- 单参数约束

  - `atten_mask` 数据类型仅支持 `UINT8`，数据格式仅支持 ND。
  - `mask_mode=3` 时，`atten_mask` 必须为二维 Tensor。
  - `mask_mode` 当前执行路径仅支持 0 和 3：0 表示无 mask，3 表示 causal mask。

- 存在性约束

  - `mask_mode=0` 时，`atten_mask` 为可选输入，可以不传或传入空指针。
  - `mask_mode=3` 时，`atten_mask` 为必选输入。

- 一致性约束

  - `mask_mode=0` 时，内核按无 mask 语义执行，不使用 `atten_mask` 的数值。
  - `mask_mode=3` 时，`atten_mask` 必须为二维 `(2048, 2048)` UINT8 causal mask，并与当前 Query/KV block 的 causal 访问窗口匹配。

- 特性交叉约束

  - `mask_mode=3` 时，`sparse_indices` 中被选中的 KV block 仍需满足 causal 语义下的有效访问范围。
  - Tiling 阶段无法读取 `atten_mask` 数值，mask 矩阵内容合法性由调用者保证。

#### Metadata参数组

- 单参数约束

  - `metadata` 数据类型仅支持 `INT32`，数据格式仅支持 ND。
  - 有效 `metadata` 为一维 Tensor，shape 为 `(metadata_size)`。
  - `metadata_size = 8 + section_num * AIC_NUM * 8 + AIV_NUM * 8`，其中第一个 8 为 head metadata 区长度，`section_num` 为实时分 section 数，生成后记录在 `metadata[0]`。

- 存在性约束

  - `metadata` 在 MXFP8 路径下为必选输入，不允许传空（None）。如传入则必须为由 `npu_quant_block_sparse_attn_metadata` 生成的有效 1D INT32 Tensor，不能为空 Tensor。

- 一致性约束

  - 如传入 `metadata`，则必须由与主算子相同的 `sparse_seq_len`、`num_heads_q`、`num_heads_kv`、`head_dim`、`sparse_block_size_q`、`sparse_block_size_k`、`quant_mode`、`mask_mode`、`layout_q`、`layout_kv`、`layout_sparse_indices` 生成。
  - 用于主算子的 metadata 生成参数中，`head_dim` 应为 128，`sparse_block_size_q`、`sparse_block_size_k` 均应为 64 或 128，`quant_mode` 应为 2，`mask_mode` 应为 0 或 3，`layout_sparse_indices` 应为 `B_N_Qb_Kb`。

- 特性交叉约束

  - metadata 与 `sparse_seq_len` 参数不匹配时，分核调度范围可能与实际稀疏任务不一致，结果不保证正确。

#### SoftmaxLSE参数组

- 单参数约束

  - `return_softmax_lse` 为 bool 属性。
  - `softmax_lse` 输出数据类型为 `FLOAT32`。

- 存在性约束

  - `return_softmax_lse=False` 时，PyTorch 接入层返回无有效 LSE 的占位 `softmax_lse` Tensor。
  - `return_softmax_lse=True` 时，返回有效 `softmax_lse` Tensor。

- 一致性约束

  - `softmax_lse` 的 token/head 维度需与 `query` 的 N1 对齐。MXFP8 路径下 `softmax_lse` shape 为 `(T1, N1)`。
  - 当 `sparse_seq_len` 对应行没有有效 KV block 时，该行 LSE 置为 `-FLT_MAX`。

- 特性交叉约束

  - `return_softmax_lse` 只影响 `softmax_lse` 是否返回有效结果，不影响 `attention_out` 的 shape、dtype 和计算语义。

## 调用示例

```python
import torch
import torch_npu
import custom_ops

torch_npu.npu.set_device(0)

# BASE_01 用例参数: B=1, N1=1, N2=1, D=128, T1=256, S2=512
# pa_block_size=128, sparse_block_size=128, block_num=4
# quant_group_size=32, D//64=2

# query: (T1, N1, D) = (256, 1, 128) FP8_E4M3FN
query = torch.randn(256, 1, 128, dtype=torch.float32).to(torch.float8_e4m3fn).npu()

# key/value: (block_num, N2, pa_block_size, D) = (4, 1, 128, 128) FP8_E4M3FN
key = torch.randn(4, 1, 128, 128, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
value = torch.randn(4, 1, 128, 128, dtype=torch.float32).to(torch.float8_e4m3fn).npu()

# q_descale: (T1, N1, D//64, 2) = (256, 1, 2, 2) E8M0
# 每个 32-D group 一个 scale，pair-packed 末维 2
q_descale = torch.full((256, 1, 2, 2), 127, dtype=torch.uint8).to(torch.float8_e8m0fnu).npu()

# k_descale: (block_num, N2, pa_block_size, D//64, 2) = (4, 1, 128, 2, 2) E8M0
k_descale = torch.full((4, 1, 128, 2, 2), 127, dtype=torch.uint8).to(torch.float8_e8m0fnu).npu()

# v_descale: (block_num, N2, pa_block_size//64, D_v, 2) = (4, 1, 2, 128, 2) E8M0
# pa_block_size=128, 128//64=2 个 S-group
v_descale = torch.full((4, 1, 2, 128, 2), 127, dtype=torch.uint8).to(torch.float8_e8m0fnu).npu()

# p_scale: (1,) E8M0, 127 = 2^0 = 1.0
p_scale = torch.tensor([127], dtype=torch.uint8).to(torch.float8_e8m0fnu).npu()

# sparse_indices: (B, N1, max_Qb, max_Kb) = (1, 1, 2, 4)
# QB0 访问 KV block 0,1,2; QB1 访问 KV block 0,1,2,3
sparse_indices = torch.tensor([[[[0, 1, 2, -1],
                                 [0, 1, 2, 3]]]], dtype=torch.int32).npu()

# sparse_seq_len: (B, N1, max_Qb) = (1, 1, 2)
sparse_seq_len = torch.tensor([[[3, 4]]], dtype=torch.int32).npu()

# atten_mask: (2048, 2048) UINT8 causal mask
atten_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.uint8), diagonal=0).npu()

# cu_seqlens_q: (B+1,) = (2,)
cu_seqlens_q = torch.tensor([0, 256], dtype=torch.int32).npu()

# seqused_kv: (B,) = (1,)
seqused_kv = torch.tensor([512], dtype=torch.int32).npu()

# block_table: (B, max_block_num_per_batch) = (1, 4)
# 逻辑 block 0→物理 0, 1→1, 2→2, 3→3
block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32).npu()

# softmax_scale = 1/sqrt(128)
softmax_scale = 0.08838834764831843

# 生成 metadata
print("[INFO] 生成 metadata ...")
metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
    sparse_seq_len,
    1,  # num_heads_q = N1
    1,  # num_heads_kv = N2
    128,  # head_dim = D
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=seqused_kv,
    batch_size=1,
    sparse_block_size_q=128,
    sparse_block_size_k=128,
    quant_mode=2,
    mask_mode=3,
    layout_q="TND",
    layout_kv="PA_BNBD",
    layout_sparse_indices="B_N_Qb_Kb",
)
print("[INFO] metadata 生成成功, shape =", tuple(metadata.shape))

print("[INFO] 开始执行 npu_quant_block_sparse_attn (quant_mode=2 MXFP8) ...")
torch_npu.npu.synchronize()

attn_out, softmax_lse = torch.ops.custom.npu_quant_block_sparse_attn(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    softmax_scale,
    128,
    128,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=seqused_kv,
    block_table=block_table,
    metadata=metadata,
    layout_kv="PA_BNBD",
    layout_q="TND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    quant_mode=2,
    mask_mode=3,
    return_softmax_lse=True,
)
torch_npu.npu.synchronize()
print("[INFO] 算子执行成功")
print("[INFO] attn_out:       shape =", tuple(attn_out.shape), "dtype =", attn_out.dtype)
print("[INFO] softmax_lse:    shape =", tuple(softmax_lse.shape), "dtype =", softmax_lse.dtype)
```
