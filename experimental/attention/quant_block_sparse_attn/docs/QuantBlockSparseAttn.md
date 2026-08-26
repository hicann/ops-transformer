# QuantBlockSparseAttn

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

QuantBlockSparseAttn（QBSA）用于 FP8 量化场景下的分块稀疏注意力计算。算子根据 `sparse_indices` 和 `sparse_seq_len` 指定的稀疏块索引，只对每个 Query block 选中的 KV block 执行注意力计算，并支持 PagedAttention 形式的 KV Cache 存储。

该算子面向大序列推理或预填充场景，通过块级稀疏选择降低 QK、PV 两次矩阵乘的计算量；同时结合 `q_descale`、`k_descale`、`v_descale` 与 `p_scale` 完成 FP8 量化数据的反量化与 softmax 后再量化计算。

计算语义如下：

$$
P = \text{softmax}((QK^T) \times q\_descale \times k\_descale \times softmax\_scale, mask)
$$

$$
O = (quant(P \times p\_scale) V) / p\_scale \times v\_descale
$$

其中 `K`、`V` 由 `block_table` 和 `sparse_indices` 从 PageAttention KV Cache 中按块寻址获得。`mask_mode=0` 表示不加 mask，`mask_mode=3` 表示 causal mask。

## 接口说明

该算子通过 PyTorch 扩展注册为 `torch.ops.custom.npu_quant_block_sparse_attn`。PyTorch 入口会创建输出 Tensor，并调用底层算子接口。

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

维度编号说明：本文中“第 N 维”按从 1 开始计数；代码表达式如 `shape[2]`、`dim[0]` 保留从 0 开始的写法。

维度符号说明：

- `B`：Batch size。
- `S1`：单个 batch 的 Query 块级序列上界，`S1 <= max_Qb * sparse_q_block_size`。
- `S2`：单个 batch 的 KV 块级序列上界，`S2 <= max_Kb * sparse_kv_block_size`。
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
      <td>不支持空 Tensor。PyTorch 接入层会将该输入转为连续 Tensor。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>(T1,N1,D); (N1,T1,D)</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>PageAttention KV Cache 中的 Key。</td>
      <td>layout_kv 仅支持 PA_BNBD。按 4D PA BNBD 视图传入，0 轴为非连续存储。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>(block_num,N2,sparse_kv_block_size,D)</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>PageAttention KV Cache 中的 Value。</td>
      <td>layout_kv 仅支持 PA_BNBD。按 4D PA BNBD 视图传入，0 轴为非连续存储。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>(block_num,N2,sparse_kv_block_size,D_v)</td>
    </tr>
    <tr>
      <td>q_descale</td>
      <td>输入</td>
      <td>Query 反量化缩放因子。</td>
      <td>PERTOKEN_PERHEAD。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T1,N1); (N1,T1)</td>
    </tr>
    <tr>
      <td>k_descale</td>
      <td>输入</td>
      <td>Key 反量化缩放因子。</td>
      <td>PERTOKEN_PERHEAD。按 4D PA BNB1 视图传入，0 轴为非连续存储。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(block_num,N2,sparse_kv_block_size,1)</td>
    </tr>
    <tr>
      <td>v_descale</td>
      <td>输入</td>
      <td>Value 反量化缩放因子。</td>
      <td>PERHEAD。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(N2)</td>
    </tr>
    <tr>
      <td>p_scale</td>
      <td>输入</td>
      <td>softmax 概率 FP8 量化缩放因子。</td>
      <td>允许传空（None 或 shape size == 0），传空时 kernel 侧使用默认值 1.0 进行量化计算；非空传入时 shape 必须为 (1)。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(1) 或空</td>
    </tr>
    <tr>
      <td>sparse_indices</td>
      <td>输入</td>
      <td>稀疏 KV block 索引。</td>
      <td>sparse_indices 仅支持 B_N_Qb_Kb，有效元素表示逻辑 KV block id。</td>
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
      <td>TND/NTD + PA_BNBD 场景必传，实际 Q 长度由相邻前缀差计算。</td>
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
      <td>TND/NTD + PA_BNBD 场景必传。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B)</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>输入</td>
      <td>PageAttention block 映射表。</td>
      <td>第 1 维必须等于 B。第二维度大于等于max_Kb。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,max_block_num_per_batch)</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输入</td>
      <td>负载均衡元数据。</td>
      <td>必须传入且不能为 nullptr 或空 Tensor；由 npu_quant_block_sparse_attn_metadata 生成，有效长度随分 section 结果动态变化。</td>
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
      <td>实现仅支持 128。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_kv_block_size</td>
      <td>输入属性</td>
      <td>KV 方向稀疏 block 大小。</td>
      <td>实现仅支持 128。</td>
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
      <td>支持 "TND"、"NTD"。</td>
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
      <td>仅支持 "TND"，预留参数。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>输入属性</td>
      <td>量化模式。</td>
      <td>当前仅支持 1，表示 A8C8_QKV_FP8_P_STATIC_SOFTMAX_FP32。</td>
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
      <td>是否返回 softmax LSE。</td>
      <td>True 时返回有效 LSE；False 时不返回有效 LSE。</td>
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
      <td>return_softmax_lse=True:(N1,T1); return_softmax_lse=False:()</td>
    </tr>
  </tbody>
  </table>

## 返回值

PyTorch 接口返回 `(attention_out, softmax_lse)`：

- `attention_out`：BF16 Tensor，最后一维为 `D_v`。PyTorch 接入层按 TND 语义返回 `(T1,N1,D_v)`；`layout_q="NTD"` 输入不会使输出保持 NTD。
- `softmax_lse`：FLOAT32 Tensor。`return_softmax_lse=True` 时返回 `(N1,T1)`；`return_softmax_lse=False` 时返回无有效 LSE 的占位 Tensor。

底层算子接口返回状态码。第一段 GetWorkspaceSize 接口完成参数校验，必选输入为空、数据类型不支持、shape 与属性不匹配时返回参数错误。

## 约束说明

### 约束类型说明

QuantBlockSparseAttn 算子约束分为 4 个档位，按约束复杂程度递增分为单参数约束、存在性约束、一致性约束和特性交叉约束，各档位约束内容如下：

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
| FP8全量化 | `query`、`key`、`value` 为 `FLOAT8_E4M3FN`，`q_descale`、`k_descale`、`v_descale`、`p_scale` 参与反量化/再量化缩放的场景。 |
| PA_BNBD | Paged Attention KV Cache 排布，逻辑形态为 `[block_num, N2, sparse_kv_block_size, D或D_v]`。 |
| 4D PA 0 轴非连续存储 | 当前支持的 KV Cache 存储形态；接口传入 4D `key`、4D `value` 和 4D `k_descale` BNB1 视图，各视图的 0 轴均为非连续存储。 |
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
| max_block_num_per_batch | 每个 batch 在 `block_table` 中可索引的最大逻辑 block 数，对应 `block_table` 第 2 维。 |
| key.stride(0) | 相邻物理 PA block 的外步长，表示 Key、Value、`k_descale` 各视图在 0 轴上的非连续存储跨度。 |

### 参数组约束

#### 公共参数组

- 单参数约束

  - 当前算子配置支持 `ascend950`。
  - `query`、`key`、`value` 数据类型仅支持 `FLOAT8_E4M3FN`，数据格式仅支持 ND。
  - `query` 仅支持 3D Tensor：
    - `layout_q="TND"` 时，shape 为 `(QueryTokenNum, N1, D)`。
    - `layout_q="NTD"` 时，shape 为 `(N1, QueryTokenNum, D)`。
  - `query` 不支持空 Tensor，且 `D` 当前固定为 128。
  - `key` 仅支持 4D PA 形态，shape 为 `(block_num, N2, sparse_kv_block_size, D)`，其中 `D` 固定为 128。
  - `value` 仅支持与 `key` 对应的 4D PA 形态，shape 为 `(block_num, N2, sparse_kv_block_size, D_v)`，其中 `D_v` 固定为 128。
  - `attention_out` 数据类型为 `BFLOAT16`，数据格式为 ND。`layout_q="TND"` 时输出 shape 为 `(QueryTokenNum, N1, D_v)`；`layout_q="NTD"` 时 PyTorch 接入层输出仍为 TND 语义的 `(QueryTokenNum, N1, D_v)`。
  - `sparse_q_block_size` 和 `sparse_kv_block_size` 当前均仅支持 128。
  - `layout_q` 当前仅支持 `TND`、`NTD`。
  - `layout_out` 为预留参数，当前不使能；传入非 `TND` 类型会被拦截。
  - `quant_mode` 当前主算子仅支持 1，表示 `A8C8_QKV_FP8_P_STATIC_SOFTMAX_FP32`。
  - `softmax_scale` 为 float 属性，取值范围必须为 `(0, 1]`，常用值为 `1 / sqrt(D)`。

- 存在性约束

  - `query`、`key`、`value`、`q_descale`、`k_descale`、`v_descale`、`sparse_indices`、`sparse_seq_len`、`metadata` 必须传入；`p_scale` 允许传空，传空时使用默认值 1.0 进行量化计算；`atten_mask` 在 `mask_mode=3` 时必须传入，在 `mask_mode=0` 时可不传或传空指针。
  - 当前算子支持 4D PA_BNBD KV Cache 输入和 BF16 attention_out 输出。

- 一致性约束

  - `quant_mode=1` 时，`BatchSize` 的取值范围必须为 `(0, 65536]`，`N1` 的取值范围必须为 `(0, 128]`；`N2`、`G` 均必须大于 0，且 `N2` <= `8`，`G` <= `16`。
  - `N1` 必须能被 `N2` 整除。
  - `sparse_indices` 必须按 `(BatchSize, N1, max_Qb, max_Kb)` 传入，`sparse_seq_len` 必须按 `(BatchSize, N1, max_Qb)` 传入；当前主算子默认这两个 Tensor 的 shape 正确，不在 host 中对max_Qb和max_Kb进行拦截校验。
  - `block_table` 必须按 `(BatchSize, max_block_num_per_batch)` 传入。主算子使用 `sparse_indices.shape[2]` 作为 `max_Qb`，使用 `block_table.shape[1]` 作为 `max_block_num_per_batch`，二者均必须大于 0。
  - 总计算块数 `BatchSize * N1 * max_Qb` 必须在 `[1, UINT32_MAX]` 范围内。

- 特性交叉约束

  - `query`、`key`、`value` 的 head dim 必须与量化参数、稀疏 block 参数和输出 head dim 保持一致。

#### 量化参数组

- 单参数约束

  - `q_descale`、`k_descale`、`v_descale`、`p_scale` 数据类型仅支持 `FLOAT32`，数据格式仅支持 ND。
  - `q_descale` 表示 Query per-token-per-head 反量化缩放，`layout_q="TND"` 时为 `(QueryTokenNum, N1)`，`layout_q="NTD"` 时为 `(N1, QueryTokenNum)`。
  - `k_descale` 表示 Key per-token-per-head 反量化缩放，需与 PA KV Cache 的物理 block、KV head 和 block 内 token 对应。
  - `v_descale` 表示 Value per-head 反量化缩放，shape 为 `(N2)`。
  - `p_scale` 表示 softmax 概率 per-tensor 静态量化缩放，允许传空，传空时使用默认值 1.0 进行量化计算；非空传入时 shape 必须为 `(1)`。

- 存在性约束

  - `quant_mode=1` FP8 场景下 `q_descale`、`k_descale`、`v_descale` 必须传入；`p_scale` 允许传空，传空时使用默认值 1.0 进行量化计算。

- 一致性约束

  - `q_descale` 的 token/head 维度必须与 `query` 的 N1 对齐。
  - `k_descale` 的 PA block、KV head、block 内 token 维度必须与 `key`、`block_table`、`sparse_kv_block_size` 对齐。
  - `v_descale` 第 1 维必须等于 `N2`。
  - `p_scale` 非空时数值应大于 0；Tiling 阶段无法读取 Tensor 数值，该数值合法性由调用者保证。

- 特性交叉约束

  - `quant_mode=1` 时，量化粒度固定为 Query per-token-per-head、Key per-token-per-head、Value per-head、P per-tensor；scale 数据类型为 `FLOAT32`。
  - `k_descale` 必须满足 Paged Attention 参数组中的 4D PA 0 轴非连续存储 stride 约束。

#### 稀疏索引参数组

- 单参数约束

  - `sparse_indices`、`sparse_seq_len` 数据类型仅支持 `INT32`，数据格式仅支持 ND。
  - `layout_sparse_indices` 当前仅支持 `B_N_Qb_Kb`。
  - `sparse_indices` 必须为 4D Tensor，shape 为 `(BatchSize, N1, max_Qb, max_Kb)`，其中 `max_Kb` 必须大于 0。
  - `sparse_seq_len` 必须为 3D Tensor，shape 为 `(BatchSize, N1, max_Qb)`。

- 存在性约束

  - `sparse_indices` 和 `sparse_seq_len` 均为必传输入。

- 一致性约束

  - `sparse_indices` 和 `sparse_seq_len` 的 shape 必须与 `BatchSize`、`N1`、`max_Qb` 语义保持一致；当前主算子 host 默认这两个 Tensor 的 shape 正确，不在 host 中对max_Qb和max_Kb的正确性进行校验，输入正确性由用户外部保证。
  - `sparse_seq_len[B,N1,max_Qb]` 表示对应 Query block 的有效 KV block 数；值为 0 时对应稀疏任务输出置零，LSE 置为 `-FLT_MAX`。
  - `sparse_indices` 的有效元素表示逻辑 KV block id。

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
  - 4D PA 输入下，接口分别传入 `key`、`value`、`k_descale` 对应的 4D/4D/4D 视图，各视图的 0 轴均为非连续存储。
  - `key` stride 必须满足 `[key.stride(0), sparse_kv_block_size * D, D, 1]`。
  - `value` stride 必须满足 `[value.stride(0), sparse_kv_block_size * D_v, D_v, 1]`。
  - `value.stride(0)` 必须等于 `key.stride(0)`，表示 Value 视图与 Key 视图使用相同的物理 PA block 外步长。
  - `k_descale` shape 必须为 `[block_num, N2, sparse_kv_block_size, 1]`；`k_descale.stride(0) * 4` 必须等于 `key.stride(0)`，表示 `k_descale` 以 FLOAT32 字节数对齐同一物理 PA block 外步长；`k_descale` 后三维 stride 必须为 `[sparse_kv_block_size, 1, 1]`。
  - `block_table` 的有效值必须在 `[0, block_num - 1]` 范围内，该逻辑由用户外部保证。

- 特性交叉约束

  - `block_table` 的 Batch 维度必须与 `sparse_indices`、`sparse_seq_len`、ActualSeqLen 参数组中的 BatchSize 一致。
  - `sparse_indices` 的有效逻辑 block id 必须小于 `max_block_num_per_batch`，否则 `block_table` 映射越界。
  - `block_table` 数值合法性由调用者保证。

#### ActualSeqLen参数组

- 单参数约束

  - `cu_seqlens_q`、`seqused_kv` 数据类型仅支持 `INT32`，数据格式仅支持 ND。
  - `cu_seqlens_q` shape 为 `(BatchSize + 1)`，`seqused_kv` shape 为 `(BatchSize)`。
  - `cu_seqlens_kv` 和 `seqused_q` 为预留参数，必须传空（None）；传入非空 Tensor 时 host 侧将拦截并报错。

- 存在性约束

  - TND/NTD + PA_BNBD 场景必须传入 `cu_seqlens_q` 和 `seqused_kv`。
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
  - 有效 `metadata` 为一维 Tensor，shape 为 `(metadata_size)`，长度随分 section 结果动态变化，不应固定为 2048。
  - `metadata_size = 8 + section_num * 36 * 8 + 72 * 8`，其中第一个 8 为 head metadata 区长度，`section_num` 为实时分 section 数，生成后记录在 `metadata[0]`；36 为 AIC core 数，72 为 AIV core 数，每个 core 的元数据长度为 8 个 `INT32`。

- 存在性约束

  - `metadata` 在主算子 schema 中为可选输入，但当前执行路径必须传入有效 `metadata`；`metadata` 不能为 nullptr 或空 Tensor。应使用 `npu_quant_block_sparse_attn_metadata` 生成并传入。

- 一致性约束

  - `metadata` 必须由与主算子相同的 `sparse_seq_len`、`num_heads_q`、`num_heads_kv`、`head_dim`、`sparse_block_size_q`、`sparse_block_size_k`、`quant_mode`、`mask_mode`、`layout_q`、`layout_kv`、`layout_sparse_indices` 生成。
  - 用于主算子的 metadata 生成参数中，`head_dim`、`sparse_block_size_q`、`sparse_block_size_k` 均应为 128，`quant_mode` 应为 1，`mask_mode` 应为 0 或 3，`layout_sparse_indices` 应为 `B_N_Qb_Kb`。

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

  - `softmax_lse` 的 token/head 维度需与 `query` 的 N1 对齐。
  - 当 `sparse_seq_len` 对应行没有有效 KV block 时，该行 LSE 置为 `-FLT_MAX`。

- 特性交叉约束

  - `return_softmax_lse` 只影响 `softmax_lse` 是否返回有效结果，不影响 `attention_out` 的 shape、dtype 和计算语义。

## 调用示例

运行以下示例前，请先完成[前置条件](../tests/pytest/README.md#前置条件)中的环境准备。示例固定使用设备 0。
`npu_quant_block_sparse_attn_metadata` 用于生成 QBSA 负载均衡元数据。调用主算子前，需使用与主算子一致的稀疏、序列长度、分块及布局参数生成 metadata，并将其传入主算子的 `metadata` 参数。

```python
import torch
import torch_npu
import custom_ops

torch_npu.npu.set_device(0)
device = torch.device("npu:0")

B, N1, N2 = 1, 1, 1
S1, S2 = 128, 128
D = D_v = 128
block_size = 128
num_blocks = 1
head_dim = D
layout_q = "TND"
layout_kv = "PA_BNBD"
layout_sparse_indices = "B_N_Qb_Kb"
quant_mode = 1
mask_mode = 0


def make_kv_views(storage):
    key_segment = num_kv_heads * block_size * head_dim
    value_segment = num_kv_heads * block_size * head_dim
    k_descale_segment = num_kv_heads * block_size * 4
    block_stride = key_segment + value_segment + k_descale_segment
    key_stride = (block_stride, block_size * head_dim, head_dim, 1)
    k_descale_stride = (block_stride // 4, block_size, 1, 1)

    fp8_storage = storage.view(torch.float8_e4m3fn)
    key = torch.as_strided(
        fp8_storage,
        (num_blocks, num_kv_heads, block_size, head_dim),
        key_stride,
        0,
    )
    value = torch.as_strided(
        fp8_storage,
        (num_blocks, num_kv_heads, block_size, D_v),
        key_stride,
        key_segment,
    )
    fp32_storage = storage.view(torch.float32)
    k_descale = torch.as_strided(
        fp32_storage,
        (num_blocks, num_kv_heads, block_size, 1),
        k_descale_stride,
        (key_segment + value_segment) // 4,
    )
    return key, value, k_descale


num_kv_heads = N2
key_segment = num_kv_heads * block_size * head_dim
value_segment = num_kv_heads * block_size * head_dim
k_descale_segment = num_kv_heads * block_size * 4
block_stride = key_segment + value_segment + k_descale_segment
storage = torch.empty(num_blocks * block_stride, dtype=torch.uint8)
key_cpu, value_cpu, k_descale_cpu = make_kv_views(storage)
key_cpu.fill_(1)
value_cpu.fill_(1)
k_descale_cpu.fill_(1.0)
storage = storage.to(device)
key, value, k_descale = make_kv_views(storage)

query = torch.ones((S1, N1, D), dtype=torch.float32).to(torch.float8_e4m3fn).to(device)
q_descale = torch.ones((S1, N1), dtype=torch.float32, device=device)
v_descale = torch.ones((N2,), dtype=torch.float32, device=device)
p_scale = torch.ones((1,), dtype=torch.float32, device=device)
sparse_indices = torch.tensor([[[[0]]]], dtype=torch.int32, device=device)
sparse_seq_len = torch.ones((B, N1, S1 // block_size), dtype=torch.int32, device=device)
cu_seqlens_q = torch.tensor([0, S1], dtype=torch.int32, device=device)
seqused_kv = torch.tensor([S2], dtype=torch.int32, device=device)
block_table = torch.zeros((B, num_blocks), dtype=torch.int32, device=device)

metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
    sparse_seq_len,
    N1,
    N2,
    head_dim,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=seqused_kv,
    batch_size=B,
    sparse_block_size_q=block_size,
    sparse_block_size_k=block_size,
    quant_mode=quant_mode,
    mask_mode=mask_mode,
    layout_q=layout_q,
    layout_kv=layout_kv,
    layout_sparse_indices=layout_sparse_indices,
)

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
    None,
    1.0 / (D ** 0.5),
    block_size,
    block_size,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=seqused_kv,
    block_table=block_table,
    metadata=metadata,
    layout_kv=layout_kv,
    layout_q=layout_q,
    layout_sparse_indices=layout_sparse_indices,
    layout_out="TND",
    quant_mode=quant_mode,
    mask_mode=mask_mode,
    return_softmax_lse=True,
)

torch_npu.npu.synchronize()
print(f"attention_out: shape={tuple(attn_out.shape)}, dtype={attn_out.dtype}")
print(f"softmax_lse: shape={tuple(softmax_lse.shape)}, dtype={softmax_lse.dtype}")
```
