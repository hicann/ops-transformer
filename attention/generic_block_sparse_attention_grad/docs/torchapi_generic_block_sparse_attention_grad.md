# generic_block_sparse_attention_grad

## 产品支持情况

<!-- npu="950" id1 -->

- <term>Ascend 950PR/Ascend 950DT</term>：支持

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持

<!-- end id3 -->

<!-- npu="310b" id4 -->

- <term>Atlas 200I/500 A2 推理产品</term>：不支持

<!-- end id4 -->

<!-- npu="310p" id5 -->

- <term>Atlas 推理系列产品</term>：不支持

<!-- end id5 -->

<!-- npu="910" id6 -->

- <term>Atlas 训练系列产品</term>：不支持

<!-- end id6 -->

## 功能说明

- **接口功能**：

  `generic_block_sparse_attention_grad`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`GenericBlockSparseAttentionGrad`算子完成通用块稀疏注意力的反向梯度计算。通过`sparse_block_idx`指定每个KV块选择的Q块/token索引，`sparse_block_count`指定每个KV块实际保留的Q数量，仅在被选中的稀疏块上计算并回传`dq`/`dk`/`dv`。

  `generic_block_sparse_attention_grad_metadata`是`generic_block_sparse_attention_grad`的元数据生成接口，用于在主算子执行前生成`metadata`。`metadata`记录AICore任务切分与负载均衡结果，主算子须传入该`metadata`以优化调度。典型调用流程如下：

  1. 准备`q`、`k`、`v`、`dout`、`attn_out`、`softmax_lse`、`sparse_block_idx`、`sparse_block_count`、`cu_seqlens_q`、`cu_seqlens_kv`等输入。
  2. 调用`generic_block_sparse_attention_grad_metadata`生成`metadata`。
  3. 调用`generic_block_sparse_attention_grad`，将上一步得到的`metadata`传入主算子。
- **计算公式**：

  稀疏块大小为$blockShapeX \times blockShapeY$。依据`sparse_block_idx`/`sparse_block_count`选取参与计算的Q–KV块对后：

  $$
  P = SimpleSoftmax(Mask(Q @ selectedK^{T} \cdot scale), lse)
  $$

  $$
  dP = dO @ selectedV^{T}
  $$

  $$
  dS = P \odot (dP - SoftmaxGrad(dO, O))
  $$

  $$
  dQ = dS @ selectedK \cdot scale
  $$

  $$
  dK = dS^{T} @ Q \cdot scale
  $$

  $$
  dV = P^{T} @ dO
  $$

  其中$scale$由`softmax_scale`指定，建议值为$D^{-0.5}$。

> [!NOTE]
> q、k、v数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim。
> TND中的N为q的headNum时记为N1（Q_N），k/v的headNum记为N2（KV_N）。T1表示所有q Batch输入样本序列长度的累加和；T2表示所有kv Batch输入样本序列长度的累加和。
> $J = \lceil max\_seqlen\_kv / blockShapeY \rceil$。与正向`generic_block_sparse_attention`不同，本反向接口的`sparse_block_idx`/`sparse_block_count`为**KV→Q**方向（每个KV块选择的Q token），而非正向的Q→KV方向。

## 函数原型

调用`generic_block_sparse_attention_grad`接口之前，请先调用前置接口`generic_block_sparse_attention_grad_metadata`，完成负载均衡与分核信息的计算。

```python
cann_ops_transformer.generic_block_sparse_attention_grad_metadata(
    sparse_block_idx,
    sparse_block_count,
    num_heads_q,
    num_heads_kv,
    head_dim,
    block_shape,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="TND",
    mask_mode=MaskMode.CAUSAL,
    softmax_precision=0,
    win_left=-1,
    win_right=-1,
) -> Tensor
```

```python
cann_ops_transformer.generic_block_sparse_attention_grad(
    q,
    k,
    v,
    dout,
    attn_out,
    softmax_lse,
    sparse_block_idx,
    sparse_block_count,
    block_shape,
    *,
    metadata=None,
    attn_mask=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="TND",
    softmax_scale=1.0,
    mask_mode=MaskMode.CAUSAL,
    softmax_precision=0,
    win_left=-1,
    win_right=-1,
) -> (Tensor, Tensor, Tensor)
```

## 枚举说明

`mask_mode` 在 Python 接口中支持传入 `IntEnum` 枚举或对应 int 值，枚举定义于 `cann_ops_transformer.ops.generic_block_sparse_attention_grad`：

### mask_mode 枚举

| 枚举名     | 值 | 含义                              |
| :--------- | :-: | :-------------------------------- |
| `CAUSAL` | 1 | Causal 模式（当前默认且唯一支持） |

> [!NOTE]
> 枚举为 `IntEnum`，可直接作为 int 传入底层算子；接口支持传入枚举、对应 int 值或枚举名字符串。当前版本仅支持 `mask_mode = 1`（`CAUSAL`）。

## 参数说明

### generic_block_sparse_attention_grad_metadata

| 参数名             | 参数类型     | 可选/必选 | 描述                                                                                                | 数据类型 | 数据格式 | 维度                                                      |
| :----------------- | :----------- | :-------- | :-------------------------------------------------------------------------------------------------- | :------- | :------- | :-------------------------------------------------------- |
| sparse_block_idx   | Tensor       | 必选      | 稀疏块索引数组，指定每个KV块选择的Q块/token索引                                                     | int32    | ND       | TND：`(B, N2, J, maxS1)`；BNSD/BSND：`(B, N2, J, S1)` |
| sparse_block_count | Tensor       | 必选      | 指定每个KV块实际选择的Q数量                                                                         | int32    | ND       | `(B, N2, J)`                                            |
| num_heads_q        | int          | 必选      | Query head数（N1）                                                                                  | int32    | -        | -                                                         |
| num_heads_kv       | int          | 必选      | Key/Value head数（N2）                                                                              | int32    | -        | -                                                         |
| head_dim           | int          | 必选      | 每个注意力头的维度，当前固定128                                                                     | int32    | -        | -                                                         |
| block_shape        | list[int]    | 必选      | 稀疏块形状 [block_x, block_y]：block_x 仅支持 1；block_y 须 >=128 且为 64 的倍数                    | int64    | -        | 长度为2                                                   |
| cu_seqlens_q       | Tensor       | 可选      | q累积序列长度，第一个元素必须为0                                                                    | int64    | ND       | `(B+1,)`                                                |
| cu_seqlens_kv      | Tensor       | 可选      | k/value累积序列长度，第一个元素必须为0                                                              | int64    | ND       | `(B+1,)`                                                |
| seqused_q          | Tensor       | 可选      | 各batch中q实际使用的序列长度                                                                        | int32    | ND       | `(B,)`                                                  |
| seqused_kv         | Tensor       | 可选      | 各batch中kv实际使用的序列长度                                                                       | int32    | ND       | `(B,)`                                                  |
| max_seqlen_q       | int          | 可选      | 所有batch中q序列长度的最大值；省略时由扩展推断（优先`seqused_q`/`cu_seqlens_q`，否则取`sparse_block_idx`最后一维）；显式传入须≥0 | int32    | -        | -                                                         |
| max_seqlen_kv      | int          | 可选      | 所有batch中kv序列长度的最大值，用于计算$J$；省略时由扩展推断（优先`seqused_kv`/`cu_seqlens_kv`，否则取`J * block_y`）；显式传入须≥0 | int32    | -        | -                                                         |
| is_packed_gqa      | bool         | 可选      | 同一group内qHead是否共享稀疏pattern，当前仅支持`True`，默认`True`                               | bool     | -        | -                                                         |
| layout_q           | string       | 可选      | q布局，支持`"TND"`/`"BNSD"`/`"BSND"`，默认`"TND"`                                           | string   | -        | -                                                         |
| layout_kv          | string       | 可选      | k/value布局，须与`layout_q`一致，默认`"TND"`                                                    | string   | -        | -                                                         |
| mask_mode          | int/MaskMode | 可选      | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」。当前仅支持1（`CAUSAL`），默认1 | int32    | -        | -                                                         |
| softmax_precision  | int          | 可选      | Softmax精度级别，当前仅支持0，默认0                                                                   | int32    | -        | -                                                         |
| win_left           | int          | 可选      | 滑窗向前包含token数，不使能时必须为-1，默认-1                                                       | int32    | -        | -                                                         |
| win_right          | int          | 可选      | 滑窗向后包含token数，不使能时必须为-1，默认-1                                                       | int32    | -        | -                                                         |

### generic_block_sparse_attention_grad

| 参数名             | 参数类型     | 可选/必选 | 描述                                                                                  | 数据类型         | 数据格式 | 维度                                                                     |
| :----------------- | :----------- | :-------- | :------------------------------------------------------------------------------------ | :--------------- | :------- | :----------------------------------------------------------------------- |
| q                  | Tensor       | 必选      | 公式中的$Q$                                                                         | bfloat16/float16 | ND       | TND：`(T1, N1, D)`；BNSD：`(B, N1, S1, D)`；BSND：`(B, S1, N1, D)` |
| k                  | Tensor       | 必选      | 公式中的$K$                                                                         | bfloat16/float16 | ND       | TND：`(T2, N2, D)`；BNSD：`(B, N2, S2, D)`；BSND：`(B, S2, N2, D)` |
| v                  | Tensor       | 必选      | 公式中的$V$，shape与k相同                                                           | bfloat16/float16 | ND       | 与k一致                                                                  |
| dout               | Tensor       | 必选      | 注意力输出梯度$dO$                                                                  | bfloat16/float16 | ND       | 与q一致                                                                  |
| attn_out           | Tensor       | 必选      | 注意力正向输出$O$                                                                   | bfloat16/float16 | ND       | 与q一致                                                                  |
| softmax_lse        | Tensor       | 必选      | 正向保存的softmax LSE                                                                 | float32          | ND       | TND：`(T1, N1, 1)`；BNSD：`(B, N1, S1, 1)`；BSND：`(B, S1, N1, 1)` |
| sparse_block_idx   | Tensor       | 必选      | 稀疏块索引数组（KV→Q）                                                               | int32            | ND       | 与metadata接口一致                                                       |
| sparse_block_count | Tensor       | 必选      | 每个KV块实际选择的Q数量                                                               | int32            | ND       | 与metadata接口一致                                                       |
| metadata           | Tensor       | 可选      | `generic_block_sparse_attention_grad_metadata`生成的任务切分结果，传入后可优化调度；须通过关键字参数传入 | int32            | ND       | `(metaSize,)`，见返回值说明                                            |
| block_shape        | list[int]    | 必选      | 稀疏块形状 [block_x, block_y]：block_x 仅支持 1；block_y 须 >=128 且为 64 的倍数      | int64            | -        | 长度为2                                                                  |
| attn_mask          | Tensor       | 可选      | 掩码矩阵，当前暂不支持，应传`None`                                                  | bool             | ND       | -                                                                        |
| cu_seqlens_q       | Tensor       | 可选      | q累积序列长度；`layout_q="TND"`时必传                                               | int64            | ND       | `(B+1,)`                                                               |
| cu_seqlens_kv      | Tensor       | 可选      | kv累积序列长度；`layout_kv="TND"`时必传                                             | int64            | ND       | `(B+1,)`                                                               |
| seqused_q          | Tensor       | 可选      | 各batch中q实际使用的序列长度                                                          | int32            | ND       | `(B,)`                                                                 |
| seqused_kv         | Tensor       | 可选      | 各batch中kv实际使用的序列长度                                                         | int32            | ND       | `(B,)`                                                                 |
| is_packed_gqa      | bool         | 可选      | Packed GQA开关，当前仅支持`True`，默认`True`                                      | bool             | -        | -                                                                        |
| layout_q           | string       | 可选      | q布局，支持`"TND"`/`"BNSD"`/`"BSND"`，默认`"TND"`                             | string           | -        | -                                                                        |
| layout_kv          | string       | 可选      | k/value布局，须与`layout_q`一致，默认`"TND"`                                      | string           | -        | -                                                                        |
| softmax_scale      | float        | 可选      | 缩放因子，建议值$1/\sqrt{D}$，默认1.0                                               | float32          | -        | -                                                                        |
| mask_mode          | int/MaskMode | 可选      | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」。当前仅支持1，默认1 | int32            | -        | -                                                                        |
| softmax_precision  | int          | 可选      | Softmax精度级别，当前仅支持0，默认0                                           | int32            | -        | -                                                                        |
| win_left           | int          | 可选      | 滑窗向前包含token数，不使能时必须为-1，默认-1                                         | int32            | -        | -                                                                        |
| win_right          | int          | 可选      | 滑窗向后包含token数，不使能时必须为-1，默认-1                                         | int32            | -        | -                                                                        |

## 返回值说明

### generic_block_sparse_attention_grad_metadata

| 参数名   | 参数类型 | 可选/必选 | 描述                     | 数据类型 | 数据格式 | 维度            |
| :------- | :------- | :-------- | :----------------------- | :------- | :------- | :-------------- |
| metadata | Tensor   | 必选      | 主算子分核与负载均衡数据 | int32    | ND       | `(metaSize,)` |

其中：

$$
metaSize = 80 + B \times N1 \times J \times 4,\quad J = \lceil max\_seqlen\_kv / blockShapeY \rceil
$$

### generic_block_sparse_attention_grad

| 参数名 | 参数类型 | 可选/必选 | 描述    | 数据类型         | 数据格式 | 维度    |
| :----- | :------- | :-------- | :------ | :--------------- | :------- | :------ |
| dq     | Tensor   | 必选      | q的梯度 | bfloat16/float16 | ND       | 与q一致 |
| dk     | Tensor   | 必选      | k的梯度 | bfloat16/float16 | ND       | 与k一致 |
| dv     | Tensor   | 必选      | v的梯度 | bfloat16/float16 | ND       | 与v一致 |

**说明**

- `dq`/`dk`/`dv`的dtype与对应输入`q`/`k`/`v`保持一致。
- `metadata`长度须满足`shape[0] ≥ 80 + B × num_heads_q × J × 4`；任务数上界$B \times N1 \times J \le 1048576$。

## 约束说明

- 确定性计算：`generic_block_sparse_attention_grad`默认为非确定性实现，暂不支持确定性实现，确定性计算配置后不会生效。
- 参数`cu_seqlens_q`、`cu_seqlens_kv`、`seqused_q`、`seqused_kv`、`sparse_block_idx`、`sparse_block_count`属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- `generic_block_sparse_attention_grad_metadata`和`generic_block_sparse_attention_grad`的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由客户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- 先成功调用`generic_block_sparse_attention_grad_metadata`生成`metadata`，再传入主算子；
- Torch扩展中`max_seqlen_q`/`max_seqlen_kv`可省略：扩展在调用底层前推断出`≥0`的值再传入host；显式传入时须`≥0`，且仍须满足`J=ceilDiv(max_seqlen_kv, block_y)`与`maxS1≥max_seqlen_q`。

### 特性参数组

|   特性参数组   |    参数字段名称    |    字段分组    |   字段类型   |
| :------------: | :----------------: | :-------------: | :----------: |
|   公共参数组   |         q         |      INPUT      |    Tensor    |
|                |         k         |      INPUT      |    Tensor    |
|                |         v         |      INPUT      |    Tensor    |
|                |        dout        |      INPUT      |    Tensor    |
|                |      attn_out      |      INPUT      |    Tensor    |
|                |    softmax_lse    |      INPUT      |    Tensor    |
|                |  sparse_block_idx  |      INPUT      |    Tensor    |
|                | sparse_block_count |      INPUT      |    Tensor    |
|                |      metadata      |      INPUT      |    Tensor    |
|                |    block_shape    | ATTR(REQUIRED) |    int[]    |
|                |   softmax_scale   | ATTR(OPTIONAL) |    float    |
|                |      layout_q      | ATTR(OPTIONAL) |    string    |
|                |     layout_kv     | ATTR(OPTIONAL) |    string    |
|                |         dq         |     OUTPUT     |    Tensor    |
|                |         dk         |     OUTPUT     |    Tensor    |
|                |         dv         |     OUTPUT     |    Tensor    |
| metadata参数组 |    max_seqlen_q    | ATTR(OPTIONAL) |     int     |
|                |   max_seqlen_kv   | ATTR(OPTIONAL) |     int     |
|                |    num_heads_q    | ATTR(REQUIRED) |     int     |
|                |    num_heads_kv    | ATTR(REQUIRED) |     int     |
|                |      head_dim      | ATTR(REQUIRED) |     int     |
|                |      metadata      |     OUTPUT     |    Tensor    |
|   Mask参数组   |     mask_mode     | ATTR(OPTIONAL) | int/MaskMode |
|                |      win_left      | ATTR(OPTIONAL) |     int     |
|                |     win_right     | ATTR(OPTIONAL) |     int     |
|                |     attn_mask     | INPUT(OPTIONAL) |    Tensor    |
| SeqLens参数组 |    cu_seqlens_q    | INPUT(OPTIONAL) |    Tensor    |
|                |   cu_seqlens_kv   | INPUT(OPTIONAL) |    Tensor    |
|                |     seqused_q     | INPUT(OPTIONAL) |    Tensor    |
|                |     seqused_kv     | INPUT(OPTIONAL) |    Tensor    |
| Softmax参数组 | softmax_precision | ATTR(OPTIONAL) |     int     |
|                |   is_packed_gqa   | ATTR(OPTIONAL) |     bool     |

### 基准信息说明

资料约束中，常见字段释义如下：

|   命名   |                          含义                          |
| :-------: | :-----------------------------------------------------: |
|     B     |               Batch,表示输入样本批量大小               |
| N1 / Q_N |                   输入q tensor的头数                   |
| N2 / KV_N |                 输入key/v tensor的头数                 |
| S1 / Q_S |                      输入q序列长度                      |
| S2 / KV_S |                  输入key/value序列长度                  |
|    T1    |             输入q所有batch序列长度的累加和             |
|    T2    |             输入kv所有batch序列长度的累加和             |
|     D     |                  headdim，当前固定128                  |
|     J     | $\lceil max\_kv\_seqlen / blockShapeY \rceil$，KV块数 |
|   maxS1   | `sparse_block_idx`最后一维容量，须≥`max_seqlen_q` |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：

  - 空Tensor指必选输入和输出的shape size为0，即有任意轴为0。
  - 触发空Tensor的用例将全部拦截报错。
- q、k、v、dout、attn_out、dq、dk、dv校验:

  <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 147px">
    <col style="width: 232px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    </colgroup>
    <thead>
    <tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>q</td>
        <td>
            <ul>
                <li>tensor_type支持bfloat16、float16</li>
                <li>TND → (T1, N1, D)；BNSD → (B, N1, S1, D)；BSND → (B, S1, N1, D)</li>
            </ul>
        </td>
        <td rowspan="5">必须存在</td>
        <td rowspan="5">
            <ul>
                <li>q、k、v、dout、attn_out、dq、dk、dv的数据类型需相同</li>
                <li>dout、attn_out、dq的shape与q一致</li>
                <li>k、v、dk、dv的shape需相同</li>
                <li>layout_q与layout_kv须一致，支持TND/BNSD/BSND</li>
            </ul>
        </td>
        <td rowspan="5">
            轴校验：
            <ul>
                <li>当前D仅支持128</li>
                <li>N1、N2取值范围[1, 128]</li>
                <li>N1 % N2 == 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k / v</td>
        <td>
            <ul>
                <li>tensor_type支持bfloat16、float16</li>
                <li>TND → (T2, N2, D)；BNSD → (B, N2, S2, D)；BSND → (B, S2, N2, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>dout / attn_out</td>
        <td>
            <ul>
                <li>dtype与q一致</li>
                <li>shape与q一致</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>dq</td>
        <td>shape/dtype与q一致</td>
    </tr>
    <tr>
        <td>dk / dv</td>
        <td>shape/dtype与k一致</td>
    </tr>
    </tbody>
    </table>
- lse校验:

  <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 147px">
    <col style="width: 232px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    </colgroup>
    <thead>
    <tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>softmax_lse</td>
        <td>
            <ul>
                <li>tensor_type仅支持float32</li>
                <li>TND：(T1, N1, 1)；BNSD：(B, N1, S1, 1)；BSND：(B, S1, N1, 1)</li>
            </ul>
        </td>
        <td>必须存在</td>
        <td>head/seq轴语义须与q布局一致</td>
        <td>无</td>
    </tr>
    </tbody>
    </table>
- sparse_block_idx、sparse_block_count、block_shape校验:

  <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 147px">
    <col style="width: 232px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    </colgroup>
    <thead>
    <tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>sparse_block_idx</td>
        <td>
            <ul>
                <li>tensor_type仅支持int32</li>
                <li>TND：(B, N2, J, maxS1)；BNSD/BSND：(B, N2, J, S1)</li>
            </ul>
        </td>
        <td rowspan="3">必须存在</td>
        <td rowspan="3">
            <ul>
                <li>metadata接口与主算子须传入相同的sparse_block_idx、sparse_block_count、block_shape</li>
                <li>J须等于ceilDiv(max_seqlen_kv, block_y)</li>
                <li>maxS1 / S1须≥max_seqlen_q；且≥sparse_block_count中所有元素的最大值</li>
            </ul>
        </td>
        <td rowspan="3">
            <ul>
                <li>block_shape：block_x=1；block_y>=128 且为 64 的倍数</li>
                <li>is_packed_gqa当前仅支持True</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_block_count</td>
        <td>
            <ul>
                <li>tensor_type仅支持int32</li>
                <li>shape为(B, N2, J)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>block_shape</td>
        <td>
            <ul>
                <li>长度为2的int列表</li>
            </ul>
        </td>
    </tr>
    </tbody>
    </table>
- metadata校验:

  <table style="undefined;table-layout: fixed; width:1625px">
        <colgroup>
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 232px">
            <col style="width: 293px">
            <col style="width: 185px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>单参数校验</th>
                <th>存在性校验</th>
                <th>一致性校验</th>
                <th>特性交叉校验</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>metadata</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持int32</li>
                        <li>shape[0] ≥ 80 + B×N1×J×4</li>
                    </ul>
                </td>
                <td>必须存在</td>
                <td>须为generic_block_sparse_attention_grad_metadata的输出</td>
                <td>任务数上界B×N1×J ≤ 1048576</td>
            </tr>
        </tbody>
    </table>

#### Mask参数组

mask_mode参数解释：

- `MaskMode.CAUSAL`（1）：当前默认且唯一支持
- win_left / win_right：不使能时必须为-1；attn_mask当前应传`None`

#### SeqLens参数组

- `layout_q="TND"`时，`cu_seqlens_q`必选；`layout_kv="TND"`时，`cu_seqlens_kv`必选。
- BNSD/BSND布局下，上述累积序列长度传`None`。
- `cu_seqlens_q`/`cu_seqlens_kv`首元素必须为0，末元素分别等于T1/T2。

#### Softmax参数组

- `softmax_precision`当前仅支持0。
- `is_packed_gqa`当前仅支持`True`。

## 调用示例

- generic_block_sparse_attention_grad_metadata + generic_block_sparse_attention_grad联合调用示例（BNSD）

  ```python
  import math
  import torch
  import torch_npu
  import cann_ops_transformer
  from cann_ops_transformer.ops.generic_block_sparse_attention_grad import MaskMode

  torch_npu.npu.set_device(0)

  B, N1, N2, S1, S2, D = 1, 1, 1, 128, 128, 128
  block_x, block_y = 1, 128
  J = math.ceil(S2 / block_y)
  scale = 1.0 / math.sqrt(D)
  block_shape = [block_x, block_y]

  q = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
  k = torch.randn(B, N2, S2, D, dtype=torch.float16, device="npu")
  v = torch.randn(B, N2, S2, D, dtype=torch.float16, device="npu")
  dout = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
  attn_out = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
  softmax_lse = torch.randn(B, N1, S1, 1, dtype=torch.float32, device="npu")

  sparse_block_idx = torch.full((B, N2, J, S1), -1, dtype=torch.int32, device="npu")
  sparse_block_count = torch.zeros((B, N2, J), dtype=torch.int32, device="npu")
  for qi in range(S1):
      sparse_block_idx[0, 0, 0, qi] = qi
  sparse_block_count[0, 0, 0] = S1

  metadata = cann_ops_transformer.generic_block_sparse_attention_grad_metadata(
      sparse_block_idx,
      sparse_block_count,
      N1,
      N2,
      D,
      block_shape,
      is_packed_gqa=True,
      layout_q="BNSD",
      layout_kv="BNSD",
      mask_mode=MaskMode.CAUSAL,
      softmax_precision=0,
      win_left=-1,
      win_right=-1,
  )

  dq, dk, dv = cann_ops_transformer.generic_block_sparse_attention_grad(
      q,
      k,
      v,
      dout,
      attn_out,
      softmax_lse,
      sparse_block_idx,
      sparse_block_count,
      block_shape,
      metadata=metadata,
      attn_mask=None,
      is_packed_gqa=True,
      layout_q="BNSD",
      layout_kv="BNSD",
      softmax_scale=scale,
      mask_mode=MaskMode.CAUSAL,
      softmax_precision=0,
      win_left=-1,
      win_right=-1,
  )
  torch_npu.npu.synchronize()
  assert dq.shape == q.shape
  assert dk.shape == k.shape
  assert dv.shape == v.shape
  assert dq.dtype == q.dtype
  ```
