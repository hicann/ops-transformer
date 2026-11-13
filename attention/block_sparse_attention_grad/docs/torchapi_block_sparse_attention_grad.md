# block_sparse_attention_grad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

  `block_sparse_attention_grad`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`BlockSparseAttentionGrad`算子完成块级稀疏注意力反向计算。通过`block_sparse_mask`指定每个Q块需要参与计算的KV块，实现高效的稀疏注意力反向计算，并一次性返回`dq`、`dk`、`dv`。

  该接口为正向接口`block_sparse_attention`的反向传播：正向的输出`attentionOut`与`softmax_lse`作为反向输入参与计算。

- **计算公式**：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。稀疏块大小为$blockShapeX \times blockShapeY$，其中$blockShapeX$表示Q方向的稀疏粒度，$blockShapeY$表示KV方向的稀疏粒度，两个方向的粒度相互独立，可以配置为不同值。`block_sparse_mask`在逻辑块粒度上定义稀疏pattern，每个mask元素控制一个$blockShapeX \times blockShapeY$的连续块是否参与计算。

  已知正向计算公式为：

  $$
  attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T + atten\_mask) \cdot value_{sparse}
  $$

  为方便表达，以变量$S$和$P$表示计算公式：

  $$
  S = scale \cdot query \cdot key_{sparse}^T + atten\_mask
  $$

  $$
  P = SoftMax(S)
  $$

  $$
  V = value_{sparse}
  $$

  $$
  Out = PV
  $$

  其中$scale$为缩放系数，一般取$scale = D^{-0.5}$，$key_{sparse}$、$value_{sparse}$为`block_sparse_mask`选取的KV块。则反向计算公式为：

  $$
  softmax\_grad = softmaxGrad(dOut, attentionOut)
  $$

  $$
  dP = dOut \cdot V^T
  $$

  $$
  dS = P \cdot (dP - softmax\_grad)
  $$

  $$
  dV = P^T \cdot dOut
  $$

  $$
  dQ = (dS \cdot K) \cdot scale
  $$

  $$
  dK = (dS^T \cdot Q) \cdot scale
  $$

> [!NOTE]
>
> dOut、query、key、value、attentionOut数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、T表示B和S合轴紧密排列的长度（Total tokens）。Tq表示输入query tensor的总token数、Nq表示输入query tensor的头数、Tkv表示输入key/value tensor的总token数、Nkv表示输入key/value tensor的头数。

## 函数原型

```python
cann_ops_transformer.block_sparse_attention_grad(
    d_out,
    query,
    key,
    value,
    attention_out,
    softmax_lse,
    block_sparse_mask,
    block_shape=None,
    *,
    q_input_layout="BNSD",
    kv_input_layout="BNSD",
    num_key_value_heads=1,
    scale_value=1.0,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    atten_mask=None,
    mask_type=0
) -> (Tensor, Tensor, Tensor)
```

> [!NOTE]
>
> `*`之后的所有参数仅支持以关键字方式传入。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| d_out | Tensor | 必选 | 公式中的dOut，即最终输出对正向attentionOut的梯度。<br>支持的shape: <ul><li>TND: [Tq, Nq, D]</li><li>BNSD: [B, Nq, Sq, D]</li></ul> | float16/bfloat16 | ND | 3~4 | × |
| query | Tensor | 必选 | 公式中的query。<br>支持的shape: <ul><li>TND: [Tq, Nq, D]</li><li>BNSD: [B, Nq, Sq, D]</li></ul> | float16/bfloat16 | ND | 3~4 | × |
| key | Tensor | 必选 | 公式中的key。<br>支持的shape: <ul><li>TND: [Tq, Nq, D]</li><li>BNSD: [B, Nq, Sq, D]</li></ul> | float16/bfloat16 | ND | 3~4 | × |
| value | Tensor | 必选 | 公式中的value。<br>支持的shape: <ul><li>TND: [Tq, Nq, D]</li><li>BNSD: [B, Nq, Sq, D]</li></ul> | float16/bfloat16 | ND | 3~4 | × |
| attention_out | Tensor | 必选 | 正向BlockSparseAttention的输出，即公式中的attentionOut。<br>支持的shape: <ul><li>TND: [Tq, Nq, D]</li><li>BNSD: [B, Nq, Sq, D]</li></ul> | float16/bfloat16 | ND | 3~4 | × |
| softmax_lse | Tensor | 必选 | Softmax计算的log-sum-exp中间结果，用于反向计算时对数和指数的逆推。<br>支持的shape: <ul><li>TND: [Tq, Nq, 1]</li><li>BNSD: [B, Nq, Sq, 1]</li></ul> | float32 | ND | 3~4 | × |
| block_sparse_mask | Tensor | 必选 | 块级稀疏mask，1表示保留该块参与计算，0表示跳过该块。决定哪些block实际参与注意力计算。<br>支持的shape:[B, Nq, ceil(Sq/block_x), ceil(Skv/block_y)] | uint8/bool | ND | 4 | × |
| block_shape | list[int] | 可选 | 稀疏块形状，必须包含两个元素[block_x, block_y]，分别表示Q方向和KV方向的稀疏粒度；不传入时按[128, 128]处理 | int64 | - | 长度为2 | - |
| q_input_layout | string | 可选 | 输入query张量的数据排布格式，默认"BNSD" | string | - | - | - |
| kv_input_layout | string | 可选 | 输入key/value张量的数据排布格式，默认"BNSD" | string | - | - | - |
| num_key_value_heads | int | 可选 | key/value的head个数，默认1。用于支持GQA（分组查询注意力）机制下的头数比例映射 | int64 | - | - | - |
| scale_value | float | 可选 | 公式中的scale，默认1.0，一般设置为D^-0.5 | double | - | - | - |
| actual_seq_lengths | list[int] | TND时必选/其余可选 | 每个batch对应的query实际序列长度 | int64 | - | 长度为B | - |
| actual_seq_lengths_kv | list[int] | TND时必选/其余可选 | 每个batch对应的key/value实际序列长度 | int64 | - | 长度为B | - |
| atten_mask | Tensor | 可选 | 公式中的atten_mask。<ul><li>`mask_type=1`时必选，为块级变长掩码（INT32 counts）<br>支持的shape:[B, Nq, maxBlockNum, 2]，其中 $maxBlockNum=\max(\lceil Sq/block\_x\rceil, \lceil Skv/block\_y\rceil)$；</li><li>`mask_type=0`时必须为None。</li></ul> | int32 | ND | 4 | × |
| mask_type | int | 可选 | attention计算中的掩码类型，默认0。<ul><li>当mask_type=0，即为不加mask场景（默认值），忽略`atten_mask`，按`block_sparse_mask`给定的块级稀疏模式计算；</li><li>当mask_type=1，即为块级变长掩码场景，按块级变长掩码计算，需配置`atten_mask`（INT32 counts）</li></ul> | int64 | - | - | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| dq | Tensor | 必选 | 公式中的dQ，即query的梯度 | float16/bfloat16 | ND | 与query一致 | √ |
| dk | Tensor | 必选 | 公式中的dK，即key的梯度 | float16/bfloat16 | ND | 与key一致 | √ |
| dv | Tensor | 必选 | 公式中的dV，即value的梯度 | float16/bfloat16 | ND | 与value一致 | √ |

## 约束说明

- q_input_layout与kv_input_layout必须保持一致，且仅支持"TND"、"BNSD"。
- d_out、query、key、value、attention_out的数据类型必须一致，支持float16和bfloat16；dq、dk、dv的数据类型与query保持一致。
- softmax_lse的数据类型必须为float32，且其layout需要与query的layout保持一致。
- query、key、value的D轴（HeadDim）必须等于128。
- query的头数N1（由query的shape推断，接口不显式传入）与num_key_value_heads（N2）需满足 N1 >= N2 && N1 % N2 == 0。
- block_shape为可选参数，传入时必须包含两个元素[block_x, block_y]，元素要求如下：
  - block_x 和 block_y 分别为 Q、KV 方向的稀疏块大小，两者均须大于 0。Ascend 950 另要求两者均为 16 的整数倍。
  - 不传入时，算子kernel接口按[128, 128]处理。
- block_sparse_mask当前必须传入，且shape为[B, Nq, qBlockNum, kvBlockNum]，其中qBlockNum = ceilDiv(Sq, block_x)、kvBlockNum = ceilDiv(Skv, block_y)，其数据类型支持uint8或bool：
  - 为1表示该block参与计算，为0表示该block不参与计算；
  - BNSD传入actual_seq_lengths和actual_seq_lengths_kv时，后两维可按实际长度最大值或Q/K shape中的S计算，分别用block_x和block_y向上取整。
- qSeqlen和kvSeqlen不需要被block_shape整除，支持非对齐场景，实际分块数通过向上取整计算；当序列长度不能被对应方向的blockShape整除时，最后一个逻辑块覆盖该方向剩余的有效token。
- actual_seq_lengths在q_input_layout为"TND"时必选；actual_seq_lengths_kv在kv_input_layout为"TND"时必选。两者必须同时配置或同时不配置，仅配置其中之一将被拦截。
- TND布局下query、key、value、attention_out、d_out为3维；BNSD布局下为4维，且block_sparse_mask为4维。
- mask_type仅支持0或1：
  - `mask_type=0`：不加mask场景，忽略atten_mask，此时atten_mask必须为None。
  - `mask_type=1`：PerBlockSizeMask场景，按块级变长掩码计算，此时atten_mask必选，且必须为int32、shape为[B, Nq, M, 2]：
    - $M$ 为块级变长掩码的最大块数，取 $M=\max(\lceil Sq/block\_x\rceil, \lceil Skv/block\_y\rceil)$；
    - 第0维B必须与query的batch一致；当q_input_layout为"TND"时，B等于actual_seq_lengths的长度；
    - 第1维Nq必须等于query的head个数；
    - [..., 0]为Q方向actualSizeX，[..., 1]为KV方向actualSizeY，需满足 0 ≤ actualSize ≤ 对应blockShape；
    - 块内[0, actualSize)有效参与计算，[actualSize, blockShape)视为padding，actualSize=0表示该块不参与计算；
    - 第3维固定为2。

## 确定性计算

- 默认不支持确定性计算，可通过PyTorch开关（torch.use_deterministic_algorithms）支持

## 调用示例

- 基础示例（BNSD布局，float16，mask_type=0）

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)
    torch.manual_seed(0)

    B, Nq, Nkv, Sq, Skv, D = 1, 32, 32, 128, 128, 128
    block_x, block_y = 128, 128
    scale = 1.0 / (D ** 0.5)

    # BNSD排布
    query = torch.randn(B, Nq, Sq, D, dtype=torch.float16)
    key = torch.randn(B, Nkv, Skv, D, dtype=torch.float16)
    value = torch.randn(B, Nkv, Skv, D, dtype=torch.float16)

    # 最终输出对attentionOut的梯度
    d_out = torch.randn(B, Nq, Sq, D, dtype=torch.float16)

    # block_sparse_mask: [B, Nq, ceil(Sq/block_x), ceil(Skv/block_y)]
    block_sparse_mask = torch.ones(B, Nq, Sq // block_x, Skv // block_y, dtype=torch.uint8)

    block_shape = [block_x, block_y]

    # attention_out与softmax_lse为正向算子的输出，实际使用时应取自block_sparse_attention正向接口
    # （return_softmax_lse=True）；此处用等价的参考实现构造，保证示例可独立运行且三个入参自洽。
    scores = (query.float() @ key.float().transpose(-1, -2)) * scale
    attention_out = (torch.softmax(scores, dim=-1) @ value.float()).to(torch.float16)
    softmax_lse = torch.logsumexp(scores, dim=-1, keepdim=True)

    dq, dk, dv = cann_ops_transformer.block_sparse_attention_grad(
        d_out.npu(), query.npu(), key.npu(), value.npu(),
        attention_out.npu(), softmax_lse.npu(), block_sparse_mask.npu(),
        block_shape,
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=Nkv,
        scale_value=scale,
        mask_type=0,
    )
    torch_npu.npu.synchronize()
    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
    ```

- 块级变长掩码示例（BNSD布局，float16，mask_type=1）

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)
    torch.manual_seed(0)

    B, Nq, Nkv, Sq, Skv, D = 1, 32, 32, 128, 128, 128
    block_x, block_y = 64, 64
    max_block_num = max((Sq + block_x - 1) // block_x, (Skv + block_y - 1) // block_y)
    scale = 1.0 / (D ** 0.5)

    query = torch.randn(B, Nq, Sq, D, dtype=torch.float16)
    key = torch.randn(B, Nkv, Skv, D, dtype=torch.float16)
    value = torch.randn(B, Nkv, Skv, D, dtype=torch.float16)

    d_out = torch.randn(B, Nq, Sq, D, dtype=torch.float16)

    block_sparse_mask = torch.ones(
        B, Nq, (Sq + block_x - 1) // block_x, (Skv + block_y - 1) // block_y,
        dtype=torch.uint8
    )

    # atten_mask: 块内有效长度表（INT32），shape为[B, Nq, max_block_num, 2]
    # [..., 0]为Q方向actualSizeX，[..., 1]为KV方向actualSizeY，须满足0 <= actualSize <= 对应blockShape
    # 此例取满块长度；actualSize=0表示该块不参与计算
    atten_mask = torch.full((B, Nq, max_block_num, 2), block_x, dtype=torch.int32)
    atten_mask[..., 1] = block_y

    block_shape = [block_x, block_y]

    # attention_out与softmax_lse为正向算子的输出，实际使用时应取自block_sparse_attention正向接口
    # （return_softmax_lse=True）；此处用等价的参考实现构造，保证示例可独立运行且三个入参自洽。
    scores = (query.float() @ key.float().transpose(-1, -2)) * scale
    attention_out = (torch.softmax(scores, dim=-1) @ value.float()).to(torch.float16)
    softmax_lse = torch.logsumexp(scores, dim=-1, keepdim=True)

    dq, dk, dv = cann_ops_transformer.block_sparse_attention_grad(
        d_out.npu(), query.npu(), key.npu(), value.npu(),
        attention_out.npu(), softmax_lse.npu(), block_sparse_mask.npu(),
        block_shape,
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=Nkv,
        scale_value=scale,
        atten_mask=atten_mask.npu(),
        mask_type=1,
    )
    torch_npu.npu.synchronize()
    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
    ```

> [!NOTE] 说明
>
> - `attention_out`与`softmax_lse`必须与`query`/`key`/`value`自洽。实际使用时请直接使用`block_sparse_attention`正向接口在`return_softmax_lse=True`时的返回值，示例中的参考实现仅用于保证示例可独立运行。
