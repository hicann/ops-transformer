# StemIndexer

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

-   API功能：StemIndexer是推理场景下稀疏attention的前处理算子，承担块级打分与动态选块职责。对每个query block，基于`qflat`与`kflat`的相关性并叠加value量值偏置`vbias`（OAM，Output-Aware Metric）进行打分，再按TPD（Token Position Decay）动态TopK预算选出关键key block，输出`sparse_indices`与`sparse_seq_len`供下游BlockSparseAttention消费。

-   计算公式：
    $$\text{score} = \text{qflat} \cdot \text{kflat}^{T} \cdot \text{rSquare} + \text{vbias}$$
    $$\text{rSquare} = \frac{1}{(\text{stem\_block\_size} / \text{stem\_stride})^{2}} = \frac{1}{(128/16)^{2}} = \frac{1}{64}$$
    主要计算过程为：
    1. `qflat`（Q块代表，$Q_{block}\in\R^{g\times d}$）与`kflat`（K块代表，$K_{block}\in\R^{S_{k}\times d}$）做矩阵乘，得到当前query block与各key block的相关性；上游每个代表聚合`stem_block_size/stem_stride=8`个token，Q×K含$8\times8=64$个token-pair，故除以`rSquare=1/64`归一化到单token-pair量级。
    2. 叠加`vbias`（value量值偏置，OAM项），避免纯QK相关性漏选对输出贡献较大的key block。
    3. 按TPD动态TopK预算（预算随query位置线性衰减）选出关键key block，并固定保留开头`initial_blocks`（sink）与末尾`window_size`（window）个块，得到输出索引`sparse_indices`与有效长度`sparse_seq_len`。

## 参数说明
| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| qflat                     | 输入      | Q块代表，shape为`[B, q_heads, max_Qb, stem_stride * D]`，不支持非连续。 | BF16 | ND         |
| kflat                   | 输入      | K块代表，shape为`[B, kv_heads, max_Kb, stem_stride * D]`，支持0轴非连续。 | BF16 | ND |
| vbias                 | 输入      | value量值偏置（OAM项），shape为`[B, kv_heads, max_Kb]`，不支持非连续。| FP32 | ND |
| q_seq_lens             | 输入      | 每个batch中query的有效token数。 | INT32     | ND         |
| kv_seq_lens            | 输入      | 每个batch中key的有效token数。 | INT32       | ND         |
| num_prompt_tokens                    | 输入      | prompt token数，用于TPD动态TopK预算分档。 | INT32       | ND         |
| metadata                    | 输入      | StemIndexerMetadata算子传入的分核调度信息，包含section数及每个核处理数据的起止点等内容。长度按最大section数动态计算并对齐到4096个INT32元素，当前不支持传空。 | INT32       | ND         |
| causal                 | 属性      | 是否因果（right-down causal mask），默认值true。 | BOOL          | -         |
| stem_block_size                 | 属性| block大小，当前仅支持128。 | INT32 | -         |
| stem_stride                 | 可选属性| 代表聚合stride，当前仅支持16。 | INT32 | -         |
| alpha      | 可选属性      | TPD衰减系数μ，用于计算预算衰减终点`k_end = k_start * alpha`，默认值1.0。 | FLOAT32          | -         |
| initial_blocks    | 可选属性      | 开头固定保留的sink块数，默认值4。 | INT32          | -         |
| window_size    | 可选属性      | 末尾固定保留的window块数，默认值4。 | INT32          | -         |
| k_block_num_rate_medium      | 可选属性      | 中等长度序列的TopK预算系数，默认值0.2。 | FLOAT32          | -         |
| k_block_num_bias_medium      | 可选属性      | 中等长度序列的TopK预算偏置，默认值30。 | INT32          | -         |
| k_block_num_rate_large      | 可选属性      | 长序列的TopK预算系数，默认值0.1。 | FLOAT32          | -         |
| k_block_num_bias_large      | 可选属性      | 长序列的TopK预算偏置，默认值30。 | INT32          | -         |
| topk_score_precision      | 可选属性      | TopK内部score精度：1表示uint32，2表示uint16，默认值1。 | INT32          | -         |
| sparse_indices     | 输出      | 选中的key block索引，shape为`[B, q_heads, max_Qb, max_Kb]`，仅前`sparse_seq_len`项有效，尾部无效区以-1填充。 | INT32          | ND         |
| sparse_seq_len           | 输出      | 每个query block选中的块数，shape为`[B, q_heads, max_Qb]`。 | INT32         | ND          |


## 约束说明
-   该接口支持图模式。
-   当前仅适配Ascend 950PR/Ascend 950DT（A5），不支持A3/A2。
-   `qflat`/`kflat`仅支持BF16，`vbias`仅支持FP32，`q_seq_lens`/`kv_seq_lens`/`num_prompt_tokens`/`metadata`及输出仅支持INT32。
-   `q_heads`仅支持32/64，`kv_heads`仅支持2/4/8，且需满足`q_heads % kv_heads == 0`（GQA）。
-   `headDim`固定为128；`stem_block_size`固定为128，`stem_stride`固定为16，因此`qflat`/`kflat`最后一维固定为`stem_stride * D = 16 * 128 = 2048`。
-   `metadata`最大section数按`B * kv_heads`计算，所需元素数为`16 + B * kv_heads * (36 + 72) * 16`，首维向上对齐到4096个INT32元素。
-   `initial_blocks`固定为4，`window_size`固定为4。
-   TopK预算分档：prompt block数较小时直接取prompt block数；中等长度使用`k_block_num_rate_medium * prompt_block + k_block_num_bias_medium`；长序列使用`k_block_num_rate_large * prompt_block + k_block_num_bias_large`；再按query位置与`alpha`线性衰减得到当前query block的动态TopK预算。
-   输出`sparse_indices`采用有效前缀契约：仅前`sparse_seq_len`项有效，尾部无效区在S2内层首轮以-1前置初始化，下游BlockSparseAttention只读取有效前缀。

## Ascend 950PR/Ascend 950DT 调用说明

-   单算子模式调用
    ```python
    import torch
    import TorchNPU
    import numpy as np
    import custom_ops   # 注册 torch.ops.custom.npu_stem_indexer(_metadata) 并挂载到 TorchNPU

    b = 4
    q_heads = 64
    kv_heads = 8
    g_size = q_heads // kv_heads
    d = 128
    stem_block_size = 128
    stem_stride = 16
    q_block_num = 16
    k_block_num = 64
    q_seq_len = q_block_num * stem_block_size
    kv_seq_len = k_block_num * stem_block_size
    num_prompt_tokens = kv_seq_len

    np.random.seed(0)
    qflat = torch.randn(b, q_heads, q_block_num, stem_stride * d, dtype=torch.bfloat16).npu()
    kflat = torch.randn(b, kv_heads, k_block_num, stem_stride * d, dtype=torch.bfloat16).npu()
    vbias = torch.randn(b, kv_heads, k_block_num, dtype=torch.float32).npu()
    q_seq_lens = torch.full((b,), q_seq_len, dtype=torch.int32).npu()
    kv_seq_lens = torch.full((b,), kv_seq_len, dtype=torch.int32).npu()
    num_prompt_tokens_tensor = torch.full((b,), num_prompt_tokens, dtype=torch.int32).npu()

    # 1. 前置分核：生成动态容量的分核调度 metadata
    metadata = torch.ops.custom.npu_stem_indexer_metadata(
        q_seq_lens, kv_seq_lens, q_heads, kv_heads,
        causal=True, stem_block_size=stem_block_size, dim_qkflat=stem_stride * d, window_size=4)

    # 2. 主算子：块级打分与动态选块
    sparse_indices, sparse_seq_len = torch.ops.custom.npu_stem_indexer(
        qflat, kflat, vbias, q_seq_lens, kv_seq_lens,
        num_prompt_tokens=num_prompt_tokens_tensor, metadata=metadata,
        causal=True, stem_block_size=stem_block_size, stem_stride=stem_stride, alpha=1.0,
        initial_blocks=4, window_size=4,
        k_block_num_rate_medium=0.2, k_block_num_bias_medium=30,
        k_block_num_rate_large=0.1, k_block_num_bias_large=30,
        topk_score_precision=1)
    ```

更多使用示例见[pytest示例](./tests/pytest/README.md)。
