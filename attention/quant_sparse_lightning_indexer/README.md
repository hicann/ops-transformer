# QuantSparseLightningIndexer

两级 TopK（two-level TopK，先按块粗筛再在块内精筛）方案的**第二级**算子（consumer，候选消费级）。它必须与第一级算子 [QuantLightningIndexerV2](../quant_lightning_indexer_v2/README.md) 配合使用：第一级按块粒度从全序列中挑出候选块并输出块级索引 `candidate_block_indices`，本算子在候选块集合内做 token 级 TopK。

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
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

- 算子定位：本算子是两级 TopK 的第二级。第一级算子 `QuantLightningIndexerV2` 以 `candidate_topk_blocks=2048` 开启 source 后，把每个 token 的行（key 序列）按 `candidate_block_size` 个位置切块，块级打分后选出 `candidate_topk_blocks` 个候选块，输出块级索引 `candidate_block_indices`；本算子接收该索引，只在候选块覆盖的位置上做 token 级 TopK，输出最终的 `sparse_indices` 供稀疏 attention 使用。
- API功能：推理场景下稀疏 attention 前处理的计算，在候选块范围内选出关键的稀疏 token，并对输入 query 和 key 进行量化实现存 8 算 8，获取最大收益。
- 计算公式：与 QuantLightningIndexerV2 相同，区别在于 TopK 的作用域被限制在候选块内：

    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\},\quad p \in \text{候选块}$$

    块域定义：位置 $p$ 属于块 $\lfloor p / \text{candidate\_block\_size} \rfloor$；只有块号出现在 `candidate_block_indices` 中且满足有效序列长度和掩码约束的位置参与 TopK，其余位置的分数置为 $-\infty$；有效位置不足 Top-k 的槽位输出 `-1`。
    主要计算过程为：
    1. 将某个 token 对应的输入参数 `q`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文 `k`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与 `q` 和 `k` 对应的反量化系数 `q_descale`（$Scale_Q$）和 `k_descale`（$Scale_K^T$）相乘，通过激活函数 $ReLU$ 过滤无效负相关信号后，得到当前 token 与前序 token 的相关性分数向量。
    3. 分数向量与权重系数 `w`（$W$）相乘，沿 g 的方向求和，再将候选范围外或不满足序列长度和掩码约束的位置置为 $-\infty$，沿 Key 位置选取前 $Top-k$ 个索引值得到输出 $out$。

- 典型调用链：QuantLightningIndexerV2生成候选块索引和有效长度→QuantSparseLightningIndexerMetadata生成本级metadata→本算子计算候选范围内的token级索引→稀疏attention。

## 参数说明

| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| q | 输入 | 公式中的$Q_{index}^{Quant}\in\R^{g\times d}$，表示输入Index Query，不支持非连续。 | INT8、FLOAT4_e2m1 | ND |
| k | 输入 | 公式中的$K_{index}^{Quant}\in\R^{S_{k}\times d}$，表示压缩后的输入Index Key，支持0轴非连续。布局限制见约束说明。 | INT8、FLOAT4_e2m1 | ND |
| w | 输入 | 公式中的$W$，表示权重系数，不支持非连续。 | FLOAT16、FLOAT32 | ND |
| q_descale | 输入 | 公式中的$Scale_Q$，表示Index Query的反量化系数，不支持非连续。`quant_mode`为5时，shape为将`q`的D轴替换为(D/64, 2)；其他场景shape与w一致 | FLOAT16、FLOAT8_e8m0 | ND |
| k_descale | 输入 | 公式中的$Scale_K$，表示Index Key的反量化系数，支持0轴非连续。`quant_mode`为5时，shape为将`k`的D轴替换为(D/64, 2)；其他场景shape为移除`k`的D轴。分页存储要求见约束说明。 | FLOAT16、FLOAT8_e8m0 | ND |
| cu_seqlens_q                    | 可选输入      | layout_q为TND时必须传入，表示每个Batch中`q`的有效token数前缀和；layout_q为BSND时不能传入。默认值为None。 | INT32       | ND         |
| cu_seqlens_k                    | 可选输入      | layout_k为TND时必须传入，表示每个Batch中`k`的有效token数前缀和；layout_k为PA_BBND或BSND时不能传入。默认值为None。 | INT32       | ND         |
| seqused_q                    | 可选输入      | layout_q为BSND时可选传入，表示每个Batch中`q`的有效token数。默认值为None。 | INT32       | ND         |
| seqused_k | 可选输入 | layout_k为PA_BBND或BSND时使用，表示每个Batch中`k`的有效token数。默认值为None。 | INT32 | ND |
| cmp_residual_k                    | 可选输入      | 压缩场景下Key的残余长度，需满足0 \<= cmp_residual_k\[i\] \< cmp_ratio。默认值为None。 | INT32       | ND         |
| block_table | 可选输入 | 表示PageAttention中KV存储使用的block映射表。默认值为None。 | INT32 | ND |
| output_idx_offset                    | 可选输入      | 输出索引的偏移量，仅加到sparse_indices上。 | INT32       | ND         |
| candidate_block_length | 可选输入 | 每行实际有效候选块个数，传入规则见约束说明。 | INT32 | ND |
| metadata                    | 可选输入      | QuantSparseLightningIndexerMetadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容，shape为(1024,)。默认值为None。 | INT32       | ND         |
| candidate_block_indices | 输入 | 第一级算子QuantLightningIndexerV2输出的候选块索引，有效性要求见约束说明。 | INT32 | ND |
| topk | 属性 | 代表topK阶段需要保留的索引数量，默认值2048。 | INT32 | - |
| quant_mode | 属性 | 用于标识输入的量化模式。 | INT32 | - |
| max_seqlen_q                 | 可选属性| Query的最大序列长度，默认值-1表示任意可能长度 | INT32 | -         |
| layout_q                 | 可选属性| 用于标识输入`q`的数据排布格式，默认值"BSND"。 | STRING | -         |
| layout_k | 可选属性 | 用于标识输入`k`的数据排布格式，默认值"PA_BBND"。 | STRING | - |
| mask_mode | 可选属性      | 表示mask的模式，默认值0。 | INT32          | -         |
| cmp_ratio      | 可选属性      | 用于稀疏计算，表示k的压缩倍数，默认值1。 | INT32          | -         |
| return_value | 可选属性 | 表示是否输出`sparse_values`，默认值0。 | INT32 | - |
| candidate_block_size      |  可选属性     | 候选块大小（每个块包含的位置数），默认值为8，当前仅支持8 | INT32          | -         |
| sparse_indices     | 输出      | 参与稀疏attention计算的token索引值。BSND shape为(B,S1,N2,topk)，TND shape为(T,N2,topk) | INT32          | ND         |
| sparse_values | 输出 | 公式中的Indices输出对应的value值。`return_value`为1时shape与`sparse_indices`一致，`return_value`为0时shape为(0,) | BFLOAT16 | ND |

## 约束说明

- 本算子用于推理场景。
- candidate_block_indices必须传入，每个Query行的候选容量固定为2048。
- candidate_block_size仅支持8，必须与第一级算子保持一致。
- metadata必须传入且不能为空。
- output_idx_offset的维数必须比sparse_indices少1。

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - `quant_mode`仅支持5，即MXFP4量化。`q`和非分页`k`支持`float4_e2m1fn_x2`或打包的`uint8`存储，物理最后一维大小为64。
  - `w`仅支持`float32`。`q_descale`和非分页`k_descale`支持`float8_e8m0fnu`或等价的`uint8`存储，shape为对应`q`或`k`的shape去掉最后一维后追加`(2,2)`。
  - `q`的N支持[1,32]，`k`的N仅支持1，逻辑head_dim仅支持128。
  - `topk`支持[1,8192]。
  - `layout_q`支持TND、BSND，`layout_k`支持TND、BSND、PA_BBND；非PA场景下，`layout_q`和`layout_k`必须相同。
  - 当`layout_k`为PA_BBND时，`k`使用`uint8`存储，shape为`(block_num,block_size/8,544)`；`block_size`支持[16,1024]且必须为16的倍数。每8个token的512字节Key数据和32字节反量化系数合并存储，`k_descale`必须传入空`uint8` Tensor。
  - 当`layout_q`为TND时，必须传入`cu_seqlens_q`；当`layout_k`为TND时，必须传入`cu_seqlens_k`，且不支持传入`block_table`。
  - BSND布局不支持传入`cu_seqlens_q`和`cu_seqlens_k`；当`layout_k`为PA_BBND时，不支持传入`cu_seqlens_k`。
  - 当`layout_k`为PA_BBND且batch数大于1时，必须传入`block_table`；batch数为1时可不传入，按物理页顺序寻址。未传入`seqused_k`时，生成metadata所用的`max_seqlen_k`必须与计算接口的逻辑Key容量一致。
  - `q`、`w`、`q_descale`、`candidate_block_indices`和`candidate_block_length`必须连续。TND布局的`k`、`k_descale`及PA_BBND布局的`k`支持0轴非连续，其余轴必须连续，0轴stride不得小于单行或单页容量；BSND布局的`k`和`k_descale`必须连续。
  - `candidate_block_size`仅支持8。`candidate_block_indices`的shape为`(T,1,2048)`或`(B,S1,1,2048)`。
  - `candidate_block_length`必须传入，shape为`(T,1)`或`(B,S1,1)`，每个元素表示对应行的实际有效候选数，取值范围为[0,2048]。对应行前`candidate_block_length`个候选块索引必须合法且互不重复，不能以候选容量代替实际有效候选数。
  - `return_value`支持0、1。为0时，`sparse_values`为空Tensor；为1时，`sparse_values`的数据类型为BFLOAT16，shape与`sparse_indices`相同，分数与索引逐项对应，输出索引不保证有序。
  - 有效位置数小于`topk`时，`sparse_indices`的剩余位置填-1，`sparse_values`的对应位置填负无穷。`output_idx_offset`仅叠加到有效索引，无效索引保持-1。
  - `mask_mode`支持0、3，`cmp_ratio`支持[1,128]内的整数。当`mask_mode`为3且`cmp_ratio`不为1时，必须传入`cmp_residual_k`，其余情况不传入。
  - 所有输入Tensor必须位于同一NPU设备。取得第一级的候选索引和实际有效候选数后，需调用本级metadata接口生成分核信息，再传入计算接口。候选有效长度或序列信息变化后，必须重新生成metadata。
<!-- end id8 -->

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - `quant_mode`仅支持2。
  - `q`、`k`支持INT8，不支持FLOAT8_e4m3fn、HIFLOAT8和FLOAT4_e2m1。
  - `q_descale`和`k_descale`支持FLOAT16，不支持FLOAT32和FLOAT8_e8m0。
  - `w`支持FLOAT16，不支持FLOAT32。
  - 不支持`return_value=1`（`sparse_values`恒为shape (0,)）。
  - `k`的N仅支持1；`q`的N与`k`的N之比仅支持64或32（即`q`的N支持64或32）。
  - `topk`支持[1, 2048]。
  - `layout_k`仅支持`PA_BBND`，因此`block_table`与`seqused_k`必传，`cu_seqlens_k`不能传入。
  - `layout_q`支持`BSND`与`TND`；为`TND`时必须传入`cu_seqlens_q`，且`layout_k`必须为`PA_BBND`。
  - `candidate_block_length`为预留接口（本期不消费），仅接受空 tensor；传入非空 tensor 会在 tiling 报错。
  - q_descale的shape与w一致，k_descale的shape为k去掉最后一维。
  - cmp_ratio仅支持1、2、4、8、16、32、64、128。
  - 逻辑head_dim为128。PA_BBND布局下，k的shape为(block_num,block_size,1,128)，block_size为16的倍数；k和k_descale支持第0轴带padding，其余轴必须连续。
<!-- end id7 -->

## 调用示例

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| PyTorch API | [test_torch_quant_sparse_lightning_indexer](examples/test_torch_quant_sparse_lightning_indexer.py) | 通过[quant_sparse_lightning_indexer](docs/torchapi_quant_sparse_lightning_indexer.md)接口调用本算子。 |
| aclnn API | [test_aclnn_quant_sparse_lightning_indexer](examples/test_aclnn_quant_sparse_lightning_indexer.cpp) | 通过[aclnnQuantSparseLightningIndexer](docs/aclnnQuantSparseLightningIndexer.md)两段式接口调用本算子。 |
