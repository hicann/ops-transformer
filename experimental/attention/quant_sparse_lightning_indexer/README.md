# QuantSparseLightningIndexer

两级 TopK（two-level TopK，先按块粗筛再在块内精筛）方案的**第二级**算子（consumer，候选消费级）。它必须与第一级算子 [QuantLightningIndexerV2](../quant_lightning_indexer_v2/README.md) 配合使用：第一级按块粒度从全序列中挑出候选块并输出块级索引 `candidate_topk_index_out`，本算子在候选块集合内做 token 级 TopK。本算子由拆分提交 P13 从 `QuantLightningIndexerV2` 的 `candidate_mode=2` 路径独立而来，因此不再注册 `candidate_mode` 属性（模式固定为 consumer）。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

> Ascend 950PR/Ascend 950DT 不支持：本分支已移除 arch35 实现（`--soc=ascend950` 构建该算子会直接报错），def 只注册 ascend910b/ascend910_93，仅保留 arch22 (910B/910_93) 实现。

## 功能说明

- 算子定位：本算子是两级 TopK 的第二级。第一级算子 `QuantLightningIndexerV2` 以 `candidate_topk_blocks=2048` 开启 source 后，把每个 token 的行（key 序列）按 `candidate_block_size` 个位置切块，块级打分后选出 `candidate_topk_blocks` 个候选块，输出块级索引 `candidate_topk_index_out`；本算子接收该索引，只在候选块覆盖的位置上做 token 级 TopK，输出最终的 `sparse_indices` 供稀疏 attention 使用。
- API功能：推理场景下稀疏 attention 前处理的计算，在候选块范围内选出关键的稀疏 token，并对输入 query 和 key 进行量化实现存 8 算 8，获取最大收益。
- 计算公式：与 QuantLightningIndexerV2 相同，区别在于 TopK 的作用域被限制在候选块内：

    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\},\quad p \in \text{候选块}$$

    块域定义：位置 $p$ 属于块 $\lfloor p / \text{candidate\_block\_size} \rfloor$；只有块号出现在 `candidate_topk_index` 中的位置参与 TopK，其余位置的分数会被降级，不会进入 TopK 结果。
    主要计算过程为：
    1. 将某个 token 对应的输入参数 `q`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文 `k`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与 `q` 和 `k` 对应的反量化系数 `q_descale`（$Scale_Q$）和 `k_descale`（$Scale_K^T$）相乘，通过激活函数 $ReLU$ 过滤无效负相关信号后，得到当前 token 与前序 token 的相关性分数向量。
    3. 分数向量按候选块掩码降级后，与权重系数 `w`（$W$）相乘，沿 g 的方向选取前 $Top-k$ 个索引值得到输出 $out$。
- 典型调用链：QuantLightningIndexerV2Metadata（生成分核信息 `metadata`）→ QuantLightningIndexerV2（`candidate_topk_blocks=2048` 开启 source，输出 `candidate_topk_index_out`）→ 本算子（输入 `candidate_topk_index`）→ 稀疏 attention。

## 参数说明

| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| q                     | 输入      | 公式中的$Q_{index}^{Quant}\in\R^{g\times d}$，表示输入Index Query，不支持非连续。| INT8 | ND         |
| k                   | 输入      | 公式中的$K_{index}^{Quant}\in\R^{S_{k}\times d}$，表示压缩后的输入Index Key，`k`/`k_descale` 支持0轴padding型非连续（仅PA_BBND布局，其余轴必须连续），stride由输入描述符携带，kernel按块号×stride寻址。| INT8 | ND |
| w                 | 输入      | 公式中的$W$，表示权重系数，不支持非连续。 | FLOAT16 | ND         |
| q_descale             | 输入      | 公式中的$Scale_Q$，表示Index Query的反量化系数，不支持非连续，shape与`w`一致。 | FLOAT16 | ND         |
| k_descale            | 输入      | 公式中的$Scale_K$，表示Index Key的反量化系数，支持0轴非连续，shape为移除`k`的D轴。 | FLOAT16 | ND         |
| cu_seqlens_q                    | 可选输入      | layout_q为TND时必须传入，表示每个Batch中`q`的有效token数前缀和；layout_q为BSND时不能传入 | INT32       | ND         |
| cu_seqlens_k                    | 可选输入      | layout_k为PA_BBND时不能传入 | INT32       | ND         |
| seqused_q                    | 可选输入      | layout_q为BSND时可选传入，表示每个Batch中`q`的有效token数 | INT32       | ND         |
| seqused_k                    | 可选输入      | 表示每个Batch中`k`的有效token数；layout_k为PA_BBND时**必传** | INT32       | ND         |
| cmp_residual_k                    | 可选输入      | 压缩场景下Key的残余长度，需满足0 \<= cmp_residual_k\[i\] \< cmp_ratio。| INT32       | ND         |
| block_table                    | 可选输入      | 表示PageAttention中KV存储使用的block映射表；layout_k为PA_BBND时**必传** | INT32       | ND         |
| output_idx_offset                    | 可选输入      | 输出索引的偏移量，仅加到`sparse_indices`上；维数比`sparse_indices`少1：BSND为(B,S1,N2)，TND为(T,N2) | INT32       | ND         |
| candidate_block_length                    | 可选输入      | **预留接口**（N4/O13，本期不消费），仅接受空 tensor；传入非空 tensor 会在 tiling 报错 | INT32       | ND         |
| metadata                    | 输入（必选）      | QuantLightningIndexerV2Metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容，**不能为空** | INT32       | ND         |
| candidate_topk_index                    | 输入（必选）      | 第一级算子QuantLightningIndexerV2在开启 source（`candidate_topk_blocks=2048`）时输出的块级候选索引。**末维即候选块个数（当前仅支持 2048，不再是本算子的入参）**；元素个数必须等于BSND：B\*S1\*N2\*末维；TND：T\*N2\*末维（T为`q`第0维）。值域\[0, numBlocks)或-1（无效槽） | INT32       | ND         |
| topk  | 属性      | 代表topK阶段需要保留的索引数量，默认值2048。 | INT32          | -         |
| quant_mode                 | 属性      | 用于标识输入的量化模式，本算子**仅支持2**（INT8量化）。def默认值1在本算子非法 | INT32          | -         |
| max_seqlen_q                 | 可选属性| Query的最大序列长度，默认值-1表示任意可能长度 | INT32 | -         |
| layout_q                 | 可选属性| 用于标识输入`q`的数据排布格式，支持"BSND"/"TND"，默认值"BSND"。 | STRING | -         |
| layout_k      | 可选属性      | 用于标识输入`k`的数据排布格式，默认值"BSND"；910B/910_93上**仅支持"PA_BBND"**（按block_table分页寻址） | STRING          | -         |
| mask_mode | 可选属性      | 表示mask的模式，仅支持0（无mask）或3（下三角），默认值0。 | INT32          | -         |
| cmp_ratio      | 可选属性      | 用于稀疏计算，表示key的压缩倍数，默认值1。 | INT32          | -         |
| return_value      |  可选属性     | 表示是否输出`sparse_values`，默认值0；910B/910_93上仅支持0（不支持输出value） | INT32          | -         |
| candidate_block_size      |  可选属性     | 候选块大小（每个块包含的位置数），须 > 0，当前仅支持8 | INT32          | -         |
| sparse_indices     | 输出      | 参与稀疏attention计算的token索引值。BSND shape为(B,S1,N2,topk)，TND shape为(T,N2,topk) | INT32          | ND         |
| sparse_values           | 输出      | 与`sparse_indices`对应的value值。910B/910_93上`return_value`固定为0，shape为(0,) | BFLOAT16         | ND          |

## 约束说明

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 本算子当前仅在此两类产品（arch22）上可用。
  - `quant_mode`仅支持2。
  - `q`、`k`支持INT8，不支持FLOAT8_e4m3fn、HIFLOAT8和FLOAT4_e2m1。
  - `q_descale`和`k_descale`支持FLOAT16，不支持FLOAT32和FLOAT8_e8m0。
  - `w`支持FLOAT16，不支持FLOAT32。
  - 不支持`return_value=1`（`sparse_values`恒为shape (0,)）。
  - `k`的N仅支持1；`q`的N与`k`的N之比仅支持64或32（即`q`的N支持64或32）。
  - `topk`支持[1, 2048]。
  - `layout_k`仅支持`PA_BBND`，因此`block_table`与`seqused_k`必传，`cu_seqlens_k`不能传入。
  - `layout_q`支持`BSND`与`TND`；为`TND`时必须传入`cu_seqlens_q`，且`layout_k`必须为`PA_BBND`。
- 候选相关约束：
  - `candidate_topk_index`为必传输入，数据类型INT32；**末维即候选块个数（当前仅支持 2048）**，元素个数必须等于BSND：`B*S1*N2*末维`，TND：`T*N2*末维`，否则tiling报错。
  - 候选块个数不再是本算子入参：tiling 从 `candidate_topk_index` 末维推导并校验（累加器、抽取与拷出逻辑均按64对齐设计，且需与第一级输出的候选维度一致）。
  - `candidate_block_size`须 > 0，当前仅支持8（BlockReduceMax以32B块即8个float32为归约粒度）。
  - `candidate_block_length`为预留接口（N4/O13，本期不消费），仅接受空 tensor，传入非空值会在 tiling 报错。
  - `metadata`为必传输入，不能为空。
  - `output_idx_offset`的维数必须比`sparse_indices`少1。
- <term>Ascend 950PR/Ascend 950DT</term>：不支持（本分支已移除 arch35 实现）。

## 调用示例

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| PyTorch API | [quant_sparse_lightning_indexer.py](torch_extension/quant_sparse_lightning_indexer.py) | 通过`torch.ops.cann_ops_transformer.quant_sparse_lightning_indexer`调用本算子（入参为`candidate_topk_index`，即第一级算子输出的块级候选索引）。图模式暂不支持。 `metadata` 需先用 metadata 算子生成后传入（本扩展不跨算子自动构建）。 |
| aclnn API | [test_aclnn_quant_sparse_lightning_indexer](examples/test_aclnn_quant_sparse_lightning_indexer.cpp) | 通过[aclnnQuantSparseLightningIndexer](docs/aclnnQuantSparseLightningIndexer.md)两段式接口调用本算子。 |
