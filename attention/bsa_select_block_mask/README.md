# BSASelectBlockMask

## 产品支持情况

| 产品                                       | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term>     |    √    |
| <term>Atlas A3训练系列产品</term>           |    √    |
| <term>Atlas A3推理系列产品</term>           |    √    |
| <term>Atlas A2训练系列产品</term>           |    √    |
| <term>Atlas A2推理系列产品</term>           |    √    |
| <term>Atlas 200I/500 A2推理产品</term>      |    ×    |
| <term>Atlas 推理系列产品</term>              |    ×    |
| <term>Atlas 训练系列产品</term>              |    ×    |

## 功能说明

- 算子功能：aclnnBSASelectBlockMask是BSA（BlockSparseAttention）的前置算子，负责根据Query和Key的内容动态生成blockSparseMask，使BSA的调用链从"手动提供掩码"变为"根据Q/K内容自适应选择稀疏模式"。
- 计算公式：

  设blockShape = [blockShapeX, blockShapeY]，Sq是query最大序列长度，Skv是key最大序列长度, 则压缩后块数：

  $$
  Xblocks = \lceil Sq / blockShapeX \rceil,\quad Yblocks = \lceil Skv / blockShapeY \rceil
  $$

  **Step1：均值池化压缩 (Mean Pooling Compression)**

  当actualBlockLenQuery / actualBlockLenKey为null时（完整压缩）：

  $$
  q\_compressed[b, n, x, d] = \frac{1}{blockShapeX} \sum_{i=0}^{blockShapeX-1} query[b, n, x \cdot blockShapeX + i, d]
  $$

  $$
  k\_compressed[b, n, y, d] = \frac{1}{blockShapeY} \sum_{j=0}^{blockShapeY-1} key[b, n, y \cdot blockShapeY + j, d]
  $$

  当actualBlockLenQuery / actualBlockLenKey非null时（部分压缩），仅对每个block内前actualBlockLen个token取均值：

  $$
  q\_compressed[b, n, x, d] = \frac{1}{actualBlockLenQ[b,x]} \sum_{i=0}^{actualBlockLenQ[b,x]-1} query[b, n, x \cdot blockShapeX + i, d]
  $$

  $$
  k\_compressed[b, n, y, d] = \frac{1}{actualBlockLenK[b,y]} \sum_{j=0}^{actualBlockLenK[b,y]-1} key[b, n, y \cdot blockShapeY + j, d]
  $$

  **Step2a：QK Matmul**

  $$
  score[b, n, x, y] = scale \cdot \sum_{d=0}^{D-1} q\_compressed[b, n, x, d] \cdot k\_compressed[b, n, y, d]
  $$

  **Step2b：Softmax**

  $$
  attn\_score[b, n, x, y] = softmax(score[b, n, x, :]) = \frac{\exp(score[b, n, x, y] - m_{final})}{l_{final}}
  $$

  **Step3：TopK选择生成索引**

  $$
  topk\_value = \text{round}(sparsity \times Xblocks \times Yblocks)
  $$

  $$
  \mathcal{indices}= \text{TopK}\left(attn\_score[b, n, x, y],\; topK\_value\right)
  $$

  其中indices为attn\_score[b, n, x, y]中topk\_value个最大值对应的索引集合。

  **Step4：生成BlockSparseMask**
  $$
  blockSparseMaskOut[b, n, x, y] =
  \begin{cases}
  1 & (b, n, x, y) \in \mathcal{indices} \\
  0 & (b, n, x, y) \notin \mathcal{indices}
  \end{cases}
  $$

- 数据排布格式：

  BSASelectBlockMask输入query、key的数据排布格式支持从多种维度排布解读，可通过queryLayout和keyLayout传入。为了方便理解后续支持的具体排布格式（如BNSD、TND等），此处先对排布格式中各缩写字母所代表的维度含义进行统一说明：

  - B：表示输入样本批量大小（Batch）
  - T：B和S合轴紧密排列的长度（Total tokens）
  - S：表示输入样本序列长度（Seq-Length）
  - H：表示隐藏层的大小（Head-Size）
  - N：表示多头数（Head-Num）
  - D：表示隐藏层最小的单元尺寸，需满足D = H / N（Head-Dim）

- 当前支持的布局：

  - queryLayout: "TND" "BNSD"
  - keyLayout: "TND" "BNSD"

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|-----|-----------|------|----------|----------|
| query | 输入 | 注意力计算中的query矩阵，即公式中的`query`。 | FLOAT16、BFLOAT16 | ND |
| key | 输入 | 注意力计算中的key矩阵，即公式中的`key`。 | 数据类型与query保持一致 | ND |
| block_shape | 输入 | 稀疏块形状数组，指定每个稀疏块的二维尺寸（行数和列数），即公式中的`blockShape`。 | INT64 | - |
| post_block_shape | 输入 | 预留参数，用于Softmax后二次压缩。 | - | - |
| actual_seq_lengths | 输入 | 每个batch的query的实际序列长度。 | INT64 | - |
| actual_seq_lengths_kv | 输入 | key的实际序列长度。 | INT64 | - |
| actual_block_len_query | 输入 | 每个query block内实际压缩的有效seq长度，用于部分压缩场景, 即公式中的`actualBlockLenQuery`。 | INT64 | - |
| actual_block_len_key | 输入 | 每个key block内实际压缩的有效seq长度，用于部分压缩场景, 即公式中的`actualBlockLenKey`。 | INT64 | - |
| q_input_layout | 输入 | query的数据排布格式。指示输入张量在内存中的具体排布。 | String | - |
| kv_input_layout | 输入 | key的数据排布格式。指示输入张量在内存中的具体排布。 | String | - |
| num_key_value_heads | 输入 | key的注意力头数。 | Int | - |
| scale_value | 输入 | 缩放系数，即公式中的`scale`。| Float | - |
| sparsity | 输入 | 稀疏度保留比例。指定公式中attn_score中需要保留的块位置占全部块位置的比例。取值范围(0.0, 1.0)。 | Float | - |
| block_sparse_mask_out | 输出 | 块状稀疏掩码输出，即公式中的`blockSparseMaskOut`。 | INT8 | ND |

## 约束说明

- actual_seq_lengths在queryLayout为"TND"时必选；actual_seq_lengths_kv在keyLayout为"TND"时必选。
- 根据算子支持的输入Layout，query张量Shape中对应的head维度大小记为N1，key张量Shape中对应的head维度大小记为N2。必须满足N1 = N2（仅支持MHA）。
- headDim = 128。
- blockShapeX和blockShapeY必须为64的倍数。
- query和key压缩后，query和key对应的Xblocks和Yblocks需满足Xblocks * Yblocks > 1。
- query和key的数据类型必须一致，仅支持FLOAT16和BFLOAT16。
- block_sparse_mask_out数据类型为INT8（二值：0或1）。
- post_block_shape当前不支持，必须传入nullptr。
- actual_block_len_query / actual_block_len_key若非null，每个元素取值范围[0, blockShapeX] / [0, blockShapeY]；为null时完整压缩。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn API | [test_aclnn_bsa_select_block_mask](examples/test_aclnn_bsa_select_block_mask.cpp) | 只支持MHA和格式为TND\BNSD的场景，通过[aclnnBSASelectBlockMask](docs/aclnnBSASelectBlockMask.md)接口方式调用BSASelectBlockMask算子。 |
