# GenericBlockSparseAttentionGradMetadata

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Description

+ Operator function: Based on the KV2Q sparse block index tables `sparseBlockIdx`/`sparseBlockCount`, GenericBlockSparseAttentionGradMetadata expands the task list in order according to the algorithm, balances the load across AIC cores, and provides the result for the subsequent GenericBlockSparseAttentionGrad operator to consume.

  - <term>Ascend 950PR/Ascend 950DT</term>:
    Expands the `(b, n2, j, g)` task list in the order of B → N2 → J → G, balances the load across AIC cores, and provides the result for the subsequent GenericBlockSparseAttentionGrad operator to consume.

  - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>:
    Expands the `(b, n2, j)` task list in the order of B → N2 → J, balances the load across AIC cores, and provides the result for the subsequent GenericBlockSparseAttentionGrad operator to consume.

+ This operator is not recommended for standalone use. It is recommended to use it together with aclnnGenericBlockSparseAttentionGrad.

+ Calculation formula of the metadata size:
  - <term>Ascend 950PR/Ascend 950DT</term>:

    $$
    \text{metaSize} = 80 + B \times N1 \times J \times 4
    $$

    where `J = ceilDiv(maxKvSeqlen, blockShapeY)`.

  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

    $$
    \text{metaSize} = 198
    $$

## Parameter Description

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Parameter Name</th>
    <th class="tg-0pky">Input/Output/Attribute</th>
    <th class="tg-0pky">Description</th>
    <th class="tg-0pky">Data Type</th>
    <th class="tg-0pky">Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">sparseBlockIdx</td>
    <td class="tg-0pky">Input</td>
    <td class="tg-0pky">Sparse block index array, specifying the indices of Q blocks/tokens selected by each KV block.</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">sparseBlockCount</td>
    <td class="tg-0pky">Input</td>
    <td class="tg-0pky">Specifies the number of Q blocks/tokens actually selected by each KV block.</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">cuSeqLengthsQOptional</td>
    <td class="tg-0pky">Optional Input</td>
    <td class="tg-0pky">Cumulative sequence length of query under the TND layout.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">cuSeqLengthsKvOptional</td>
    <td class="tg-0pky">Optional Input</td>
    <td class="tg-0pky">Cumulative sequence length of key/value under the TND layout.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">sequsedQOptional</td>
    <td class="tg-0pky">Optional Input</td>
    <td class="tg-0pky">Actual sequence length of query in each batch. This parameter takes effect only when the layout is TND; pass nullptr for BNSD/BSND, and the actual length is the same as maxQSeqlen/the S dimension of query.</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">sequsedKvOptional</td>
    <td class="tg-0pky">Optional Input</td>
    <td class="tg-0pky">Actual sequence length of kv in each batch. This parameter takes effect only when the layout is TND; pass nullptr for BNSD/BSND, and the actual length is taken from maxKvSeqlen (which must be the same as the S dimension of key/value).</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">maxQSeqlen</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Maximum value of the query sequence lengths in all batches.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">maxKvSeqlen</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Maximum value of the kv sequence lengths in all batches.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">numQHeads</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Number of heads of query (N1).</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">numKvHeads</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Number of heads of key/value (N2).</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">headDim</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Embedding dimension. Currently fixed to 128.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">blockShape</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Sparse block shape [blockShapeX, blockShapeY]. Currently [1, 128] is supported.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">layoutQ</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Layout format on the query side. TND/BNSD/BSND are supported.</td>
    <td class="tg-0pky">STRING</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">layoutKv</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Layout format on the key/value side. It must be the same as layoutQ.</td>
    <td class="tg-0pky">STRING</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">layoutSparsePattern</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Data layout of sparseBlockIdx and sparseBlockCount. Currently only 1 (BNKQ) is supported.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">maskType</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Mask mode. Currently only 1 (CAUSAL) is supported.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">softmaxPrecision</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Softmax precision level. Currently only 0 is supported.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">winLeft</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Number of tokens included forward in the sliding window. It must be -1 when this function is disabled.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">winRight</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Number of tokens included backward in the sliding window. It must be -1 when this function is disabled.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">residualBlockMode</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Processing mode of the trailing incomplete KV block. Currently only 0 is supported, where the sparse index determines whether the block participates in the calculation.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">isConsistentTopk</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Indicates whether the maximum number of KV blocks selected by each Q block within the same batch and the same head is consistent during preceding sparse selection. false/true are supported, and both values use the generic calculation path.</td>
    <td class="tg-0pky">BOOL</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">metadata</td>
    <td class="tg-0pky">Output</td>
    <td class="tg-0pky">Core allocation information, where the supported metaSize length varies by product. For details, see the constraints.</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
</tbody></table>

## Constraints

* This operator must be used together with aclnnGenericBlockSparseAttentionGrad; this operator must be successfully executed before the main operator is called.
* layoutQ and layoutKv must be the same, with the value TND/BNSD/BSND. cuSeqLengthsQOptional/cuSeqLengthsKvOptional are required under the TND layout.
* sequsedQOptional/sequsedKvOptional take effect only for TND; pass nullptr for BNSD/BSND, and the actual sequence lengths are taken from maxQSeqlen/maxKvSeqlen (which must be the same as the S dimensions of Q/K).
* HeadDim is fixed to 128. numQHeads/numKvHeads range from [1, 128], and numQHeads % numKvHeads == 0.
* blockShape: blockShapeX supports only 1, and blockShapeY must be greater than or equal to 128 and be a multiple of 64; layoutSparsePattern supports only 1 currently; residualBlockMode supports only 0 currently; maskType supports only 1 currently.
* winLeft/winRight must be -1 when they are disabled.
* The upper bound of the number of tasks is B × numQHeads × J ≤ 1048576, and the length of metadata must meet the following requirements:
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: the length of metadata ≥ 198.
  - <term>Ascend 950PR&950DT products</term>: the length of metadata must satisfy shape[0] ≥ 80 + B × numQHeads × J × 4.

## Calling Instructions

| Calling Method  | Sample Code                                                                | Description                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn API | [test_aclnn_generic_block_sparse_attention_grad_metadata](./examples/test_aclnn_generic_block_sparse_attention_grad_metadata.cpp) | Calls the GenericBlockSparseAttentionGradMetadata operator by using the [aclnnGenericBlockSparseAttentionGradMetadata](docs/aclnnGenericBlockSparseAttentionGradMetadata_en.md) API. |
