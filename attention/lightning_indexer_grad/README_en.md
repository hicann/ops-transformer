# LightningIndexerGrad

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

- Operator function: `lightning_indexer_grad` is the backward operator of `lightning_indexer`. It computes the gradients of `query`, `key`, and `weights` based on the output `sparseIndices` of the forward operator.

- Computation formula:
  The backward computation formulas of LightningIndexer are as follows:

  $$
  S = Relu(Matmul(Query, Gather(Key, Indices)))
  $$

  $$
  Y = Dy*Weights
  $$

  $$
  dW = Reduce(S * dy)
  $$

  $$
  dQ = Matmul(ReluGrad(Y, S), Gather(Key, Indices))
  $$

  $$
  dK = ScatterAdd(Matmul(ReluGrad(Y, S), Q), Indices)
  $$

## Parameter Description

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 150px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>Input</td>
      <td>Input Q in the formula. Empty tensors and non-contiguous tensors are not supported. When layout is BSND, the shape is (B,S1,N1,D). When layout is TND, the shape is (T1,N1,D).</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Input K in the formula. Empty tensors and non-contiguous tensors are not supported. When layout is BSND, the shape is (B,S2,N2,D). When layout is TND, the shape is (T2,N2,D).</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Input dY in the formula, indicating the output gradient. Empty tensors and non-contiguous tensors are not supported. When layout is BSND, the shape is (B,S1,N1,D). When layout is TND, the shape is (T1,N1,D).</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseIndices</td>
      <td>Input</td>
      <td>Input Indices in the formula. It is the sparseIndicesOut output of the forward LightningIndexer operator. Empty tensors and non-contiguous tensors are not supported. When layout is BSND, the shape is (B,S1,N2,K). When layout is TND, the shape is (T1,N2,K), where K is the number of blocks retained by topK.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>Input</td>
      <td>Input W in the formula. Empty tensors and non-contiguous tensors are not supported. When layout is BSND, the shape is (B,S1,N1). When layout is TND, the shape is (T1,N1).</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQuery</td>
      <td>Input</td>
      <td>Number of valid tokens of Query in each batch. Empty tensors and non-contiguous tensors are not supported. None can be passed to indicate that the value is the same as the S length of query. A 1D tensor with length B is supported, and the number of valid tokens in each batch must not exceed the dimension S size of query and must not be less than 0. When layout is TND, this parameter must be passed, and the number of elements is used as the value of B. Each element indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKey</td>
      <td>Input</td>
      <td>Number of valid tokens of Key in each batch. Empty tensors and non-contiguous tensors are not supported. None can be passed to indicate that the value is the same as the S length of key. A 1D tensor with length B is supported, and the number of valid tokens in each batch must not exceed the dimension S size of key and must not be less than 0. When layout is TND, this parameter must be passed. Each element indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>Input</td>
      <td>Identifies the data layout format of the input Query/Key. The default value is "BSND". Currently, BSND and TND are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNum</td>
      <td>Input</td>
      <td>Represents the number of heads.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Input</td>
      <td>Represents the sparse mode. When sparse_mode is 0, the defaultMask mode is used. When sparse_mode is 3, the mask of the rightDownCausal mode is used, corresponding to the lower triangular scenario divided by the right vertex.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>Input</td>
      <td>Used for sparse computation. It indicates the number of previous tokens that attention needs to compute correlations with. Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>Input</td>
      <td>Used for sparse computation. It indicates the number of following tokens that attention needs to compute correlations with. Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dQuery</td>
      <td>Output</td>
      <td>dQ output in the formula, indicating the gradient of query. Empty tensors and non-contiguous tensors are not supported. The data type is the same as that of query, and the shape is the same as that of query.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dKey</td>
      <td>Output</td>
      <td>dK output in the formula, indicating the gradient of key. Empty tensors and non-contiguous tensors are not supported. The data type is the same as that of key, and the shape is the same as that of key.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dWeights</td>
      <td>Output</td>
      <td>dW output in the formula, indicating the gradient of weights. Empty tensors and non-contiguous tensors are not supported. The data type is the same as that of weights, and the shape is the same as that of weights.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>deterministic</td>
      <td>Input</td>
      <td>Specifies whether deterministic computation is currently supported. The default value is False.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## Constraints

- inputLayout supports TND/BSND.
- For constraints on data shapes, BSND of Layout is used as an example:

  - B (Batchsize): the value range is 1\~1024.
  - N (Head-Num): the value is 1\~64.
  - G (Group): the value is N.
  - S1 (Seq-LengthQ): the value range is 1\~128K.
  - S2 (Seq-LengthK): the value range is topK\~128K.
  - D (Head-Dim): the value is 128.
  - TopK: the value is 2048.

## Calling Example

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Calling Method</th>
    <th class="tg-0pky">Sample Code</th>
    <th class="tg-0pky">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn API</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_lightning_indexer_grad.cpp">test_aclnn_lightning_indexer_grad
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    Invokes the operator through the
    <a href="./docs/aclnnLightningIndexerGrad.md">aclnnLightningIndexerGrad
    </a>
    API
    </td>
  </tr>
</tbody></table>
