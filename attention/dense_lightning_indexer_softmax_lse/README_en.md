# DenseLightningIndexerSoftmaxLse

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 training products</term>|      √     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: The DenseLightningIndexerSoftmaxLse operator is a branch operator used by the DenseLightningIndexerGradKlLoss operator to compute Softmax inputs.

- Computation formula:

$$
\text{res}=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right)\right)
$$

$$
\text{maxIndex}=\text{max}\left(res\right)
$$

$$
\text{sumIndex}=\text{ReduceSum}\left(\text{exp}\left(res-maxIndex\right)\right)
$$

maxIndex and sumIndex are passed as outputs to the DenseLightningIndexerGradKlLoss operator and used as inputs for Softmax computation.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>queryIndex</td>
      <td>Input</td>
      <td>Input queryIndex of the lightningIndexer structure.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>keyIndex</td>
      <td>Input</td>
      <td>Input keyIndex of the lightningIndexer structure.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>weights</td>
      <td>Input</td>
      <td>Weight.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>actualSeqLengthsQuery</td>
      <td>Input</td>
      <td>Number of valid Query tokens in each batch.</td>
      <td>INT64</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>actualSeqLengthsKey</td>
      <td>Input</td>
      <td>Number of valid Key tokens in each batch.</td>
      <td>INT64</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>layout</td>
      <td>Input</td>
      <td>Layout format.</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>sparseMode</td>
      <td>Input</td>
      <td>Sparse mode.</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
     <tr>
       <td>preTokens</td>
       <td>Input</td>
       <td>Used for sparse computation. Indicates the number of preceding tokens with which Attention computes correlation.</td>
       <td>INT64</td>
       <td>-</td>
      </tr>
     <tr>
       <td>nextTokens</td>
       <td>Input</td>
       <td>Used for sparse computation. Indicates the number of succeeding tokens with which Attention computes correlation.</td>
       <td>INT64</td>
       <td>-</td>
     </tr>
     <tr>
      <td>softmaxMaxOut</td>
      <td>Output</td>
      <td>Max value used for Softmax computation.</td>
      <td>FLOAT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>softmaxSumOut</td>
      <td>Output</td>
      <td>Sum value used for Softmax computation.</td>
      <td>FLOAT32</td>
      <td>ND</td>
     </tr>
     </tbody>
    </table>

## Constraints

None

## Calling Description

| Calling Method | Calling Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn call | [test_aclnn_dense_lightning_indexer_softmax_lse](examples/test_aclnn_dense_lightning_indexer_softmax_lse.cpp) | Calls the dense_lightning_indexer_softmax_lse operator through the [aclnnDenseLightningIndexerSoftmaxLse](docs/aclnnDenseLightningIndexerSoftmaxLse_en.md) API. |
