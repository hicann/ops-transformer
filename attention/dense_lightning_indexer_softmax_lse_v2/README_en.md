# DenseLightningIndexerSoftmaxLseV2

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 training products</term>|      ×     |
|<term>Atlas A2 training products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: The DenseLightningIndexerSoftmaxLseV2 operator is a branch operator used by the DenseLightningIndexerGradKlLoss operator to compute Softmax inputs. Compared with DenseLightningIndexerSoftmaxLse, it adds support for Compressed Attention and supports core-level load balancing through a preceding metadata operator.

- Computation formula:

$$
\text{res}=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right)\right)
$$

$$
\text{lse}=\text{ReduceMax}\left(\text{res}\right)+\text{log}\left(\text{ReduceSum}\left(\text{exp}\left(\text{res}-\text{ReduceMax}\left(\text{res}\right)\right)\right)\right)
$$

lse is passed as an output to the DenseLightningIndexerGradKlLoss operator and used as an input for Softmax computation.

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
      <td>weight</td>
      <td>Input</td>
      <td>Weight tensor.</td>
      <td>FLOAT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>cuSeqLensQOptional</td>
      <td>Optional input</td>
      <td>Cumulative sum of the numbers of valid q tokens in the current batch and all preceding batches. Mandatory in TND scenarios. The first value is fixed to 0. Supports empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>cuSeqLensKOptional</td>
      <td>Optional input</td>
      <td>Cumulative sum of the numbers of valid k tokens in the current batch and all preceding batches. Mandatory in TND scenarios. The first value is fixed to 0. Supports empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>seqUsedQOptional</td>
      <td>Optional input</td>
      <td>Actually used length of q in different batches. Supports empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>seqUsedKOptional</td>
      <td>Optional input</td>
      <td>Actually used length of k in different batches. Supports empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>cmpResidualKOptional</td>
      <td>Optional input</td>
      <td>Represents the residual of the k sequence length related to cmpRatio. Mandatory when maskMode=3 and cmpRatio>1. Supports empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>metadataOptional</td>
      <td>Input</td>
      <td>Core-level load balancing information output by the preceding AICPU operator. It is output by the aclnnDenseLightningIndexerSoftmaxLseV2Metadata operator. Supports empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>layoutQ</td>
      <td>Attribute</td>
      <td>Represents the layout format of the query side. Supports BSND and TND. If a null pointer is passed, BSND is used. layoutQ must be consistent with layoutK.</td>
      <td>STRING</td>
      <td>-</td>
     </tr>
     <tr>
      <td>layoutK</td>
      <td>Attribute</td>
      <td>Represents the layout format of the key side. Supports BSND and TND. If a null pointer is passed, BSND is used. layoutK must be consistent with layoutQ.</td>
      <td>STRING</td>
      <td>-</td>
     </tr>
     <tr>
      <td>maskMode</td>
      <td>Attribute</td>
      <td>Represents the mask mode. 0: No mask; 3: Mask in rightDownCausal mode.</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
     <tr>
      <td>cmpRatio</td>
      <td>Attribute</td>
      <td>Represents the compression multiple of keys. Value range: [1, 128], indicating no compression.</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
     <tr>
      <td>softmaxLseOut</td>
      <td>Output</td>
      <td>LSE (log-sum-exp) value used for Softmax computation.</td>
      <td>FLOAT32</td>
      <td>ND</td>
     </tr>
     </tbody>
    </table>

## Constraints

  - In BSND scenarios, batch_size and max_seqlen_q related information must be passed (reflected in the tensor shape).
  - In TND scenarios, cuSeqLensQ and cuSeqLensK must be passed.
  - When maskMode=3 and cmpRatio>1, cmpResidualK must be passed, and each value of cmpResidualK must be less than cmpRatio.
  - Each value of seqUsedQ is not greater than the actual seqlen_q of each batch.
  - Each value of seqUsedK is not greater than the actual seqlen_k of each batch.

  <table style="undefined;table-layout: fixed; width: 909px"><colgroup>
  <col style="width: 125px">
  <col style="width: 182px">
  <col style="width: 602px">
  </colgroup>
  <thead>
  <tr>
    <th>Specification Item</th>
    <th>Specification</th>
    <th>Specification Description</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>B</td>
    <td>1~256</td>
    <td>-</td>
  </tr>
  <tr>
    <td>S1, S2</td>
    <td>0~128K</td>
    <td>S1 and S2 support different lengths.</td>
  </tr>
  <tr>
    <td>N1</td>
    <td>1~128</td>
    <td>Must be divisible by N2.</td>
  </tr>
  <tr>
    <td>N2</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>D</td>
    <td>128</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layout</td>
    <td>BSND/TND</td>
    <td>layoutQ must be consistent with layoutK.</td>
  </tr>
  <tr>
    <td>cmpRatio</td>
    <td>1~128</td>
    <td>1 indicates no compression.</td>
  </tr>
  <tr>
    <td>maskMode</td>
    <td>0, 3</td>
    <td>0=No mask, 3=rightDownCausal.</td>
  </tr>
  </tbody>
  </table>

## Calling Description

| Calling Method | Calling Sample                                                                          | Description                                                                  |
|--------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------|
| aclnn call | [test_aclnn_dense_lightning_indexer_softmax_lse_v2](examples/test_aclnn_dense_lightning_indexer_softmax_lse_v2.cpp) | Calls the dense_lightning_indexer_softmax_lse_v2 operator through the [aclnnDenseLightningIndexerSoftmaxLseV2](docs/aclnnDenseLightningIndexerSoftmaxLseV2_en.md) API. |
| PyTorch API | - | Calls the dense_lightning_indexer_softmax_lse operator through the [dense_lightning_indexer_softmax_lse](docs/torchapi_dense_lightning_indexer_softmax_lse_en.md) API. |
