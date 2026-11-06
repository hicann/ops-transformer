# DenseLightningIndexerKLLossGradMetadata

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      x     |
|<term>Atlas A2 products</term>|      x     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: The `DenseLightningIndexerKLLossGradMetadata` operator computes and outputs core-splitting metadata based on the input shape, layout, mask, and compression ratio information of the `DenseLightningIndexerKLLossGrad` operator. The metadata is used by the subsequent `DenseLightningIndexerKLLossGrad` operator.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 180px">
  <col style="width: 100px">
  <col style="width: 700px">
  <col style="width: 90px">
  <col style="width: 80px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>cu_seqlens_q</td>
      <td>Optional input</td>
      <td>Valid Sequence Length of Query in different batches, with shape (B+1, ). Required only when layout_q is TND. The first value is fixed to 0.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_k</td>
      <td>Optional input</td>
      <td>Valid Sequence Length of Key in different batches, with shape (B+1, ). Required only when layout_k is TND. The first value is fixed to 0.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional input</td>
      <td>Sequence Length of Query that actually participates in the computation in different batches, with shape (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_k</td>
      <td>Optional input</td>
      <td>Sequence Length of Key that actually participates in the computation in different batches, with shape (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_k</td>
      <td>Optional input</td>
      <td>Remainder of the compressed Sequence Length of cmp_kv in different batches. Used together with cmp_ratio to implement the mask and workload computation of the cmp_kv part, with shape (B, ). Required when cmp_ratio is not 1 and mask_mode is 3.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>Attribute</td>
      <td>Number of Query heads. Currently [1, 128] is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_k</td>
      <td>Attribute</td>
      <td>Number of Key heads. Currently only 1 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>Attribute</td>
      <td>Dimension of the attention head. Currently only 128 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional attribute</td>
      <td>Number of batches. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional attribute</td>
      <td>Maximum Sequence Length of Query. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_k</td>
      <td>Optional attribute</td>
      <td>Maximum Sequence Length of Key. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional attribute</td>
      <td>Arrangement format of Query. BSND and TND are supported. The default value is BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_k</td>
      <td>Optional attribute</td>
      <td>Arrangement format of Key. BSND and TND are supported. The default value is BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>Optional attribute</td>
      <td>Mask mode. 0 indicates No mask, and 3 indicates rightDownCausal mode. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional attribute</td>
      <td>Compression ratio of Key. The value range is [1, 128]. The default value is 1, indicating no compression.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Load balancing result output. The shape is fixed to [64].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- The DenseLightningIndexerKLLossGradMetadata operator must be used together with the DenseLightningIndexerKLLossGrad operator.
- B (Batch) indicates the batch size of input samples.
- The values of parameters cu_seqlens_q and cu_seqlens_k must be the accumulated number of valid tokens in the current batch and all preceding batches. The value of each element must be greater than or equal to that of the preceding element.
- The values of parameters seqused_q and seqused_k indicate the number of valid tokens in each batch.
- Parameter cmp_residual_k must satisfy cmp_residual_k[i] < cmp_ratio.
- For details about the mask modes indicated by mask_mode, see [sparse_mode parameter description](../../docs/zh/context/sparse_mode_introduction_en.md).

## Problem Locating Instructions

- For common execution problems or exception errors of the AI CPU operator kernel, see "[Troubleshooting Cases > Operator Execution Problems > AI CPU Operator Kernel Execution Errors](https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/troubleshooting/troubleshooting_0151.html)" in the Troubleshooting guide for the problem locating method.

## Invocation Description

| Invocation Method  | Sample Code                                                  | Description                                                  |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_dense_lightning_indexer_kl_loss_grad_metadata](./examples/test_aclnn_dense_lightning_indexer_kl_loss_grad_metadata.cpp) | Calls the DenseLightningIndexerKLLossGradMetadata operator through the [aclnnDenseLightningIndexerKLLossGradMetadata](docs/aclnnDenseLightningIndexerKLLossGradMetadata_en.md) API. |
| PyTorch API | [test_torch_dense_lightning_indexer_kl_loss_grad_metadata](./examples/test_torch_dense_lightning_indexer_kl_loss_grad_metadata.py) | Calls the DenseLightningIndexerKLLossGradMetadata operator through the [dense_lightning_indexer_kl_loss_grad_metadata](../dense_lightning_indexer_kl_loss_grad/docs/torchapi_dense_lightning_indexer_kl_loss_grad_en.md) API. |
