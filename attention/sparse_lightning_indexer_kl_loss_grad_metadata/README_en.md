# SparseLightningIndexerKLLossGradMetadata

English | [简体中文](./README.md)

## Product Support

| Product                                                      | Supported or Not |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: Based on the input shape, layout, mask, and compression ratio information of the `SparseLightningIndexerKLLossGrad` operator, the `SparseLightningIndexerKLLossGradMetadata` operator computes and outputs core-splitting metadata for use by the subsequent `SparseLightningIndexerKLLossGrad` operator.

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
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>cu_seqlens_q</td>
      <td>Optional Input</td>
      <td>Indicates the valid sequence length of q in different batches. The shape is (B+1, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_k</td>
      <td>Optional Input</td>
      <td>Indicates the valid sequence length of k in different batches. The shape is (B+1, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>Optional Input</td>
      <td>Indicates the sequence length of q that actually participates in the computation in different batches. The shape is (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_k</td>
      <td>Optional Input</td>
      <td>Indicates the sequence length of k that actually participates in the computation in different batches. The shape is (B, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_k</td>
      <td>Optional Input</td>
      <td>Indicates the remainder of the sequence length of cmp_k after compression in different batches. Together with cmp_ratio, it implements the mask and load computation for the cmp_k part. The shape is (B, ). This parameter is required when cmp_ratio is not 1 and mask_mode is 3.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>Attribute</td>
      <td>Indicates the number of heads of q. Currently, [1, 128] is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_k</td>
      <td>Attribute</td>
      <td>Indicates the number of heads of k. Currently, only 1 is supported</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>Attribute</td>
      <td>Indicates the dimension of the attention head. Currently, only 128 is supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>Optional Attribute</td>
      <td>Indicates the number of key sparse tokens selected from q. Currently, [1, 2048], 4096, and 8192 are supported.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>Optional Attribute</td>
      <td>Indicates the number of batches. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>Optional Attribute</td>
      <td>Indicates the maximum sequence length of q. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_k</td>
      <td>Optional Attribute</td>
      <td>Indicates the maximum sequence length of k. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>Optional Attribute</td>
      <td>Indicates the arrangement format of q. BSND and TND are supported. The default value is BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_k</td>
      <td>Optional Attribute</td>
      <td>Indicates the arrangement format of k. BSND and TND are supported. The default value is BSND.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>Optional Attribute</td>
      <td>Indicates the sparse mode. 0 indicates No mask, and 3 indicates the rightDownCausal mode. The default value is 0.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Optional Attribute</td>
      <td>Indicates the compression ratio of k. The value range is [1, 128]. The default value is 1, indicating no compression.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Indicates the load balancing result output. The shape is fixed to (64, ).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

  <ul>
    <li><term>Ascend 950PR&950DT products</term>: topk supports only [1, 2048].</li>
    <li><term>Atlas A3 products</term>: seqused_q, seqused_k, and cmp_residual_k are supported; num_heads_q supports only 8/16/32/64; topk supports only 512/1024/2048/4096/8192.</li>
    <li><term>Atlas A2 products</term>: seqused_q, seqused_k, and cmp_residual_k are supported; num_heads_q supports only 8/16/32/64; topk supports only 512/1024/2048/4096/8192.</li>
  </ul>

## Constraints

- The SparseLightningIndexerKLLossGradMetadata operator must be used together with the SparseLightningIndexerKLLossGrad operator.
- B (batch) indicates the batch size of input samples. q is an input parameter of the paired SparseLightningIndexerKLLossGrad operator, and S1 indicates the size of the S axis in the q shape when layout_q=BSND.
- layout_q and layout_k must be the same.
- The values of cu_seqlens_q and cu_seqlens_k must be the cumulative numbers of valid tokens of the current batch and preceding batches. The first element is fixed to 0, and the value of each subsequent element must be greater than or equal to that of the previous element.
- The values of seqused_q and seqused_k must indicate the number of valid tokens in each batch.
- When seqused_q, seqused_k, and cmp_residual_k are passed, each must be a one-dimensional INT32 tensor with shape (B,), and the length must equal the actual number of batches B.
- cmp_residual_k must satisfy cmp_residual_k[i] < cmp_ratio.
- For details about the mask modes indicated by mask_mode, see [sparse_mode Parameter Description](../../docs/zh/context/sparse_mode_introduction_en.md).
- Constraints for Ascend 950PR&950DT products:
  - layout_q=BSND scenario
    - max_seqlen_q must be passed the value of S1.
  - layout_q=TND scenario
    - cu_seqlens_q must be passed.
  - Batch value rules
    - When layout_q is BSND, the batch is derived preferentially from the shape of seqused_q; if seqused_q is not passed, the number of batches is obtained from batch_size.
    - When layout_q is TND, the batch is derived preferentially from the shape of seqused_q; if seqused_q is not passed, the batch is derived from the shape of cu_seqlens_q.
  - q Seqlen value rules
    - When layout_q is BSND, the seqlen is obtained preferentially from the elements in seqused_q; if seqused_q is not passed, the seqlen is obtained from max_seqlen_q.
    - When layout_q is TND, the seqlen is obtained preferentially from the elements in seqused_q; if seqused_q is not passed, the seqlen is obtained from the elements in cu_seqlens_q.

- Constraints for <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
  - Batch value rules
    - When batch_size is greater than 0, batch_size is used preferentially.
    - When batch_size is 0, the batch is derived preferentially from the shape of seqused_q; when seqused_q is not passed and layout_q is TND, the batch is derived from the shape of cu_seqlens_q.
    - An error is reported when a positive batch cannot be derived.
  - Valid sequence length rules
    - When seqused_q and seqused_k are passed, their corresponding elements are used as the actual valid lengths of q/k, and the values must not exceed the allocated lengths of the corresponding batches.
    - When the corresponding seqused parameter is not passed, max_seqlen_q/max_seqlen_k is used in the BSND scenario, and the difference between adjacent elements of the corresponding cu_seqlens is used in the TND scenario.

## Calling Instructions

| Calling Method | Sample Code                                                  | Description                                                  |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_sparse_lightning_indexer_kl_loss_grad_metadata](./examples/test_aclnn_sparse_lightning_indexer_kl_loss_grad_metadata.cpp) | The SparseLightningIndexerKLLossGradMetadata operator is called through the [aclnnSparseLightningIndexerKLLossGradMetadata](docs/aclnnSparseLightningIndexerKLLossGradMetadata_en.md) API. |
| PyTorch API | [test_torch_sparse_lightning_indexer_kl_loss_grad_metadata](./examples/test_torch_sparse_lightning_indexer_kl_loss_grad_metadata.py) | The SparseLightningIndexerKLLossGradMetadata operator is called through the [sparse_lightning_indexer_kl_loss_grad_metadata](../sparse_lightning_indexer_kl_loss_grad/docs/torchapi_sparse_lightning_indexer_kl_loss_grad_en.md) API. |
