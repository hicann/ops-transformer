# SparseFlashMlaSoftmaxL1NormMetadata

English | [简体中文](./README.md)

## Product Support

| Product                                                      | Supported |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

- Operator function: This operator is an AI CPU operator and a preceding operator of the `SparseFlashMlaSoftmaxL1Norm` operator. Based on the input shape, layout, mask, and compression ratio information of the `SparseFlashMlaSoftmaxL1Norm` operator, it computes and outputs core partitioning metadata, reducing host array access in the tiling phase of the main operator.

  **This operator is not recommended for standalone use. It is recommended to use this operator together with the `SparseFlashMlaSoftmaxL1Norm` operator to form a complete workflow.**

  Computation process:
  1. Receives the shape information of the main operator, including batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ, numHeadsK, headDim, topk, layout, and mask information.
  2. Computes the total number of seqs involved in load balancing based on the layout and the optional input seqUsedQOptional, and evenly partitions tasks to available AIC cores in a strided manner.
  3. Outputs metadata, which is used as the metadataOptional input of the `SparseFlashMlaSoftmaxL1Norm` operator.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ------ | -------------- | ---- | -------- | -------- |
| cu_seqlens_q | Optional input | Cumulative sum of the number of valid Query tokens in each batch. This parameter is required in the TND scenario, and the first value is fixed to 0. | INT32 | ND |
| cu_seqlens_k | Optional input | Cumulative sum of the number of valid Key tokens in each batch. This parameter is required in the TND scenario, and the first value is fixed to 0. | INT32 | ND |
| seqused_q | Optional input | Number of tokens of q that actually participate in the computation in each batch. When this parameter exists, the total number of seqs is accumulated based on the seqs actually used in each batch. | INT32 | ND |
| seqused_k | Optional input | Reserved API parameter, indicating the number of tokens of k that actually participate in the computation in each batch. The current kernel path does not use this parameter yet. | INT32 | ND |
| cmp_residual_k | Optional input | Reserved API parameter, indicating the remainder related to cmpRatio in the key sequence length in each batch. The current kernel path does not use this parameter yet. | INT32 | ND |
| topk_length | Optional input | Actual selectable topk length of k corresponding to each row of q. This parameter is required when maskMode=0 and sparse indices exist. | INT32 | ND |
| batch_size | Optional attribute | Number of batches. Non-negative numbers are supported. In the TND scenario, you can set this parameter to 0 and infer the value from cu_seqlens_q. The default value is 0. | INT64 | - |
| max_seqlen_q | Optional attribute | Maximum sequence length of q. Non-negative numbers are supported. This parameter must be a positive number in the BSND scenario. The default value is 0. | INT64 | - |
| max_seqlen_k | Optional attribute | Maximum sequence length of k. Non-negative numbers are supported. This parameter must be a positive number in the BSND scenario. The default value is 0. | INT64 | - |
| num_heads_q | Required attribute | Number of heads of q. This parameter must be a positive number and divisible by num_heads_k. | INT64 | - |
| num_heads_k | Required attribute | Number of heads of k. This parameter must be a positive number. | INT64 | - |
| head_dim | Required attribute | Head dimension of q/k. This parameter must be a positive number. | INT64 | - |
| topk | Optional attribute | Number of key tokens selected from k. Non-negative numbers are supported. 0 indicates no sparsity. The default value is 0. | INT64 | - |
| cmp_ratio | Optional attribute | Compression ratio of k. The value range is [1, 128]. The default value is 1. | INT64 | - |
| mask_mode | Optional attribute | Indicates the sparse mask mode. 0: No mask. 3: rightDownCausal mode. The default value is 0. | INT64 | - |
| layout_q | Optional attribute | Data layout format on the q side. BSND/TND is supported. The default value is "BSND". | STRING | - |
| layout_k | Optional attribute | Data layout format on the k side. BSND/TND is supported. The default value is "BSND". | STRING | - |
| metadata | Output | Load balancing result output. The shape is fixed to (64,). The output is used as the metadataOptional input of `SparseFlashMlaSoftmaxL1Norm`. | INT32 | ND |

## Constraints

- Deterministic computation: aclnnSparseFlashMlaSoftmaxL1NormMetadata uses the deterministic implementation by default.
- BSND scenario: batch_size, max_seqlen_q, and max_seqlen_k must be passed in to obtain shape information.
- TND scenario: cu_seqlens_q and cu_seqlens_k must be passed in to obtain correct shape information. When batch_size is 0, the batch is inferred from the shape of cu_seqlens_q.
- Layout constraints: layout_q must be BSND or TND, layout_k supports BSND and TND, and layout_k must be consistent with layout_q.
- Head constraints: num_heads_q and num_heads_k must be positive numbers, num_heads_q must be divisible by num_heads_k, and head_dim only supports 512.
- Load balancing constraints:
    - topk supports non-negative numbers, and 0 indicates no sparsity.
    - The value range of cmp_ratio is [1, 128].
    - Currently, mask_mode supports only 0 and 3.

## Troubleshooting

- For common execution problems or abnormal errors of the AI CPU operator Kernel, refer to the "[Fault Case Collection > Operator Execution Problems > AI CPU Operator Kernel Execution Errors](https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/troubleshooting/troubleshooting_0151.html)" section in the Troubleshooting documentation for the troubleshooting method.

## Metadata Output Layout

The metadata output is an INT32 tensor with the shape fixed to (64,). The field layout is as follows:

| Field | Index | Description |
| ---- | ----- | ---- |
| totalNum | 0 | Total number of seqs involved in load balancing. In the TND scenario, this value is the cumulative value of seqQ of all batches. In the BSND scenario, this value is B×S. When seqused_q exists, this value is accumulated based on the seqs actually used in each batch. |
| formerCoreProcessNum | 1 | Number of seqs processed by regular cores, that is, ceil(totalNum / totalCoreNum). |
| remainCoreProcessNum | 2 | Number of seqs processed by tail cores, that is, floor(totalNum / totalCoreNum). |
| remainCoreNum | 3 | Number of tail cores. This value is 0 when totalNum is divisible by totalCoreNum, and is totalCoreNum - (totalNum % totalCoreNum) otherwise. |
| totalCoreNum | 4 | Number of AIC cores actually used, which is min(totalNum, aicCoreNum, 36). |

The strided partitioning method is used. The seq indexes processed by each core are arranged in steps of totalCoreNum across cores.

## Calling Example

| Calling Method | Sample Code | Description |
| -------- | -------- | ---- |
| aclnn API | - | Calls the operator through the [aclnnSparseFlashMlaSoftmaxL1NormMetadata](docs/aclnnSparseFlashMlaSoftmaxL1NormMetadata_en.md) API, and outputs metadata for use by `SparseFlashMlaSoftmaxL1Norm`. |
| PyTorch API | - | Calls the operator through [sparse_flash_mla_softmax_l1_norm_metadata](../sparse_flash_mla_softmax_l1_norm/docs/torchapi_sparse_flash_mla_softmax_l1_norm_en.md) (built in the main operator API). |
