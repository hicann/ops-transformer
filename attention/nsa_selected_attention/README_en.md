# NsaSelectedAttention

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |

## Function Description

- Operator function: Implements the computation of selected-attention in the NativeSparseAttention algorithm in the training scenario.

- Calculation formula:
  The forward computation formula of selected attention is as follows:

  $$
  selected\_key = Gather(key, topk\_indices[i]),0<=i<selected\_block\_count \\
  selected\_value = Gather(value, topk\_indices[i]),0<=i<selected\_block\_count
  $$

  $$
  attention\_out = Softmax(Mask(scale * (query @ selected\_key^T), atten\_mask)) @ selected\_value
  $$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| query | Input | Input query in the formula. | BFLOAT16, FLOAT16 | ND |
| key | Input | Input key in the formula. | BFLOAT16, FLOAT16 | ND |
| value | Input | Input value in the formula. | BFLOAT16, FLOAT16 | ND |
| topkIndices | Input | topk_indices in the formula, indicating the indices of the selected data. | INT32 | ND |
| attenMaskOptional | Optional Input | atten_mask in the formula, indicating the attention mask. The value 1 indicates that the bit does not participate in the computation, and the value 0 indicates that the bit participates in the computation. | BOOL, UINT8 | ND |
| scaleValue | Optional Attribute | scale in the formula, indicating the scaling coefficient.<br>The default value is 1.0. | DOUBLE | - |
| selectedBlockCount | Attribute | selected_block_count in the formula, indicating the number of select blocks. | INT64 | - |
| softmaxMaxOut | Output | Max intermediate result of the Softmax computation, used for backward computation. | FLOAT | ND |
| softmaxSumOut | Output | Sum intermediate result of the Softmax computation, used for backward computation. | FLOAT | ND |
| attentionOut | Output | attention_out in the formula. | BFLOAT16, FLOAT16 | ND |

## Constraints

- When this API is used together with PyTorch, ensure that the versions of the CANN-related packages match those of the PyTorch-related packages.
- The batch sizes of input query, key, and value must be the same. That is, actualSeqQLenOptional and actualSeqKvLenOptional passed in must have the same length.
- The D (Head-Dim) of input query, key, and value must satisfy (`D_q == D_k && D_k >= D_v`).
- The data types of input query, key, and value must be the same.
- The inputLayout of input query, key, and value must be the same.
- sparseMode currently supports 0 and 2.
- selectedBlockSize supports values <=128 and must be a multiple of 16.
- selectBlockCount: Supports [1~128]. The total selected size `selectBlockCount * selectBlockSize` < 128*64(8K)
- When the layout is TND, the S2 of each batch must be greater than the total selected size `selectBlockCount * selectBlockSize`
- inputLayout currently supports only TND.
- The N of input query and the N of key / value can be different but must be proportional. That is, N_q / N_kv must be a non-zero integer, called G (group), and `G <= 32` must be satisfied.
- When the input attenMaskOptional is nullptr, the sparseMode parameter does not take effect and the full computation is performed.
- For constraints on the data shape, take TND of inputLayout as an example (Note: T is the cumulative sum of the S lengths of all batches. When the S values of all batches are equal, `T = B * S`). Specifically:

  - B (Batchsize): The value range is 1\~1024.
  - N (Head-Num): The value range is 1\~128.
  - G (Group): The value range is 1\~32.
  - S (Seq-Length): The value range is 1\~128K. In addition, S_kv >= selectedBlockSize * selectedBlockCount must be satisfied, and the length of S_kv must be a multiple of selectedBlockSize.
  - D (Head-Dim): `D_qk=192` and `D_v=128`.

## Calling Description

| Calling Method           | Calling Sample                                                                                    | Description                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn call | [test_aclnn_nsa_selected_attention](./examples/test_aclnn_nsa_selected_attention.cpp) | In non-TND scenarios, calls the NsaSelectedAttention operator through the [aclnnNsaSelectedAttention](docs/aclnnNsaSelectedAttention_en.md) API.             |
