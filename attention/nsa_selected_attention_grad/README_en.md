# NsaSelectedAttentionGrad

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |

## Function Description

- Operator function: Rearranges data of the selectedBlockSize size selected from key and value based on topkIndices, and then computes the backward output of attention in the training scenario.

- Calculation formula:

    Rearranges selectedBlockCount pieces of data of the selectedBlockSize size selected from key and value based on the passed-in topkIndices. The formula is as follows:

    $$
    selectedKey = Gather(key, topkIndices[i]),0<=i<selectedBlockCount \\
    selectedValue = Gather(value, topkIndices[i]),0<=i<selectedBlockCount
    $$

    Then, performs the backward computation of the attention mechanism. The computation formulas are as follows:

    $$
    V=P^TdY
    $$

    $$
    Q=\frac{((dS)*K)}{\sqrt{d}}
    $$

    $$
    K=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| query | Input | Input Q in the formula. | BFLOAT16, FLOAT16 | ND |
| key | Input | Input K in the formula. | BFLOAT16, FLOAT16 | ND |
| value | Input | Input V in the formula. | BFLOAT16, FLOAT16 | ND |
| attentionOut | Optional Input | Final output of the forward attention computation. | BFLOAT16, FLOAT16 | ND |
| attentionOutGrad | Input | Input dY in the formula. | BFLOAT16, FLOAT16 | ND |
| softmaxMax | Input | Intermediate output of the forward attention computation. | FLOAT | ND |
| softmaxSum | Input | Intermediate output of the forward attention computation. | FLOAT | ND |
| topkIndices | Input | Indices of the selected KV in the formula. | INT32 | ND |
| scaleValue | Attribute | Reciprocal of the square root of d in the formula, representing the scaling coefficient. | DOUBLE | - |
| dqOut | Output | dQ in the formula, indicating the gradient of query. | BFLOAT16, FLOAT16 | ND |
| dkOut | Output | dK in the formula, indicating the gradient of key. | BFLOAT16, FLOAT16 | ND |
| dvOut | Output | dV in the formula, indicating the gradient of value. | BFLOAT16, FLOAT16 | ND |

## Constraints

- When this API is used together with PyTorch, ensure that the versions of the CANN-related packages match those of the PyTorch-related packages.
- The B (batchsize) of input query, key, value, attentionOut, and attentionOutGrad must be the same.
- The N (numHead) of input key and value must be the same.
- The N (numHead) of input query, attentionOut, and attentionOutGrad must be the same.
- The D (HeadDim) of input value, attentionOut, and attentionOutGrad must be the same.
- The inputLayout of input query, key, value, attentionOut, and attentionOutGrad must be the same.
- For constraints on the data shape, take TND of inputLayout as an example. Specifically:
  - T1: The value range is 1\~2M. T1 indicates the sum of S of all batches of query.
  - T2: The value range is 1\~2M. T2 indicates the sum of S of all batches of key and value.
  - B: The value range is 1\~2M.
  - N1: The value range is 1\~128. Indicates the headNum of query. N1 must be a multiple of N2.
  - N2: The value range is 1\~128. Indicates the headNum of key and value.
  - G: The value range is 1\~32. `G = N1 / N2`.
  - S: The value range is 1\~128K. The S of key and value must be greater than or equal to selectedBlockSize * selectedBlockCount and must be a multiple of selectedBlockSize.
  - D: The value range is 192 or 128. The D (HeadDim) of K and V can be different.
  - selectedBlockSize supports values <=128 and must be a multiple of 16.
  - selectBlockCount: Supports [1~128]. The total selected size `selectBlockCount * selectBlockSize` < 128*64(8K)
  - When the layout is TND, the S2 of each batch must be greater than the total selected size `selectBlockCount * selectBlockSize`
- Constraint on the shapes of the softmaxMax and softmaxSum parameters: \[T1, N1, 8\].
- Constraint on the shape of the topkIndices parameter: [T1, N2, selectedBlockCount].

## Calling Description

| Calling Method           | Calling Sample                                                                                    | Description                                                                                                 |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn call | [test_aclnn_nsa_selected_attention_grad](./examples/test_aclnn_nsa_selected_attention_grad.cpp) | In non-TND scenarios, calls the NsaSelectedAttentionGrad operator through the [aclnnNsaSelectedAttentionGrad](docs/aclnnNsaSelectedAttentionGrad_en.md) API.             |
