# NsaCompressAttention

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |

## Function Description

- **Operator function**: Computes the compress attention and the select topk index calculation in NSA. Paper: <https://arxiv.org/pdf/2502.11089>

- **Computation formula**: compression block size: $l$, select block size: $l'$, compression stride size: $d$

$$
P_{cmp} = Softmax(query*key^T) \\
$$

$$
attentionOut = Softmax(atten\_mask(scale*query*key^T, atten\_mask))*value
$$

$$
P_{slc}[j] = \sum_{m=0}^{l'/d-1}\sum_{n=0}^{l/d-1}P_{cmp} [l'/d*j-m-n],
$$

$$
P_{slc'} = \sum_{h=1}^{H}P_{slc}^{h}
$$

$$
P_{slc'} = topk\_mask(P_{slc'})
$$

$$
topkIndices = topk(P_{slc'})
$$

The data layout formats of the query, key, and value inputs of NsaCompressAttention can be interpreted from multiple dimensions. The format can be passed in through inputLayout. Currently, only TND is supported.

- B: Indicates the batch size of input samples (Batch)
- T: The length formed by tightly combining B and S
- S: Indicates the sequence length of input samples (Seq-Length)
- H: Indicates the size of the hidden layer (Head-Size)
- N: Indicates the number of heads (Head-Num)
- D: Indicates the size of the smallest unit of the hidden layer, which must satisfy D=H/N (Head-Dim)

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| query | Input | query in the formula. | BFLOAT16, FLOAT16 | ND |
| key | Input | key in the formula. | BFLOAT16, FLOAT16 | ND |
| value | Input | value in the formula. | BFLOAT16, FLOAT16 | ND |
| attenMaskOptional | Input | atten_mask in the formula. | BOOL | ND |
| actualSelSeqKvLenOptional | Input | The 0th dimension of P_slc' in the formula. | INT64 | ND |
| topkMaskOptional | Input | topk_mask in the formula. | BOOL | ND |
| scaleValue | Input | scale in the formula, the scaling coefficient, usually D^-0.5. | DOUBLE | - |
| compressBlockSize | Input | l in the formula, the compression sliding window size. | INT64 | - |
| compressStride | Input | d in the formula, the interval between two compression sliding windows. | INT64 | - |
| selectBlockSize | Input | l' in the formula, the select block size. | INT64 | - |
| selectBlockCount | Input | The number of selections by topK in the formula. | INT64 | - |
| attentionOut | Output | attentionOut in the formula; except for the last dimension, the output shape follows query, and the last dimension takes the headDim of value. | BFLOAT16, FLOAT16 | ND |
| topkIndicesOut | Output | topkIndices in the formula. | INT32 | - |

## Constraints

- When this API is used with PyTorch, the versions of CANN-related packages and PyTorch-related packages must match.
- compressBlockSize, compressStride, and selectBlockSize must be multiples of 16 and satisfy: `compressBlockSize >= compressStride && selectBlockSize >= compressBlockSize && selectBlockSize % compressStride == 0`
- compressBlockSize: 16-aligned, supported up to 128.
- compressStride: 16-aligned, supported up to 64.
- selectBlockSize: 16-aligned, supported up to 128.
- selectBlockCount: supports [1~32]. `selectBlockCount <= min(SelSkv)`.
- actualSelSeqKvLenOptional must be in prefix-sum mode; and it must be passed in the TND format.
- Due to UB limitations, CmpSkv must satisfy the following constraint: `CmpSkv <= 14000`.
- `SelSkv = CeilDiv(CmpSkv, selectBlockSize // compressStride)`.
- layoutOptional currently supports only TND.
- The data types of query, key, and value inputs must be the same.
- The batchSizes of query, key, and value inputs must be equal.
- The headDims of query, key, and value inputs must satisfy: `qD == kD && kD >= vD`.
- The inputLayouts of query, key, and value inputs must be the same.
- If the headNum of the query input is N1 and the headNum of the key and value inputs is N2, `N1 >= N2 && N1 % N2 == 0`.
- Let `G = N1 / N2`. G must satisfy the following constraints: `G < 128 && 128 % G == 0`.
- The usage of attenMask and topkMask must comply with the paper description.

## Calling Description

| Calling Method | Calling Sample | Description |
|----------------|----------------|------------------------------------------------------|
| aclnn call | [test_aclnn_nsa_compress_attention](./examples/test_aclnn_nsa_compress_attention.cpp) | Non-TND scenario: the NsaCompressAttention operator is called through the [aclnnNsaCompressAttention](docs/aclnnNsaCompressAttention_en.md) API. |
