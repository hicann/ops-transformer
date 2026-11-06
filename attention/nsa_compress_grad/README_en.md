# NsaCompressGrad

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |

## Function Description

- Operator function: Backward computation of the aclnnNsaCompress operator.

- Computation formula:
  The forward computation formula of selective attention is as follows:

    $$
    \text{dw} = \text{dk\_cmp} \cdot K^\top
    $$

    $$
    \text{dk} = W^\top \cdot \text{dk\_cmp}
    $$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| outputGrad | Input | The backward gradient of the forward operator output; supported shape: [T, N, D]. | BFLOAT16, FLOAT16 | ND |
| input | Input | Tensor to be compressed; supported shape: [T, N, D]. | BFLOAT16, FLOAT16 | ND |
| weight | Input | Compression weight; the shape is [compressBlockSize, N], in a broadcast relationship with input. | BFLOAT16, FLOAT16 | ND |
| actSeqLenOptional | Input | The S size corresponding to each batch. Currently, a parameter that is not nullptr must be explicitly passed in. When the length of a single batch is S, [S] is passed in. | INT64 | ND |
| compressBlockSize | Input | Compression sliding window size. | INT64 | - |
| compressStride | Input | Interval size between two compression sliding windows. | INT64 | - |
| actSeqLenType | Input | Sequence length type. The value is 0 or 1. 0 indicates the cumulative sum of the sequence lengths of batches, and 1 indicates the independent sequence length of each batch. Currently, 0 is recommended. For example, when the batch lengths are 64 and 80, the value 0 corresponds to [64, 144], and the value 1 corresponds to [64, 80]. | INT64 | - |
| layoutOptional | Input | Input data layout format; only "TND" is supported. When nullptr is passed in, the default value "TND" is used. | String | - |
| inputGrad | Output | The gradient of input; the shape is the same as input. | BFLOAT16, FLOAT16 | ND |
| weightGrad | Output | The gradient of weight; the shape is the same as weight. | BFLOAT16, FLOAT16 | ND |

## Constraints

- compressBlockSize and compressStride must be multiples of 16, and compressBlockSize >= compressStride.

## Calling Description

| Calling Method | Calling Sample | Description |
|----------------|----------------|------------------------------------------------------|
| aclnn call | [test_aclnn_nsa_compress_grad](./examples/test_aclnn_nsa_compress_grad.cpp) | Non-TND scenario: the NsaCompressGrad operator is called through the [aclnnNsaCompressGrad](docs/aclnnNsaCompressGrad_en.md) API. |
