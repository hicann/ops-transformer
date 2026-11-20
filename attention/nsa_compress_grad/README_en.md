# NsaCompressGrad

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |

## Function Description

- Operator function: Backward computation of the aclnnNsaCompress operator.

- Computation formula:

  Let \(a_b\) be the start of batch \(b\) on input, \(o_b\) be the start of that batch on outputGrad, \(s\) be compressStride, and \(L\) be compressBlockSize. \(i\) iterates only over the complete windows of that batch.

    $$
    \text{weightGrad}[r,n] = \sum_{b,i,d} \text{outputGrad}[o_b+i,n,d] \cdot \text{input}[a_b+i\cdot s+r,n,d], \quad 0 \le r < L
    $$

    $$
    \text{inputGrad}[a_b+t,n,d] = \sum_{i} \text{outputGrad}[o_b+i,n,d] \cdot \text{weight}[t-i\cdot s,n]
    $$

  inputGrad sums only over complete windows that satisfy \(0 \le t-i\cdot s < L\). The weight gradient is reduced along the feature dimension, then accumulated across batches and windows.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| outputGrad | Input | Backward gradient of the forward operator output. The shape is [T_compressed, N, D]. T_compressed = Σ_b max(0, floor((S_b - compressBlockSize) / compressStride) + 1), where S_b is the sequence length of batch b. | BFLOAT16, FLOAT16 | ND |
| input | Input | Tensor to be compressed. The shape is [T, N, D]. T is the sum of the sequence lengths of all batches and is not necessarily equal to T_compressed. | BFLOAT16, FLOAT16 | ND |
| weight | Input | Compression weight; the shape is [compressBlockSize, N], in a broadcast relationship with input. | BFLOAT16, FLOAT16 | ND |
| actSeqLenOptional | Input | The S size corresponding to each batch. Currently, a parameter that is not nullptr must be explicitly passed in. When the length of a single batch is S, [S] is passed in. | INT64 | ND |
| compressBlockSize | Input | Compression sliding window size. | INT64 | - |
| compressStride | Input | Interval size between two compression sliding windows. | INT64 | - |
| actSeqLenType | Input | Sequence length type. The value is 0 or 1. 0 indicates the cumulative sum of the sequence lengths of batches, and 1 indicates the independent sequence length of each batch. Currently, 0 is recommended. For example, when the batch lengths are 64 and 80, the value 0 corresponds to [64, 144], and the value 1 corresponds to [64, 80]. | INT64 | - |
| layoutOptional | Input | Input data layout format; only "TND" is supported. When nullptr is passed in, the default value "TND" is used. | String | - |
| inputGrad | Output | The gradient of input. The shape is [T, N, D], the same as input. | BFLOAT16, FLOAT16 | ND |
| weightGrad | Output | The gradient of weight; the shape is the same as weight. | BFLOAT16, FLOAT16 | ND |

## Constraints

- compressBlockSize and compressStride must be multiples of 16, and compressBlockSize >= compressStride.
- The last dimension D of outputGrad and input must be a multiple of 16.

## Calling Description

| Calling Method | Calling Sample | Description |
|----------------|----------------|------------------------------------------------------|
| aclnn call | [test_aclnn_nsa_compress_grad](./examples/test_aclnn_nsa_compress_grad.cpp) | TND scenario: the NsaCompressGrad operator is called through the [aclnnNsaCompressGrad](docs/aclnnNsaCompressGrad_en.md) API. |
