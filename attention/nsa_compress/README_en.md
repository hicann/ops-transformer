# NsaCompress

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      ×     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: In the training scenario, the operator uses the Nsa Compress algorithm to reduce the attention computation for long contexts and achieves compression along the KV sequence dimension.

- Computation formula:

    Each batch computes complete windows with W=compressBlockSize and S=compressStride as follows:

$$
\mathrm{output}_{i,n,d}=\sum_{j=0}^{W-1}\mathrm{input}_{iS+j,n,d}\cdot\mathrm{weight}_{j,n}
$$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| input | Input | Tensor to be compressed; supported shape: [T, N, D]. | BFLOAT16, FLOAT16 | ND |
| weight | Input | Compression weight; supported shape: [compressBlockSize, N], broadcast along D within each window. | BFLOAT16, FLOAT16 | ND |
| actSeqLenOptional | Input | Cumulative sequence lengths of all batches. | INT64 | ND |
| layoutOptional | Input | Input data layout format; only TND is supported. | String | - |
| compressBlockSize | Input | Compression sliding window size. | INT64 | - |
| compressStride | Input | Interval size between two compression sliding windows. | INT64 | - |
| actSeqLenType | Input | Sequence length type. Only 0 (cumulative lengths) is supported. | INT64 | - |
| output | Output | Compression result with shape [T_compressed, N, D]. T_compressed is the total number of complete windows across batches. | BFLOAT16, FLOAT16 | ND |

## Constraints

- When this API is used with PyTorch, the versions of CANN-related packages and PyTorch-related packages must match.
- weight is broadcast along D within each window, `input.shape[1] = weight.shape[1]`; empty input and weight are not supported.
- actSeqLenType currently supports only the value 0, which means actSeqLenOptional must be in prefix-sum mode.
- actSeqLenOptional cannot currently be empty.
- layoutOptional currently supports only TND. In this case, input.shape[0] must equal actSeqLenOptional[-1].
- `input.shape[1] = weight.shape[1]`, and it must be less than or equal to 128.
- input.shape[2] must be a multiple of 16, with an upper limit of 256.
- `weight.shape[0] = compressBlockSize`, which must be a multiple of 16, with an upper limit of 128.
- compressStride must be a multiple of 16, and satisfy `16 <= compressStride <= compressBlockSize`.

## Calling Description

| Calling Method | Calling Sample | Description |
|----------------|----------------|------------------------------------------------------|
| aclnn call | [test_aclnn_nsa_compress](./examples/test_aclnn_nsa_compress.cpp) | TND scenario: the NsaCompress operator is called through the [aclnnNsaCompress](docs/aclnnNsaCompress_en.md) API. |
