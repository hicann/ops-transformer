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

    The forward computation formula of Nsa Compress is as follows:

$$
\tilde{K}_t^{\text{cmp}} = f_K^{\text{cmp}}(k_{:t}) = \left\{ \varphi(k_{id+1:id+l}) \bigg| 0 \leq i \leq \left\lfloor \frac{t-l}{d} \right\rfloor \right\}
$$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| input | Input | Tensor to be compressed; supported shape: [T, N, D]. | BFLOAT16, FLOAT16 | ND |
| weight | Input | Compression weight; supported shape: [compressBlockSize, N], in a broadcast relationship with input. | BFLOAT16, FLOAT16 | ND |
| actSeqLenOptional | Input | The S size corresponding to each batch. | INT64 | ND |
| layoutOptional | Input | Input data layout format; supports BSH, SBH, BSND, BNSD, and TND. Currently, only TND is supported. | String | - |
| compressBlockSize | Input | Compression sliding window size. | INT64 | - |
| compressStride | Input | Interval size between two compression sliding windows. | INT64 | - |
| actSeqLenType | Input | Sequence length type. The value 0 indicates the cumsum result, and 1 indicates the sequence size of each batch. Currently, only 0 is supported. | INT64 | - |
| output | Output | Compression result; supported shape: [T, N, D]. | BFLOAT16, FLOAT16 | ND |

## Constraints

- When this API is used with PyTorch, the versions of CANN-related packages and PyTorch-related packages must match.
- input and weight must be in a broadcast relationship, `input.shape[1] = weight.shape[1]`; empty input and weight are not supported.
- actSeqLenType currently supports only the value 0, which means actSeqLenOptional must be in prefix-sum mode.
- actSeqLenOptional cannot currently be empty.
- layoutOptional currently supports only TND. In this case, input.shape[0] must equal actSeqLenOptional[-1].
- `input.shape[1] = weight.shape[1]`, and it must be less than or equal to 128.
- input.shape[2] must be a multiple of 16, with an upper limit of 256.
- `weight.shape[0] = compressBlockSize`, which must be a multiple of 16, with an upper limit of 128.
- compressStride must be a multiple of 16, and `compressBlockSize >= compressStride`.

## Calling Description

| Calling Method | Calling Sample | Description |
|----------------|----------------|------------------------------------------------------|
| aclnn call | [test_aclnn_nsa_compress](./examples/test_aclnn_nsa_compress.cpp) | TND scenario: the NsaCompress operator is called through the [aclnnNsaCompress](docs/aclnnNsaCompress_en.md) API. |
