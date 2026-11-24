# CompressorGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: CompressorGrad is the backward operator of the Compressor operator. It computes the gradients of the input $X$, the weights $W^{KV}$/$W^{Gate}$, and the position encoding $Ape$. The forward pass exports softmax\_score (grouped softmax result) and kv (the raw KV organized by compression window, i.e. kv\_state before softmax weighting) as intermediate results when gradEnabled is true, which serve as the inputs of this operator. The main computation process is as follows:
    1. Computes the Hadamard product backward block by block: Multiplies the upstream gradient $dC$ with softmax\_score and kv element-wise to obtain $dK$ and $dS^\prime$;
    2. Computes the softmax backward: Performs the softmax backward on $dS^\prime$ along the compression axis to obtain $dZ$;
    3. Computes the APE gradient: Accumulates $dZ$ by token position to obtain $dApe$;
    4. Computes the matmul backward: Performs the matrix multiplication backward on $dK$, $dZ$, and the weights to obtain $dX$, $dW^{KV}$, and $dW^{Gate}$.

- Calculation formulas:

    1. Computes the Hadamard product backward, where $N$ is the total number of compression blocks, $i$ is the compression block index, $dC_i$ is the upstream gradient of the $i$-th block, and $S_i$ and $K_i$ are the $i$-th block of softmax\_score and kv, respectively:

        $$
        dK_i = dC_i \odot S_i,~ i=1,\cdots,N
        $$

        $$
        dS^\prime_i = dC_i \odot K_i,~ i=1,\cdots,N
        $$

    2. Computes the softmax backward (summing along the compression axis), where $k$ is the row index within a block:

        $$
        dZ_i = S_i \odot \left(dS^\prime_i - \sum_{k=1}^{coff \cdot cmp\_ratio} \left(S_i \odot dS^\prime_i\right)_{k,:}\right),~ i=1,\cdots,N
        $$

    3. Computes the APE gradient, where $pos$ is the global position of the token corresponding to each row of $dZ$:

        $$
        dApe = ScatterAdd\left(dZ,~ pos \% cmp\_ratio\right)
        $$

        When coff=2, the second dimension of softmax\_score, kv, and the backward intermediate $dZ$ is interleaved row by row with row index $j = r \times coff + e$ ($r$ is the row index of the token within the compression block; $e=0$ for the prev slot and $e=1$ for the cur slot); $dApe$ has shape $[cmp\_ratio, coff \cdot D]$ and is accumulated from $dZ[r \cdot coff+e, d]$ into $dApe[pos\%cmp\_ratio, e \cdot D+d]$, that is, the prev slot is accumulated into columns $[0,D)$ and the cur slot into columns $[D,2D)$; the prev slot of the first block contains cache rows and does not contribute.

    4. Computes the matrix multiplication backward, where $dNewKv$ and $dNewScore$ are the results of mapping $dK$ and $dZ$ back to global token rows by compression block (when coff=2, the prev/cur halves correspond to the token rows of the previous block and the current block respectively, matching the forward $W^{aKV}$/$W^{bKV}$):

        $$
        dX = dNewKv @ W^{KV} + dNewScore @ W^{Gate}
        $$

        $$
        dW^{KV} = dNewKv^T @ X,~ dW^{Gate} = dNewScore^T @ X
        $$
## Parameter Description

| Parameter                   | Input/Output/Attribute | Description                                                          | Data Type      | Data Format |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| x | Input | $X$ in the formulas. The raw data of the forward input. | FLOAT16, BFLOAT16 | ND         |
| wkv | Input | $W^{KV}$ in the formulas. The kv compression weight of the forward pass.  | FLOAT16, BFLOAT16 | ND |
| wgate | Input | $W^{Gate}$ in the formulas. The gate compression weight of the forward pass. | FLOAT16, BFLOAT16 | ND |
| d\_cmp\_kv | Input | $dC$ in the formulas. The upstream gradient of the forward output cmp\_kv. | FLOAT16, BFLOAT16 | ND         |
| softmax\_score | Input | $S$ in the formulas. The grouped softmax intermediate result exported by the forward pass when gradEnabled is true. | FLOAT32     | ND         |
| kv | Input | $K$ in the formulas. The raw KV intermediate result exported by the forward pass when gradEnabled is true (before softmax weighting, organized by compression window, and stored separately from softmax\_score). | FLOAT32       | ND         |
| cu\_seqlens | Optional input | Number of valid tokens in different batches.<br>Required when the shape of x is [T,H]; must be empty when the shape of x is [B,S,H]. | INT32 | ND         |
| seqused | Optional input | Number of tokens that actually participate in compression in different batches.<br>If set to None, the value is the same as the Sequence Length of each batch. | INT32          | ND         |
| start\_pos | Optional input | Starting position of the computation.<br>If set to None, the computation starts from 0. | INT32          | ND         |
| cmp\_ratio | Attribute | Used for sparse computation. Indicates the data compression ratio. The value is an integer in the range [2, 128]. | INT32          | -         |
| coff | Optional attribute | Indicates whether to perform overlap data rearrangement. <br>coff=1: overlap data rearrangement is not required; coff=2: overlap data rearrangement is required.<br>The default value is 1. | INT32          | -         |
| enable\_ape | Optional attribute | Indicates whether to compute the dApe gradient.<br>The default value is true. When it is false, the dApe computation is skipped and the content of dApe is invalid. | BOOL          | -         |
| d\_x | Output | $dX$ in the formulas. The gradient of the input x. | FLOAT16, BFLOAT16         | ND          |
| d\_wkv | Output | $dW^{KV}$ in the formulas. The gradient of the weight wkv. | FLOAT16, BFLOAT16         | ND          |
| d\_wgate | Output | $dW^{Gate}$ in the formulas. The gradient of the weight wgate. | FLOAT16, BFLOAT16         | ND          |
| d\_ape | Output | $dApe$ in the formulas. The gradient of the APE position encoding. When enable\_ape is false, the operator skips the computation and the content is invalid. | FLOAT32         | ND          |

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: enable_ape does not take effect, and dApe is still computed normally.

## Constraints

- Meanings of the x parameter dimensions: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, D (Head Dim) indicates the minimum unit size of the hidden layer, and T indicates the sum of sequence lengths of input samples across all batches.
- Supported layouts: two layouts, BSH (x is [B,S,H]) and TH (x is [T,H] variable-length sequences). The Atlas A2 products and A3 products share the same operator implementation.
- Data type constraint: x, wkv, wgate, dCmpKv, dX, dWkv, and dWgate must use the same data type (all FLOAT16 or all BFLOAT16); softmaxScore, kv, and dApe are fixed to FLOAT32. Mixed data types are rejected during tiling.
- Input shape restrictions:
    - wkv supports the input shape [coff* D,H]
    - wgate supports the input shape [coff* D,H]
    - softmax\_score supports the input shape: [min(T,T//cmp_ratio+B), coff*cmp_ratio, D] when BS is merged into one axis, and [B,ceil(S/cmp_ratio),coff*cmp_ratio,D] when BS is not merged
    - kv supports the input shape: same as softmax\_score
    - d\_cmp\_kv supports the input shape: [min(T,T//cmp_ratio+B),D] when BS is merged into one axis, and [B,ceil(S/cmp_ratio),D] when BS is not merged
    - d\_x supports the output shape: same as x, which is [T,H] when BS is merged into one axis and [B,S,H] when BS is not merged
    - d\_wkv and d\_wgate support the output shape [coff* D,H]
    - d\_ape supports the output shape [cmp_ratio,coff* D]
    - start\_pos supports the input shape [B,]
    - If the dimensions of x use BS merged into one axis, that is, the input shape of x is [T,H]
        - The input shape of cu\_seqlens must be [B+1,]. Each element in this parameter indicates the total number of tokens in the current batch and all preceding batches, that is, a prefix sum. Therefore, the value of each element must be greater than or equal to that of the preceding element, and the first element must be 0.
        - seqused supports the input shape [B,]. The number of valid tokens in each batch must be less than or equal to the corresponding Sequence Length, that is, seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n], and must not be less than 0.
    - If the dimensions of x do not use BS merged into one axis, that is, the input shape of x is [B,S,H]
        - cu\_seqlens: the parameter must be empty.
        - seqused supports the input shape [B,]. The number of valid tokens in each batch must be less than or equal to the corresponding Sequence Length, that is, seqused[n] <= S, and must not be less than 0.

- Input value range restrictions:
    - This API supports generalization of B and S, with the following scenario restrictions:
        - **Empty tensors with B, S, or T of 0 are not supported**: Different from the forward Compressor operator, none of the inputs/outputs of CompressorGrad support empty tensors. shapeSize must be greater than 0.
        - In some long-sequence scenarios, an excessively large computation workload may cause an error indicating that NPU memory is exceeded. Note: The computation workload is affected by the input shape of x. The larger the values, the greater the computation workload.
- Input attribute restrictions:
    - Supports D of 128/512.
    - Supports H from 1K to 10K, aligned to 512.
    - Supports coff of 1/2.
    - Supports cmp\_ratio from 2 to 128.
- enableApe description:
    - enable\_ape defaults to true, and dApe is generated with shape [cmp\_ratio, coff\*D]. When it is false, the dApe computation is skipped; d\_ape remains a required output but its content is invalid. On the PyTorch side, the shared backward API controls whether dApe is generated through the `enable_ape` parameter.

## Invocation Description

| Invocation Method | Sample Code                                                                       | Description                                                                    |
| --------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| aclnn API | -| Calls the CompressorGrad operator through [aclnnCompressorGrad](docs/aclnnCompressorGrad_en.md). |
| PyTorch API | - | Calls the Compressor operator through [cann_ops_transformer.compressor](../compressor/docs/torchapi_compressor_en.md) or [compressor_v2](../compressor/docs/torchapi_compressor_v2_en.md). CompressorGrad is automatically called in the backward pass through its autograd. V2 additionally returns d_ape when ape requires gradients. |
